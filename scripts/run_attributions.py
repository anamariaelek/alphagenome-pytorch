#!/usr/bin/env python
"""Integrated-gradients attributions for model-verified divergent splice sites.

For every (species, site, tissue) target in the manifest, attributes the
fine-tuned model's splice-usage logit at that site and condition back to the
input DNA.  The condition is ``<Tissue>_<timepoint>`` at the target's timepoint
of maximum observed divergence (``tp_max_departure``), looked up in the
species' own ``usage.json`` condition table -- the same name->index mapping the
training and evaluation code uses, so the attributed output is the same head
channel that produced the stored predictions.

Windows are 131,072 bp (the model's input length) **centred on the site**,
which is not how the evaluation windows were tiled; ``--verify`` quantifies the
resulting difference by comparing the centred-window prediction against the
stored evaluation prediction for the same site, condition and species.

Integrated gradients (Sundararajan et al. 2017) are computed against
dinucleotide-shuffled baselines (Altschul-Erikson shuffle, the DeepLIFT/BPNet
convention) and averaged over ``--baselines`` independent shuffles; a zeros
baseline is available via ``--baseline zeros`` for comparison.

Outputs, per target, an ``.npz`` with base-resolution attributions in a
+/-``--report-bp`` window around the site plus a 128-bp-binned profile over the
whole input window, and one summary row per target in ``attr_summary.csv``.

Usage:
    python run_attributions.py --manifest attr_manifest_human.csv \
        --checkpoint $BM --out-dir attributions --verify 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfaidx
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "scripts"))
sys.path.insert(0, str(CODE / "src"))

from evaluate_splice import build_model, load_config, resolve_checkpoint  # noqa: E402
from alphagenome_pytorch.attributions import (  # noqa: E402
    _autocast_context,
    integrated_gradients,
    summarize_attributions,
    usage_target,
)
from alphagenome_pytorch.utils.sequence import sequence_to_onehot  # noqa: E402

SPECIES_DIR = {
    "human": "Homo_sapiens",
    "mouse": "Mus_musculus",
    "rat": "Rattus_norvegicus",
    "rabbit": "Oryctolagus_cuniculus",
    "opossum": "Monodelphis_domestica",
    "macaque": "Macaca_mulatta",
    "chicken": "Gallus_gallus",
}
BIN = 128  # bp per bin for the whole-window coarse profile


# ── dinucleotide-preserving shuffle ────────────────────────────────────────
def dinuc_shuffle_tokens(tokens: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Altschul-Erikson dinucleotide shuffle of an integer token sequence.

    Preserves the exact dinucleotide (and hence mononucleotide) composition.
    Implements the DeepLIFT formulation: shuffle each character's successor
    list while keeping its last entry fixed, then walk the Eulerian path.
    """
    n = len(tokens)
    chars, idx = np.unique(tokens, return_inverse=True)
    next_inds: list[list[int]] = [[] for _ in chars]
    for i in range(n - 1):
        next_inds[idx[i]].append(i + 1)
    for t in range(len(chars)):
        lst = next_inds[t]
        if len(lst) > 1:
            head = rng.permutation(len(lst) - 1)
            next_inds[t] = [lst[i] for i in head] + [lst[-1]]
    counters = [0] * len(chars)
    out = np.empty(n, dtype=tokens.dtype)
    pos = 0
    out[0] = tokens[pos]
    for j in range(1, n):
        t = idx[pos]
        pos = next_inds[t][counters[t]]
        counters[t] += 1
        out[j] = tokens[pos]
    return out


def onehot_to_tokens(oh: np.ndarray) -> np.ndarray:
    """One-hot (S,4) -> tokens 0..3, with 4 for all-zero (N) positions."""
    tok = oh.argmax(axis=1).astype(np.int64)
    tok[oh.sum(axis=1) == 0] = 4
    return tok


def tokens_to_onehot(tok: np.ndarray) -> np.ndarray:
    oh = np.zeros((len(tok), 4), dtype=np.float32)
    m = tok < 4
    oh[np.where(m)[0], tok[m]] = 1.0
    return oh


# ── model / data plumbing ──────────────────────────────────────────────────
def load_model(checkpoint: str, device: torch.device):
    ckpt_path, cfg_path = resolve_checkpoint(checkpoint)
    cfg = load_config(cfg_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(cfg, ckpt, device)
    model.eval()
    return model, cfg


def species_table(cfg: dict, data_dir: Path, genomes_dir: Path) -> dict:
    """species name -> {organism_index, fasta path, condition_labels}."""
    out = {}
    for spec in cfg["species_specs"]:
        name = spec["name"]
        sub = SPECIES_DIR[name]
        meta = json.load(open(data_dir / sub / "usage.json"))
        fa = Path(spec["genome"])
        if not fa.exists():  # config paths point at the training user's home
            fa = genomes_dir / "/".join(fa.parts[-3:])
        out[name] = dict(org=int(spec["organism_index"]), fasta=str(fa),
                         conds=meta["condition_labels"])
    return out


def window(fasta: pyfaidx.Fasta, chrom: str, pos: int, seq_len: int):
    """One-hot window of ``seq_len`` bp centred on 0-based ``pos``."""
    half = seq_len // 2
    start = pos - half
    clen = len(fasta[chrom])
    lpad = max(0, -start)
    rpad = max(0, start + seq_len - clen)
    seq = str(fasta[chrom][max(0, start): min(clen, start + seq_len)])
    if lpad or rpad:
        seq = "N" * lpad + seq + "N" * rpad
    oh = sequence_to_onehot(seq).astype(np.float32)
    assert oh.shape == (seq_len, 4), oh.shape
    return oh, half  # site index within the window


def predict_at(model, x: torch.Tensor, org: int, site: int, cond: int, device):
    with torch.no_grad(), _autocast_context(model, device):
        out = model.forward(x, torch.tensor([org], device=device),
                            resolutions=(1,), channels_last=True)
    lg = out["splice_sites_usage"]["logits"][0, site, cond].float().item()
    return lg, float(1.0 / (1.0 + np.exp(-lg)))


# ── verification ───────────────────────────────────────────────────────────
def verify(model, cfg, spt, man: pd.DataFrame, traj: Path, n: int, device):
    """Compare centred-window predictions with the stored evaluation ones."""
    T = pd.read_parquet(traj)
    print("trajectory table columns:", list(T.columns))
    keys = {c.lower(): c for c in T.columns}
    c_sp, c_ti = keys.get("species"), keys.get("tissue")
    c_tp = keys.get("timepoint", keys.get("tp"))
    c_st = keys.get("site", keys.get("site_key"))
    c_pr = keys.get("pred", keys.get("sse_pred", keys.get("usage_pred")))
    if c_pr is None and "kind" in keys and "value" in keys:
        kinds = sorted(T[keys["kind"]].unique())
        print("kinds in table:", kinds)
        pk = [k for k in kinds if str(k).lower().startswith("pred")]
        T = T[T[keys["kind"]] == pk[0]].copy()
        c_pr = keys["value"]
    rows = []
    sub = man.groupby("species", group_keys=False).head(max(1, n // man.species.nunique()))
    for _, r in sub.iterrows():
        s = spt[r.species]
        cname = f"{r.Tissue}_{int(r.tp_max_departure)}"
        if cname not in s["conds"]:
            rows.append(dict(species=r.species, site=r.site, cond=cname, note="condition absent"))
            continue
        cidx = int(s["conds"][cname])
        fa = pyfaidx.Fasta(s["fasta"], as_raw=True, sequence_always_upper=True)
        oh, si = window(fa, str(r.Chromosome), int(r.Position), cfg["sequence_length"])
        x = torch.from_numpy(oh)[None].to(device)
        t0 = time.time()
        lg, pr = predict_at(model, x, s["org"], si, cidx, device)
        dt = time.time() - t0
        stored = np.nan
        if all([c_sp, c_ti, c_tp, c_st, c_pr]):
            m = T[(T[c_sp] == r.species) & (T[c_ti] == r.Tissue)
                  & (T[c_st].astype(str) == str(r.site))
                  & (T[c_tp].astype(int) == int(r.tp_max_departure))]
            if len(m):
                stored = float(m[c_pr].iloc[0])
        rows.append(dict(species=r.species, site=r.site, tissue=r.Tissue, cond=cname,
                         cond_idx=cidx, logit=lg, pred_centred=pr, pred_stored=stored,
                         fwd_s=dt))
    V = pd.DataFrame(rows)
    print("\n" + V.to_string(index=False))
    ok = V.dropna(subset=["pred_centred", "pred_stored"]) if "pred_stored" in V else V.iloc[:0]
    if len(ok) >= 3:
        d = (ok.pred_centred - ok.pred_stored).abs()
        print("\ncentred vs stored prediction: n=%d  r=%.3f  median |diff|=%.4f  max %.4f"
              % (len(ok), ok.pred_centred.corr(ok.pred_stored), d.median(), d.max()))
    return V


# ── attribution ────────────────────────────────────────────────────────────
def attribute(model, cfg, spt, man, out_dir: Path, args, device):
    seq_len = cfg["sequence_length"]
    rng = np.random.default_rng(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, fastas = [], {}
    nb = seq_len // BIN
    for k, (_, r) in enumerate(man.iterrows(), 1):
        s = spt[r.species]
        cname = f"{r.Tissue}_{int(r.tp_max_departure)}"
        if cname not in s["conds"]:
            rows.append(dict(target=f"{r.species}|{r.site}|{cname}", species=r.species,
                             note="condition absent"))
            continue
        cidx = int(s["conds"][cname])
        if r.species not in fastas:
            fastas[r.species] = pyfaidx.Fasta(s["fasta"], as_raw=True,
                                              sequence_always_upper=True)
        oh, si = window(fastas[r.species], str(r.Chromosome), int(r.Position), seq_len)
        x = torch.from_numpy(oh)[None].to(device)
        tfn = usage_target(si, cidx)
        lg, pr = predict_at(model, x, s["org"], si, cidx, device)

        tok = onehot_to_tokens(oh)
        acc = torch.zeros_like(x)
        t0 = time.time()
        nbase = 1 if args.baseline == "zeros" else args.baselines
        for b in range(nbase):
            if args.baseline == "zeros":
                base = torch.zeros_like(x)
            else:
                base = torch.from_numpy(tokens_to_onehot(
                    dinuc_shuffle_tokens(tok, rng)))[None].to(device)
            acc += integrated_gradients(model, x, s["org"], tfn, baseline=base,
                                        steps=args.steps)
        attr = (acc / nbase)[0].float().cpu().numpy()           # (S, 4)
        contrib = summarize_attributions(
            torch.from_numpy(attr)[None], torch.from_numpy(oh)[None])[0].numpy()
        dt = time.time() - t0

        w = args.report_bp
        lo, hi = si - w, si + w + 1
        prof = contrib[: nb * BIN].reshape(nb, BIN).sum(axis=1)
        name = f"{r.species}_{str(r.site).replace(':', '_')}_{cname}"
        np.savez_compressed(
            out_dir / f"{name}.npz",
            attr=attr[lo:hi].astype(np.float32),
            contrib=contrib[lo:hi].astype(np.float32),
            profile_128bp=prof.astype(np.float32),
            onehot=oh[lo:hi].astype(np.int8),
            offsets=np.arange(-w, w + 1, dtype=np.int32),
            meta=json.dumps(dict(species=r.species, site=str(r.site), tissue=r.Tissue,
                                 condition=cname, condition_idx=cidx,
                                 organism_index=s["org"], position=int(r.Position),
                                 chrom=str(r.Chromosome), strand=str(r.Strand),
                                 human_anchor=str(r.human_anchor), symbol=str(r.symbol),
                                 role=str(r.role), steps=args.steps,
                                 baselines=nbase, baseline=args.baseline,
                                 logit=lg, pred=pr)),
        )
        cw = contrib[lo:hi]
        rows.append(dict(
            target=name, species=r.species, site=str(r.site), tissue=r.Tissue,
            symbol=r.symbol, human_anchor=r.human_anchor, role=r.role,
            div_class=r.div_class, lineage=r.lineage, condition=cname, cond_idx=cidx,
            logit=lg, pred=pr,
            abs_sum_window=float(np.abs(cw).sum()),
            abs_sum_total=float(np.abs(contrib).sum()),
            frac_in_window=float(np.abs(cw).sum() / max(1e-9, np.abs(contrib).sum())),
            peak_offset=int(np.argmax(np.abs(cw)) - w),
            peak_contrib=float(cw[np.argmax(np.abs(cw))]),
            secs=dt))
        if k % 5 == 0 or k == len(man):
            print(f"[{k}/{len(man)}] {name} {dt:.1f}s", flush=True)
            pd.DataFrame(rows).to_csv(out_dir / "attr_summary.csv", index=False)
    S = pd.DataFrame(rows)
    S.to_csv(out_dir / "attr_summary.csv", index=False)
    print("\nwrote", out_dir / "attr_summary.csv", len(S), "rows")
    return S


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--genomes-dir", required=True)
    p.add_argument("--out-dir", default="attributions")
    p.add_argument("--traj", default=None, help="candidate_trajectories.parquet (verify)")
    p.add_argument("--verify", type=int, default=0, help="verify N targets and exit")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--baselines", type=int, default=3)
    p.add_argument("--baseline", choices=["dinuc", "zeros"], default="dinuc")
    p.add_argument("--report-bp", type=int, default=512)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    man = pd.read_csv(args.manifest)
    if args.limit:
        man = man.head(args.limit)
    model, cfg = load_model(args.checkpoint, device)
    print("sequence_length", cfg["sequence_length"], "| usage_coord_base",
          cfg.get("usage_coord_base"), "| usage_delta_from_mean",
          cfg.get("usage_delta_from_mean"), flush=True)
    spt = species_table(cfg, Path(args.data_dir), Path(args.genomes_dir))

    if args.verify:
        verify(model, cfg, spt, man, Path(args.traj), args.verify, device)
        return
    attribute(model, cfg, spt, man, Path(args.out_dir), args, device)


if __name__ == "__main__":
    main()
