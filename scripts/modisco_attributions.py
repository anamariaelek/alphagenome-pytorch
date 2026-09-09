#!/usr/bin/env python
"""TF-MoDISco motif discovery on the human-branch attribution set.

An independent check on the 6-mer analysis in ``analyze_attributions.py``: instead
of ranking fixed-width k-mers by mean contribution, this clusters high-attribution
seqlets into motifs (CWMs) and asks whether the human (gained, dynamic) and static
partner sides carry different motif vocabularies.

Two runs, answering two different questions:

  joint    all 206 regions in one run, so both sides share one motif vocabulary.
           Each pattern's seqlets are then split by side and tested with Fisher's
           exact test against the pooled side split.  This is the direct analogue
           of the per-side k-mer comparison and the run to read for "does the
           k-mer result hold".
  by-side  one run per side, so each side's motifs emerge independently; patterns
           are matched across sides with tomtom to separate shared from
           side-specific motifs.

Discovered CWMs are matched with tomtom against the same published splicing
consensus set used for the k-mer annotation (``analyze_attributions.MOTIFS``), so
the two analyses speak the same motif vocabulary.

Outputs
  modisco_joint.h5 / modisco_human.h5 / modisco_partner.h5   raw modisco results
  modisco_patterns.csv          joint run: seqlet counts per side, Fisher p/q,
                                log2 odds, median offset from the splice site,
                                consensus, tomtom match
  modisco_patterns_by_side.csv  per-side runs, with the cross-side tomtom match
  modisco_motifs.png            joint patterns: CWM logo + human/partner seqlet
                                split per pattern
  modisco_sides.png             independently discovered patterns, side by side

Attribution is of the usage logit at the site in the tissue and timepoint of
maximum observed divergence; hypothetical contributions are the per-base
attributions mean-centred over A/C/G/T, matching the house
``hypothetical_contributions`` helper.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

import modiscolite

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from analyze_attributions import MOTIFS, seq_of  # same consensus vocabulary

try:
    from score_pairs_gw import style
except Exception:
    def style():
        plt.rcParams.update({"figure.dpi": 150, "font.size": 7,
                             "axes.spines.top": False, "axes.spines.right": False})

BASES = np.array(list("ACGT"))
IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "AC": "M", "AG": "R", "AT": "W",
         "CG": "S", "CT": "Y", "GT": "K", "ACG": "V", "ACT": "H", "AGT": "D",
         "CGT": "B", "ACGT": "N"}
SIDE_COL = {"human": "#1f78b4", "partner": "#e31a1c"}


# ---------------------------------------------------------------- data loading
def load_side(attr_dir: Path):
    """Return (one_hot, hyp_contribs, offsets, meta) stacked over targets, per side."""
    out = {}
    for f in sorted(attr_dir.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        side = "human" if m["role"] == "focal" else "partner"
        oh = z["onehot"].astype(np.float32)
        attr = z["attr"].astype(np.float32)
        hyp = attr - attr.mean(axis=-1, keepdims=True)
        out.setdefault(side, {"oh": [], "hyp": [], "meta": []})
        out[side]["oh"].append(oh)
        out[side]["hyp"].append(hyp)
        out[side]["meta"].append(dict(file=f.name, **m))
        offsets = z["offsets"]
    for side, d in out.items():
        d["oh"] = np.stack(d["oh"])
        d["hyp"] = np.stack(d["hyp"])
        d["offsets"] = offsets
    return out


# -------------------------------------------------------------------- modisco
def run_modisco(oh, hyp, args, tag: str):
    print("  modisco[%s]: %d regions x %d bp" % (tag, oh.shape[0], oh.shape[1]),
          flush=True)
    pos, neg = modiscolite.tfmodisco.TFMoDISco(
        one_hot=oh, hypothetical_contribs=hyp,
        sliding_window_size=args.window, flank_size=args.flank,
        target_seqlet_fdr=args.fdr, min_metacluster_size=args.min_metacluster,
        max_seqlets_per_metacluster=args.max_seqlets,
        n_leiden_runs=args.n_leiden,
        trim_to_window_size=args.trim_window,
        initial_flank_to_add=args.flank_to_add,
        min_ic_in_window=args.min_ic, min_ic_windowsize=args.min_ic_window,
        verbose=False)
    return pos, neg


def read_h5(path: Path):
    """Flatten a modisco h5 into a list of pattern dicts."""
    import h5py
    pats = []
    with h5py.File(path, "r") as h:
        for grp in ("pos_patterns", "neg_patterns"):
            if grp not in h:
                continue
            for name in sorted(h[grp], key=lambda s: int(s.split("_")[-1])):
                g = h[grp][name]
                s = g["seqlets"]
                sign = "pos" if grp == "pos_patterns" else "neg"
                pats.append(dict(
                    sign=sign,
                    name="%s_%s" % (sign, name),
                    ppm=g["sequence"][:], cwm=g["contrib_scores"][:],
                    n_seqlets=int(s["n_seqlets"][0]) if "n_seqlets" in s
                    else len(s["start"]),
                    ex=s["example_idx"][:], start=s["start"][:], end=s["end"][:]))
    return pats


# ------------------------------------------------------------ motif utilities
def trim(cwm, ppm, frac: float = 0.3):
    """Trim a pattern to the span carrying the bulk of its contribution."""
    w = np.abs(cwm).sum(axis=1)
    if w.max() <= 0:
        return cwm, ppm, 0, len(w)
    keep = np.where(w >= frac * w.max())[0]
    lo, hi = int(keep[0]), int(keep[-1]) + 1
    return cwm[lo:hi], ppm[lo:hi], lo, hi


def per_position_ic(ppm, pseudo: float = 1e-3):
    p = np.clip(ppm, pseudo, None)
    p = p / p.sum(axis=1, keepdims=True)
    return (p * np.log2(p / 0.25)).sum(axis=1)


def consensus(ppm, thresh: float = 0.25) -> str:
    out = []
    for row in ppm:
        sel = "".join(BASES[np.argsort(-row)][: max(1, int((row > thresh).sum()))])
        out.append(IUPAC.get("".join(sorted(sel)), "N"))
    return "".join(out)


def consensus_informative(ppm, min_ic: float = 0.25, pad: int = 1) -> str:
    """Consensus over the informative span only, so the string stays readable."""
    ic = per_position_ic(ppm)
    keep = np.where(ic >= min_ic)[0]
    if not len(keep):
        return consensus(ppm)
    lo, hi = max(0, keep[0] - pad), min(len(ic), keep[-1] + 1 + pad)
    return consensus(ppm[lo:hi])


IUPAC_SETS = {c: set(b) for b, c in IUPAC.items()}


def parse_consensus_pattern(pat: str) -> list[set]:
    """A MOTIFS regex -> one allowed-base set per position."""
    core = pat.replace("$", "").replace("(", "").replace(")", "").split("|")[0]
    out, i = [], 0
    while i < len(core):
        if core[i] == "[":
            j = core.index("]", i)
            out.append(set(core[i + 1:j]))
            i = j + 1
        else:
            out.append({core[i]})
            i += 1
    return out


def iupac_match(cons: str) -> str:
    """Consensus motifs compatible with this pattern's IUPAC string, at any offset."""
    obs = [IUPAC_SETS.get(c, set("ACGT")) for c in cons]
    hits = []
    for name, pat in MOTIFS.items():
        want = parse_consensus_pattern(pat)
        if len(want) > len(obs):
            continue
        if any(all(want[k] & obs[i + k] for k in range(len(want)))
               for i in range(len(obs) - len(want) + 1)):
            hits.append(name)
    return ";".join(hits)


def write_meme(path: Path, motifs: list[tuple[str, np.ndarray]]):
    with open(path, "w") as fh:
        fh.write("MEME version 4\n\nALPHABET= ACGT\n\n"
                 "strands: + -\n\nBackground letter frequencies\n"
                 "A 0.25 C 0.25 G 0.25 T 0.25\n\n")
        for name, ppm in motifs:
            p = np.clip(ppm, 1e-4, None)
            p = p / p.sum(axis=1, keepdims=True)
            fh.write("MOTIF %s\n" % name.replace(" ", "_"))
            fh.write("letter-probability matrix: alength= 4 w= %d nsites= 100 E= 0\n"
                     % len(p))
            for row in p:
                fh.write(" %.6f %.6f %.6f %.6f\n" % tuple(row))
            fh.write("\n")


def consensus_db() -> list[tuple[str, np.ndarray]]:
    """Turn the IUPAC consensus vocabulary into PPMs for tomtom."""
    db = []
    for name, pat in MOTIFS.items():
        core = pat.replace("$", "").replace("(", "").replace(")", "")
        core = core.split("|")[0]
        rows, i = [], 0
        while i < len(core):
            if core[i] == "[":
                j = core.index("]", i)
                allowed = core[i + 1:j]
                i = j + 1
            else:
                allowed = core[i]
                i += 1
            row = np.full(4, 0.02)
            for b in allowed:
                row[int(np.where(BASES == b)[0][0])] = 1.0
            rows.append(row / row.sum())
        db.append((name, np.array(rows)))
    return db


def tomtom(query: list[tuple[str, np.ndarray]], target: list[tuple[str, np.ndarray]],
           thresh: float = 0.5) -> pd.DataFrame:
    """Best tomtom match per query motif."""
    if not query or not target:
        return pd.DataFrame(columns=["query", "match", "q"])
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_meme(td / "q.meme", query)
        write_meme(td / "t.meme", target)
        cmd = ["tomtom", "-no-ssc", "-oc", str(td / "out"), "-verbosity", "1",
               "-min-overlap", "3", "-dist", "pearson", "-thresh", str(thresh),
               str(td / "q.meme"), str(td / "t.meme")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        f = td / "out" / "tomtom.tsv"
        if r.returncode != 0 or not f.exists():
            print("    tomtom failed:", r.stderr.strip()[:200])
            return pd.DataFrame(columns=["query", "match", "q"])
        t = pd.read_csv(f, sep="\t", comment="#")
    t = t.dropna(subset=["Target_ID"]).sort_values("q-value")
    best = t.groupby("Query_ID").first().reset_index()
    return best.rename(columns={"Query_ID": "query", "Target_ID": "match",
                                "q-value": "q", "Orientation": "orient"})[
        ["query", "match", "q", "orient"]]


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


# ----------------------------------------------------------------- pattern table
def joint_table(pats, n_human, offsets) -> pd.DataFrame:
    rows = []
    for p in pats:
        cwm, ppm, lo, hi = trim(p["cwm"], p["ppm"])
        h = int((p["ex"] < n_human).sum())
        q = int((p["ex"] >= n_human).sum())
        mid = ((p["start"] + p["end"]) // 2).clip(0, len(offsets) - 1)
        rows.append(dict(pattern=p["name"], sign=p["sign"], n_seqlets=p["n_seqlets"],
                         n_human=h, n_partner=q,
                         frac_human=h / max(1, h + q),
                         med_offset=float(np.median(offsets[mid])),
                         med_abs_offset=float(np.median(np.abs(offsets[mid]))),
                         consensus=(cs := consensus_informative(ppm)),
                         iupac_match=iupac_match(cs),
                         width=hi - lo, ic_max=float(per_position_ic(ppm).max()),
                         ic_sum=float(per_position_ic(ppm).sum()),
                         cwm_mass=float(np.abs(cwm).sum())))
    T = pd.DataFrame(rows)
    if T.empty:
        return T
    tot_h, tot_p = T.n_human.sum(), T.n_partner.sum()
    pv = [fisher_exact([[r.n_human, r.n_partner],
                        [tot_h - r.n_human, tot_p - r.n_partner]])[1]
          for r in T.itertuples()]
    T["p_side"] = pv
    T["q_side"] = bh(pv)
    T["log2_odds_human"] = np.log2(
        ((T.n_human + 0.5) / (tot_h - T.n_human + 0.5))
        / ((T.n_partner + 0.5) / (tot_p - T.n_partner + 0.5)))
    return T.sort_values("n_seqlets", ascending=False)


# --------------------------------------------------------------------- figures
def logo(ax, cwm, title=None):
    import logomaker
    df = pd.DataFrame(cwm, columns=list("ACGT"))
    logomaker.Logo(df, ax=ax, color_scheme="classic", show_spines=False)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=5.8, loc="left", pad=1.5)


def fig_joint(T: pd.DataFrame, pats, out: Path, base_frac: float, top: int = 6):
    style()
    sig = T[T.q_side < 0.05]
    rest = T[~T.pattern.isin(sig.pattern)].head(max(0, top - len(sig)))
    sel = pd.concat([sig, rest]).sort_values("n_seqlets", ascending=False)
    by_name = {p["name"]: p for p in pats}
    fig, axes = plt.subplots(len(sel), 2, figsize=(6.6, 0.72 * len(sel)),
                             gridspec_kw=dict(width_ratios=[2.1, 1.0], wspace=0.22,
                                              hspace=0.85))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(sel.itertuples()):
        p = by_name[r.pattern]
        cwm, _, _, _ = trim(p["cwm"], p["ppm"])
        ann = r.match if isinstance(r.match, str) else (
            r.iupac_match.split(";")[0] + "?" if isinstance(r.iupac_match, str)
            and r.iupac_match else "")
        lab = "%s  %s" % (r.consensus[:16], ann)
        logo(axes[i, 0], cwm, "%s  ·  n=%d  ·  %+.0f bp" %
             (lab.strip(), r.n_seqlets, r.med_offset))
        ax = axes[i, 1]
        fh = r.n_human / max(1, r.n_human + r.n_partner)
        ax.barh([0], [fh], color=SIDE_COL["human"], height=0.62)
        ax.barh([0], [1 - fh], left=[fh], color=SIDE_COL["partner"], height=0.62)
        ax.axvline(base_frac, color="0.25", lw=0.8, ls=":")
        star = "" if r.q_side > 0.05 else ("*" if r.q_side > 0.01 else "**")
        ax.text(1.02, 0, "%.0f%% human%s" % (100 * fh, star), va="center",
                fontsize=5.6, transform=ax.get_yaxis_transform())
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1] if i == len(sel) - 1 else [])
        if i == len(sel) - 1:
            ax.set_xticklabels(["0", "50", "100%"], fontsize=5.6)
            ax.set_xlabel("seqlets from the human side", fontsize=6.0)
    fig.suptitle("TF-MoDISco motifs, joint run over both sides: largest patterns "
                 "plus every side-biased one\n(dotted line = pooled human share; "
                 "* q<0.05, ** q<0.01, Fisher)", fontsize=7.0, y=1.02)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_sides(SB: pd.DataFrame, pats_by_side, out: Path, top: int = 5):
    style()
    fig, axes = plt.subplots(top, 2, figsize=(7.0, 0.80 * top),
                             gridspec_kw=dict(wspace=0.30, hspace=1.15))
    axes = np.atleast_2d(axes)
    for j, side in enumerate(["human", "partner"]):
        d = SB[SB.side == side].head(top)
        by_name = {p["name"]: p for p in pats_by_side.get(side, [])}
        for i in range(top):
            ax = axes[i, j]
            if i >= len(d):
                ax.axis("off")
                continue
            r = d.iloc[i]
            p = by_name[r.pattern]
            cwm, _, _, _ = trim(p["cwm"], p["ppm"])
            shared = "shared" if isinstance(r.cross_match, str) else "side-specific"
            logo(ax, cwm, "%s  ·  n=%d  ·  %s%s" %
                 (r.consensus[:16], r.n_seqlets, shared,
                  ("  ·  " + r["match"]) if isinstance(r["match"], str) else ""))
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0, j].annotate(
            "%s side%s" % (side, " (gained, dynamic)" if side == "human" else " (static)"),
            xy=(0, 1), xycoords="axes fraction", xytext=(0, 26),
            textcoords="offset points", fontsize=7.2, color=SIDE_COL[side],
            ha="left", va="bottom", annotation_clip=False)
    fig.suptitle("TF-MoDISco motifs discovered independently per side",
                 fontsize=7.4, y=1.02)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attr-dir", required=True)
    p.add_argument("--out-dir", "--tab-dir", dest="out_dir", default=".")
    p.add_argument("--fig-dir", default=None)
    p.add_argument("--h5-dir", default=None)
    p.add_argument("--window", type=int, default=11)
    p.add_argument("--flank", type=int, default=3)
    p.add_argument("--fdr", type=float, default=0.15)
    p.add_argument("--min-metacluster", type=int, default=50)
    p.add_argument("--max-seqlets", type=int, default=20000)
    p.add_argument("--n-leiden", type=int, default=20)
    p.add_argument("--trim-window", type=int, default=12,
                   help="modisco trim_to_window_size; 30 (default) is TF-scale")
    p.add_argument("--flank-to-add", type=int, default=3)
    p.add_argument("--min-ic", type=float, default=0.4)
    p.add_argument("--min-ic-window", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    np.random.seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    figd = Path(args.fig_dir) if args.fig_dir else out
    h5d = Path(args.h5_dir) if args.h5_dir else out
    figd.mkdir(parents=True, exist_ok=True); h5d.mkdir(parents=True, exist_ok=True)

    D = load_side(Path(args.attr_dir))
    offsets = D["human"]["offsets"]
    print("regions: " + ", ".join("%s %d" % (s, len(D[s]["oh"])) for s in D))

    db = consensus_db()

    # ---- joint run: human regions first, then partner, so example_idx splits cleanly
    n_human = len(D["human"]["oh"])
    oh = np.concatenate([D["human"]["oh"], D["partner"]["oh"]])
    hyp = np.concatenate([D["human"]["hyp"], D["partner"]["hyp"]])
    pos, neg = run_modisco(oh, hyp, args, "joint")
    h5j = h5d / "modisco_joint.h5"
    if h5j.exists():
        h5j.unlink()
    modiscolite.io.save_hdf5(h5j, pos, neg, window_size=args.window)
    pats = read_h5(h5j)
    print("joint patterns: %d (%d seqlets)"
          % (len(pats), sum(p["n_seqlets"] for p in pats)))
    T = joint_table(pats, n_human, offsets)
    m = tomtom([(p["name"], trim(p["cwm"], p["ppm"])[1]) for p in pats], db)
    T = T.merge(m.rename(columns={"query": "pattern", "q": "match_q"}),
                on="pattern", how="left")
    T.attrs["base_frac_human"] = T.n_human.sum() / max(1, T.n_human.sum() + T.n_partner.sum())
    T.insert(0, "run", "joint")
    T.to_csv(out / "modisco_patterns.csv", index=False)

    print("\njoint run, per-pattern side split (pooled human share %.3f):"
          % T.attrs["base_frac_human"])
    cols = ["pattern", "consensus", "ic_max", "n_seqlets", "n_human", "n_partner",
            "log2_odds_human", "q_side", "med_offset", "match", "match_q",
            "iupac_match"]
    print(T[cols].head(12).to_string(index=False, float_format=lambda v: "%.3g" % v))

    # ---- per-side runs
    rows, pats_by_side = [], {}
    for side in ("human", "partner"):
        ps, ns = run_modisco(D[side]["oh"], D[side]["hyp"], args, side)
        h5s = h5d / ("modisco_%s.h5" % side)
        if h5s.exists():
            h5s.unlink()
        modiscolite.io.save_hdf5(h5s, ps, ns, window_size=args.window)
        pl = read_h5(h5s)
        pats_by_side[side] = pl
        for q in pl:
            cwm, ppm, lo, hi = trim(q["cwm"], q["ppm"])
            mid = ((q["start"] + q["end"]) // 2).clip(0, len(offsets) - 1)
            rows.append(dict(run="by_side", side=side, pattern=q["name"],
                             sign=q["sign"], n_seqlets=q["n_seqlets"],
                             consensus=(cs := consensus_informative(ppm)),
                             iupac_match=iupac_match(cs), width=hi - lo,
                             ic_max=float(per_position_ic(ppm).max()),
                             med_offset=float(np.median(offsets[mid]))))
    SB = pd.DataFrame(rows)
    for side in ("human", "partner"):
        qy = [(p["name"], trim(p["cwm"], p["ppm"])[1]) for p in pats_by_side[side]]
        mm = tomtom(qy, db)
        SB.loc[SB.side == side, "match"] = SB.loc[SB.side == side, "pattern"].map(
            dict(zip(mm["query"], mm["match"])))
        SB.loc[SB.side == side, "match_q"] = SB.loc[SB.side == side, "pattern"].map(
            dict(zip(mm["query"], mm["q"])))
        other = "partner" if side == "human" else "human"
        tg = [(p["name"], trim(p["cwm"], p["ppm"])[1]) for p in pats_by_side[other]]
        cm = tomtom(qy, tg, thresh=0.05)
        SB.loc[SB.side == side, "cross_match"] = SB.loc[SB.side == side, "pattern"].map(
            dict(zip(cm["query"], cm["match"])))
        SB.loc[SB.side == side, "cross_q"] = SB.loc[SB.side == side, "pattern"].map(
            dict(zip(cm["query"], cm["q"])))
    SB = SB.sort_values(["side", "n_seqlets"], ascending=[True, False])
    SB.to_csv(out / "modisco_patterns_by_side.csv", index=False)
    print("\nper-side runs:")
    print(SB[["side", "pattern", "consensus", "ic_max", "n_seqlets", "med_offset",
              "match", "iupac_match", "cross_match"]].head(16)
          .to_string(index=False, float_format=lambda v: "%.3g" % v))
    print("\nshared vs side-specific (cross-side tomtom q<0.05):")
    print(SB.assign(shared=SB.cross_match.notna())
          .groupby(["side", "shared"]).size().to_string())

    fig_joint(T, pats, figd / "modisco_motifs.png",
              base_frac=T.n_human.sum() / max(1, T.n_human.sum() + T.n_partner.sum()))
    fig_sides(SB, pats_by_side, figd / "modisco_sides.png")
    print("\nwrote", out / "modisco_patterns.csv", out / "modisco_patterns_by_side.csv",
          "and 2 figures in", figd)


if __name__ == "__main__":
    main()
