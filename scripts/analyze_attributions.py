#!/usr/bin/env python
"""Summarize integrated-gradients attributions for human-branch divergent sites.

Reads the per-target ``.npz`` files written by ``run_attributions.py`` (base-
resolution attributions in a +/-512 bp window plus a 128-bp-binned profile over
the whole 131 kb input) and produces:

  attr_stats.csv        one row per target: attribution mass by distance band,
                        splice-motif contribution, peak positions
  attr_kmers.csv        6-mer contribution enrichment, per side (human vs the
                        static partner), with matches to published RBP/splicing
                        consensus motifs
  attr_profiles.png     mean |contribution| vs distance to the site, human
                        (dynamic, gained) side vs partner (static) side, and
                        the distribution of attribution mass by band
  attr_examples.png     per-base attribution tracks for the top anchors, human
                        side above and partner side below
  attr_kmers.png        top contributing 6-mers per side, annotated with motif
                        matches

The "human side" of every pair is the species that gained the dynamic
trajectory on the human branch; the "partner side" is the species whose
trajectory is static at the orthologous site.  Attribution is of the usage
logit at the site, in the tissue and timepoint of maximum observed divergence.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
try:
    from score_pairs_gw import style
except Exception:  # standalone use
    def style():
        plt.rcParams.update({"figure.dpi": 150, "font.size": 7,
                             "axes.spines.top": False, "axes.spines.right": False})

BASES = np.array(list("ACGT"))
BANDS = [(0, 10, "site +/-10"), (10, 50, "10-50"), (50, 100, "100"),
         (100, 300, "300"), (300, 512, "512")]

# Published consensus motifs for splicing regulators and core splice signals,
# written as DNA with IUPAC degeneracy.
MOTIFS = {
    "5'SS (GTRAGT)": "GT[AG]AG",
    "3'SS (YAG)": "[CT]AG$",
    "polypyrimidine (T-rich)": "TTTT",
    "branchpoint (YTNAY)": "[CT]T[ACGT]A[CT]",
    "RBFOX (TGCATG)": "TGCATG",
    "NOVA (YCAY)": "[CT]CA[CT]",
    "PTBP1 (TCTT/CTCTCT)": "(TCTTC|CTCTCT)",
    "hnRNPA1 (TAGGGA/T)": "TAGGG[AT]",
    "MBNL (YGCY)": "[CT]GC[CT]",
    "QKI (ACTAAY)": "ACTAA[CT]",
    "CELF (TGTGTG)": "TGTGTG",
    "ESRP (GGTGGT)": "GGTGG",
    "SRSF1 (GGAGGA)": "GGAG",
    "SRSF2 (GGCCTC)": "GGCC",
    "hnRNPC (TTTTT)": "TTTTT",
}


def seq_of(onehot: np.ndarray) -> str:
    tok = onehot.argmax(axis=1)
    s = BASES[tok]
    s[onehot.sum(axis=1) == 0] = "N"
    return "".join(s)


def load_targets(d: Path) -> list[dict]:
    out = []
    for f in sorted(d.glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        out.append(dict(meta=m, contrib=z["contrib"], attr=z["attr"],
                        onehot=z["onehot"], offsets=z["offsets"],
                        profile=z["profile_128bp"], file=f.name))
    return out


def per_target_stats(T: list[dict]) -> pd.DataFrame:
    rows = []
    for t in T:
        m, c, off = t["meta"], t["contrib"], t["offsets"]
        a = np.abs(c)
        tot = a.sum()
        r = dict(target=t["file"][:-4], species=m["species"], site=m["site"],
                 tissue=m["tissue"], condition=m["condition"], role=m["role"],
                 symbol=m["symbol"], human_anchor=m["human_anchor"],
                 strand=m["strand"], pred=m["pred"], logit=m["logit"],
                 abs_total_window=float(tot))
        for lo, hi, name in BANDS:
            sel = (np.abs(off) >= lo) & (np.abs(off) < hi)
            r[f"frac_{name}"] = float(a[sel].sum() / max(tot, 1e-12))
        # signed contribution of the core splice signal: +/-3 bp of the site
        core = np.abs(off) <= 3
        r["core_signed"] = float(c[core].sum())
        r["core_frac"] = float(a[core].sum() / max(tot, 1e-12))
        # distal mass from the coarse whole-window profile
        p = np.abs(t["profile"])
        nb = len(p)
        mid = nb // 2
        near = p[mid - 4: mid + 4].sum()          # +/-512 bp
        r["frac_distal_131kb"] = float(1.0 - near / max(p.sum(), 1e-12))
        r["peak_offset"] = int(off[np.argmax(a)])
        r["peak_signed"] = float(c[np.argmax(a)])
        rows.append(r)
    return pd.DataFrame(rows)


def kmer_table(T: list[dict], k: int = 6, min_count: int = 40) -> pd.DataFrame:
    acc: dict[tuple[str, str], list[float]] = {}
    for t in T:
        side = "human" if t["meta"]["role"] == "focal" else "partner"
        s = seq_of(t["onehot"])
        c = t["contrib"].astype(np.float64)
        cs = np.concatenate([[0.0], np.cumsum(c)])
        for i in range(len(s) - k + 1):
            km = s[i:i + k]
            if "N" in km:
                continue
            v = cs[i + k] - cs[i]
            key = (side, km)
            if key in acc:
                acc[key][0] += v
                acc[key][1] += 1
            else:
                acc[key] = [v, 1]
    rows = [dict(side=sd, kmer=km, total=v, count=n, mean=v / n)
            for (sd, km), (v, n) in acc.items() if n >= min_count]
    K = pd.DataFrame(rows)
    K["motifs"] = [";".join(nm for nm, pat in MOTIFS.items() if re.search(pat, km))
                   for km in K.kmer]
    return K.sort_values(["side", "mean"], ascending=[True, False])


def fig_profiles(S: pd.DataFrame, T: list[dict], out: Path):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3),
                             gridspec_kw=dict(wspace=0.33))
    off = T[0]["offsets"]
    for side, col, lab in (("focal", "#1f78b4", "human side (gained, dynamic)"),
                           ("partner", "#e31a1c", "partner side (static)")):
        A = np.array([np.abs(t["contrib"]) for t in T if t["meta"]["role"] == side])
        if not len(A):
            continue
        med = np.median(A, axis=0)
        w = 11
        sm = np.convolve(med, np.ones(w) / w, mode="same")
        axes[0].plot(off, sm, color=col, lw=1.1, label=f"{lab} (n={len(A)})")
    axes[0].set_xlabel("distance from splice site (bp)")
    axes[0].set_ylabel("median |contribution|\n(11 bp mean)")
    axes[0].set_title("Attribution around the site", fontsize=7.2)
    axes[0].legend(frameon=False, fontsize=5.5, loc="center right",
               handlelength=1.2, borderaxespad=0.2)

    bandcols = [f"frac_{n}" for _, _, n in BANDS]
    x = np.arange(len(bandcols))
    for side, col, lab in (("focal", "#1f78b4", "human"), ("partner", "#e31a1c", "partner")):
        d = S[S.role == side]
        mu = [d[c].mean() for c in bandcols]
        se = [d[c].std(ddof=1) / np.sqrt(max(len(d), 1)) for c in bandcols]
        axes[1].bar(x + (0.2 if side == "partner" else -0.2), mu, 0.38, yerr=se,
                    color=col, alpha=0.85, error_kw=dict(lw=0.7), label=lab)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n for _, _, n in BANDS], rotation=30, ha="right", fontsize=6.0)
    axes[1].set_ylabel("fraction of |attribution|")
    axes[1].set_title("Attribution mass by distance band", fontsize=7.2)
    axes[1].legend(frameon=False, fontsize=6.0)

    for side, col in (("focal", "#1f78b4"), ("partner", "#e31a1c")):
        d = S[S.role == side]
        axes[2].scatter(d.pred, d.core_frac, s=9, color=col, alpha=0.7, lw=0)
    axes[2].set_xlabel("predicted usage at the site")
    axes[2].set_ylabel("fraction within +/-3 bp\n(core splice signal)")
    axes[2].set_title("Core-signal reliance vs usage", fontsize=7.2)

    fig.suptitle("Integrated-gradients attribution of human-branch divergent splice sites",
                 fontsize=8.0, y=1.02)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_examples(S: pd.DataFrame, T: list[dict], out: Path, n: int = 6):
    style()
    byt = {t["file"][:-4]: t for t in T}
    anchors = (S[S.role == "focal"].sort_values("abs_total_window", ascending=False)
               .human_anchor.tolist())
    seen, chosen = set(), []
    for a in anchors:
        sym = S[S.human_anchor == a].symbol.iloc[0]
        if sym in seen:
            continue
        seen.add(sym)
        chosen.append(a)
        if len(chosen) == n:
            break
    fig, axes = plt.subplots(len(chosen), 1, figsize=(6.4, 1.05 * len(chosen)),
                             sharex=True, gridspec_kw=dict(hspace=0.45))
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, chosen):
        d = S[S.human_anchor == a]
        for _, r in d.iterrows():
            t = byt[r.target]
            c = t["contrib"]
            sgn = 1.0 if r.role == "focal" else -1.0
            col = "#1f78b4" if r.role == "focal" else "#e31a1c"
            ax.fill_between(t["offsets"], 0, sgn * np.abs(c), color=col, lw=0,
                            alpha=0.85, step="mid")
        f = d[d.role == "focal"].iloc[0]
        p = d[d.role == "partner"]
        ax.axhline(0, color="#444444", lw=0.5)
        ax.axvline(0, color="#999999", lw=0.6, zorder=0)
        ax.set_ylabel("|contrib|", fontsize=6.0)
        ax.set_title("%s  %s   human vs %s   (%s)"
                     % (f.symbol, f.tissue,
                        p.species.iloc[0] if len(p) else "n/a", f.condition),
                     fontsize=6.6, loc="left", pad=2)
        ax.tick_params(labelsize=6.5)
    axes[-1].set_xlabel("distance from splice site (bp)")
    fig.suptitle("Per-base attribution: human side (up) vs static partner (down)",
                 fontsize=8.0, y=0.995)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_kmers(K: pd.DataFrame, out: Path, top: int = 18):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1),
                             gridspec_kw=dict(wspace=0.55))
    for ax, side, col, lab in ((axes[0], "human", "#1f78b4", "human side (gained)"),
                               (axes[1], "partner", "#e31a1c", "partner side (static)")):
        d = K[K.side == side].head(top).iloc[::-1]
        y = np.arange(len(d))
        ax.barh(y, d["mean"], color=col, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{km}  {mo.split(';')[0] if mo else ''}"
                            for km, mo in zip(d.kmer, d.motifs)], fontsize=5.8)
        ax.set_xlabel("mean contribution per 6-mer occurrence")
        ax.set_title(lab, fontsize=7.2)
    fig.suptitle("Top contributing 6-mers, annotated with published splicing motifs",
                 fontsize=8.0, y=1.01)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attr-dir", required=True)
    p.add_argument("--out-dir", "--tab-dir", dest="out_dir", default=".",
                   help="directory for attr_stats.csv / attr_kmers.csv")
    p.add_argument("--fig-dir", default=None)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figd = Path(args.fig_dir) if args.fig_dir else out

    T = load_targets(Path(args.attr_dir))
    print("loaded %d targets" % len(T))
    S = per_target_stats(T)
    S["role"] = S.role.map({"focal": "focal", "partner": "partner"})
    S.to_csv(out / "attr_stats.csv", index=False)

    print("\nattribution mass by band (mean fraction):")
    cols = [f"frac_{n}" for _, _, n in BANDS] + ["core_frac", "frac_distal_131kb"]
    print(S.groupby("role")[cols].mean().round(4).to_string())
    print("\ncore |contribution| fraction, human vs partner:")
    for r, d in S.groupby("role"):
        print("  %-8s n=%3d  median %.4f  IQR %.4f-%.4f"
              % (r, len(d), d.core_frac.median(),
                 d.core_frac.quantile(.25), d.core_frac.quantile(.75)))
    print("\npeak offset distribution (|offset| <= 10 / <= 50 / > 50):")
    for r, d in S.groupby("role"):
        a = d.peak_offset.abs()
        print("  %-8s %.2f / %.2f / %.2f" % (r, (a <= 10).mean(), (a <= 50).mean(), (a > 50).mean()))

    K = kmer_table(T)
    K.to_csv(out / "attr_kmers.csv", index=False)
    print("\ntop 10 6-mers per side:")
    for side, d in K.groupby("side"):
        print(" ", side, ", ".join("%s(%.3g%s)" % (r.kmer, r["mean"],
                                                   "*" if r.motifs else "")
                                   for _, r in d.head(10).iterrows()))

    fig_profiles(S, T, figd / "attr_profiles.png")
    fig_examples(S, T, figd / "attr_examples.png")
    fig_kmers(K, figd / "attr_kmers.png")
    print("\nwrote", out / "attr_stats.csv", out / "attr_kmers.csv", "and 3 figures in", figd)


if __name__ == "__main__":
    main()
