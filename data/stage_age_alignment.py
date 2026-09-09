#!/usr/bin/env python
"""Aligned-timepoint -> absolute-age table and figure for the evo-devo panel.

Builds stage_age_table.csv from the three spliser sample manifests and renders
stage_age_alignment.png next to it. This is the one report figure with no
dependency on the AlphaGenome prediction store, which is why it lives here with
its input table rather than under devas/ with the model-evaluation figures.

Timepoint indices in the manifests are pipeline positions, not ages: each species
is aligned onto its own 1..N grid. This converts every sample's developmental
stage to days post conception (dpc) so the grids can be compared, using
species-specific gestation length to place postnatal stages.

    python stage_age_alignment.py                       # defaults below
    python stage_age_alignment.py --samples-dir DIR --out-dir DIR

Outputs (both written to --out-dir):
    stage_age_table.csv        one row per species x aligned timepoint
    stage_age_alignment.png    dpc vs aligned timepoint, log y, one line per species
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLES_DIR = os.path.expanduser("~/sds/sd17d003/Anamaria/spliser/data")
MANIFESTS = ["samples.txt", "samples_hg38.txt", "samples_mmul_10.txt"]

# Gestation / incubation length in days, used to convert postnatal stages to dpc.
GEST = {"Human": 280, "Macaque": 165, "Mouse": 20, "Rat": 21,
        "Rabbit": 30, "Opossum": 15, "Chicken": 21}

SP_ORDER = ["human", "macaque", "rabbit", "rat", "mouse", "opossum", "chicken"]
SP_COL = dict(zip(SP_ORDER, ["#1f6fb4", "#4f9ad0", "#7b4fa0", "#d1721a",
                             "#e2a33c", "#2e8b6b", "#8c8c8c"]))
# Vertical nudge for the end-of-line species labels, so they don't collide.
NUDGE = {"human": 1.0, "macaque": 1.0, "rabbit": 1.28, "rat": 0.95,
         "mouse": 1.0, "opossum": 0.80, "chicken": 1.18}
# Chicken's last timepoint sits under macaque's label, so its label goes left.
LABEL_LEFT = {"chicken"}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlelocation": "left", "legend.frameon": False,
        "figure.dpi": 200, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def age_dpc(species, stage, detail):
    """Days post conception for one sample, from its stage label and free-text detail."""
    g = GEST[species]
    st = (stage or "").strip()
    dt = re.sub(r"\s+", " ", (detail or "").strip())

    m = re.fullmatch(r"e([\d.]+)", st)          # embryonic day
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"P([\d.]+)", st)          # postnatal day
    if m:
        return g + float(m.group(1))
    m = re.fullmatch(r"([\d.]+)wpc", st)        # weeks post conception
    if m:
        return 7 * float(m.group(1))

    for rgx, f in [(r"([\d.]+) ?days?", lambda v: v),
                   (r"([\d.]+) ?weeks?", lambda v: 7 * v)]:
        m = re.fullmatch(rgx, dt)
        if m:
            return f(float(m.group(1)))
    for rgx, f in [(r"([\d.]+) ?days? post ?birth", lambda v: g + v),
                   (r"([\d.]+) ?weeks? post ?birth", lambda v: g + 7 * v),
                   (r"([\d.]+) ?months? post ?birth", lambda v: g + 30.44 * v),
                   (r"([\d.]+) ?years? post ?birth", lambda v: g + 365.25 * v)]:
        m = re.fullmatch(rgx, dt)
        if m:
            return f(float(m.group(1)))
    return float("nan")


def build_table(samples_dir):
    frames = []
    for m in MANIFESTS:
        S = pd.read_csv(os.path.join(samples_dir, m), sep="\t", dtype=str)
        S.columns = [c.replace("\ufeff", "") for c in S.columns]
        frames.append(S)
    ALL = pd.concat(frames, ignore_index=True)
    ALL["Timepoint"] = pd.to_numeric(ALL["Timepoint"], errors="coerce")

    A = ALL.dropna(subset=["Timepoint"])
    A = A[A["Species"].isin(GEST)].copy()
    A["dpc"] = [age_dpc(s, st, dt) for s, st, dt
                in zip(A["Species"], A["Developmental_stage"], A["Stage_detail"])]
    A = A.dropna(subset=["dpc"])

    AGE = (A.groupby(["Species", "Timepoint"])
            .agg(n_samples=("dpc", "size"),
                 dpc_median=("dpc", "median"),
                 dpc_min=("dpc", "min"),
                 dpc_max=("dpc", "max"),
                 dpc_geomean=("dpc", lambda v: float(np.exp(np.log(v).mean()))),
                 stages=("Developmental_stage", lambda s: ",".join(sorted(set(s)))))
            .reset_index())
    AGE["Timepoint"] = AGE["Timepoint"].astype(int)
    AGE["prenatal"] = AGE["dpc_median"] < AGE["Species"].map(GEST)
    for k in ["dpc_median", "dpc_min", "dpc_max", "dpc_geomean"]:
        AGE[k] = AGE[k].round(2)

    out = AGE.sort_values(["Species", "Timepoint"]).rename(columns={"dpc_median": "dpc"})
    out.insert(0, "species", out["Species"].str.lower())
    out["log_dpc"] = np.log(out["dpc"]).round(4)
    out["gestation_days"] = out["Species"].map(GEST)
    return out[["species", "Species", "Timepoint", "dpc", "log_dpc", "dpc_geomean",
                "dpc_min", "dpc_max", "gestation_days", "prenatal", "n_samples", "stages"]]


def fig_alignment(out, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for s in SP_ORDER:
        d = out[out["species"] == s].sort_values("Timepoint")
        if d.empty:
            continue
        ax.plot(d["Timepoint"], d["dpc"], "-", color=SP_COL[s], lw=1.4, zorder=3)
        post = ~d["prenatal"]
        ax.scatter(d.loc[post, "Timepoint"], d.loc[post, "dpc"],
                   s=20, color=SP_COL[s], zorder=4)
        ax.scatter(d.loc[~post, "Timepoint"], d.loc[~post, "dpc"], s=26,
                   facecolor="white", edgecolor=SP_COL[s], linewidth=1.2, zorder=4)
        left = s in LABEL_LEFT
        ax.text(d["Timepoint"].iloc[-1] + (-0.3 if left else 0.3),
                d["dpc"].iloc[-1] * NUDGE[s], s.capitalize(), color=SP_COL[s],
                fontsize=7, va="center", ha="right" if left else "left")

    ax.set_yscale("log")
    ax.set_xlabel("Aligned timepoint (pipeline index)")
    ax.set_ylabel("Age (days from conception)")
    ax.set_title("Aligned timepoints correspond to very different absolute ages")
    ax.set_xticks(range(1, 16))
    ax.set_xlim(0.4, 18.0)
    ax.set_yticks([10, 30, 100, 1000, 10000])
    ax.set_yticklabels(["10", "30", "100", "1k", "10k"])
    ax.scatter([], [], s=26, facecolor="white", edgecolor="0.35", linewidth=1.2,
               label="prenatal")
    ax.scatter([], [], s=20, color="0.35", label="postnatal")
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(out_dir, "stage_age_alignment.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples-dir", default=SAMPLES_DIR,
                   help="directory holding %s (default: %%(default)s)" % ", ".join(MANIFESTS))
    p.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="where to write the table and figure (default: this script's directory)")
    a = p.parse_args()

    style()
    out = build_table(a.samples_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    out.to_csv(os.path.join(a.out_dir, "stage_age_table.csv"), index=False)
    fig_alignment(out, a.out_dir)
    print("%d species x timepoint rows from %d samples; dpc %.0f-%.0f"
          % (len(out), out["n_samples"].sum(), out["dpc"].min(), out["dpc"].max()),
          flush=True)
    print(out.groupby("Species")
             .agg(n_tp=("Timepoint", "size"), dpc_first=("dpc", "min"),
                  dpc_last=("dpc", "max"), n_prenatal=("prenatal", "sum"))
             .to_string(), flush=True)


if __name__ == "__main__":
    main()
