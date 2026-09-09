#!/usr/bin/env python
"""Final candidate set, trajectory panels, and the attribution manifest.

Takes divergence_candidates_polarized.csv (polarize_lineage.py, with gene
symbols added off-host) and:

  1. collapses it to one row per human anchor x tissue, keeping the
     highest-scoring partner contrast, and writes the full deduplicated set as
     divergence_candidates_final.csv with a `selected` flag;
  2. selects a stratified illustration set - up to --per-cell per
     (lineage, div_class) cell, polarized branches first, ordered by score;
  3. renders candidates_trajectories.png: one panel per selected candidate,
     observed (solid) and predicted (dashed) usage for both species over the 15
     developmental timepoints, with the departure timepoint marked;
  4. writes attr_manifest.csv - two rows per selected candidate (focal and
     partner), the units an attribution run would take, with the site, strand,
     tissue, and the timepoint of maximum departure.

    python select_candidates.py [--polarized F] [--traj F] [--per-cell N]
                                [--out-dir D] [--fig-dir D]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_pairs_gw import AG, TPS  # noqa: E402
from rank_divergence import XS as XS_DEFAULT, style  # noqa: E402

SPCOL = {"human": "#1f6fb4", "macaque": "#4c9f70", "rabbit": "#c77dff",
         "rat": "#e07a5f", "mouse": "#d1495b", "opossum": "#f0a202",
         "chicken": "#6a4c93"}
POLAR = ["human", "euarchontoglires", "glires", "rodentia", "rabbit", "mouse",
         "rat", "opossum"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--polarized",
                   default=f"{XS_DEFAULT}/divergence_candidates_polarized.csv")
    p.add_argument("--traj", default=f"{XS_DEFAULT}/candidate_trajectories.parquet")
    p.add_argument("--per-cell", type=int, default=3)
    p.add_argument("--per-gene", type=int, default=1,
                   help="max candidates per gene x tissue x branch")
    p.add_argument("--n-panels", type=int, default=24)
    p.add_argument("--min-multi", type=int, default=10,
                   help="panel slots reserved for human-vs-multiple-species cases")
    p.add_argument("--out-dir", default=XS_DEFAULT)
    p.add_argument("--fig-dir", default=f"{AG}/devas/figures")
    a = p.parse_args()

    D = pd.read_csv(a.polarized)
    D["anchor_tissue"] = D["human_site"] + "|" + D["Tissue"]
    F = (D.sort_values("score", ascending=False)
          .drop_duplicates("anchor_tissue").reset_index(drop=True))
    print("passing rows %d -> %d anchor x tissue candidates | %d genes"
          % (len(D), len(F), F["gene_id"].nunique()), flush=True)

    # ── stratified selection ──
    F["pol"] = np.where(F["lineage"].isin(POLAR), 0, 1)
    F = F.sort_values(["pol", "score"], ascending=[True, False])
    take = (F.groupby(["gene_id", "Tissue", "lineage"], sort=False).head(a.per_gene)
             .groupby(["lineage", "div_class"], sort=False).head(a.per_cell).sort_values(["pol", "score"], ascending=[True, False]))
    # how many distinct partner species diverge from the same focal at this anchor
    npart = (D.groupby(["anchor_tissue", "focal"])["partner"].nunique()
              .rename("n_partners").reset_index())
    for X in (F, take):
        X["n_partners"] = X.merge(npart, on=["anchor_tissue", "focal"], how="left")["n_partners"].fillna(1).astype(int).to_numpy()
        X["multi_species"] = (X["n_verified_states"] >= 3) | (X["n_partners"] >= 2)
    # reserve panel slots for human-vs-multiple-species cases, then fill by score
    multi = take[take["multi_species"]].head(a.min_multi)
    rest = take[~take["anchor_tissue"].isin(multi["anchor_tissue"])]
    sel = (pd.concat([multi, rest]).head(a.n_panels)
             .sort_values(["pol", "score"], ascending=[True, False]).copy())
    print("panels: %d multi-species (>=3 verified species or >=2 divergent partners) of %d"
          % (int(sel["multi_species"].sum()), len(sel)), flush=True)
    F["selected"] = F["anchor_tissue"].isin(sel["anchor_tissue"])
    F["sel_rank"] = np.nan
    F.loc[F["selected"], "sel_rank"] = range(1, int(F["selected"].sum()) + 1)
    out = f"{a.out_dir}/divergence_candidates_final.csv"
    F.drop(columns=["pol"]).to_csv(out, index=False)
    # human-vs-multiple-species cases, as a table of their own
    MS = F[F["multi_species"] & (F["focal"] == "human")].sort_values("score", ascending=False)
    mscols = [c for c in ["human_site", "Tissue", "symbol", "gene_id", "focal", "partner",
                          "div_class", "lineage", "n_states", "n_verified_states", "states",
                          "verified_states", "n_partners", "tp_max_departure", "div_mag",
                          "quality", "score", "selected"] if c in MS.columns]
    msp = f"{a.out_dir}/multi_species_candidates.csv"
    MS[mscols].to_csv(msp, index=False)
    print("human-vs-multiple: %d anchor x tissue rows | %d genes | %d with >=2 divergent "
          "partners | verified-species counts %s"
          % (len(MS), MS["gene_id"].nunique(), int((MS["n_partners"] >= 2).sum()),
             MS["n_verified_states"].value_counts().sort_index().to_dict()), flush=True)
    print("selected %d for panels | by lineage %s"
          % (len(sel), sel["lineage"].value_counts().to_dict()), flush=True)

    # ── trajectory panels ──
    TR = pd.read_parquet(a.traj)
    piv = TR.pivot_table(index=["species", "site", "Tissue", "kind"],
                         columns="Timepoint", values="value").reindex(columns=TPS)
    # anchor-keyed view: every species with an ortholog at the human anchor
    pivA = TR.pivot_table(index=["human_site", "Tissue", "species", "kind"],
                          columns="Timepoint", values="value").reindex(columns=TPS)
    style()
    n = len(sel)
    ncol = 6
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.85 * ncol, 1.55 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    def short(sym):
        parts = [x for x in str(sym).split("/")]
        keep = [x for x in parts if not x.startswith("ENSG")] or parts[:1]
        lab = "/".join(keep[:2])
        return lab if len(lab) <= 16 else lab[:15] + "\u2026"

    for k, r in enumerate(sel.itertuples()):
        ax = axes[k // ncol, k % ncol]
        try:
            block = pivA.loc[(r.human_site, r.Tissue)]
            spp = list(dict.fromkeys(block.index.get_level_values("species")))
        except KeyError:
            spp = []
        # focal first, then the divergent partner, then the remaining witnesses
        order = ([r.focal] + [r.partner]
                 + [s for s in spp if s not in (r.focal, r.partner)])
        drawn = 0
        for sp in order:
            if sp not in spp:
                continue
            col = SPCOL.get(sp, "#666666")
            emph = sp in (r.focal, r.partner)
            for kind, ls, lw in [("true", "-", 1.5 if emph else 0.9),
                                 ("pred", "--", 1.0 if emph else 0.7)]:
                try:
                    y = block.loc[(sp, kind)].to_numpy(float)
                except KeyError:
                    continue
                m = np.isfinite(y)
                if m.sum() < 2:
                    continue
                ax.plot(np.array(TPS)[m], y[m], ls, color=col, lw=lw,
                        alpha=1.0 if emph else 0.75,
                        label=sp if kind == "true" else None)
                if kind == "true":
                    drawn += 1
        if np.isfinite(r.tp_max_departure):
            ax.axvline(r.tp_max_departure, color="#bbbbbb", lw=0.7, zorder=0)
        contrast = ("%s v %d spp" % (r.focal[:3], drawn - 1) if drawn > 2
                    else "%s v %s" % (r.focal[:3], r.partner[:3]))
        ax.set_title("%s  %s\n%s  %s  \u00b7  %s" %
                     (short(r.symbol), r.Tissue, contrast,
                      {"gain_in_focal": "gain", "loss_in_focal": "loss",
                       "direction_reversal": "reversal",
                       "profile_shift": "shift"}.get(r.div_class, r.div_class),
                      r.lineage),
                     fontsize=6, pad=2)
        ax.legend(fontsize=4.6, handlelength=1.0, handletextpad=0.35,
                  borderpad=0.1, labelspacing=0.15, loc="lower right", frameon=False)
        ax.set_ylim(-0.02, 1.02)
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    for i in range(nrow):
        axes[i, 0].set_ylabel("splice-site usage")
    for j in range(ncol):
        axes[nrow - 1, j].set_xlabel("developmental timepoint")
    fig.suptitle("Model-verified divergent developmental trajectories "
                 "(observed solid, predicted dashed)", fontsize=8, y=1.0)
    fig.text(0.005, -0.02,
             "one panel per candidate anchor x tissue; every species with an "
             "ortholog at the anchor is drawn (focal and divergent partner in bold, "
             "remaining witnesses thin); panel slots are reserved for cases where the "
             "focal species diverges from several witnesses; grey line marks the "
             "timepoint of maximum observed departure; branch label from Fitch parsimony",
             fontsize=5.6, color="#666666")
    fig.tight_layout()
    figp = f"{a.fig_dir}/candidates_trajectories.png"
    fig.savefig(figp, bbox_inches="tight")
    fig.savefig(figp.replace(".png", ".pdf"), bbox_inches="tight")

    # ── attribution manifest ──
    rows = []
    for r in sel.itertuples():
        for role, sp, site in [("focal", r.focal, r.focal_site),
                               ("partner", r.partner, r.partner_site)]:
            ch, pos, strand = site.split(":")
            rows.append(dict(sel_rank=int(F.loc[F["anchor_tissue"] == r.anchor_tissue,
                                                "sel_rank"].iloc[0]),
                             human_anchor=r.human_site, symbol=r.symbol,
                             gene_id=r.gene_id, Tissue=r.Tissue, role=role,
                             species=sp, site=site, Chromosome=ch, Position=int(pos),
                             Strand=strand, div_class=r.div_class, lineage=r.lineage,
                             tp_max_departure=r.tp_max_departure,
                             div_mag=r.div_mag, quality=r.quality, score=r.score))
    M = pd.DataFrame(rows).sort_values(["sel_rank", "role"])
    mp = f"{a.out_dir}/attr_manifest.csv"
    M.to_csv(mp, index=False)
    print("manifest %d targets (%d species-tissue combinations)"
          % (len(M), M.groupby(["species", "Tissue"]).ngroups))
    print(M.groupby("species").size().to_string())
    print("\nselected candidates:")
    print(sel[["sel_rank" if "sel_rank" in sel else "score", "symbol", "Tissue",
               "focal", "partner", "div_class", "lineage", "div_mag", "quality",
               "tp_max_departure"]].to_string(index=False))
    print("\nwrote", out, mp, "and", figp)


if __name__ == "__main__":
    main()
