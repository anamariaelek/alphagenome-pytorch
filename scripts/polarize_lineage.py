#!/usr/bin/env python
"""Polarize each divergence candidate onto a branch of the species tree.

Input is divergence_ranked.csv (rank_divergence.py).  For every candidate
(human anchor x tissue) the observed dynamic/static state of each species that
has a shape call at that anchor is collected, and Fitch parsimony over

    (((human,(rabbit,(mouse,rat))),opossum)

locates the branch(es) on which the state changed.  Species without an ortholog
or without a shape call in that tissue are pruned rather than scored, so the
lineage assignment is only as deep as the ortholog set allows; n_states records
how many species testified and lineage is 'multiple' when parsimony needs more
than one change.  Macaque and chicken are absent from the map by construction.

Also recorded per candidate: the timepoint of maximum observed departure between
the two species, whether opossum can testify at that timepoint (its
early-development coverage gap means it often cannot), and the human gene the
site sits in, taken from the GTF-annotated splice-site table.

    python polarize_lineage.py [--ranked F] [--pairs F] [--pred-dir D]
                               [--out-dir D] [--fig-dir D] [--passing-only]

Outputs:
    divergence_candidates_polarized.csv   one row per passing candidate
    candidate_trajectories.parquet        tidy observed+predicted trajectories
                                          for every species at every candidate
                                          anchor, for the step-5 panels
    divergence_by_lineage.png             candidates per branch, by class
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_pairs_gw import (AG, PRED, TIS, TPS, SP_ORDER,  # noqa: E402
                            DYNSET, load_species, members)
from rank_divergence import XS as XS_DEFAULT, style  # noqa: E402

TREE = {"root": ["euarchontoglires", "opossum"],
        "euarchontoglires": ["human", "glires"],
        "glires": ["rabbit", "rodentia"],
        "rodentia": ["mouse", "rat"]}
TIPS = ["human", "rabbit", "mouse", "rat", "opossum"]
BRANCH_ORDER = ["human", "euarchontoglires", "glires", "rabbit", "rodentia",
                "mouse", "rat", "opossum", "multiple", "unpolarized"]
DCOL = {"gain_in_focal": "#1f6fb4", "loss_in_focal": "#d1495b",
        "direction_reversal": "#6a4c93", "profile_shift": "#f0a202"}


def children(n):
    return TREE.get(n, [])


def fitch(states):
    """Fitch parsimony. states maps tip name -> 0/1; absent tips are pruned.

    Returns (list of branches carrying a change, assigned state per node).
    """
    up = {}

    def post(n):
        ch = [c for c in children(n)]
        if not ch:
            return up.setdefault(n, {states[n]} if n in states else None)
        sets = [post(c) for c in ch]
        sets = [s for s in sets if s]
        if not sets:
            up[n] = None
        elif len(sets) == 1:
            up[n] = set(sets[0])
        else:
            inter = set.intersection(*sets)
            up[n] = inter if inter else set.union(*sets)
        return up[n]

    post("root")
    if not up.get("root"):
        return [], {}
    # downpass: prefer the parent's state when ambiguous; root prefers the
    # outgroup state when opossum testifies
    asg = {}
    root_set = up["root"]
    if len(root_set) == 1:
        asg["root"] = next(iter(root_set))
    elif "opossum" in states:
        asg["root"] = states["opossum"]
    else:
        asg["root"] = sorted(root_set)[0]

    def pre(n):
        for cnode in children(n):
            s = up.get(cnode)
            if not s:
                continue
            asg[cnode] = asg[n] if asg[n] in s else sorted(s)[0]
            pre(cnode)

    pre("root")
    ch = []
    for n, kids in TREE.items():
        if n not in asg:
            continue
        for k in kids:
            if k in asg and asg[k] != asg[n]:
                ch.append(k)
    return ch, asg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ranked", default=f"{XS_DEFAULT}/divergence_ranked.csv")
    p.add_argument("--map", default=f"{XS_DEFAULT}/xspecies_site_map_genomewide.parquet")
    p.add_argument("--pred-dir", default=PRED)
    p.add_argument("--out-dir", default=XS_DEFAULT)
    p.add_argument("--fig-dir", default=f"{AG}/devas/figures")
    p.add_argument("--gtf", default=(f"{AG}/data/Homo_sapiens/"
                                     "splice_sites_intersect_protein_coding_"
                                     "with_gtf_annotation.parquet"))
    p.add_argument("--passing-only", action="store_true", default=True)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.fig_dir, exist_ok=True)

    R = pd.read_csv(a.ranked)
    C = R[R["passes_cut"]].copy() if a.passing_only else R.copy()
    anchors = set(C["human_site"])
    print("candidates %d | human anchors %d" % (len(C), len(anchors)), flush=True)

    # ── every species' state at every candidate anchor ──
    m = pd.read_parquet(a.map)
    mem = members(m)
    mem = mem[mem["human_site"].isin(anchors)]
    need = {sp: set(g["site"]) for sp, g in mem.groupby("species")}
    S, TW, PW = {}, {}, {}
    for sp in SP_ORDER:
        if sp not in need:
            continue
        S[sp], TW[sp], PW[sp] = load_species(sp, a.pred_dir, need[sp])

    site2anchor = {(r.species, r.site): r.human_site for r in mem.itertuples()}
    st = []
    for sp in S:
        d = S[sp].copy()
        d["species"] = sp
        d["human_site"] = [site2anchor.get((sp, s)) for s in d["site"]]
        st.append(d[["human_site", "species", "site", "Tissue", "obs_shape_site",
                     "pred_shape_site", "same_shape_site", "well", "dyn", "amp",
                     "mean_true", "pred_r", "pred_rmse", "n_tp"]])
    ST = pd.concat(st, ignore_index=True).dropna(subset=["human_site"])
    ST["verified_call"] = ST["well"] & ST["same_shape_site"].astype("boolean").fillna(False)
    print("state rows %d | species %s" % (len(ST), sorted(ST["species"].unique())),
          flush=True)

    # ── tidy trajectories for the panels ──
    tr = []
    for sp in TW:
        for kind, W in [("true", TW[sp]), ("pred", PW[sp])]:
            w = W.copy()
            w.index.names = ["site", "Tissue"]
            t = w.reset_index().melt(id_vars=["site", "Tissue"], var_name="Timepoint",
                                     value_name="value")
            t["species"] = sp
            t["kind"] = kind
            tr.append(t)
    TR = pd.concat(tr, ignore_index=True)
    TR["human_site"] = [site2anchor.get((s, x)) for s, x in zip(TR["species"], TR["site"])]
    TR = TR.dropna(subset=["human_site"])
    trp = f"{a.out_dir}/candidate_trajectories.parquet"
    TR.to_parquet(trp, index=False)
    print("trajectory rows %d -> %s" % (len(TR), trp), flush=True)

    # ── polarize per anchor x tissue ──
    obs = {(r.human_site, r.Tissue, r.species): (int(r.dyn), bool(r.verified_call))
           for r in ST.itertuples()}
    keys = sorted({(r.human_site, r.Tissue) for r in C.itertuples()})
    pol = {}
    for hs, t in keys:
        stt, ver = {}, []
        for sp in TIPS:
            v = obs.get((hs, t, sp))
            if v is None:
                continue
            stt[sp] = v[0]
            if v[1]:
                ver.append(sp)
        ch, asg = fitch(stt)
        if len(stt) < 3:
            lab = "unpolarized"
        elif len(ch) == 1:
            lab = ch[0]
        elif len(ch) == 0:
            lab = "unpolarized"
        else:
            lab = "multiple"
        pol[(hs, t)] = dict(lineage=lab, n_changes=len(ch), n_states=len(stt),
                            n_verified_states=len(ver),
                            states=json.dumps({k: stt[k] for k in sorted(stt)}),
                            verified_states=",".join(sorted(ver)),
                            root_state=asg.get("root"))
    for k in ["lineage", "n_changes", "n_states", "n_verified_states", "states",
              "verified_states", "root_state"]:
        C[k] = [pol[(h, t)][k] for h, t in zip(C["human_site"], C["Tissue"])]

    # ── timepoint of maximum departure, and whether opossum testifies there ──
    piv = TR.pivot_table(index=["species", "site", "Tissue", "kind"],
                         columns="Timepoint", values="value")
    piv = piv.reindex(columns=TPS)
    tp_max, dep, opo_tp, opo_state = [], [], [], []
    for r in C.itertuples():
        try:
            f = piv.loc[(r.focal, r.focal_site, r.Tissue, "true")].to_numpy(float)
            pp = piv.loc[(r.partner, r.partner_site, r.Tissue, "true")].to_numpy(float)
        except KeyError:
            tp_max.append(np.nan); dep.append(np.nan)
            opo_tp.append(False); opo_state.append(None)
            continue
        d = np.abs(f - pp)
        if np.all(~np.isfinite(d)):
            tp_max.append(np.nan); dep.append(np.nan)
        else:
            i = int(np.nanargmax(d))
            tp_max.append(TPS[i]); dep.append(float(d[i]))
        o = obs.get((r.human_site, r.Tissue, "opossum"))
        opo_state.append(None if o is None else o[0])
        ok = False
        if o is not None and len(tp_max) and np.isfinite(tp_max[-1] or np.nan):
            osite = mem[(mem["species"] == "opossum")
                        & (mem["human_site"] == r.human_site)]["site"]
            if len(osite):
                try:
                    ov = piv.loc[("opossum", osite.iloc[0], r.Tissue, "true")]
                    ok = bool(np.isfinite(ov.get(tp_max[-1], np.nan)))
                except KeyError:
                    ok = False
        opo_tp.append(ok)
    C["tp_max_departure"] = tp_max
    C["max_departure"] = dep
    C["opossum_state"] = opo_state
    C["opossum_testifies_at_tp"] = opo_tp

    # ── gene annotation (ids here; symbols resolved off-host) ──
    g = pd.read_parquet(a.gtf)
    gcol = next((c for c in g.columns if c.lower() in
                 ("gene_id", "gene", "gene_ids", "geneid")), None)
    print("gtf cols", list(g.columns)[:14], "| gene col", gcol, flush=True)
    if gcol:
        g["Chromosome"] = g["Chromosome"].astype(str)
        gm = {(c, int(p)): v for c, p, v in
              zip(g["Chromosome"], g["Position"], g[gcol])}
        cp = C["human_site"].str.rsplit(":", n=1).str[0]
        C["gene_id"] = [gm.get((s.split(":")[0], int(s.split(":")[1]))) for s in cp]
        print("gene ids resolved %d / %d" % (C["gene_id"].notna().sum(), len(C)),
              flush=True)

    out = f"{a.out_dir}/divergence_candidates_polarized.csv"
    C.to_csv(out, index=False)
    print("\nlineage counts:")
    print(C["lineage"].value_counts().to_string())
    print("\nn_states distribution:", C["n_states"].value_counts().to_dict())
    print("opossum testifies at the departure timepoint: %d of %d"
          % (int(C["opossum_testifies_at_tp"].sum()), len(C)))
    print("timepoint of max departure:",
          C["tp_max_departure"].value_counts().sort_index().to_dict())

    # ── figure ──
    style()
    order = [b for b in BRANCH_ORDER if (C["lineage"] == b).any()]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw=dict(width_ratios=[1, 1], wspace=0.55))
    ax = axes[0]
    bot = np.zeros(len(order))
    for k, col in DCOL.items():
        v = np.array([int(((C["lineage"] == b) & (C["div_class"] == k)).sum())
                      for b in order], float)
        if v.sum() == 0:
            continue
        ax.bar(range(len(order)), v, bottom=bot, color=col, width=0.72,
               label="%s (n=%d)" % (k.replace("_", " "), int(v.sum())))
        bot += v
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("candidates")
    ax.set_title("Branch carrying the change (n=%d)" % len(C))
    ax.legend(handletextpad=0.4, borderpad=0.2)
    for i, b in enumerate(order):
        ax.text(i, bot[i] + max(bot) * 0.015, "%d" % bot[i], ha="center", fontsize=6)

    ax = axes[1]
    tt = (C.groupby(["lineage", "Tissue"]).size().unstack(fill_value=0)
           .reindex(index=order).reindex(columns=[t for t in TIS], fill_value=0))
    im = ax.imshow(tt.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(tt.shape[1]))
    ax.set_xticklabels(tt.columns, rotation=45, ha="right")
    ax.set_yticks(range(tt.shape[0]))
    ax.set_yticklabels(tt.index)
    for i in range(tt.shape[0]):
        for j in range(tt.shape[1]):
            v = int(tt.to_numpy()[i, j])
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=5.8,
                        color="white" if v > tt.to_numpy().max() * 0.6 else "#222222")
    ax.set_title("Branch x tissue")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label("candidates",
                                                                fontsize=6.5)
    fig.text(0.005, -0.16,
             "state = observed dynamic/static shape class per species at the anchor; "
             "Fitch parsimony on (((human,(rabbit,(mouse,rat))),opossum); species "
             "without an ortholog or a shape call in that tissue are pruned, so "
             "'unpolarized' means fewer than three species testified or parsimony "
             "found no change among those that did",
             fontsize=5.6, color="#666666")
    figp = f"{a.fig_dir}/divergence_by_lineage.png"
    fig.savefig(figp, bbox_inches="tight")
    fig.savefig(figp.replace(".png", ".pdf"), bbox_inches="tight")
    print("\nwrote", out, "and", figp)


if __name__ == "__main__":
    main()
