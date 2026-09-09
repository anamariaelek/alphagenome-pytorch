#!/usr/bin/env python3
"""
plot_prediction_clusters.py
---------------------------
Heatmap + per-cluster profile plots for the PREDICTED splice-usage trajectories,
grouped by the reference cluster each prediction was assigned to
(``cluster_predictions.py`` output).

For each ``<species>/pred_gp_splice_usage/<organ>/`` it reads the assignment
parquet (``..._prediction_clusters.parquet``: Chromosome / Position / pred_cluster
/ pred_shape) and the model's predicted usage (``usage_<species>.parquet`` →
``SSE_pred`` per timepoint), builds a per-site × 15-timepoint matrix of *predicted*
usage, and renders — with the same helper used for the reference clustering:
  <prefix>_pred_heatmap.png    sites (rows) ordered by assigned cluster
  <prefix>_pred_profiles.png   mean predicted trajectory per cluster, shape-badged

It also renders a **parallel OBSERVED (true-trajectory) pair** for the *same sites* under the
*same* pred_cluster grouping (from the saved ``<prefix>_obs_gp_features.npy``):
  <prefix>_obs_heatmap.png     true trajectories, same rows/clusters as the predicted heatmap
  <prefix>_obs_profiles.png    mean true trajectory per (predicted) cluster
so each cluster block holds the same sites in both figures — a direct true-vs-predicted
comparison. Disable with ``--no-reference``. (The obs pair needs the saved obs GP features, so
it is skipped in the raw-SSE fallback path.)

The predicted panels show the GP-smoothed predicted trajectories (or, in the fallback, raw
predicted SSE); the cluster membership and shape come from the reference-based assignment, so
the panels are directly comparable to the reference profiles in ``gp_splice_usage/``.

Usage
-----
  python plot_prediction_clusters.py \\
      --species human mouse rat rabbit opossum \\
      --organ Brain Cerebellum Liver Testis
"""

import os
import sys
import argparse
import logging

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from alphagenome_pytorch.plotting.splicing import save_cluster_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

TPS = list(range(1, 16))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds-dir",
                   default=("/home/elek/sds/sd17d003/Anamaria/alphagenome_genomicsxai/"
                            "lora_32_human_mouse_rat_rabbit_opossum/"
                            "preds_all_intersect_protein_coding"),
                   help="Root holding <species>/usage_<species>.parquet and "
                        "<species>/pred_gp_splice_usage/<organ>/.")
    p.add_argument("--species", nargs="+",
                   default=["human", "mouse", "rat", "rabbit", "opossum"])
    p.add_argument("--organ", nargs="+",
                   default=["Brain", "Cerebellum", "Liver", "Testis"])
    p.add_argument("--split", default="test")
    p.add_argument("--heatmap-height", type=float, default=9.0)
    p.add_argument("--no-reference", action="store_true",
                   help="Skip the parallel OBSERVED (true-trajectory) heatmap/profiles that use "
                        "the same sites and pred_cluster grouping as the predicted plots.")
    return p.parse_args()


def prefix(sp, organ, split):
    return f"{sp}_{organ.replace(' ', '_')}_{split}"


def pred_clusters_path(root, sp, organ, split):
    return os.path.join(root, sp, "pred_gp_splice_usage", organ.replace(" ", "_"),
                        f"{prefix(sp, organ, split)}_prediction_clusters.parquet")


def main():
    args = parse_args()
    n_ok = 0
    for sp in args.species:
        organs = [o for o in args.organ
                  if os.path.exists(pred_clusters_path(args.preds_dir, sp, o, args.split))]
        if not organs:
            log.warning("%s: no prediction-cluster parquets for %s — skipping", sp, args.organ)
            continue

        usage_path = os.path.join(args.preds_dir, sp, f"usage_{sp}.parquet")
        usg = None  # loaded lazily only if a tissue lacks saved GP features

        for organ in organs:
            pcp = pred_clusters_path(args.preds_dir, sp, organ, args.split)
            out_dir = os.path.dirname(pcp)
            pfx = prefix(sp, organ, args.split)
            pred_feat_path = os.path.join(out_dir, f"{pfx}_pred_gp_features.npy")
            obs_feat_path = os.path.join(out_dir, f"{pfx}_obs_gp_features.npy")

            import pyarrow.parquet as _pq
            _have = set(_pq.ParquetFile(pcp).schema.names)
            _cols = ["Chromosome", "Position", "pred_cluster", "pred_shape", "ref_cluster"]
            for _extra in ("pred_shape_site", "obs_shape_site"):   # per-site shapes (newer outputs)
                if _extra in _have:
                    _cols.append(_extra)
            meta = pd.read_parquet(pcp, columns=_cols)
            if meta.empty:
                log.warning("  [%s/%s] %s has 0 rows (no trajectories cleared "
                            "cluster_predictions.py's --min-timepoints) — skipping", sp, organ, pcp)
                continue
            meta["Chromosome"] = meta["Chromosome"].astype(str)
            # Total size of the FIXED reference clustering predictions were assigned into
            # (Ward cluster IDs are contiguous 1..N) — used so a given cluster ID gets the
            # same tab20 color here as in the reference heatmap, even when (as is typical)
            # not every reference cluster has a nearest predicted trajectory.
            n_ref_clusters = int(meta["ref_cluster"].max())

            # feats_obs (the OBSERVED trajectories for these exact sites) is rendered as a
            # parallel reference plot using the SAME pred_cluster grouping, so each cluster
            # block holds the same sites in both — directly comparable true vs predicted.
            feats_obs = None
            if os.path.exists(pred_feat_path):
                # Preferred: GP-smoothed features (row-aligned to the parquet), so the panels
                # exactly match the reference GP pipeline.
                feats = np.load(pred_feat_path).astype(np.float32)
                if len(feats) != len(meta):
                    log.warning("  [%s/%s] features/parquet length mismatch (%d vs %d) — skipping",
                                sp, organ, len(feats), len(meta)); continue
                pc = meta["pred_cluster"].to_numpy()
                shape_of = meta.drop_duplicates("pred_cluster").set_index("pred_cluster")["pred_shape"].to_dict()
                site_shapes = meta["pred_shape_site"].to_numpy() if "pred_shape_site" in meta.columns else None
                obs_site_shapes = meta["obs_shape_site"].to_numpy() if "obs_shape_site" in meta.columns else None
                if not args.no_reference and os.path.exists(obs_feat_path):
                    feats_obs = np.load(obs_feat_path).astype(np.float32)
                    if len(feats_obs) != len(meta):
                        log.warning("  [%s/%s] obs features length mismatch — skipping reference plot", sp, organ)
                        feats_obs = None
                src = "GP features"
            else:
                # Fallback: raw predicted SSE (gaps interpolated), matched by (chrom,pos).
                # The reference (observed) parallel plot needs the saved obs GP features, so
                # it's skipped in this legacy path.
                if not os.path.exists(usage_path):
                    log.warning("  [%s/%s] no GP features and no usage parquet — skipping", sp, organ); continue
                if usg is None:
                    log.info("[%s] loading predicted usage (no saved GP features) ...", sp)
                    usg = pd.read_parquet(usage_path,
                                          columns=["Chromosome", "Position", "Tissue", "Timepoint", "SSE_pred"])
                    usg["Chromosome"] = usg["Chromosome"].astype(str)
                sub = usg[usg["Tissue"] == organ]
                wide = (sub.pivot_table(index=["Chromosome", "Position"], columns="Timepoint",
                                        values="SSE_pred", aggfunc="mean").reindex(columns=TPS)
                           .interpolate(axis=1, limit_direction="both").dropna())
                m2 = meta.drop_duplicates(["Chromosome", "Position"]).set_index(["Chromosome", "Position"])
                common = wide.index.intersection(m2.index)
                if len(common) == 0:
                    log.warning("  [%s/%s] no usage/assignment overlap — skipping", sp, organ); continue
                feats = wide.loc[common].to_numpy(dtype=np.float32)
                pc = m2.loc[common, "pred_cluster"].to_numpy()
                shape_of = m2.reset_index().drop_duplicates("pred_cluster").set_index("pred_cluster")["pred_shape"].to_dict()
                site_shapes = (m2.loc[common, "pred_shape_site"].to_numpy()
                               if "pred_shape_site" in m2.columns else None)
                obs_site_shapes = None
                if not args.no_reference:
                    log.warning("  [%s/%s] no saved obs GP features — skipping reference plot "
                                "(re-run cluster_predictions.py to save them)", sp, organ)
                src = "raw predicted SSE"

            # Keep the original reference-cluster IDs (no renumbering) so labels and
            # colors line up exactly with the reference heatmap/profiles — only the
            # clusters that actually received a predicted trajectory are plotted.
            present = sorted(pd.unique(pc))
            cluster_shapes = {old: shape_of[old] for old in present}

            save_cluster_plots(feats, pc, cluster_shapes, len(present), out_dir,
                               f"{pfx}_pred", f"{sp}/{organ} [{args.split}] PREDICTED",
                               heatmap_height=args.heatmap_height,
                               cluster_ids=present, color_denom=n_ref_clusters,
                               site_shapes=site_shapes)
            made = f"{pfx}_pred"
            if feats_obs is not None:
                # Same sites, same pred_cluster grouping, same archetype (pred) shapes — only
                # the trajectories (and the per-site strip, obs_shape_site) differ, so the two
                # figures line up cluster-for-cluster for true-vs-predicted comparison.
                save_cluster_plots(feats_obs, pc, cluster_shapes, len(present), out_dir,
                                   f"{pfx}_obs", f"{sp}/{organ} [{args.split}] OBSERVED (same sites/clustering)",
                                   heatmap_height=args.heatmap_height,
                                   cluster_ids=present, color_denom=n_ref_clusters,
                                   site_shapes=obs_site_shapes)
                made = f"{pfx}_{{pred,obs}}"
            log.info("  [%s/%s] %s sites, %d clusters (%s) -> %s_{heatmap,profiles}.png",
                     sp, organ, f"{len(feats):,}", len(present), src, made)
            n_ok += 1
    log.info("=== Done: %d species/organ plot sets ===", n_ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
