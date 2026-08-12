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

These show the RAW predicted trajectories (not GP-smoothed); the cluster membership
and shape come from the reference-based assignment, so the panels are directly
comparable to the reference profiles in ``gp_splice_usage/``.

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
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TPS = list(range(1, 16))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-root",
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
                  if os.path.exists(pred_clusters_path(args.pred_root, sp, o, args.split))]
        if not organs:
            log.warning("%s: no prediction-cluster parquets for %s — skipping", sp, args.organ)
            continue

        usage_path = os.path.join(args.pred_root, sp, f"usage_{sp}.parquet")
        usg = None  # loaded lazily only if a tissue lacks saved GP features

        for organ in organs:
            pcp = pred_clusters_path(args.pred_root, sp, organ, args.split)
            out_dir = os.path.dirname(pcp)
            pfx = prefix(sp, organ, args.split)
            feat_path = os.path.join(out_dir, f"{pfx}_pred_gp_features.npy")

            meta = pd.read_parquet(pcp, columns=["Chromosome", "Position", "pred_cluster", "pred_shape"])
            meta["Chromosome"] = meta["Chromosome"].astype(str)

            if os.path.exists(feat_path):
                # Preferred: GP-smoothed predicted features (row-aligned to the parquet),
                # so the panels exactly match the reference GP pipeline.
                feats = np.load(feat_path).astype(np.float32)
                if len(feats) != len(meta):
                    log.warning("  [%s/%s] features/parquet length mismatch (%d vs %d) — skipping",
                                sp, organ, len(feats), len(meta)); continue
                pc = meta["pred_cluster"].to_numpy()
                shape_of = meta.drop_duplicates("pred_cluster").set_index("pred_cluster")["pred_shape"].to_dict()
                src = "GP features"
            else:
                # Fallback: raw predicted SSE (gaps interpolated), matched by (chrom,pos).
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
                src = "raw predicted SSE"

            # Remap assigned reference-cluster ids to a contiguous 1..K over the clusters
            # that actually received predictions (save_cluster_plots iterates 1..K).
            present = sorted(pd.unique(pc))
            remap = {old: i + 1 for i, old in enumerate(present)}
            labels = np.array([remap[c] for c in pc], dtype=int)
            cluster_shapes = {remap[old]: shape_of[old] for old in present}

            save_cluster_plots(feats, labels, cluster_shapes, len(present), out_dir,
                               f"{pfx}_pred", f"{sp}/{organ} [{args.split}] PREDICTED",
                               heatmap_height=args.heatmap_height)
            log.info("  [%s/%s] %s sites, %d clusters (%s) -> %s_pred_{heatmap,profiles}.png",
                     sp, organ, f"{len(feats):,}", len(present), src, pfx)
            n_ok += 1
    log.info("=== Done: %d species/organ plot sets ===", n_ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
