#!/usr/bin/env python3
"""
relabel_cluster_shapes.py
--------------------------
Re-label an *existing* reference clustering (and its downstream prediction-cluster
annotations) with the new hierarchical dynamic/direction shape scheme
(``alphagenome_pytorch.clustering.classify_dynamic_direction``), without
re-running GP smoothing, Ward linkage, or nearest-centroid assignment -- only the
human-readable shape LABEL attached to each (already-fixed) cluster ID changes.

For each reference clustering found under --ref-dir (a
``<prefix>_clustering_metadata.parquet`` + `<prefix>_gp_features.npy`` pair):
  1. Recompute each cluster's centroid (mean GP feature vector) from the existing
     Cluster assignments -- exactly what cluster_trajectories.py itself computes,
     just reusing already-saved features instead of the raw data.
  2. Reclassify each centroid with classify_dynamic_direction -> a new
     {cluster_id: shape} map.
  3. Back up the original metadata parquet (once, as *.legacy_bak.parquet) and
     overwrite its ClusterShape column with the new labels.

For each prediction-cluster parquet found under --preds-dir (matched to a
reference by species/tissue), the pred_cluster/obs_cluster/ref_cluster ID
assignments are unchanged (clustering itself is untouched), so this just remaps
pred_shape/obs_shape/ref_shape through the corresponding reference's new
{cluster_id: shape} map and recomputes same_shape/same_shape_ref.

Usage
-----
  python relabel_cluster_shapes.py \\
      --ref-dir /path/to/gp_splice_usage \\
      --preds-dir /path/to/<model>/preds_intersect_protein_coding \\
      --split test
"""

import os
import sys
import glob
import logging
import argparse
import shutil

import numpy as np
import pandas as pd

from alphagenome_pytorch.clustering import (
    cluster_centroids,
    classify_dynamic_direction,
    classify_site_shapes,
    shape_fraction_summary,
    reference_prefix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-dir", required=True, help="Root of the reference clusterings (per species/tissue).")
    p.add_argument("--preds-dir", default=None,
                   help="Root of the prediction-cluster outputs to also relabel "
                        "(<preds-dir>/<species>/pred_gp_splice_usage/<Tissue>/). Optional.")
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--exc-floor", type=float, default=0.08)
    p.add_argument("--exc-median-win", type=int, default=3)
    p.add_argument("--reversal-fraction", type=float, default=0.30)
    p.add_argument("--biphasic-abs-leg", type=float, default=0.20)
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change without writing anything.")
    return p.parse_args()


def relabel_reference(meta_path, feats_path, args):
    """Returns the new {cluster_id: shape} map for this reference clustering."""
    meta = pd.read_parquet(meta_path)
    feats = np.load(feats_path)
    if len(feats) != len(meta):
        raise ValueError(f"features ({len(feats)}) and metadata ({len(meta)}) not aligned for {meta_path}")

    labels = meta["Cluster"].to_numpy()
    centroids, cluster_ids = cluster_centroids(feats, labels)
    new_shapes = {}
    for cid, centroid in zip(cluster_ids.tolist(), centroids):
        new_shapes[cid] = classify_dynamic_direction(
            centroid, exc_floor=args.exc_floor, median_win=args.exc_median_win,
            reversal_fraction=args.reversal_fraction, biphasic_abs_leg=args.biphasic_abs_leg,
        )

    old_dist = meta["ClusterShape"].value_counts()
    new_col = meta["Cluster"].map(new_shapes)
    new_dist = new_col.value_counts()
    # Per-site labels (each trajectory on its own GP curve) — the source for shape stats.
    site_shapes = classify_site_shapes(
        feats, exc_floor=args.exc_floor, median_win=args.exc_median_win,
        reversal_fraction=args.reversal_fraction, biphasic_abs_leg=args.biphasic_abs_leg)
    log.info("  %s clusters, %s sites", f"{len(cluster_ids)}", f"{len(meta):,}")
    log.info("  old shape distribution (cluster): %s", dict(old_dist))
    log.info("  new shape distribution (cluster): %s", dict(new_dist))
    log.info("  per-site shape distribution:      %s", dict(pd.Series(site_shapes).value_counts()))

    if not args.dry_run:
        backup_path = meta_path.replace(".parquet", ".legacy_bak.parquet")
        if not os.path.exists(backup_path):
            shutil.copy2(meta_path, backup_path)
            log.info("  backed up -> %s", backup_path)
        meta["ClusterShape"] = new_col
        meta["ShapeSite"] = site_shapes
        meta.to_parquet(meta_path, index=False)
        log.info("  wrote -> %s", meta_path)
        # refresh the shape-fraction summary (computed from the per-site labels)
        frac = shape_fraction_summary(meta, shape_col="ShapeSite")
        frac_path = meta_path.replace("_clustering_metadata.parquet", "_shape_fractions.csv")
        frac.to_csv(frac_path, index=False)
        log.info("  shape fractions -> %s  (from ShapeSite)", frac_path)

    return new_shapes


def relabel_predictions(pred_path, new_shapes, args):
    df = pd.read_parquet(pred_path)
    old_same_shape = df["same_shape"].mean() if "same_shape" in df.columns else float("nan")

    df["pred_shape"] = df["pred_cluster"].map(new_shapes)
    df["obs_shape"] = df["obs_cluster"].map(new_shapes)
    df["ref_shape"] = df["ref_cluster"].map(new_shapes)
    df["same_shape"] = df["pred_shape"] == df["obs_shape"]
    df["same_shape_ref"] = df["pred_shape"] == df["ref_shape"]

    new_same_shape = df["same_shape"].mean()
    log.info("  same_shape: %.3f -> %.3f  (n=%s)", old_same_shape, new_same_shape, f"{len(df):,}")

    if not args.dry_run:
        backup_path = pred_path.replace(".parquet", ".legacy_bak.parquet")
        if not os.path.exists(backup_path):
            shutil.copy2(pred_path, backup_path)
            log.info("  backed up -> %s", backup_path)
        df.to_parquet(pred_path, index=False)
        log.info("  wrote -> %s", pred_path)


def main():
    args = parse_args()

    meta_paths = sorted(glob.glob(os.path.join(args.ref_dir, "*", "*", f"*_{args.split}_clustering_metadata.parquet")))
    if not meta_paths:
        sys.exit(f"No *_{args.split}_clustering_metadata.parquet found under {args.ref_dir}")
    log.info("Found %d reference clustering(s)", len(meta_paths))

    ref_shape_maps = {}  # (species, tissue) -> {cluster_id: shape}
    for meta_path in meta_paths:
        # <ref_dir>/<species>/<Tissue>/<species>_<Tissue>_<split>_clustering_metadata.parquet
        tissue_dir = os.path.dirname(meta_path)
        species = os.path.basename(os.path.dirname(tissue_dir))
        tissue = os.path.basename(tissue_dir)
        prefix = reference_prefix(species, tissue, args.split)
        feats_path = os.path.join(tissue_dir, f"{prefix}_gp_features.npy")
        if not os.path.exists(feats_path):
            log.warning("skip %s/%s: no gp_features.npy next to %s", species, tissue, meta_path)
            continue
        log.info("=== reference: %s / %s ===", species, tissue)
        ref_shape_maps[(species, tissue)] = relabel_reference(meta_path, feats_path, args)

    if args.preds_dir:
        pred_paths = sorted(glob.glob(os.path.join(
            args.preds_dir, "*", "pred_gp_splice_usage", "*", f"*_{args.split}_prediction_clusters.parquet")))
        log.info("Found %d prediction-cluster file(s)", len(pred_paths))
        for pred_path in pred_paths:
            tissue_dir = os.path.dirname(pred_path)
            tissue = os.path.basename(tissue_dir)
            species = os.path.basename(os.path.dirname(os.path.dirname(tissue_dir)))
            key = (species, tissue)
            if key not in ref_shape_maps:
                log.warning("skip %s: no matching reference relabeled for %s/%s", pred_path, species, tissue)
                continue
            log.info("=== predictions: %s / %s ===", species, tissue)
            relabel_predictions(pred_path, ref_shape_maps[key], args)

    log.info("=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
