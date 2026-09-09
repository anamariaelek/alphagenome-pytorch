#!/usr/bin/env python3
"""
cluster_predictions.py
----------------------
Annotate *predicted* splice-site trajectories with cluster / shape labels by
assigning each prediction to the **fixed reference clustering of the observed
(true) trajectories** — never re-clustering the predictions independently.

For every (species, tissue):
  1. Load the saved true-trajectory clustering as a reference
     (``cluster_trajectories.py`` outputs: ``<prefix>_gp_features.npy`` +
     ``<prefix>_clustering_metadata.parquet``) and compute its per-cluster
     centroids in GP-feature space.
  2. GP-smooth the predicted trajectories with the *identical* transform used
     for the reference (same kernel / length-scale / noise, and the true read
     counts as observation weights) so predictions live in the same space.
  3. Label shapes two ways:
       - **per-site** (``*_shape_site``): run the shape classifier on each
         trajectory's OWN GP curve — the faithful, cluster-free label used for
         accuracy (matches the reference's ``ShapeSite``);
       - **cluster-mean** (``*_shape``): via nearest reference centroid →
         ``pred_cluster``/``pred_shape``, kept for the archetype/heatmap view.
  4. Join the site's true reference labels and write one row per site.

Output (per species/tissue): ``<prefix>_prediction_clusters.parquet`` with
  Chromosome, Position, Strand,
  ref_cluster, ref_shape, ref_shape_site,
  obs_cluster, obs_shape, obs_shape_site,
  pred_cluster, pred_shape, pred_shape_site,
  same_cluster, same_shape, same_shape_site,          (predicted vs observed)
  same_cluster_ref, same_shape_ref, same_shape_site_ref,  (predicted vs reference)
  pred_centroid_dist, obs_centroid_dist, n_obs
plus a logged nearest-centroid self-assignment accuracy (the classifier ceiling:
how well nearest-centroid reproduces the reference Ward partition).

``same_shape_site`` (per-site predicted vs observed shape) is the primary
shape-accuracy metric; ``same_shape`` is its cluster-mean counterpart.

The GP settings MUST match the reference run (defaults below mirror
``cluster_trajectories.py``); pass overrides if the reference used others.

Usage
-----
  # one species/tissue
  python cluster_predictions.py --species human --tissue Brain \\
      --ref-dir /home/.../gp_splice_usage --preds-dir /home/.../preds_all_intersect_protein_coding \\
      --usage-template /home/.../data/combined_usage_data_{species}.parquet \\
      --output /home/.../pred_clusters

  # all species x tissues found under --ref-dir
  python cluster_predictions.py --species all --tissue all \\
      --ref-dir /home/.../gp_splice_usage --preds-dir /home/.../preds_all_intersect_protein_coding \\
      --usage-template /home/.../data/combined_usage_data_{species}.parquet \\
      --output /home/.../pred_clusters --n-jobs 16
"""

import os
import sys
import glob
import logging
import argparse

import numpy as np
import pandas as pd

from alphagenome_pytorch.clustering import (
    T_GRID,
    smooth_all_trajectories,
    load_reference,
    cluster_centroids,
    assign_to_centroids,
    self_assignment_accuracy,
    load_pred_true_trajectories,
    classify_site_shapes,
    reference_prefix,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("Data")
    g.add_argument("--species", default="human",
                   help="Species, or 'all' to process every species under --ref-dir.")
    g.add_argument("--tissue", default="Brain",
                   help="Tissue, or 'all' to process every tissue under --ref-dir.")
    g.add_argument("--split", default="test", choices=["test", "val", "train"],
                   help="Reference split suffix (matches cluster_trajectories.py). "
                        "Default: test")
    g.add_argument("--ref-dir", required=True,
                   help="Root of the true-clustering outputs. Per species/tissue the "
                        "files are expected under <ref-dir>/<species>/<Tissue>/.")
    g.add_argument("--preds-dir", required=True,
                   help="Root of the predicted-usage parquets, one per species: "
                        "<preds-dir>/<species>/usage_<species>.parquet.")
    g.add_argument("--usage-template", required=True,
                   help="Combined true-usage parquet path template with {species} "
                        "(supplies Strand + Reads).")
    g.add_argument("--min-timepoints", type=int, default=5)
    g.add_argument("--min-reads", type=int, default=1)

    g = p.add_argument_group("GP smoothing (must match the reference run)")
    g.add_argument("--gp-length-scale", type=float, default=0.20)
    g.add_argument("--gp-noise-level", type=float, default=0.05)
    g.add_argument("--gp-n-restarts", type=int, default=2)
    g.add_argument("--gp-bound-tol", type=float, default=0.01)
    g.add_argument("--n-jobs", type=int, default=-1)

    g = p.add_argument_group("Per-site shape classification (must match the reference run)")
    g.add_argument("--exc-floor", type=float, default=0.08,
                   help="Excursion floor for the dynamic gate. Default: 0.08")
    g.add_argument("--exc-median-win", type=int, default=3,
                   help="Median-filter window for the excursion gate. Default: 3")
    g.add_argument("--reversal-fraction", type=float, default=0.30,
                   help="Min biphasic leg as a fraction of amplitude. Default: 0.30")
    g.add_argument("--biphasic-abs-leg", type=float, default=0.20,
                   help="Absolute min biphasic leg. Default: 0.20")

    g = p.add_argument_group("Output")
    g.add_argument("--output", required=True, help="Output directory.")
    g.add_argument("--overwrite", action="store_true",
                   help="Recompute even if the output parquet already exists.")

    return p.parse_args()


def _ref_files_exist(ref_dir, species, tissue, split):
    prefix = reference_prefix(species, tissue, split)
    d = os.path.join(ref_dir, species, tissue.replace(" ", "_"))
    return (os.path.exists(os.path.join(d, f"{prefix}_gp_features.npy"))
            and os.path.exists(os.path.join(d, f"{prefix}_clustering_metadata.parquet")))


def _discover(ref_dir, species, tissue, split):
    """Enumerate (species, tissue) pairs that have reference clustering outputs."""
    sp_list = ([species] if species != "all"
               else sorted(d for d in os.listdir(ref_dir)
                           if os.path.isdir(os.path.join(ref_dir, d))))
    pairs = []
    for sp in sp_list:
        sp_dir = os.path.join(ref_dir, sp)
        if not os.path.isdir(sp_dir):
            continue
        if tissue != "all":
            tissues = [tissue]
        else:
            tissues = sorted(t for t in os.listdir(sp_dir)
                             if os.path.isdir(os.path.join(sp_dir, t)))
        for t in tissues:
            if _ref_files_exist(ref_dir, sp, t, split):
                pairs.append((sp, t))
            else:
                log.warning("skip %s/%s: no reference clustering under %s", sp, t, ref_dir)
    return pairs


def annotate(species, tissue, args):
    prefix = reference_prefix(species, tissue, args.split)
    ref_subdir = os.path.join(args.ref_dir, species, tissue.replace(" ", "_"))
    out_path = os.path.join(args.output, f"{prefix}_prediction_clusters.parquet")
    if os.path.exists(out_path) and not args.overwrite:
        log.info("%s/%s: exists, skipping (%s)", species, tissue, out_path)
        return out_path

    log.info("=== %s / %s ===", species, tissue)

    # 1. reference clustering + centroids
    ref = load_reference(ref_subdir, species, tissue, split=args.split, prefix=prefix)
    acc = self_assignment_accuracy(ref)
    log.info("reference: %s sites, %d clusters; nearest-centroid self-assignment "
             "accuracy = %.3f", f"{len(ref['meta']):,}", len(ref["cluster_ids"]), acc)

    # 2. predicted AND observed trajectories, GP-smoothed with the identical transform
    preds_parquet = os.path.join(args.preds_dir, species, f"usage_{species}.parquet")
    usage_parquet = args.usage_template.format(species=species)
    if not os.path.exists(preds_parquet):
        log.warning("%s/%s: predictions not found (%s), skipping", species, tissue, preds_parquet)
        return None
    sites, true_wide, pred_wide, reads_wide = load_pred_true_trajectories(
        preds_parquet, usage_parquet, species, tissue,
        min_timepoints=args.min_timepoints, min_reads=args.min_reads)
    log.info("trajectories: %s (>= %d observed timepoints)", f"{len(sites):,}", args.min_timepoints)

    gp = dict(length_scale=args.gp_length_scale, noise_level=args.gp_noise_level,
              n_restarts=args.gp_n_restarts, n_jobs=args.n_jobs, bound_tol=args.gp_bound_tol)
    log.info("GP smoothing observed trajectories ...")
    feat_true, *_ = smooth_all_trajectories(true_wide, reads_wide, T_GRID, **gp)
    log.info("GP smoothing predicted trajectories ...")
    feat_pred, *_ = smooth_all_trajectories(pred_wide, reads_wide, T_GRID, **gp)

    valid = ~(np.isnan(feat_true).any(axis=1) | np.isnan(feat_pred).any(axis=1))
    feat_true, feat_pred = feat_true[valid], feat_pred[valid]
    sites = sites.iloc[np.where(valid)[0]].reset_index(drop=True)

    # 3a. cluster membership: assign predicted & observed trajectories to reference
    #     centroids (kept for the archetype/heatmap view — plot_prediction_clusters.py).
    c2s = ref["cluster_to_shape"]
    pc, pdist = assign_to_centroids(feat_pred, ref["centroids"], ref["cluster_ids"])
    tc, tdist = assign_to_centroids(feat_true, ref["centroids"], ref["cluster_ids"])

    # 3b. per-site shapes: label each trajectory on its OWN GP curve (the source of
    #     truth for accuracy — no cluster-mean averaging; matches the reference's ShapeSite).
    shp = dict(exc_floor=args.exc_floor, median_win=args.exc_median_win,
               reversal_fraction=args.reversal_fraction, biphasic_abs_leg=args.biphasic_abs_leg)
    pred_shape_site = classify_site_shapes(feat_pred, **shp)
    obs_shape_site = classify_site_shapes(feat_true, **shp)

    out = sites.copy()
    out["Chromosome"] = out["Chromosome"].astype(str)
    out["_ridx"] = np.arange(len(out))          # row index into feat_pred/feat_true
    out["pred_cluster"] = pc
    out["pred_shape"] = pd.Series(pc).map(c2s).to_numpy()      # cluster-mean shape (viz)
    out["obs_cluster"] = tc                                    # observed traj, same pipeline
    out["obs_shape"] = pd.Series(tc).map(c2s).to_numpy()
    out["pred_shape_site"] = pred_shape_site                   # per-site shape (stats)
    out["obs_shape_site"] = obs_shape_site
    out["pred_centroid_dist"] = pdist
    out["obs_centroid_dist"] = tdist
    out["n_obs"] = (reads_wide.values[valid] > 0).sum(axis=1)

    # 4a. apples-to-apples: predicted vs observed
    out["same_cluster"] = out["pred_cluster"] == out["obs_cluster"]
    out["same_shape"] = out["pred_shape"] == out["obs_shape"]              # cluster-mean
    out["same_shape_site"] = out["pred_shape_site"] == out["obs_shape_site"]  # per-site

    # 4b. vs the canonical reference labels of the observed trajectory
    ref_meta = ref["meta"][["Chromosome", "Position", "Strand", "Cluster",
                            "ClusterShape", "ShapeSite"]].copy()
    ref_meta["Chromosome"] = ref_meta["Chromosome"].astype(str)
    ref_meta = ref_meta.rename(columns={"Cluster": "ref_cluster",
                                        "ClusterShape": "ref_shape",
                                        "ShapeSite": "ref_shape_site"})
    out = out.merge(ref_meta, on=["Chromosome", "Position", "Strand"], how="left")
    out["same_cluster_ref"] = out["pred_cluster"] == out["ref_cluster"]
    out["same_shape_ref"] = out["pred_shape"] == out["ref_shape"]
    out["same_shape_site_ref"] = out["pred_shape_site"] == out["ref_shape_site"]

    # GP-smoothed predicted / observed features, aligned to the final parquet row order
    # (via _ridx, so they survive the ref_meta merge). Saved for downstream plotting.
    order = out["_ridx"].to_numpy()
    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, f"{prefix}_pred_gp_features.npy"), feat_pred[order])
    np.save(os.path.join(args.output, f"{prefix}_obs_gp_features.npy"), feat_true[order])

    cols = ["Chromosome", "Position", "Strand",
            "ref_cluster", "ref_shape", "ref_shape_site",
            "obs_cluster", "obs_shape", "obs_shape_site",
            "pred_cluster", "pred_shape", "pred_shape_site",
            "same_cluster", "same_shape", "same_shape_site",
            "same_cluster_ref", "same_shape_ref", "same_shape_site_ref",
            "pred_centroid_dist", "obs_centroid_dist", "n_obs"]
    out = out[cols]
    out.to_parquet(out_path, index=False)
    log.info("%s/%s: %s sites | per-site pred-vs-obs same_shape_site=%.3f "
             "(pred-vs-ref=%.3f) | cluster-mean same_shape=%.3f -> %s",
             species, tissue, f"{len(out):,}",
             out["same_shape_site"].mean(), out["same_shape_site_ref"].mean(),
             out["same_shape"].mean(), out_path)
    return out_path


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    pairs = _discover(args.ref_dir, args.species, args.tissue, args.split)
    if not pairs:
        sys.exit("No (species, tissue) pairs with reference clustering found.")
    log.info("Processing %d (species, tissue) pair(s)", len(pairs))
    for sp, t in pairs:
        try:
            annotate(sp, t, args)
        except Exception as e:
            log.exception("%s/%s failed: %s", sp, t, e)
    log.info("=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
