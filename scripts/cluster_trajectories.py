#!/usr/bin/env python3
"""
cluster_trajectories.py
-----------------------
Cluster developmental splice site SSE trajectories using Gaussian process
regression + Ward-linkage hierarchical clustering, then annotate each cluster
with a shape label (up_early, down_late, flat_high, …).

The reusable building blocks (data prep, GP smoothing, shape classification,
k-selection, plotting) live in ``alphagenome_pytorch.plotting.splicing`` so they
are shared with the splice_trajectory_clustering / splice_trajectory_type_eval
notebooks. This script is the command-line driver around them.

Outputs
-------
  <prefix>_clustering_metadata.parquet   — per-trajectory cluster + shape labels + GP stats
  <prefix>_gp_features.npy               — GP posterior means (n_trajectories × 15)
  <prefix>_linkage.npy                   — Ward linkage matrix (for further analysis)
  <prefix>_gp_diagnostics.parquet        — per-trajectory GP hyperparameter diagnostics
  <prefix>_heatmap.png                   — (optional) heatmap
  <prefix>_profiles.png                  — (optional) cluster mean trajectory grid

Usage examples
--------------
  # Human, all tissues, 30 clusters, 16 parallel workers
  python cluster_trajectories.py \\
      --species human --n-clusters 30 \\
      --n-jobs 16 --output results/ --save-plots

  # All species, Brain only, auto-select k, save plots
  python cluster_trajectories.py \\
      --species all --tissue Brain \\
      --auto-k --n-jobs -1 --output results/ --save-plots

  # Resume: skip GP (reuse saved features), just re-cluster with different k
  python cluster_trajectories.py \\
      --species human --tissue Brain --n-clusters 50 \\
      --load-features results/human_brain_gp_features.npy \\
      --load-sites    results/human_brain_sites.parquet \\
      --output results/ --save-plots
"""

import os
import sys
import logging
import argparse
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # headless: set before importing the plotting utilities

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

from alphagenome_pytorch.plotting.splicing import (
    T_GRID,
    SHAPE_ORDER,
    prepare_trajectories,
    filter_to_split,
    smooth_all_trajectories,
    classify_cluster_shape,
    select_k_gap,
    save_cluster_plots,
)


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--parquet-template",
                   default=("/home/elek/sds/sd17d003/Anamaria/"
                            "alphagenome_genomicsxai/data/"
                            "combined_usage_data_{species}.parquet"),
                   help="Path template with {species} placeholder, or a direct path.")
    g.add_argument("--species", default="human",
                   help="Species (e.g. human) or 'all' for all species. "
                        "Default: human")
    g.add_argument("--tissue", default=None,
                   help="Tissue to filter (e.g. Brain). "
                        "Omit to use all tissues as separate trajectory dimensions.")
    g.add_argument("--data-config", default=None,
                   metavar="JSON",
                   help="Path to data_config.json. "
                        "When provided together with --split, sites are "
                        "filtered to those inside the split's genomic windows.")
    g.add_argument("--split", default="test",
                   choices=["test", "val", "train"],
                   help="Which split to keep (requires --data-config). "
                        "Default: test")
    g.add_argument("--min-timepoints", type=int, default=5,
                   help="Min detected timepoints per trajectory. Default: 5")
    g.add_argument("--min-reads", type=int, default=1)
    g.add_argument("--max-sites", type=int, default=None,
                   help="Random subsample limit applied *after* split filtering "
                        "(default: all sites).")
    g.add_argument("--random-seed", type=int, default=42)

    # GP
    g = p.add_argument_group("GP smoothing")
    g.add_argument("--gp-length-scale", type=float, default=0.20,
                   help="RBF kernel length scale in normalised time [0,1]. "
                        "0.20 ≈ 2-3 timepoint window. Default: 0.20")
    g.add_argument("--gp-noise-level", type=float, default=0.05)
    g.add_argument("--gp-n-restarts", type=int, default=2,
                   help="Hyperparameter optimiser restarts. Default: 2")
    g.add_argument("--gp-bound-tol", type=float, default=0.01,
                   help="Fraction of a bound's range within which a fitted "
                        "hyperparameter is flagged as 'hitting' that bound. "
                        "Default: 0.01 (1%%)")
    g.add_argument("--n-jobs", type=int, default=-1,
                   help="Parallel workers for GP fitting. "
                        "-1 = all cores (default). 1 = sequential.")
    g.add_argument("--load-features",
                   help="Path to previously saved .npy GP features array. "
                        "Skips GP fitting when provided.")
    g.add_argument("--load-sites",
                   help="Path to previously saved sites parquet (required with "
                        "--load-features).")

    # Clustering
    g = p.add_argument_group("Clustering")
    g.add_argument("--n-clusters", type=int, default=None,
                   help="Number of clusters. Omit to auto-select via gap statistic.")
    g.add_argument("--auto-k-min", type=int, default=5)
    g.add_argument("--auto-k-max", type=int, default=80)

    # Shape classification
    g = p.add_argument_group("Shape classification")
    g.add_argument("--flat-high", type=float, default=0.80,
                   help="Mean SSE >= this → flat_high. Default: 0.80")
    g.add_argument("--flat-low",  type=float, default=0.20,
                   help="Mean SSE <= this → flat_low. Default: 0.20")
    g.add_argument("--knee-frac", type=float, default=0.65,
                   help="Fraction of net change in one half → 'early'/'late'; "
                        "otherwise 'mid' (gradual). Default: 0.65")
    g.add_argument("--min-net-change", type=float, default=0.12,
                   help="Min absolute start-to-end SSE change for an up/down "
                        "label. Default: 0.12")
    g.add_argument("--amplitude-threshold", type=float, default=0.08,
                   help="Max amplitude to be considered flat. Default: 0.08")
    g.add_argument("--monotone-frac", type=float, default=0.65,
                   help="Min fraction of steps in one direction for monotone "
                        "classification. Default: 0.65")
    g.add_argument("--reversal-fraction", type=float, default=0.30,
                   help="Min size of each leg (as fraction of amplitude) for "
                        "up-down/down-up classification. Default: 0.30")

    # Output
    g = p.add_argument_group("Output")
    g.add_argument("--output", required=True,
                   help="Output directory.")
    g.add_argument("--prefix", default="",
                   help="Optional file prefix. Auto-generated from species/tissue "
                        "if empty.")
    g.add_argument("--save-plots", action="store_true",
                   help="Save heatmap and cluster profile plots.")

    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    species = None if args.species == "all" else args.species
    tissue  = args.tissue

    sp_part  = species or "all"
    tis_part = (tissue.replace(" ", "_") if tissue else "all_tissues")
    label    = f"{sp_part}/{tis_part}"
    prefix   = args.prefix or f"{sp_part}_{tis_part}"

    # ── 1. Load / prepare data ─────────────────────────────────────────────
    log.info("=== Data ===")
    if args.load_features:
        # Resume mode: skip GP, reuse saved features
        if not args.load_sites:
            sys.exit("--load-sites is required when using --load-features")
        log.info("Loading saved GP features from %s", args.load_features)
        features_v = np.load(args.load_features)
        sites_v    = pd.read_parquet(args.load_sites)
        lmls_v     = np.full(len(features_v), np.nan)
        stds_v     = np.zeros_like(features_v)
        diagnostics_v = None
        log.info("Loaded %s trajectories × %d timepoints",
                 f"{len(features_v):,}", features_v.shape[1])
    else:
        sites, sse_wide, reads_wide = prepare_trajectories(
            args.parquet_template,
            species=species, tissue=tissue,
            min_timepoints=args.min_timepoints,
            min_reads=args.min_reads,
            max_sites=None,           # subsample *after* split filtering
            random_seed=args.random_seed,
        )

        # ── 1b. Filter to split ────────────────────────────────────────────
        if args.data_config is not None:
            split_mask = filter_to_split(
                sites, args.data_config,
                species=species, split=args.split,
            )
            sites      = sites[split_mask].reset_index(drop=True)
            sse_wide   = sse_wide.iloc[np.where(split_mask)[0]]
            reads_wide = reads_wide.iloc[np.where(split_mask)[0]]
            label      = f"{sp_part}/{tis_part} [{args.split}]"
            prefix     = args.prefix or f"{sp_part}_{tis_part}_{args.split}"

        # ── 1c. Subsample if requested ─────────────────────────────────────
        if args.max_sites is not None and len(sites) > args.max_sites:
            rng_ss = np.random.default_rng(args.random_seed)
            idx_ss = np.sort(rng_ss.choice(len(sites), size=args.max_sites,
                                           replace=False))
            sites      = sites.iloc[idx_ss].reset_index(drop=True)
            sse_wide   = sse_wide.iloc[idx_ss]
            reads_wide = reads_wide.iloc[idx_ss]
            log.info("Subsampled to %s trajectories", f"{args.max_sites:,}")

        log.info("Final dataset: %s trajectories", f"{len(sites):,}")

        # ── 2. GP smoothing ────────────────────────────────────────────────
        log.info("=== GP smoothing ===")
        features, stds, lmls, diagnostics = smooth_all_trajectories(
            sse_wide, reads_wide, T_GRID,
            length_scale=args.gp_length_scale,
            noise_level=args.gp_noise_level,
            n_restarts=args.gp_n_restarts,
            n_jobs=args.n_jobs,
            bound_tol=args.gp_bound_tol,
        )

        valid      = ~np.isnan(features).any(axis=1)
        features_v = features[valid].astype(np.float32)
        stds_v     = stds[valid].astype(np.float32)
        lmls_v     = lmls[valid]
        sites_v    = sites.iloc[np.where(valid)[0]].reset_index(drop=True)
        diagnostics_v = diagnostics.iloc[np.where(valid)[0]].reset_index(drop=True)

        # Save GP features (allows re-clustering without re-fitting)
        out_feats = os.path.join(args.output, f"{prefix}_gp_features.npy")
        out_sites = os.path.join(args.output, f"{prefix}_sites.parquet")
        out_diag  = os.path.join(args.output, f"{prefix}_gp_diagnostics.parquet")
        np.save(out_feats, features_v)
        sites_v.to_parquet(out_sites, index=False)
        diagnostics.to_parquet(out_diag, index=False)
        log.info("Saved GP features → %s", out_feats)
        log.info("Saved sites       → %s", out_sites)
        log.info("Saved GP diagnostics → %s", out_diag)

    n = len(features_v)
    log.info("Working with %s valid trajectories", f"{n:,}")

    # ── 3. Hierarchical clustering ─────────────────────────────────────────
    log.info("=== Ward-linkage clustering ===")
    Z = linkage(features_v, method="ward", metric="euclidean")

    # Save linkage matrix
    out_Z = os.path.join(args.output, f"{prefix}_linkage.npy")
    np.save(out_Z, Z)
    log.info("Saved linkage → %s", out_Z)

    if args.n_clusters is not None:
        n_clusters = args.n_clusters
        log.info("Using k=%d (user-specified)", n_clusters)
    else:
        n_clusters = select_k_gap(Z, k_min=args.auto_k_min, k_max=args.auto_k_max)
        log.info("Auto-selected k=%d  (range %d–%d)",
                 n_clusters, args.auto_k_min, args.auto_k_max)

    cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # ── 4. Shape classification ────────────────────────────────────────────
    log.info("=== Shape classification ===")
    cluster_shapes = {}
    for k in range(1, n_clusters + 1):
        mean_k = features_v[cluster_labels == k].mean(axis=0)
        cluster_shapes[k] = classify_cluster_shape(
            mean_k,
            amplitude_threshold=args.amplitude_threshold,
            monotone_frac=args.monotone_frac,
            reversal_fraction=args.reversal_fraction,
            min_net_change=args.min_net_change,
            knee_frac=args.knee_frac,
            flat_high=args.flat_high,
            flat_low=args.flat_low,
        )

    log.info("%-8s  %-16s  %8s", "Cluster", "Shape", "N sites")
    log.info("-" * 38)
    for k, shape in cluster_shapes.items():
        nc = int((cluster_labels == k).sum())
        log.info("C%-7d  %-16s  %8s", k, shape, f"{nc:,}")

    # ── 5. Build metadata parquet ──────────────────────────────────────────
    log.info("=== Building metadata ===")
    meta = sites_v.copy()
    meta["Cluster"]      = cluster_labels
    meta["ClusterShape"] = meta["Cluster"].map(cluster_shapes)
    meta["GP_mean_SSE"]  = features_v.mean(axis=1)
    meta["GP_max_SSE"]   = features_v.max(axis=1)
    meta["GP_min_SSE"]   = features_v.min(axis=1)
    meta["GP_range"]     = meta["GP_max_SSE"] - meta["GP_min_SSE"]
    meta["GP_lml"]       = lmls_v

    if diagnostics_v is not None:
        meta["GP_fit_length_scale"]      = diagnostics_v["fit_length_scale"].values
        meta["GP_fit_noise_level"]       = diagnostics_v["fit_noise_level"].values
        meta["GP_hit_length_scale_lo"]   = diagnostics_v["hit_length_scale_lo"].values
        meta["GP_hit_length_scale_hi"]   = diagnostics_v["hit_length_scale_hi"].values
        meta["GP_hit_noise_level_lo"]    = diagnostics_v["hit_noise_level_lo"].values
        meta["GP_hit_noise_level_hi"]    = diagnostics_v["hit_noise_level_hi"].values

    _cmean  = {}
    _crange = {}
    for k in range(1, n_clusters + 1):
        mk = features_v[cluster_labels == k].mean(axis=0)
        _cmean[k]  = float(mk.mean())
        _crange[k] = float(mk.max() - mk.min())
    meta["ClusterMeanSSE"]  = meta["Cluster"].map(_cmean)
    meta["ClusterRangeSSE"] = meta["Cluster"].map(_crange)

    out_meta = os.path.join(args.output, f"{prefix}_clustering_metadata.parquet")
    meta.to_parquet(out_meta, index=False)
    log.info("Metadata → %s  (%s rows)", out_meta, f"{len(meta):,}")

    # Shape summary
    ss = Counter()
    for k, sh in cluster_shapes.items():
        ss[sh] += int((cluster_labels == k).sum())
    log.info("Shape distribution:")
    for sh in SHAPE_ORDER:
        if ss.get(sh, 0) > 0:
            log.info("  %-16s  %s trajectories", sh, f"{ss[sh]:,}")

    # ── 6. Plots ───────────────────────────────────────────────────────────
    if args.save_plots:
        log.info("=== Saving plots ===")
        save_cluster_plots(features_v, cluster_labels, cluster_shapes,
                           n_clusters, args.output, prefix, label,
                           random_seed=args.random_seed)

    log.info("=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
