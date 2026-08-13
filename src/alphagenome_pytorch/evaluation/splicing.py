"""Splicing evaluation utilities: loading predictions/usage stats, computing
accuracy metrics (AUPRC, Pearson r), genomic-feature overlap, and developmental-
dynamics classification.

Pure data loading and computation — no matplotlib. Plot functions that consume
this module's outputs live in ``alphagenome_pytorch.plotting.splicing``.
"""

import json
import os

import numpy as np
import pandas as pd

from alphagenome_pytorch.plotting.splicing import SPECIES_SCI


def load_splice_predictions(model_dir, subsets=("gtf", "usage", "union", "intersect"),
                             species_list=("human", "mouse", "rat", "rabbit", "opossum")):
    """Load per-species/per-subset classification metrics.json files into a nested dict."""
    evals = {}
    if subsets is None:
        subsets = ["preds"]

    for sub_key in subsets:
        pd_key = "preds" if sub_key == "preds" else "preds_" + sub_key
        data_dir = os.path.join(model_dir, pd_key)
        print(f"Loading predictions from {data_dir}")

        for species in species_list:
            pred_file = os.path.join(data_dir, species, "metrics.json")
            if not os.path.exists(pred_file):
                print(f"Predictions for {species} in {pd_key} are missing.")
                continue
            with open(pred_file, "r") as f:
                metrics = json.load(f)
            evals.setdefault(species, {})
            evals[species].setdefault(pd_key, {"sites": {}, "auprc": {}, "usage": {}})
            sp_metrics = metrics.get(species, {})
            per_class_auprc = sp_metrics.get("per_class_auprc", {}) or {}
            per_class_positives = sp_metrics.get("per_class_n_positives", {}) or {}
            evals[species][pd_key]["sites"] = per_class_positives
            auprc = dict(per_class_auprc)
            auprc["binary"] = sp_metrics.get("binary_auprc", None)
            evals[species][pd_key]["auprc"] = auprc
            overall_cor = sp_metrics.get("usage_by_head", {}).get("same_species", None)
            tissue_cor = sp_metrics.get("per_tissue_by_head", {}).get("same_species", None)
            evals[species][pd_key]["usage"] = {"overall": overall_cor, "tissue": tissue_cor}

    return evals


def pearson_r_from_stats(n, sum_p, sum_t, sum_p2, sum_t2, sum_pt):
    """Compute Pearson r from sufficient statistics."""
    denom = np.sqrt((n * sum_p2 - sum_p**2) * (n * sum_t2 - sum_t**2))
    if denom < 1e-12:
        return float("nan")
    return float((n * sum_pt - sum_p * sum_t) / denom)


def split_condition_label(label):
    """Split 'Tissue_Timepoint' -> (tissue, timepoint)."""
    if "_" in label:
        tissue, timepoint = label.rsplit("_", 1)
        return tissue, timepoint
    return label, None


def load_splice_usage_stats(model_dir, ann_data_dir, subsets=None,
                             species_list=("human", "mouse", "rat", "rabbit", "opossum")):
    """Load per-species/per-subset usage_*.npz sufficient statistics, aggregate
    Pearson r per condition and per tissue, and cache the result as JSON next to
    ``model_dir``."""
    if subsets is None:
        subsets = ["preds"]

    usage_results = {}

    for sps in species_list:
        metadata_file = os.path.join(ann_data_dir, SPECIES_SCI[sps], "usage.json")
        if not os.path.exists(metadata_file):
            print(f"Metadata missing for {sps}: {metadata_file}")
            continue

        with open(metadata_file) as f:
            metadata = json.load(f)

        idx_to_label = {int(v): k for k, v in metadata.get("condition_labels", {}).items()}
        usage_results[sps] = {}

        for sub_key in subsets:
            pd_key = "preds" if sub_key == "preds" else f"preds_{sub_key}"

            usage_file = os.path.join(model_dir, pd_key, sps, f"usage_{sps}.npz")
            if not os.path.exists(usage_file):
                print(f"  Usage file missing: {sps} / {pd_key}")
                continue

            data = np.load(usage_file)
            cond_ids = data["stats_cond_ids"]
            n_arr = data["stats_n"]
            sum_p = data["stats_sum_pred"]
            sum_t = data["stats_sum_true"]
            sum_p2 = data["stats_sum_pred2"]
            sum_t2 = data["stats_sum_true2"]
            sum_pt = data["stats_sum_prod"]

            per_condition = {}
            tissue_stats = {}

            for i, cond_idx in enumerate(cond_ids):
                label = idx_to_label.get(int(cond_idx), f"cond_{cond_idx}")
                tissue, timepoint = split_condition_label(label)

                r_cond = pearson_r_from_stats(
                    n_arr[i], sum_p[i], sum_t[i], sum_p2[i], sum_t2[i], sum_pt[i]
                )

                per_condition[label] = {
                    "tissue": tissue,
                    "timepoint": timepoint,
                    "pearson_r": r_cond,
                    "r_squared": r_cond**2 if np.isfinite(r_cond) else float("nan"),
                    "n": int(n_arr[i]),
                }

                if tissue not in tissue_stats:
                    tissue_stats[tissue] = {
                        "n": 0,
                        "sum_p": 0.0,
                        "sum_t": 0.0,
                        "sum_p2": 0.0,
                        "sum_t2": 0.0,
                        "sum_pt": 0.0,
                        "r_per_cond": [],
                        "timepoints": {},
                    }

                ts = tissue_stats[tissue]
                ts["n"] += int(n_arr[i])
                ts["sum_p"] += float(sum_p[i])
                ts["sum_t"] += float(sum_t[i])
                ts["sum_p2"] += float(sum_p2[i])
                ts["sum_t2"] += float(sum_t2[i])
                ts["sum_pt"] += float(sum_pt[i])
                ts["r_per_cond"].append(r_cond)
                ts["timepoints"][str(timepoint)] = per_condition[label]

            per_tissue = {}
            for tissue, ts in sorted(tissue_stats.items()):
                r_pooled = pearson_r_from_stats(
                    ts["n"], ts["sum_p"], ts["sum_t"], ts["sum_p2"], ts["sum_t2"], ts["sum_pt"]
                )
                valid_r = [r for r in ts["r_per_cond"] if np.isfinite(r)]

                per_tissue[tissue] = {
                    "pearson_r": r_pooled,
                    "r_squared": r_pooled**2 if np.isfinite(r_pooled) else float("nan"),
                    "std_pearson_r": float(np.std(valid_r)) if valid_r else float("nan"),
                    "mean_pearson_r": float(np.mean(valid_r)) if valid_r else float("nan"),
                    "n": ts["n"],
                    "n_conditions": len(ts["r_per_cond"]),
                    "timepoints": dict(sorted(ts["timepoints"].items(), key=lambda x: x[0])),
                }

            r_overall = pearson_r_from_stats(
                n_arr.sum(), sum_p.sum(), sum_t.sum(),
                sum_p2.sum(), sum_t2.sum(), sum_pt.sum()
            )

            usage_results[sps][pd_key] = {
                "overall_pearson_r": r_overall,
                "overall_r_squared": r_overall**2 if np.isfinite(r_overall) else float("nan"),
                "per_tissue": per_tissue,
                "per_condition": per_condition,
            }

            tissue_str = ", ".join(f"{t}={v['pearson_r']:.3f}" for t, v in per_tissue.items())
            print(f"  {sps} {pd_key}: overall r={r_overall:.3f}  [{tissue_str}]")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    usage_results_path = os.path.join(model_dir, "prediction_splice_usage_results.json")
    with open(usage_results_path, "w") as f:
        json.dump(usage_results, f, indent=4, cls=NumpyEncoder)

    print(f"\nSaved to {usage_results_path}")
    return usage_results


def build_interval_index(gtf_df):
    """Build per-chromosome NCLS interval trees, keyed by (chrom, strand)."""
    from ncls import NCLS

    index = {}
    for (chrom, strand), grp in gtf_df.groupby(['Chromosome', 'Strand']):
        starts = grp['Start'].values.astype(np.int64)
        ends   = grp['End'].values.astype(np.int64)
        ids    = grp.index.values.astype(np.int64)
        index[(chrom, strand)] = NCLS(starts, ends, ids)
    return index


def ovl_feature_vectorized(ann_df, gtf_df, gtf_index):
    """Overlap each row of ``ann_df`` (a splice site, using its 1bp Position and the
    strand encoded in its SiteType suffix) with GTF features via the NCLS index
    built by :func:`build_interval_index`."""
    from collections import defaultdict

    results = {
        'Overlapping_Feature': ['Intergenic'] * len(ann_df),
        'gene_type':    [None] * len(ann_df),
        'gene':         [None] * len(ann_df),
        'transcript':   [None] * len(ann_df),
        'exon':         [None] * len(ann_df)
    }

    # Derive strand from SiteType
    ann_df = ann_df.copy()
    ann_df['_strand'] = ann_df['SiteType'].str[-1]  # '+' or '-'

    for (chrom, strand), grp in ann_df.groupby(['Chromosome', '_strand']):
        key = (chrom, strand)
        if key not in gtf_index:
            continue

        pos_array = grp['Position'].values.astype(np.int64)
        row_idx   = grp.index.values.astype(np.int64)  # index into ann_df

        # Batch query: returns parallel arrays of (ann_idx, gtf_idx)
        ann_hits, gtf_hits = gtf_index[key].all_overlaps_both(
            pos_array,
            pos_array + 1,
            row_idx           # <-- these are carried through as the query IDs
        )

        if len(ann_hits) == 0:
            continue

        # Group gtf row hits by ann_df row index
        hit_map = defaultdict(list)
        for a_idx, g_idx in zip(ann_hits, gtf_hits):
            hit_map[a_idx].append(g_idx)

        for a_idx, g_idxs in hit_map.items():
            ovl = gtf_df.loc[g_idxs]
            features = ";".join(ovl['Feature'].unique())
            results['Overlapping_Feature'][a_idx] = features
            results['gene_type'][a_idx]    = ";".join(ovl.loc[ovl['Feature'] == 'gene_type',   'gene_type'   ].dropna().unique()) or None
            results['gene'][a_idx]         = ";".join(ovl.loc[ovl['Feature'] == 'gene',        'gene_id'      ].dropna().unique()) or None
            results['transcript'][a_idx]   = ";".join(ovl.loc[ovl['Feature'] == 'transcript',  'transcript_id'].dropna().unique()) or None
            results['exon'][a_idx]         = ";".join(ovl.loc[ovl['Feature'] == 'exon',        'exon_number'  ].dropna().astype(str).unique()) or None

    return pd.DataFrame(results, index=ann_df.index)


def fast_auprc(y_true, y_scores):
    """Vectorized AUPRC (trapezoidal precision-recall integration)."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_scores = np.asarray(y_scores, dtype=np.float32)

    desc_idx = np.argsort(y_scores)[::-1]
    y_true = y_true[desc_idx]

    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)

    precision = tp / (tp + fp)
    recall = tp / tp[-1] if tp[-1] > 0 else tp

    return np.trapezoid(precision, recall)


def load_species_data(species_list, base_dir, blacklist=None):
    """Load per-species ``predictions_{sp}_auprc_by_feature_category.csv`` files."""
    dfs = {}
    for sp in species_list:
        fn = os.path.join(base_dir, sp, f"predictions_{sp}_auprc_by_feature_category.csv")
        if os.path.exists(fn):
            df = pd.read_csv(fn)
            if blacklist is not None:
                df = df[~df['category'].isin(blacklist)]
            print(f"Loaded {sp}: {len(df)} rows")
            dfs[sp] = df
        else:
            print(f"Not found, skipping: {fn}")
    return dfs


def get_categories_by_auprc(species_dfs, min_positives=0, blacklist=None):
    """Return union of feature-overlap categories across species, sorted by median AUPRC."""
    all_cats = None
    for sp, df in species_dfs.items():
        df_filt = df[df['n_positives'] > min_positives]
        cats = set(df_filt['category'].unique())
        all_cats = cats if all_cats is None else all_cats | cats  # union instead of intersection

    records = []
    for sp, df in species_dfs.items():
        df_filt = df[df['n_positives'] > min_positives]
        for cat in all_cats:
            median_auprc = df_filt[df_filt['category'] == cat]['auprc'].median()
            if not np.isnan(median_auprc):  # skip if category absent in this species
                records.append({'category': cat, 'auprc': median_auprc})

    order = (pd.DataFrame(records)
             .groupby('category')['auprc']
             .median()
             .sort_values(ascending=False)
             .index.tolist())
    return order


def load_pearson_r_data(species_list, base_dir, blacklist=None):
    """Load per-species ``usage_{sp}_pearson_r_by_feature_category.csv`` files."""
    dfs = {}
    for sp in species_list:
        fn = os.path.join(base_dir, sp, f"usage_{sp}_pearson_r_by_feature_category.csv")
        if os.path.exists(fn):
            df = pd.read_csv(fn)
            if blacklist is not None:
                df = df[~df['category'].isin(blacklist)]
            print(f"Loaded {sp}: {len(df)} rows")
            dfs[sp] = df
        else:
            print(f"Not found, skipping: {fn}")
    return dfs


def get_categories_by_pearson_r(species_dfs, min_positives=0, blacklist=None):
    """Return union of feature-overlap categories across species, sorted by median Pearson r."""
    all_cats = None
    for sp, df in species_dfs.items():
        df_filt = df[df['n_sites'] > min_positives]
        cats = set(df_filt['category'].unique())
        all_cats = cats if all_cats is None else all_cats | cats  # union instead of intersection

    records = []
    for sp, df in species_dfs.items():
        df_filt = df[df['n_sites'] > min_positives]
        for cat in all_cats:
            median_pearson_r = df_filt[df_filt['category'] == cat]['pearson_r'].median()
            if not np.isnan(median_pearson_r):  # skip if category absent in this species
                records.append({'category': cat, 'pearson_r': median_pearson_r})

    order = (pd.DataFrame(records)
             .groupby('category')['pearson_r']
             .median()
             .sort_values(ascending=False)
             .index.tolist())
    return order


def classify_sites_by_true_usage_variance(
    usage_data,
    tissue,
    min_timepoints=5,
    aggfunc="mean",
    low_threshold=0.2,
    high_threshold=0.8,
    amplitude_threshold=0.2,
    direction_change_threshold=0.2,
    n_interp=1000,
):
    """
    Classify each unique site (chr_pos) in a tissue as:
      - dynamics_class: 'dynamic' or 'stable'
      - dynamics_subclass:
          stable -> 'high' / 'low' / 'median'
          dynamic -> 'up' / 'down' / 'up-down' / 'down-up'

    Rules:
      - stable if spline amplitude <= amplitude_threshold
      - stable if total directional change < direction_change_threshold
      - dynamic only if both amplitude and directional-change thresholds are met
    """

    from scipy.interpolate import CubicSpline

    required_keys = {"chr_pos", "tissues", "timepoints", "trues"}
    missing = required_keys - set(usage_data.keys())
    if missing:
        raise ValueError(f"usage_data is missing required keys: {sorted(missing)}")
    if not (0 <= low_threshold < high_threshold <= 1):
        raise ValueError("Expected 0 <= low_threshold < high_threshold <= 1")

    df = pd.DataFrame({
        "chr_pos": usage_data["chr_pos"],
        "tissue": usage_data["tissues"],
        "timepoint": usage_data["timepoints"],
        "true_usage": usage_data["trues"],
    })

    df = df[df["tissue"] == tissue].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "chr_pos", "tissue", "n_obs", "n_timepoints",
            "mean_true_usage", "true_usage_variance",
            "dynamics_class", "dynamics_subclass"
        ])

    df["timepoint_num"] = pd.to_numeric(df["timepoint"], errors="coerce")
    df = df.dropna(subset=["timepoint_num"]).copy()

    site_tp = (
        df.groupby(["chr_pos", "timepoint_num"], as_index=False)["true_usage"]
        .agg(aggfunc)
        .sort_values(["chr_pos", "timepoint_num"])
    )

    summary = (
        site_tp.groupby("chr_pos")["true_usage"]
        .agg(
            n_obs="size",
            n_timepoints="count",
            mean_true_usage="mean",
            true_usage_variance="var",
        )
        .reset_index()
    )

    summary["tissue"] = tissue
    summary = summary[summary["n_timepoints"] >= min_timepoints].copy()

    if summary.empty:
        summary["dynamics_class"] = pd.Series(dtype="object")
        summary["dynamics_subclass"] = pd.Series(dtype="object")
        return summary[[
            "chr_pos", "tissue", "n_obs", "n_timepoints",
            "mean_true_usage", "true_usage_variance",
            "dynamics_class", "dynamics_subclass"
        ]]

    summary["true_usage_variance"] = summary["true_usage_variance"].fillna(0.0)

    def _spline_change_stats(g):
        g = g.sort_values("timepoint_num")
        x = g["timepoint_num"].to_numpy(dtype=float)
        y = g["true_usage"].to_numpy(dtype=float)

        if x.size < 2 or np.unique(x).size < 2:
            return pd.Series({
                "sse_amplitude": 0.0,
                "up": 0.0,
                "down": 0.0,
                "total_change": 0.0,
                "up_timing": np.nan,
                "down_timing": np.nan,
                "ratio_up": np.nan,
                "dynamic_direction": "none",
            })

        spline = CubicSpline(x, y, bc_type="natural")
        x_new = np.linspace(x.min(), x.max(), n_interp)
        y_new = spline(x_new)

        dy = np.diff(y_new)
        t = x_new[1:]

        pos = dy > 0
        neg = dy < 0

        up = float(dy[pos].sum()) if np.any(pos) else 0.0
        down = float((-dy[neg]).sum()) if np.any(neg) else 0.0
        total = up + down

        up_timing = float((dy[pos] * t[pos]).sum() / up) if up > 0 else np.nan
        down_timing = float(((-dy[neg]) * t[neg]).sum() / down) if down > 0 else np.nan
        ratio_up = float(up / total) if total > 0 else np.nan
        amplitude = float(np.max(y_new) - np.min(y_new))

        if total < direction_change_threshold or np.isnan(ratio_up):
            direction = "none"
        elif ratio_up < 0.3:
            direction = "down"
        elif ratio_up > 0.7:
            direction = "up"
        else:
            if np.isnan(up_timing) or np.isnan(down_timing):
                direction = "none"
            else:
                direction = "up-down" if up_timing < down_timing else "down-up"

        return pd.Series({
            "sse_amplitude": amplitude,
            "up": up,
            "down": down,
            "total_change": total,
            "up_timing": up_timing,
            "down_timing": down_timing,
            "ratio_up": ratio_up,
            "dynamic_direction": direction,
        })

    trajectory = site_tp.groupby("chr_pos").apply(_spline_change_stats).reset_index()
    summary = summary.merge(trajectory, on="chr_pos", how="left")

    summary["dynamics_class"] = np.where(
        (summary["sse_amplitude"] > amplitude_threshold)
        & (summary["dynamic_direction"] != "none"),
        "dynamic",
        "stable",
    )

    summary["dynamics_subclass"] = np.where(
        summary["dynamics_class"] == "stable",
        np.where(
            summary["mean_true_usage"] > high_threshold,
            "high",
            np.where(summary["mean_true_usage"] < low_threshold, "low", "median"),
        ),
        summary["dynamic_direction"],
    )

    return summary[[
        "chr_pos", "tissue", "n_obs", "n_timepoints",
        "mean_true_usage", "true_usage_variance",
        "dynamics_class", "dynamics_subclass"
    ]]
