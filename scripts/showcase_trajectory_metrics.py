#!/usr/bin/env python3
"""
Showcase real examples of developmental trajectories with different correlation/RMSE combinations.

Loads real model predictions and observed trajectories from traj_eval output, finds examples of:
1. Good correlation + Good RMSE (model captures both shape and magnitude)
2. Good correlation + Poor RMSE (model captures shape but wrong magnitude)
3. Poor correlation + Good RMSE (model gets magnitude but wrong shape)

Reuses plotting functions from alphagenome_pytorch.plotting.splicing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# Constants for tissue colors and ordering (from splicing.py)
TISSUE_COLORS = {
    'Brain': '#3399cc',
    'Cerebellum': '#34ccff',
    'Heart': '#cc0100',
    'Kidney': '#cc9900',
    'Liver': '#339900',
    'Ovary': '#cc329a',
    'Testis': '#ff6600'
}

TISSUE_ORDER = ["Brain", "Cerebellum", "Heart", "Kidney", "Liver", "Ovary", "Testis"]


def load_real_trajectories(preds_dir='/Volumes/sd17d003/Anamaria/alphagenome_genomicsxai/best_model/preds_intersect_protein_coding_all',
                          min_timepoints=12):
    """Load real usage data and find examples of each correlation/RMSE profile.

    Only considers trajectories with at least min_timepoints observed measurements
    for better statistical power.
    """

    # Load usage data for a species (use human as reference)
    usage_file = os.path.join(preds_dir, 'human', 'usage_human.parquet')
    if not os.path.exists(usage_file):
        log.warning(f"Usage file not found: {usage_file}")
        return None

    df = pd.read_parquet(usage_file)
    log.info(f"Loaded {len(df)} usage records from {usage_file}")

    # Group by site and tissue to get trajectories
    trajectories = []
    for (species, chrom, pos, tissue), group in df.groupby(['Species', 'Chromosome', 'Position', 'Tissue']):
        group = group.sort_values('Timepoint')

        # Need at least min_timepoints
        if len(group) < min_timepoints:
            continue

        true_vals = group['SSE_true'].values
        pred_vals = group['SSE_pred'].values

        # Skip if mostly NaN
        if np.isnan(true_vals).sum() > len(true_vals) * 0.3 or np.isnan(pred_vals).sum() > len(pred_vals) * 0.3:
            continue

        timepoints = group['Timepoint'].values
        r = compute_metrics(true_vals, pred_vals)['r']
        rmse = compute_metrics(true_vals, pred_vals)['rmse']

        trajectories.append({
            'species': species, 'chrom': chrom, 'pos': pos, 'tissue': tissue,
            'true': true_vals, 'pred': pred_vals, 'timepoints': timepoints,
            'r': r, 'rmse': rmse, 'n_tp': len(group)
        })

    if not trajectories:
        log.warning(f"No valid trajectories found with ≥{min_timepoints} timepoints")
        return None

    log.info(f"Found {len(trajectories)} trajectories with ≥{min_timepoints} timepoints")

    # Find representatives of each scenario
    examples = {}

    # 1. Good correlation + Good RMSE: r > 0.6, rmse < 0.15
    candidates = [t for t in trajectories if t['r'] > 0.6 and t['rmse'] < 0.15]
    if candidates:
        best = max(candidates, key=lambda x: x['r'])
        examples['good_corr_good_rmse'] = {
            **best,
            'title': 'Good Correlation + Good RMSE',
            'description': 'Model captures both shape and magnitude'
        }
        log.info(f"Found good_corr_good_rmse example: {best['species']} {best['chrom']}:{best['pos']} {best['tissue']}")

    # 2. Good correlation + Poor RMSE (amplitude error): r > 0.6, rmse >= 0.15
    candidates = [t for t in trajectories if t['r'] > 0.6 and t['rmse'] >= 0.15]
    if candidates:
        best = max(candidates, key=lambda x: x['r'])
        examples['good_corr_poor_rmse'] = {
            **best,
            'title': 'Good Correlation + Poor RMSE',
            'description': 'Model captures shape but wrong magnitude'
        }
        log.info(f"Found good_corr_poor_rmse example: {best['species']} {best['chrom']}:{best['pos']} {best['tissue']}")

    # 3. Vertical shift: high r but shifted baseline (look for high r but moderate rmse due to mean difference)
    candidates = [t for t in trajectories if t['r'] > 0.7 and 0.08 < t['rmse'] < 0.18]
    if candidates:
        def has_vertical_shift(t):
            """Check if trajectory has consistent vertical shift."""
            true_mean = np.mean(t['true'])
            pred_mean = np.mean(t['pred'])
            # Good correlation but different means
            return abs(true_mean - pred_mean) > 0.15

        candidates = [c for c in candidates if has_vertical_shift(c)]
        if candidates:
            best = max(candidates, key=lambda x: x['r'])
            examples['vertical_shift'] = {
                **best,
                'title': 'Vertical Shift (Same Shape, Different Baseline)',
                'description': 'Model captures direction but shifts baseline'
            }
            log.info(f"Found vertical_shift example: {best['species']} {best['chrom']}:{best['pos']} {best['tissue']}")

    # 4. Poor correlation + Good RMSE (pattern error): r <= 0.3, rmse < 0.15
    candidates = [t for t in trajectories if t['r'] <= 0.3 and t['rmse'] < 0.15]
    if candidates:
        def score_candidate(t):
            """Score based on excursion and consistency of deviation."""
            true_centered = t['true'] - np.mean(t['true'])
            excursion = np.max(np.abs(true_centered))
            pred_centered = t['pred'] - np.mean(t['pred'])
            pointwise_errors = np.abs(true_centered - pred_centered)
            max_error = np.max(pointwise_errors)
            rmse = t['rmse']
            consistency = min(1.0, rmse / (max_error + 1e-6))
            return excursion * 0.6 + consistency * 0.4 - 0.05 * max(-1, t['r'])

        candidates_scored = [(c, score_candidate(c)) for c in candidates]
        candidates_scored.sort(key=lambda x: x[1], reverse=True)

        # Take the best one
        best = candidates_scored[0][0]
        examples['pattern_mismatch'] = {
            **best,
            'title': 'Poor Correlation + Good RMSE',
            'description': 'Model captures magnitude but wrong timing'
        }
        log.info(f"Found pattern_mismatch example: {best['species']} {best['chrom']}:{best['pos']} {best['tissue']}")

        # Also take second best for an additional example
        if len(candidates_scored) > 1:
            best2 = candidates_scored[1][0]
            examples['pattern_mismatch_2'] = {
                **best2,
                'title': 'Poor Correlation + Good RMSE',
                'description': 'Model captures magnitude but wrong timing'
            }
            log.info(f"Found pattern_mismatch_2 example: {best2['species']} {best2['chrom']}:{best2['pos']} {best2['tissue']}")

    # 5. Opposite direction (negative correlation): r < -0.6
    candidates = [t for t in trajectories if t['r'] < -0.6]
    if candidates:
        best = min(candidates, key=lambda x: x['r'])  # Pick most negative
        examples['opposite_direction'] = {
            **best,
            'title': 'Opposite Direction (Inverted)',
            'description': 'Model predicts opposite developmental pattern'
        }
        log.info(f"Found opposite_direction example: {best['species']} {best['chrom']}:{best['pos']} {best['tissue']}")

    # If we don't have all three, relax criteria
    if len(examples) < 3:
        log.warning(f"Only found {len(examples)} examples, relaxing criteria...")

        if 'good_corr_good_rmse' not in examples:
            candidates = [t for t in trajectories if t['r'] > 0.5]
            if candidates:
                best = max(candidates, key=lambda x: x['r'])
                examples['good_corr_good_rmse'] = {
                    **best,
                    'title': f'Good Correlation + Good RMSE\n(r={best["r"]:.3f}, RMSE={best["rmse"]:.3f})',
                    'description': 'Model captures both shape and magnitude'
                }

        if 'good_corr_poor_rmse' not in examples:
            candidates = sorted(trajectories, key=lambda x: x['rmse'], reverse=True)
            for t in candidates:
                if t['r'] > 0.4:
                    examples['good_corr_poor_rmse'] = {
                        **t,
                        'title': f'Good Correlation + Poor RMSE\n(r={t["r"]:.3f}, RMSE={t["rmse"]:.3f})',
                        'description': 'Model captures shape but misses magnitude'
                    }
                    break

        if 'poor_corr_good_rmse' not in examples:
            candidates = [t for t in trajectories if t['rmse'] < 0.2]
            if candidates:
                best = min(candidates, key=lambda x: x['r'])
                examples['poor_corr_good_rmse'] = {
                    **best,
                    'title': f'Poor Correlation + Good RMSE\n(r={best["r"]:.3f}, RMSE={best["rmse"]:.3f})',
                    'description': 'Model captures magnitude but misses shape/direction'
                }

    return examples


def compute_metrics(true_traj, pred_traj):
    """Compute Pearson r and RMSE metrics (centered, like in evaluate_splice.py)."""
    # Center trajectories (remove mean)
    true_centered = true_traj - np.mean(true_traj)
    pred_centered = pred_traj - np.mean(pred_traj)

    # Pearson r
    cov = np.mean(true_centered * pred_centered)
    std_true = np.std(true_traj)
    std_pred = np.std(pred_traj)
    if std_true > 0 and std_pred > 0:
        r = cov / (std_true * std_pred)
    else:
        r = 0.0

    # RMSE (on centered values, to focus on dynamic component)
    rmse = np.sqrt(np.mean((true_centered - pred_centered) ** 2))

    # RMSE^2 decomposition: amplitude vs pattern
    amplitude_term = (std_pred - std_true) ** 2
    pattern_term = 2 * std_true * std_pred * (1 - max(-1, min(1, r)))  # Clamp r to [-1,1]

    return {
        'r': r,
        'rmse': rmse,
        'amplitude_term': amplitude_term,
        'pattern_term': pattern_term,
    }


def plot_trajectory_metrics_showcase(examples=None, output_path=None):
    """Create a 5-panel figure showcasing different correlation/RMSE scenarios from real data."""

    if examples is None:
        examples = load_real_trajectories()

    if examples is None or len(examples) < 3:
        log.error("Could not load sufficient real trajectory examples")
        return None

    n_examples = len(examples)
    # Layout: 5 panels in 2 rows (3 on top, 2 on bottom), or 5 in a row if user wants
    if n_examples == 5:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
        axes = axes.flatten()
    else:
        ncols = min(3, n_examples)
        nrows = (n_examples + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 3.75 * nrows))
        if n_examples == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

    for idx, (scenario_name, data) in enumerate(examples.items()):
        ax = axes[idx]

        true_traj = data['true']
        pred_traj = data['pred']
        timepoints = data['timepoints']

        # Compute metrics
        metrics = compute_metrics(true_traj, pred_traj)
        r = metrics['r']
        rmse = metrics['rmse']

        # Plot trajectories
        ax.plot(timepoints, true_traj, 'o-', color='#3399cc', linewidth=2.5,
                markersize=6, label='Observed', zorder=2)
        ax.plot(timepoints, pred_traj, 'x--', color='#cc0100', linewidth=2,
                markersize=6, label='Predicted', zorder=2)

        # Styling - minimal grid
        ax.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.5)
        ax.set_xlim(0.5, 15.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Timepoint', fontsize=16, fontweight='normal')
        ax.set_ylabel('Usage', fontsize=16, fontweight='normal')
        ax.tick_params(labelsize=13)
        # Make x-axis ticks integers, but only show every other label
        ax.set_xticks(range(1, 16))
        ax.set_xticklabels([str(i) if i % 2 == 1 else '' for i in range(1, 16)])

        # Title with site info on top, metrics below
        site_info = f"{data['species']} {data['chrom']}:{data['pos']} {data['tissue']}"
        ax.set_title(f'{site_info}\nr = {r:.2f}  RMSE = {rmse:.3f}',
                    fontsize=12, fontweight='normal', pad=10)

    # Add legend outside of first plot (no box frame)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01),
              ncol=2, fontsize=13, frameon=False, borderpad=0)

    # Hide unused subplots
    for idx in range(len(examples), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        'Developmental Trajectory Failure Modes: Correlation + RMSE',
        fontsize=16, fontweight='bold', y=0.995
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        log.info(f"Saved trajectory metrics showcase to {output_path}")

    return fig


def plot_rmse_decomposition(examples=None, output_path=None):
    """Create a figure showing RMSE = amplitude + pattern decomposition for each scenario."""

    if examples is None:
        examples = load_real_trajectories()

    if examples is None or len(examples) < 3:
        log.error("Could not load sufficient trajectory examples")
        return None

    n_examples = len(examples)
    if n_examples == 5:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
        axes = axes.flatten()
    else:
        ncols = min(3, n_examples)
        nrows = (n_examples + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 3.75 * nrows))
        if n_examples == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

    for idx, (scenario_name, data) in enumerate(examples.items()):
        ax = axes[idx]

        true_traj = data['true']
        pred_traj = data['pred']

        metrics = compute_metrics(true_traj, pred_traj)
        r = metrics['r']
        rmse = metrics['rmse']
        amplitude_term = metrics['amplitude_term']
        pattern_term = metrics['pattern_term']

        # RMSE^2 = amplitude + pattern
        rmse_sq = rmse ** 2

        # Create bar chart
        components = ['Amplitude\nError', 'Pattern\nError']
        values = [amplitude_term, pattern_term]
        colors_bar = ['#FF6B6B', '#4ECDC4']

        bars = ax.bar(components, values, color=colors_bar, alpha=0.75, edgecolor='black', linewidth=1.5, width=0.6)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add total RMSE line
        ax.axhline(y=rmse_sq, color='black', linestyle='--', linewidth=2.5, label=f'Total RMSE² = {rmse_sq:.4f}')

        ax.set_ylabel('Squared Error', fontsize=16, fontweight='normal')
        ax.tick_params(labelsize=13)
        ax.set_title(f'r = {r:.2f}', fontsize=12, fontweight='normal', pad=10)
        ax.legend(loc='upper right', fontsize=12, frameon=False)
        ax.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.5)
        ax.set_ylim(0, max([rmse_sq * 1.2, 0.02]))

    # Hide unused subplots
    for idx in range(len(examples), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        'RMSE Decomposition: Amplitude vs Pattern Error',
        fontsize=16, fontweight='bold', y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        log.info(f"Saved RMSE decomposition to {output_path}")

    return fig


if __name__ == '__main__':
    # Create output directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    figures_dir = os.path.join(project_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Load real trajectory examples
    print("Loading real trajectory examples from model predictions...")
    examples = load_real_trajectories()

    if examples and len(examples) >= 3:
        print(f"Found {len(examples)} example scenarios")

        # Generate plots with real data
        fig1_path = os.path.join(figures_dir, 'trajectory_metrics_showcase.png')
        fig2_path = os.path.join(figures_dir, 'rmse_decomposition.png')

        print("Generating trajectory metrics showcase with real data...")
        plot_trajectory_metrics_showcase(examples, fig1_path)

        print("Generating RMSE decomposition figure with real data...")
        plot_rmse_decomposition(examples, fig2_path)

        print(f"\nFigures saved to: {figures_dir}/")
        print(f"  - trajectory_metrics_showcase.png")
        print(f"  - rmse_decomposition.png")

        # Print example details
        print("\nReal trajectory examples used:")
        for scenario, data in examples.items():
            print(f"  {scenario}:")
            print(f"    Site: {data['species']} {data['chrom']}:{data['pos']} {data['tissue']}")
            print(f"    r={data['r']:.3f}, RMSE={data['rmse']:.3f}")
    else:
        print("ERROR: Could not find sufficient real trajectory examples")
        print("Falling back to synthetic examples...")

        # Fallback to synthetic
        examples = {}
        # Generate synthetic examples for fallback
        timepoints = np.arange(1, 16)
        true_1 = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                           0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
        pred_1 = true_1 + np.random.normal(0, 0.03, len(true_1))
        pred_1 = np.clip(pred_1, 0, 1)

        true_2 = true_1.copy()
        pred_2 = 0.1 + (true_2 - 0.1) * 0.4 + np.random.normal(0, 0.02, len(true_2))
        pred_2 = np.clip(pred_2, 0, 1)

        true_3 = true_1.copy()
        pred_3 = 0.45 - (true_3 - 0.45) + np.random.normal(0, 0.02, len(true_3))
        pred_3 = np.clip(pred_3, 0, 1)

        examples = {
            'good_corr_good_rmse': {
                'true': true_1, 'pred': pred_1, 'timepoints': timepoints,
                'r': compute_metrics(true_1, pred_1)['r'],
                'rmse': compute_metrics(true_1, pred_1)['rmse'],
                'title': 'Good Correlation + Good RMSE\n(Model captures shape and magnitude)',
            },
            'good_corr_poor_rmse': {
                'true': true_2, 'pred': pred_2, 'timepoints': timepoints,
                'r': compute_metrics(true_2, pred_2)['r'],
                'rmse': compute_metrics(true_2, pred_2)['rmse'],
                'title': 'Good Correlation + Poor RMSE\n(Model captures shape, misses magnitude)',
            },
            'poor_corr_good_rmse': {
                'true': true_3, 'pred': pred_3, 'timepoints': timepoints,
                'r': compute_metrics(true_3, pred_3)['r'],
                'rmse': compute_metrics(true_3, pred_3)['rmse'],
                'title': 'Poor Correlation + Good RMSE\n(Model gets magnitude, misses shape)',
            }
        }

        fig1_path = os.path.join(figures_dir, 'trajectory_metrics_showcase.png')
        fig2_path = os.path.join(figures_dir, 'rmse_decomposition.png')

        print("Generating trajectory metrics showcase with synthetic data...")
        plot_trajectory_metrics_showcase(examples, fig1_path)

        print("Generating RMSE decomposition figure with synthetic data...")
        plot_rmse_decomposition(examples, fig2_path)

        print(f"\nFigures saved to: {figures_dir}/")
        print(f"  (Using synthetic data as real examples not available)")
