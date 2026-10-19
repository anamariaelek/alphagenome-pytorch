"""Compare splice-usage Pearson r of two models, one panel per evaluation species.

Minimal reproduction of the "Human vs multi species model" barplot (cell [11])
from splice_model_eval_compare_multi_species.ipynb, reduced to two models.

Per (model, species): per-condition Pearson r is computed from the summary
statistics in usage_*.npz, one value is taken per tissue, and bars show
mean ± std across tissues. The bracket gives Δr = model B - model A.

Usage:
    python plot_usage_single_vs_multi_species.py \
        [--models lora_16_h lora_8_hmrbo] [--labels Human Multi-species] \
        [--species human mouse] [--out usage_single_vs_multi_species.png]
    python plot_usage_single_vs_multi_species.py --example   # hard-coded Human vs Human+Mouse example
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path("/Volumes/sd17d003/Anamaria/alphagenome_genomicsxai")
EVALS_DIR = BASE_DIR / "evals"
PREDS_SUBSET = "preds_intersect_protein_coding"

SPECIES_ORDER = ["human", "macaque", "mouse", "rat", "rabbit", "opossum", "chicken"]
SPECIES_DIRS = {
    "human": "Homo_sapiens",
    "macaque": "Macaca_mulatta",
    "mouse": "Mus_musculus",
    "rat": "Rattus_norvegicus",
    "rabbit": "Oryctolagus_cuniculus",
    "opossum": "Monodelphis_domestica",
    "chicken": "Gallus_gallus",
}
DEFAULT = {
    "models": ["lora_16_h", "lora_8_hmrbo"],
    "labels": ["Human", "Multi-species"],
    "species": None,  # all species evaluated for both models
    "colors": ["#b3cbeb", "#05416C"],
    "out": "usage_single_vs_multi_species.png",
}
# --example: hard-coded Human vs Human+Mouse values (human target), no evals loaded
EXAMPLE = {
    "labels": ["Human", "Human+Mouse"],
    "means": [0.847, 0.882],
    "errors": [0.030, 0.026],
    "colors": ["#b3cbeb", "#165B8D"],
    "out": "usage_human_vs_human_mouse.png",
}
FONT_SIZE = 10


def usage_file(model, species, fallback_head):
    """The species' own head if the model has one, else the fallback head."""
    eval_dir = EVALS_DIR / model / PREDS_SUBSET / species
    for fn in (f"usage_{species}.npz", f"usage_{species}_from_{fallback_head}.npz"):
        if (eval_dir / fn).exists():
            return eval_dir / fn
    return None


def per_tissue_r(npz_path, species):
    """Pearson r per tissue (first condition of each tissue, as in the notebook)."""
    with open(BASE_DIR / "data" / SPECIES_DIRS[species] / "usage.json") as f:
        idx_to_name = {int(v): k for k, v in json.load(f)["condition_labels"].items()}

    d = np.load(npz_path)
    n = d["stats_n"].astype(float)
    sp, st = d["stats_sum_pred"], d["stats_sum_true"]
    num = n * d["stats_sum_prod"] - sp * st
    dp = n * d["stats_sum_pred2"] - sp**2
    dt = n * d["stats_sum_true2"] - st**2
    ok = (n > 1) & (dp > 0) & (dt > 0)
    r = num[ok] / np.sqrt(dp[ok] * dt[ok])

    by_tissue = {}
    for cond, val in zip(d["stats_cond_ids"][ok].astype(int), r):
        tissue = idx_to_name.get(cond, f"unknown_{cond}").rsplit("_", 1)[0]
        by_tissue.setdefault(tissue, float(val))
    return np.array(list(by_tissue.values()))


def model_stats(model, species, fallback_head):
    """Mean and std of per-tissue Pearson r for one model on one species."""
    path = usage_file(model, species, fallback_head)
    r = per_tissue_r(path, species) if path else np.array([])
    return (r.mean() if len(r) else np.nan), (r.std(ddof=1) if len(r) > 1 else np.nan)


def plot_pair(ax, means, errors, colors):
    """Two bars (mean ± std) and a Δr bracket; returns the bracket height."""
    ax.bar([0, 1], means, yerr=errors, width=0.8, color=colors, alpha=0.85, capsize=3,
           edgecolor="#333333", linewidth=0.8)

    y = np.nanmax(np.add(means, np.nan_to_num(errors))) + 0.04
    if not np.isnan(y):
        delta = means[1] - means[0]
        ax.plot([0, 0, 1, 1], [y, y + 0.008, y + 0.008, y], color="#333333", lw=0.9)
        ax.text(0.5, y + 0.013, rf"$\Delta r$={delta * 100:+.1f}%", ha="center", va="bottom",
                fontsize=FONT_SIZE, color="#333333")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FONT_SIZE)
    return y


def plot_example(labels, out):
    """Single panel with model names as x tick labels, no title or top/right spines."""
    fig, ax = plt.subplots(figsize=(2.0, 4.0))
    plot_pair(ax, EXAMPLE["means"], EXAMPLE["errors"], EXAMPLE["colors"])
    ax.set_ylim(0.45, 1.02)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Pearson r", fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs=2, default=DEFAULT["models"],
                        help="Two eval dirs under evals/: baseline, then comparison model")
    parser.add_argument("--labels", nargs=2, default=None, help="Display names")
    parser.add_argument("--species", nargs="+", default=None,
                        help="Evaluation species (default: all evaluated for both models)")
    parser.add_argument("--fallback-head", default="human",
                        help="Head used when a model has none for the evaluated species")
    parser.add_argument("--example", action="store_true",
                        help="Draw the Human vs Human+Mouse example from hard-coded values")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.example:
        plot_example(args.labels or EXAMPLE["labels"], args.out or EXAMPLE["out"])
        return

    models = args.models
    species_list = args.species or [
        s for s in SPECIES_ORDER
        if all(usage_file(m, s, args.fallback_head) for m in models)
    ]
    if not species_list:
        raise SystemExit("No evaluation species shared by both models")

    fig, axes = plt.subplots(1, len(species_list), figsize=(1.3 * len(species_list) + 0.5, 4.0),
                             sharey=True, squeeze=False)
    y_max = 0.0
    for ax, species in zip(axes[0], species_list):
        means, errors = zip(*(model_stats(m, species, args.fallback_head) for m in models))
        y_max = np.nanmax([y_max, plot_pair(ax, means, errors, DEFAULT["colors"])])
        ax.set_title(species.capitalize(), fontsize=FONT_SIZE, fontweight="bold", pad=10)
        ax.set_xticks([])
    axes[0, 0].set_ylim(0.45, max(1.09, y_max + 0.11))
    axes[0, 0].set_ylabel("Pearson r", fontsize=FONT_SIZE)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.85, ec="#333333") for c in DEFAULT["colors"]]
    fig.legend(handles, args.labels or DEFAULT["labels"], loc="lower center", ncol=2,
               frameon=False, fontsize=FONT_SIZE)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(args.out or DEFAULT["out"], dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
