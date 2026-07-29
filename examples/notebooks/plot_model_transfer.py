import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# --- 1. Data and Parameters ---
df_usage_ = pd.read_csv('splice_model_usage.csv') # at least the following columns: ['model', 'model_species', 'pred_species', 'tissue', 'correlation']
LEFTOUT_SPECIES = 'rat'
BASELINE_MODEL = 'human'
MODEL_WITH_LEFTOUT = 'human_mouse_rat_rabbit_opossum'

# --- 2. Fixed Metric Fetcher ---
def get_tissue_metrics(df, tissue_item):
    # Standardize to list format so .isin() works for both single tissues and lists of tissues
    tissues = tissue_item if isinstance(tissue_item, list) else [tissue_item]
    
    baseline_sub = df[
        (df['model'] == BASELINE_MODEL) & 
        (df['pred_species'] == LEFTOUT_SPECIES) &
        (df['tissue'].isin(tissues))
    ]
    leftout_model_sub = df[
        (df['model'] == MODEL_WITH_LEFTOUT) & 
        (df['pred_species'] == LEFTOUT_SPECIES) & 
        (df['model_species'] == LEFTOUT_SPECIES) &
        (df['tissue'].isin(tissues))
    ]
        
    # Calculate overall mean Pearson r across filtered subset
    r_baseline = baseline_sub['correlation'].mean() if not baseline_sub.empty else 0.0
    r_full = leftout_model_sub['correlation'].mean() if not leftout_model_sub.empty else 0.0
    delta = r_full - r_baseline
    return r_baseline, r_full, delta

# --- 3. Generate Multi-Page PDF ---
pdf_filename = f'splicing_model_transfer_{LEFTOUT_SPECIES}.pdf'

# Tissues and title pairs
pages_to_generate = [
    ('Liver', 'Liver'),
    ('Cerebellum', 'Cerebellum'),
    ('Testis', 'Testis'),
    (['Liver', 'Cerebellum', 'Testis'], 'Liver, Cerebellum, Testis (Mean)')
]

# Fixed font size
FONT_SIZE = 11

# Colors
TISSUE_COLORS = {
    'Brain': '#3399cc',
    'Midbrain': '#34b3e6',
    'Cerebellum': '#34ccff',
    'Heart': '#cc0100',
    'Kidney': '#cc9900',
    'Liver': '#339900',
    'Ovary': '#cc329a',
    'Testis': '#ff6600'
}

with PdfPages(pdf_filename) as pdf:
    for tissue_item, title_label in pages_to_generate:
        # Fetch r_baseline, r_full, and delta (averages over list if tissue_item is a list)
        r_baseline, r_full, delta = get_tissue_metrics(df_usage_, tissue_item)
        
        # Color Shading logic
        if isinstance(tissue_item, list):
            # Neutral slate tones for combined multi-tissue mean
            color_human = '#555555'
            color_rat = '#888888'
        else:
            base_hex = TISSUE_COLORS.get(tissue_item, '#333333')
            rgb_base = mcolors.to_rgb(base_hex)
            color_human = tuple(0.75 * c for c in rgb_base)
            color_rat = base_hex

        fig, ax = plt.subplots(figsize=(3, 2), dpi=300)
        x_pos = 0
        bar_width = 0.55

        # Base Model Bar (Rat -> Human)
        ax.bar(x_pos, r_baseline, width=bar_width, color=color_human, edgecolor='black', linewidth=0.8, zorder=3)

        # Delta Improvement Bar (Rat -> Rat)
        ax.bar(x_pos, delta, bottom=r_baseline, width=bar_width, color=color_rat, edgecolor='black', hatch='///', linewidth=0.8, zorder=3)

        # External Annotations (Side-aligned)
        right_edge_x = x_pos + (bar_width / 2) + 0.05
        plot_baseline = max(r_baseline, 0.5)
        plot_full = max(r_full, 0.5)
        plot_delta = max((delta / 2), 0.04)
        ax.text(right_edge_x, plot_baseline + 2*plot_delta, f"  $r = {r_full:.3f}$", ha='left', va='bottom', fontsize=FONT_SIZE)
        ax.text(right_edge_x, plot_baseline + plot_delta, f"$\Delta r = {delta:+.3f}$", ha='left', va='center', fontsize=FONT_SIZE)
        ax.text(right_edge_x, plot_baseline, f"  $r = {r_baseline:.3f}$", ha='left', va='top', fontsize=FONT_SIZE)

        # Legend (Bottom Right Corner)
        patch_human = mpatches.Patch(facecolor=color_human, edgecolor='black', linewidth=0.8, label='rat → human')
        patch_rat = mpatches.Patch(facecolor=color_rat, edgecolor='black', hatch='///', linewidth=0.8, label='rat → rat')
        
        # Expand x-axis slightly on the right to give the legend room
        ax.set_xlim(-0.35, 1.5)

        # Legend shifted further right
        leg = ax.legend(
            handles=[patch_rat, patch_human],
            title=r'$\mathrm{cis} \rightarrow \mathrm{trans}$',
            loc='lower left',
            bbox_to_anchor=(0.35, 0),  # Anchors the legend past the annotations
            frameon=False,
            fontsize=FONT_SIZE,
            title_fontsize=FONT_SIZE,
            handletextpad=0.5, handlelength=1.2, labelspacing=0.1
        )
        leg._legend_box.align = "left"

        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_ylabel('Pearson r', fontsize=FONT_SIZE)
        ax.set_title(title_label, fontsize=FONT_SIZE, pad=8)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)

        ax.tick_params(axis='both', labelsize=FONT_SIZE, length=2)
        ax.grid(axis='y', alpha=0.3, linestyle='--', color='#aaaaaa', zorder=1)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis='x', bottom=False)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print(f"Successfully generated all pages into {pdf_filename}")