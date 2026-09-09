"""Plotting utilities for splicing data.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
import logging

log = logging.getLogger(__name__)


SPLICE_CLASS_NAMES = ["Donor+", "Acceptor+", "Donor-", "Acceptor-"]
BACKGROUND_CLASS = 4
CLASS_LABELS = {0: 'donor +', 1: 'acceptor +', 2: 'donor -', 3: 'acceptor -', 4: 'no splice site'}
CLASS_COLORS  = {0: '#ff7f00', 1: '#33a02c', 2: '#fdbf6f', 3: '#b2df8a', 4: '#1f78b4'}

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

SPECIES_ORDER = ['human', 'macaque', 'mouse', 'rat', 'rabbit', 'opossum', 'chicken']
TISSUE_ORDER = ["Brain", "Midbrain", "Cerebellum", "Heart", "Kidney", "Liver", "Ovary", "Testis"]

# The species to process
SPECIES = ["human", "macaque", "mouse", "rat", "rabbit", "opossum", "chicken"]
SPECIES_SCI = {
    'human': 'Homo_sapiens',
    'macaque': 'Macaca_mulatta',
    'mouse': 'Mus_musculus',
    'rat': 'Rattus_norvegicus',
    'rabbit': 'Oryctolagus_cuniculus',
    'opossum': 'Monodelphis_domestica',
    'chicken': 'Gallus_gallus'
}

# Chromosome sizes
CHR_SIZES = {
    # Homo sapiens	GRCh38
    "human": {
        "1": 248956422,
        "2": 242193529,
        "3": 198295559,
        "4": 190214555,
        "5": 181538259,
        "6": 170805979,
        "7": 159345973,
        "8": 145138636,
        "9": 138394717,
        "10": 133797422,
        "11": 135086622,
        "12": 133275309,
        "13": 114364328,
        "14": 107043718,
        "15": 101991189,
        "16": 90338345,
        "17": 83257441,
        "18": 80373285,
        "19": 58617616,
        "20": 64444167,
        "21": 46709983,
        "22": 50818468,
        "X": 156040895,
        "Y": 57227415,
    },
    # Mus musculus	GRCm38
    "mouse": {
        "1": 195471971,
        "2": 182113224,
        "3": 160039680,
        "4": 156508116,
        "5": 151834684,
        "6": 149736546,
        "7": 145441459,
        "8": 129401213,
        "9": 124595110,
        "10": 130694993,
        "11": 122082543,
        "12": 120129022,
        "13": 120421639,
        "14": 124902244,
        "15": 104043685,
        "16": 98207768,
        "17": 94987271,
        "18": 90702639,
        "19": 61431566,
        "X": 171031299,
        "Y": 91744698,
    },
    # Rattus norvegicus	Rnor_5.0
    "rat": {
        "1": 290094216,
        "2": 285068071,
        "3": 183740530,
        "4": 248343840,
        "5": 177180328,
        "6": 156897508,
        "7": 143501887,
        "8": 132457389,
        "9": 121549591,
        "10": 112200500,
        "11": 93518069,
        "12": 54450796,
        "13": 118718031,
        "14": 115151701,
        "15": 114627140,
        "16": 90051983,
        "17": 92503511,
        "18": 87229863,
        "19": 72914587,
        "20": 57791882,
        "X": 154597545,
    },
    # Oryctolagus cuniculus	OryCun2.0
    "rabbit": {
        "1": 194850757,
        "2": 174332312,
        "3": 155691105,
        "4": 91394100,
        "5": 37992211,
        "6": 27502587,
        "7": 173684459,
        "8": 111795807,
        "9": 116251907,
        "10": 47997241,
        "11": 87554214,
        "12": 155355395,
        "13": 143360832,
        "14": 163896628,
        "15": 109054052,
        "16": 84478945,
        "17": 85008467,
        "18": 69800736,
        "19": 57279966,
        "20": 33191332,
        "21": 15578276,
        "X": 111700775,
    },
    # Monodelphis domestica	BROADO5
    "opossum": {
        "1": 748055161,
        "2": 541556283,
        "3": 527952102,
        "4": 435153693,
        "5": 304825324,
        "6": 292091736,
        "7": 260857928,
        "8": 312544902,
        "X": 79335909,
    },
    # Macaca mulatta	MMUL_1
    "macaque": {
        "1": 228252215,
        "2": 189746636,
        "3": 196418989,
        "4": 167655696,
        "5": 182086969,
        "6": 178205221,
        "7": 169801366,
        "8": 147794981,
        "9": 133323859,
        "10": 94855758,
        "11": 134511895,
        "12": 106505843,
        "13": 138028943,
        "14": 133002572,
        "15": 110119387,
        "16": 78773432,
        "17": 94452569,
        "18": 73567989,
        "19": 64391591,
        "20": 88221753,
        "X": 153947521
    },
    # Gallus gallus	Galgal4
    "chicken": {
        "1": 195276750,
        "2": 148809762,
        "3": 110447801,
        "4": 90216835,
        "5": 59580361,
        "6": 34951654,
        "7": 36245040,
        "8": 28767244,
        "9": 23441680,
        "10": 19911089,
        "11": 19401079,
        "12": 19897011,
        "13": 17760035,
        "14": 15161805,
        "15": 12656803,
        "16": 535270,
        "17": 10454150,
        "18": 11219875,
        "19": 9983394,
        "20": 14302601,
        "21": 6802778,
        "22": 4081097,
        "23": 5723239,
        "24": 6323281,
        "25": 2191139,
        "26": 5329985,
        "27": 5209285,
        "28": 4742627,
        "32": 1028,
        "W": 1248174,
        "Z": 82363669
    }
}


# Trajectory-clustering constants used by save_cluster_plots() below.
from alphagenome_pytorch.clustering import (
    T_GRID, SHAPE_COLORS, SHAPE_ORDER, DYNAMIC_SHAPE_COLORS, DYNAMIC_SHAPE_ORDER,
)

def _normalize_tissue_subset(tissue_subset):
    """Return a de-duplicated list of tissue names or None."""
    if tissue_subset is None:
        return None

    if isinstance(tissue_subset, str):
        tissues = [tissue_subset]
    elif isinstance(tissue_subset, (list, tuple, set, np.ndarray, pd.Index)):
        tissues = list(tissue_subset)
    else:
        raise TypeError("tissue_subset must be a string or a list-like of strings")

    normalized = []
    for tissue in tissues:
        if pd.isna(tissue):
            continue
        tissue_name = str(tissue).strip()
        if tissue_name:
            normalized.append(tissue_name)

    normalized = list(dict.fromkeys(normalized))
    if not normalized:
        raise ValueError("tissue_subset is empty after normalization")

    return normalized


def plot_splice_site_dynamics(
    df,
    site_coords,
    metric='SSE',
    tissue_order=TISSUE_ORDER,
    tissue_colors=TISSUE_COLORS,
    tissue_subset=None,
    figsize=None,
    title=None,
    jitter=0.0,
    size=None,
    verbose=False,
    alpha_threshold=None,
    beta_threshold=None,
    reads_threshold=None,
    min_conditions=None,
    n_cols=None,
    n_rows=None
):
    """
    Plot splice site dynamics with all tissues on one plot per site.

    Args:
        df: DataFrame with all splice site data (Species, Chromosome, Position, Strand,
                condition columns, Alpha, Beta, SSE, Condition_Name).
                Performance tips for very large DataFrames (>100M rows):
                - Convert string columns to categorical: df['Species'] = df['Species'].astype('category')
                  (also do this for Chromosome, Strand) - This speeds up filtering by 2-5x
                - Or set a MultiIndex for maximum speed: df.set_index([...]) - 100x faster
                  but takes time to create. The function uses optimized .query() by default.
        site_coords: List of site coordinate strings or single string. Formats:
                     - Single site: "species chrom:pos[:strand]" (e.g., "mouse 1:15188:+")
                     - Range: "species chrom:start-end[:strand]" (e.g., "human X:70492000-70501000:+")
                     Range format will plot all splice sites found in that region.
                     Strand is optional; if not specified, all strands included (or auto-detected for single sites).
        metric: Column name to plot (default: 'SSE')
        tissue_order: List of tissues in desired order
        tissue_colors: Dict mapping tissue names to colors
        tissue_subset: Tissue name or list of tissue names to display
            (e.g., 'Brain' or ['Brain', 'Heart']). If None, all tissues are shown.
        figsize: Tuple of (width, height) or None for auto
        title: Overall figure title
        jitter: Amount of horizontal jitter to apply to points (default: 0.0, typical: 0.1-0.3)
        size: Optional column name in df used to scale point size (e.g. 'Alpha', 'Beta')
        verbose: If True, print condition information (tissues and timepoints) for each site (default: False)
        alpha_threshold: Minimum Alpha value to include (default: None, no filtering)
        beta_threshold: Minimum Beta value to include (default: None, no filtering)
        reads_threshold: Minimum Alpha+Beta value to include (default: None, no filtering)
        min_conditions: Minimum number of conditions (unique Condition_Name values) required to plot a site (default: None, no filtering)
        n_cols: Number of columns in subplot grid (default: None, auto-calculated as min(3, n_sites))
        n_rows: Number of rows in subplot grid (default: None, auto-calculated from n_cols)

    Returns:
        matplotlib Figure object
    """
    if isinstance(site_coords, str):
        site_coords = [site_coords]


    # --- Normalize column casing ---------------------------------------
    rename_targets = ['species', 'chromosome', 'position', 'strand', 'tissue', 'timepoint']
    df.columns = [col.capitalize() if col.lower() in rename_targets else col for col in df.columns]
    if verbose:
        print(f"Debug: DataFrame columns after capitalization: {df.columns.tolist()}")

    if 'Condition_Name' not in df.columns:
        if set(['Tissue', 'Timepoint']).issubset(df.columns):
            df['Condition_Name'] = df['Tissue'].astype(str) + '_' + df['Timepoint'].astype(str)
            if verbose:
                print("Debug: Created 'Condition_Name' column by combining 'Tissue' and 'Timepoint'")
        else:
            raise ValueError(
                "DataFrame must contain 'Condition_Name' column or both 'Tissue' "
                "and 'Timepoint' columns to create it."
            )

    # Check if DataFrame has optimal index for fast lookups
    index_cols = ['Species', 'Chromosome', 'Position']
    
    # Check if 'Strand' exists either as a column or as an index level name
    has_strand = 'Strand' in df.columns or (
        isinstance(df.index, pd.MultiIndex) and 'Strand' in df.index.names
    )
    if has_strand:
        index_cols.append('Strand')

    # Determine if the DataFrame is already properly MultiIndexed
    has_index = (
        isinstance(df.index, pd.MultiIndex)
        and all(name in df.index.names for name in ['Species', 'Chromosome', 'Position'])
    )

    site_filters = []
    for coord in site_coords:
        parts = coord.strip().split()
        if len(parts) != 2:
            print(f"Warning: Invalid coordinate format '{coord}'. Expected 'species chrom:pos[:strand]' or 'species chrom:start-end[:strand]'")
            continue
        species = parts[0].lower()
        chrom_pos_strand = parts[1].split(':')
        if len(chrom_pos_strand) < 2 or len(chrom_pos_strand) > 3:
            print(f"Warning: Invalid chrom:pos[:strand] format in '{coord}'")
            continue

        chrom = chrom_pos_strand[0]
        strand = chrom_pos_strand[2] if len(chrom_pos_strand) == 3 else None

        if '-' in chrom_pos_strand[1]:
            try:
                start, end = chrom_pos_strand[1].split('-')
                # Remove underscores from position strings (e.g., "70_493_375" -> "70493375")
                start = int(start.replace('_', ''))
                end = int(end.replace('_', ''))
            except ValueError:
                print(f"Warning: Invalid range in '{coord}'")
                continue

            # Fast path for indexed DataFrames
            if has_index:
                try:
                    # Check if index is sorted (required for slicing)
                    if not df.index.is_monotonic_increasing:
                        print(f"Warning: MultiIndex is not sorted. Slicing may not work correctly.")
                        print(f"         Run: df = df.sort_index() before plotting.")
                    
                    if strand is not None:
                        # Specific strand requested
                        level_sel = df.loc[pd.IndexSlice[species, str(chrom), start:end, strand], :]
                    else:
                        # All strands - use slice(None) to select all values in the Strand level
                        level_sel = df.loc[pd.IndexSlice[species, str(chrom), start:end, :], :]
                    
                    if isinstance(level_sel, pd.Series):
                        level_sel = level_sel.to_frame().T
                    sites_in_range = level_sel.index.to_frame(index=False)[index_cols[2:]].drop_duplicates() # Position (and Strand if present)
                    sites_in_range['Species'] = species
                    sites_in_range['Chromosome'] = str(chrom)
                except (KeyError, IndexError) as e:
                    if verbose:
                        print(f"Debug: KeyError/IndexError for range query: {e}")
                        print(f"Debug: Looking for species={species}, chrom={chrom}, range={start}-{end}")
                        print(f"Debug: Available chromosomes in index: {df.index.get_level_values('Chromosome').unique().tolist()[:10]}")
                    sites_in_range = pd.DataFrame()
            else:
                # Slow path: boolean masking
                mask = (
                    (df['Species'] == species) &
                    (df['Chromosome'] == str(chrom)) &
                    (df['Position'] >= start) &
                    (df['Position'] <= end)
                )
                if strand is not None:
                    mask = mask & (df['Strand'] == strand)
                sites_in_range = df[mask][index_cols].drop_duplicates()

            if sites_in_range.empty:
                print(f"Warning: No sites found in range {coord}")
                continue

            sites_in_range = sites_in_range.sort_values(index_cols[2:])  # Sort by Position (and Strand if present)
            for _, row in sites_in_range.iterrows():
                strand_val = row['Strand'] if 'Strand' in sites_in_range.columns else '?'
                site_filters.append((row['Species'], row['Chromosome'], row['Position'], strand_val))

            print(f"Found {len(sites_in_range)} sites in {coord}")
        else:
            try:
                # Remove underscores from position strings (e.g., "70_493_375" -> "70493375")
                pos = int(chrom_pos_strand[1].replace('_', ''))
            except ValueError:
                print(f"Warning: Invalid position in '{coord}'")
                continue

            if strand is None:
                # Auto-detect strand
                if has_index:
                    try:
                        # Need to include all strands in the query
                        level_sel = df.loc[pd.IndexSlice[species, str(chrom), pos, :], :]
                        if isinstance(level_sel, pd.Series):
                            strand = level_sel.name[-1] if isinstance(level_sel.name, tuple) else '?'
                        else:
                            strand = level_sel.index[0][-1] if len(level_sel) > 0 else '?'
                            
                    except (KeyError, IndexError):
                        strand = '?'
                else:
                    mask = (
                        (df['Species'] == species) &
                        (df['Chromosome'] == str(chrom)) &
                        (df['Position'] == pos)
                    )
                    matching_rows = df[mask]
                    if not matching_rows.empty:
                        if "Strand" in matching_rows.columns:
                            strand = matching_rows.iloc[0]['Strand']
                        else:
                            strand = '?'

            if strand is None:          # site not found / no strand info -> don't filter on Strand
                strand = '?'
            site_filters.append((species, chrom, pos, strand))

    if not site_filters:
        print("No valid site coordinates provided")
        return None

    # Optimization: Use multi-index for fast lookups on large DataFrames
    # (has_index already checked above)
    if has_index:
        # Fast path: Use index-based lookups
        df_filtered_list = []
        for species, chrom, pos, strand in site_filters:
            try:
                if strand != '?':
                    subset = df.loc[(species, str(chrom), pos, strand), :]
                else:
                    subset = df.loc[(species, str(chrom), pos), :]
                if isinstance(subset, pd.Series):
                    subset = subset.to_frame().T
                df_filtered_list.append(subset)
            except KeyError:
                continue
        
        if not df_filtered_list:
            print("No data found for any of the requested sites")
            return None
        df_filtered = pd.concat(df_filtered_list, axis=0)
        # Reset index so columns are accessible for subsequent filtering
        df_filtered = df_filtered.reset_index()
    else:
        # Optimized path: Use .query() with combined condition
        # This is much faster than boolean masking for large DataFrames
        query_parts = []
        for species, chrom, pos, strand in site_filters:
            if strand != '?':
                query_parts.append(
                    f"(Species == {repr(species)} and Chromosome == {repr(str(chrom))} and "
                    f"Position == {pos} and Strand == {repr(strand)})"
                )
            else:
                query_parts.append(
                    f"(Species == {repr(species)} and Chromosome == {repr(str(chrom))} and "
                    f"Position == {pos})"
                )
        
        query_str = " or ".join(query_parts)
        
        try:
            df_filtered = df.query(query_str).copy()
        except Exception as e:
            # Fallback to boolean masking if query fails
            if verbose:
                print(f"Query failed ({e}), falling back to boolean indexing...")
            site_masks = []
            for species, chrom, pos, strand in site_filters:
                mask = (
                    (df['Species'] == species) &
                    (df['Chromosome'] == str(chrom)) &
                    (df['Position'] == pos)
                )
                if strand != '?':
                    mask = mask & (df['Strand'] == strand)
                site_masks.append(mask)
            
            combined_mask = site_masks[0]
            for mask in site_masks[1:]:
                combined_mask = combined_mask | mask
            
            df_filtered = df[combined_mask].copy()
        
        if df_filtered.empty:
            print("No data found for any of the requested sites")
            return None

    if tissue_order is None:
        tissue_order = TISSUE_ORDER
    if tissue_colors is None:
        tissue_colors = TISSUE_COLORS

    selected_tissues = _normalize_tissue_subset(tissue_subset)
    if selected_tissues is not None:
        if 'Tissue' not in df_filtered.columns:
            raise ValueError("tissue_subset requires a 'Tissue' column")

        df_filtered = df_filtered[df_filtered['Tissue'].astype(str).isin(selected_tissues)].copy()
        if df_filtered.empty:
            print("No data found for requested tissue_subset")
            return None

        selected_set = set(selected_tissues)
        ordered_selected = [t for t in tissue_order if t in selected_set]
        ordered_selected.extend([t for t in selected_tissues if t not in ordered_selected])
        tissue_order = ordered_selected

    all_tissues_found = set()

    # Pre-filter sites to determine which will actually be plotted (if min_conditions is set)
    if min_conditions is not None:
        valid_site_filters = []
        for species, chrom, pos, strand in site_filters:
            mask = (
                (df_filtered['Species'] == species) &
                (df_filtered['Chromosome'] == str(chrom)) &
                (df_filtered['Position'] == pos)
            )
            if strand != '?':
                mask = mask & (df_filtered['Strand'] == strand)
            df_site_check = df_filtered[mask]
            
            if not df_site_check.empty:
                n_conditions = df_site_check['Condition_Name'].nunique()
                if n_conditions >= min_conditions:
                    valid_site_filters.append((species, chrom, pos, strand))
                elif verbose:
                    print(f"Skipping {species} {chrom}:{pos} - only {n_conditions} conditions (minimum: {min_conditions})")
        site_filters = valid_site_filters
    
    if not site_filters:
        print("No sites to plot after applying filters")
        return None

    n_sites = len(site_filters)
    
    # Calculate subplot layout
    if n_cols is None and n_rows is None:
        # Auto-calculate: default to max 3 columns
        n_cols = min(3, n_sites)
        n_rows = (n_sites + n_cols - 1) // n_cols
    elif n_cols is not None and n_rows is None:
        # User specified columns, calculate rows
        n_rows = (n_sites + n_cols - 1) // n_cols
    elif n_rows is not None and n_cols is None:
        # User specified rows, calculate columns
        n_cols = (n_sites + n_rows - 1) // n_rows
    # else: both specified, use as-is

    if figsize is None:
        figsize = (n_cols * 3, n_rows * 2.2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for site_idx, (species, chrom, pos, strand) in enumerate(site_filters):
        ax = axes[site_idx]

        # Now filter from the already-filtered DataFrame (much faster)
        mask = (
            (df_filtered['Species'] == species) &
            (df_filtered['Chromosome'] == str(chrom)) &
            (df_filtered['Position'] == pos)
        )
        if strand != '?':
            mask = mask & (df_filtered['Strand'] == strand)
        df_site = df_filtered[mask].copy()

        if alpha_threshold is not None:
            df_site = df_site[df_site['Alpha'] >= alpha_threshold]
        if beta_threshold is not None:
            df_site = df_site[df_site['Beta'] >= beta_threshold]
        if reads_threshold is not None:
            df_site = df_site[(df_site['Alpha'] + df_site['Beta']) >= reads_threshold]

        label_to_type = {0: 'Donor+', 1: 'Acceptor+', 2: 'Donor-', 3: 'Acceptor-', 4: 'None'}
        if not df_site.empty and 'Label' in df_site.columns:
            site_label = df_site.iloc[0]['Label']
            site_type = label_to_type.get(site_label, f'{strand}')
        else:
            site_type = f'{strand}'

        if df_site.empty:
            ax.text(0.5, 0.5, f'No data\n{species} {chrom}:{pos} ({site_type})',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue

        df_site = df_site.dropna(subset=['Tissue', 'Timepoint'])
        df_site['Timepoint'] = df_site['Timepoint'].astype(int)
        tissues = [t for t in tissue_order if t in df_site['Tissue'].values]
        all_tissues_found.update(tissues)
        all_tps = sorted(df_site['Timepoint'].unique())

        if verbose:
            print(f"\n{species} {chrom}:{pos} ({site_type})")
            print(f"  Total data points: {len(df_site)}")
            print(f"  Tissues present: {', '.join(tissues)}")
            print(f"  Timepoints: {', '.join(map(str, all_tps))}")
            tissue_tp_summary = df_site.groupby(['Condition_Name', 'Tissue', 'Timepoint']).size().reset_index(name='count')
            tissue_tp_summary = tissue_tp_summary.sort_values(['Tissue', 'Timepoint'])
            print("  Conditions:")
            for _, row in tissue_tp_summary.iterrows():
                print(f"    {row['Tissue']} {row['Timepoint']} ({row['Condition_Name']}) n={row['count']}")

        n_tissues = len(tissues)
        
        if jitter > 0 and n_tissues > 1:
            tissue_offsets = {
                tissue: jitter * (i - (n_tissues - 1) / 2) / (n_tissues - 1)
                for i, tissue in enumerate(tissues)
            }
        else:
            tissue_offsets = {tissue: 0.0 for tissue in tissues}

        # Validate optional size column once per site
        size_col = size if isinstance(size, str) and size in df_site.columns else None
        if size is not None and size_col is None and verbose:
            print(f"  Warning: size='{size}' is not a valid df column for this site. Using fixed marker size.")

        if size_col is not None:
            site_size_vals = pd.to_numeric(df_site[size_col], errors='coerce')
            smin = site_size_vals.min()
            smax = site_size_vals.max()
            has_size_range = pd.notna(smin) and pd.notna(smax) and (smax > smin)
        else:
            has_size_range = False

        for tissue in tissues:
            tissue_data = df_site[df_site['Tissue'] == tissue]

            agg_dict = {'mean': (metric, 'mean'), 'std': (metric, 'std'), 'count': (metric, 'count')}
            if size_col is not None:
                agg_dict['size_value'] = (size_col, 'mean')

            grouped = tissue_data.groupby('Timepoint').agg(**agg_dict).reset_index()
            grouped['std'] = grouped['std'].fillna(0)

            color = tissue_colors.get(tissue, '#808080')
            x_coords = grouped['Timepoint'] + tissue_offsets[tissue]

            ax.fill_between(
                x_coords,
                grouped['mean'] - grouped['std'],
                grouped['mean'] + grouped['std'],
                color=color,
                alpha=0.1,
                linewidth=0
            )

            if size_col is not None:
                ax.plot(x_coords, grouped['mean'],
                        color=color, linewidth=2, alpha=0.9, label=tissue, zorder=2)

                if has_size_range:
                    point_sizes = 20 + 100 * (grouped['size_value'] - smin) / (smax - smin)
                    point_sizes = point_sizes.fillna(10).clip(lower=10, upper=200)
                else:
                    point_sizes = pd.Series(60, index=grouped.index)

                ax.scatter(
                    x_coords,
                    grouped['mean'],
                    s=point_sizes,
                    color=color,
                    alpha=0.9,
                    edgecolors='white',
                    linewidths=0.5,
                    zorder=3
                )
            else:
                ax.plot(x_coords, grouped['mean'],
                        color=color, linewidth=2, marker='o', markersize=4,
                        alpha=0.9, label=tissue, zorder=2)

        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xlim(0.5, 15.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if metric == 'SSE':
            ax.set_ylim(-0.05, 1.05)

        ax.set_xlabel('Timepoint', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f'{species} {chrom}:{pos} ({site_type})', fontsize=11)

    # Hide any unused subplots
    for idx in range(n_sites, len(axes)):
        axes[idx].set_visible(False)

    if all_tissues_found:
        legend_tissues = [t for t in tissue_order if t in all_tissues_found]
        handles = [
            plt.Line2D([0], [0], color=tissue_colors.get(t, '#808080'),
                       linewidth=2, marker='o', markersize=4, label=t)
            for t in legend_tissues
        ]
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5),
                   fontsize=10, framealpha=0.9, title='Tissue')

    if title is not None:
        fig.suptitle(title, fontsize=14, y=0.995)

    plt.tight_layout(rect=[0, 0, 0.85, 1] if all_tissues_found else None)
    return fig

def plot_splice_site_predictions(
    df,
    site_coords,
    true_col='true_usage',
    pred_col='pred_usage',
    tissue_order=TISSUE_ORDER,
    tissue_colors=TISSUE_COLORS,
    tissue_subset=None,
    figsize=None,
    title=None,
    jitter=0.0,
    size=None,
    verbose=False,
    alpha_threshold=None,
    beta_threshold=None,
    reads_threshold=None,
    min_conditions=None,
    n_cols=None,
    n_rows=None,
    split_by_tissue=False,
    highlight_timepoints=None,
    highlight_color='#ffd54a',
    show_missing=False,
):
    """
    Plot true vs. predicted splice site usage over time. Works efficiently with
    DataFrames indexed by ['Species', 'Chromosome', 'Position'] or
    ['Species', 'Chromosome', 'Position', 'Strand'].

    Args:
        tissue_subset: Tissue name or list of tissue names to display
            (e.g., 'Brain' or ['Brain', 'Heart']). If None, all tissues are shown.
        highlight_timepoints: Timepoint(s) to highlight with a shaded vertical band.
            May be a single int, a list of ints (applied to every panel), or a dict
            mapping a site key to a list of ints for per-panel highlighting. Accepted
            site keys: 'species chrom:pos', 'chrom:pos', (species, chrom, pos), or the
            integer position.
        highlight_color: Color of the highlight band.
        show_missing: If True, a requested site with no matching data still gets a
            placeholder panel (labeled with its species, body "no data") instead of
            being dropped — keeps a fixed panel grid across figures.
    """
    def _highlights_for(species, chrom, pos):
        """Resolve which timepoints to highlight for a given panel/site."""
        if highlight_timepoints is None:
            return []
        if isinstance(highlight_timepoints, dict):
            for key in ((str(species).lower(), str(chrom), int(pos)),
                        f'{str(species).lower()} {chrom}:{pos}',
                        f'{chrom}:{pos}', int(pos)):
                if key in highlight_timepoints:
                    tps = highlight_timepoints[key]
                    break
            else:
                return []
        else:
            tps = highlight_timepoints
        if np.isscalar(tps):
            tps = [tps]
        return [int(t) for t in tps]
    if isinstance(site_coords, str):
        site_coords = [site_coords]

    # --- Normalize column casing (only for non-index columns if indexed) -
    rename_targets = ['species', 'chromosome', 'position', 'strand', 'tissue', 'timepoint']
    if isinstance(df.index, pd.MultiIndex):
        # Normalize index names if they exist
        df.index.names = [col.capitalize() if col.lower() in rename_targets else col for col in df.index.names]
    
    if hasattr(df, 'columns') and df.columns is not None:
        df.columns = [col.capitalize() if col.lower() in rename_targets else col for col in df.columns]

    # Quick helper to safely get columns/index names
    def get_all_dims(dataframe):
        dims = []
        if isinstance(dataframe.index, pd.MultiIndex):
            dims.extend([name for name in dataframe.index.names if name is not None])
        else:
            if dataframe.index.name is not None:
                dims.append(dataframe.index.name)
        if hasattr(dataframe, 'columns'):
            dims.extend(dataframe.columns.tolist())
        return dims

    all_dims = get_all_dims(df)

    if 'Condition_Name' not in all_dims:
        if set(['Tissue', 'Timepoint']).issubset(all_dims):
            # Temporarily working with columns is safer for transformations
            if isinstance(df.index, pd.MultiIndex):
                df = df.copy() # Avoid mutations
                # Ensure we can add column safely if it's in index vs columns
                tissue_series = df.index.get_level_values('Tissue') if 'Tissue' in df.index.names else df['Tissue']
                tp_series = df.index.get_level_values('Timepoint') if 'Timepoint' in df.index.names else df['Timepoint']
                df['Condition_Name'] = tissue_series.astype(str) + '_' + tp_series.astype(str)
            else:
                df['Condition_Name'] = df['Tissue'].astype(str) + '_' + df['Timepoint'].astype(str)
            if verbose:
                print("Debug: Created 'Condition_Name' column")
        else:
            raise ValueError("DataFrame must contain 'Condition_Name' or both 'Tissue' and 'Timepoint'.")

    # Verify true/pred exist
    for col in (true_col, pred_col):
        if col not in all_dims:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Check if DataFrame has optimal index for fast lookups
    has_index = isinstance(df.index, pd.MultiIndex) and set(['Species', 'Chromosome', 'Position']).issubset(df.index.names)
    has_strand_in_index = has_index and 'Strand' in df.index.names

    # --- Parse site_coords into (species, chrom, pos, strand) tuples ----
    site_filters = []
    for coord in site_coords:
        parts = coord.strip().split()
        if len(parts) != 2:
            print(f"Warning: Invalid coordinate format '{coord}'")
            continue
        species = parts[0].lower()
        chrom_pos_strand = parts[1].split(':')
        if len(chrom_pos_strand) < 2 or len(chrom_pos_strand) > 3:
            print(f"Warning: Invalid chrom:pos[:strand] format in '{coord}'")
            continue
        chrom = chrom_pos_strand[0]
        strand = chrom_pos_strand[2] if len(chrom_pos_strand) == 3 else None

        if '-' in chrom_pos_strand[1]:
            try:
                start, end = chrom_pos_strand[1].split('-')
                start = int(start.replace('_', ''))
                end = int(end.replace('_', ''))
            except ValueError:
                print(f"Warning: Invalid range in '{coord}'")
                continue

            if has_index:
                try:
                    if not df.index.is_monotonic_increasing:
                        print("Warning: MultiIndex is not sorted. Slicing may not work correctly.")
                        print("         Run: df = df.sort_index() before plotting.")
                    
                    # Target slicing using cross-section or index slicers
                    idx_slicer = (species, str(chrom), slice(start, end))
                    if has_strand_in_index and strand is not None:
                        idx_slicer = (species, str(chrom), slice(start, end), strand)
                    
                    level_sel = df.loc[idx_slicer, :]
                    if isinstance(level_sel, pd.Series):
                        level_sel = level_sel.to_frame().T
                    
                    # Dynamically figure out remaining positions
                    sites_in_range = level_sel.index.to_frame(index=False)[['Species', 'Chromosome', 'Position']].drop_duplicates()
                    if has_strand_in_index:
                        sites_in_range['Strand'] = level_sel.index.get_level_values('Strand')
                except (KeyError, IndexError) as e:
                    if verbose:
                        print(f"Debug: KeyError/IndexError for range query: {e}")
                    sites_in_range = pd.DataFrame()
            else:
                mask = (
                    (df['Species'] == species)
                    & (df['Chromosome'] == str(chrom))
                    & (df['Position'] >= start)
                    & (df['Position'] <= end)
                )
                if strand is not None and 'Strand' in df.columns:
                    mask = mask & (df['Strand'] == strand)
                sites_in_range = df[mask][['Species', 'Chromosome', 'Position', 'Strand' if 'Strand' in df.columns else 'Position']].drop_duplicates()

            if sites_in_range.empty:
                print(f"Warning: No sites found in range {coord}")
                continue
            
            sites_in_range = sites_in_range.sort_values(['Position'])
            for _, row in sites_in_range.iterrows():
                strand_val = row['Strand'] if 'Strand' in sites_in_range.columns else (strand if strand else '?')
                site_filters.append((row['Species'], row['Chromosome'], row['Position'], strand_val))
            print(f"Found {len(sites_in_range)} sites in {coord}")
        else:
            try:
                pos = int(chrom_pos_strand[1].replace('_', ''))
            except ValueError:
                print(f"Warning: Invalid position in '{coord}'")
                continue

            if strand is None:
                if has_index:
                    try:
                        # Extract cross section safely to check strand labels
                        idx_check = (species, str(chrom), pos)
                        level_sel = df.loc[idx_check, :]
                        if has_strand_in_index:
                            strand = level_sel.index.get_level_values('Strand')[0] if len(level_sel) > 0 else '?'
                        else:
                            strand = '?'
                    except (KeyError, IndexError):
                        strand = '?'
                else:
                    mask = ((df['Species'] == species) & (df['Chromosome'] == str(chrom)) & (df['Position'] == pos))
                    matching_rows = df[mask]
                    strand = matching_rows.iloc[0]['Strand'] if not matching_rows.empty and 'Strand' in matching_rows.columns else '?'
            site_filters.append((species, chrom, pos, strand))

    if not site_filters:
        print("No valid site coordinates provided")
        return None

    # --- Build df_filtered (union of all requested sites) ---------------
    if has_index:
        df_filtered_list = []
        for species, chrom, pos, strand in site_filters:
            try:
                # Explicit index selection match based on index structure
                if has_strand_in_index and strand != '?':
                    subset = df.loc[[(species, str(chrom), pos, strand)], :]
                else:
                    subset = df.loc[[(species, str(chrom), pos)], :]
                df_filtered_list.append(subset)
            except KeyError:
                continue
        if not df_filtered_list:
            print("No data found for any of the requested sites")
            return None
        # Crucial fix: reset_index drops standard multiindex names directly into clean columns!
        df_filtered = pd.concat(df_filtered_list, axis=0).reset_index()
    else:
        query_parts = []
        for species, chrom, pos, strand in site_filters:
            if strand != '?' and 'Strand' in df.columns:
                query_parts.append(f"(Species == {repr(species)} and Chromosome == {repr(str(chrom))} and Position == {pos} and Strand == {repr(strand)})")
            else:
                query_parts.append(f"(Species == {repr(species)} and Chromosome == {repr(str(chrom))} and Position == {pos})")
        query_str = " or ".join(query_parts)
        try:
            df_filtered = df.query(query_str).copy()
        except Exception:
            site_masks = []
            for species, chrom, pos, strand in site_filters:
                mask = ((df['Species'] == species) & (df['Chromosome'] == str(chrom)) & (df['Position'] == pos))
                if strand != '?' and 'Strand' in df.columns:
                    mask = mask & (df['Strand'] == strand)
                site_masks.append(mask)
            combined_mask = site_masks[0]
            for mask in site_masks[1:]:
                combined_mask = combined_mask | mask
            df_filtered = df[combined_mask].copy()

    if df_filtered.empty:
        print("No data found for any of the requested sites")
        return None

    if tissue_order is None:
        tissue_order = TISSUE_ORDER
    if tissue_colors is None:
        tissue_colors = TISSUE_COLORS

    selected_tissues = _normalize_tissue_subset(tissue_subset)
    if selected_tissues is not None:
        if 'Tissue' not in df_filtered.columns:
            raise ValueError("tissue_subset requires a 'Tissue' column")

        df_filtered = df_filtered[df_filtered['Tissue'].astype(str).isin(selected_tissues)].copy()
        if df_filtered.empty:
            print("No data found for requested tissue_subset")
            return None

        selected_set = set(selected_tissues)
        ordered_selected = [t for t in tissue_order if t in selected_set]
        ordered_selected.extend([t for t in selected_tissues if t not in ordered_selected])
        tissue_order = ordered_selected

    label_to_type = {0: 'Donor+', 1: 'Acceptor+', 2: 'Donor-', 3: 'Acceptor-', 4: 'None'}

    def site_mask(d, species, chrom, pos, strand):
        m = (
            (d['Species'].astype(str).str.lower() == str(species).lower())
            & (d['Chromosome'].astype(str) == str(chrom))
            & (d['Position'] == pos)
        )
        if strand != '?' and 'Strand' in d.columns:
            m = m & (d['Strand'] == strand)
        return m

    # --- Optional min_conditions pre-filter ------------------------------
    if min_conditions is not None:
        valid_site_filters = []
        for species, chrom, pos, strand in site_filters:
            df_site_check = df_filtered[site_mask(df_filtered, species, chrom, pos, strand)]
            if not df_site_check.empty:
                n_conditions = df_site_check['Condition_Name'].nunique()
                if n_conditions >= min_conditions:
                    valid_site_filters.append((species, chrom, pos, strand))
        site_filters = valid_site_filters

    if not site_filters:
        print("No sites to plot after applying filters")
        return None

    # --- Per-site filtering (thresholds) + site_type --------------------
    site_data = []
    for species, chrom, pos, strand in site_filters:
        df_site = df_filtered[site_mask(df_filtered, species, chrom, pos, strand)].copy()

        if df_site.empty and not show_missing:
            continue

        if alpha_threshold is not None and 'Alpha' in df_site.columns:
            df_site = df_site[df_site['Alpha'] >= alpha_threshold]
        if beta_threshold is not None and 'Beta' in df_site.columns:
            df_site = df_site[df_site['Beta'] >= beta_threshold]
        if reads_threshold is not None and {'Alpha', 'Beta'}.issubset(df_site.columns):
            df_site = df_site[(df_site['Alpha'] + df_site['Beta']) >= reads_threshold]

        if not df_site.empty and 'Label' in df_site.columns:
            site_type = label_to_type.get(df_site.iloc[0]['Label'], f'{strand}')
        else:
            site_type = f'{strand}'

        if not df_site.empty:
            df_site = df_site.dropna(subset=['Tissue', 'Timepoint']).copy()
            if not df_site.empty:
                df_site['Timepoint'] = df_site['Timepoint'].astype(int)

        site_data.append({
            'species': species, 'chrom': chrom, 'pos': pos, 'strand': strand,
            'site_type': site_type, 'df_site': df_site,
        })

    # --- Build the panel list -------------------------------------------
    panels = []
    all_tissues_found = set()
    for sd in site_data:
        df_site = sd['df_site']
        tissues_present = [] if df_site.empty else [t for t in tissue_order if t in df_site['Tissue'].values]
        all_tissues_found.update(tissues_present)

        if split_by_tissue and tissues_present:
            for t in tissues_present:
                panels.append({**sd, 'tissue': t})
        else:
            panels.append({**sd, 'tissue': None})

    n_panels = len(panels)
    if n_panels == 0:
        print("No valid panels to plot.")
        return None

    # --- Grid layout -------------------------------------------------------
    if n_cols is None and n_rows is None:
        n_cols = min(3, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
    elif n_cols is not None and n_rows is None:
        n_rows = (n_panels + n_cols - 1) // n_cols
    elif n_rows is not None and n_cols is None:
        n_cols = (n_panels + n_rows - 1) // n_rows

    if figsize is None:
        figsize = (n_cols * 3, n_rows * 2.2)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for panel_idx, panel in enumerate(panels):
        ax = axes[panel_idx]
        species, chrom, pos = panel['species'], panel['chrom'], panel['pos']
        site_type, df_site, tissue = panel['site_type'], panel['df_site'], panel['tissue']

        label = f'{species} {chrom}:{pos} ({site_type})'
        label = label.replace('(?)', '')
        if split_by_tissue and tissue is not None:
            label = f'{label}\n{tissue}'

        df_plot = df_site if tissue is None else df_site[df_site['Tissue'] == tissue]

        if df_plot.empty:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, color='#999999', fontsize=9)
            ax.set_title(str(species) if pos in (0, None) else label, fontsize=11)
            ax.set_xlim(0.5, 15.5); ax.set_ylim(-0.05, 1.05)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel('Timepoint', fontsize=10); ax.set_ylabel('Usage', fontsize=10)
            continue

        plot_tissues = [tissue] if tissue is not None else [t for t in tissue_order if t in df_plot['Tissue'].values]

        n_t = len(plot_tissues)
        offsets = {t: jitter * (i - (n_t - 1) / 2) / (n_t - 1) if jitter > 0 and n_t > 1 else 0.0 for i, t in enumerate(plot_tissues)}

        size_col = size if isinstance(size, str) and size in df_plot.columns else None
        has_size_range = False
        if size_col:
            size_vals = pd.to_numeric(df_plot[size_col], errors='coerce')
            smin, smax = size_vals.min(), size_vals.max()
            has_size_range = pd.notna(smin) and pd.notna(smax) and (smax > smin)

        for t in plot_tissues:
            t_data = df_plot[df_plot['Tissue'] == t]
            agg_dict = {
                'true_mean': (true_col, 'mean'), 'true_std': (true_col, 'std'),
                'pred_mean': (pred_col, 'mean'), 'pred_std': (pred_col, 'std'),
            }
            if size_col:
                agg_dict['size_value'] = (size_col, 'mean')
            
            grouped = t_data.groupby('Timepoint').agg(**agg_dict).reset_index()
            grouped[['true_std', 'pred_std']] = grouped[['true_std', 'pred_std']].fillna(0)

            color = tissue_colors.get(t, '#808080')
            x = grouped['Timepoint'] + offsets[t]

            point_sizes = 20 + 100 * (grouped['size_value'] - smin) / (smax - smin) if has_size_range else pd.Series(20, index=grouped.index)
            point_sizes = point_sizes.fillna(10).clip(lower=10, upper=200)

            # True -- plotted only over timepoints where a true value was actually
            # observed, so the line connects across whatever timepoints have ground
            # truth instead of breaking (matplotlib splits a line at NaN y-values) at
            # every timepoint that's merely missing an observation. Predicted values
            # are unaffected -- if predictions exist for every timepoint (e.g. because
            # they were freshly computed rather than looked up from an observation-only
            # table), the pred line below still spans the full x range.
            true_present = grouped['true_mean'].notna()
            x_true = x[true_present]
            true_mean = grouped['true_mean'][true_present]
            true_std = grouped['true_std'][true_present]
            true_sizes = point_sizes[true_present]
            ax.fill_between(x_true, true_mean - true_std, true_mean + true_std, color=color, alpha=0.1, linewidth=0)
            ax.plot(x_true, true_mean, color=color, linewidth=1.5, alpha=0.9, zorder=2)
            ax.scatter(x_true, true_mean, s=true_sizes, color=color, marker='o', edgecolors='white', linewidths=0.5, alpha=0.9, zorder=3)

            # Pred
            ax.plot(x, grouped['pred_mean'], color=color, linewidth=1, alpha=0.9, zorder=2, linestyle='--')
            ax.scatter(x, grouped['pred_mean'], s=point_sizes, color=color, marker='x', linewidths=0.5, alpha=0.9, zorder=3)

        for htp in _highlights_for(species, chrom, pos):
            ax.axvspan(htp - 0.4, htp + 0.4, color=highlight_color, alpha=0.35, zorder=0, linewidth=0)

        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xlim(0.5, 15.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Timepoint', fontsize=10)
        ax.set_ylabel('Usage', fontsize=10)
        ax.set_title(label, fontsize=11)

    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    # --- Legends -----------------------------------------------------------
    style_handles = [
        plt.Line2D([0], [0], color='black', linewidth=1.5, marker='o', markersize=4, label='True'),
        plt.Line2D([0], [0], color='black', linewidth=1, marker='x', markersize=4, label='Predicted', linestyle='--'),
    ]
    if all_tissues_found and not split_by_tissue:
        legend_tissues = [t for t in tissue_order if t in all_tissues_found]
        tissue_handles = [plt.Line2D([0], [0], color=tissue_colors.get(t, '#808080'), linewidth=2, marker='o', markersize=4, label=t) for t in legend_tissues]
        fig.legend(handles=tissue_handles, loc='center left', bbox_to_anchor=(1.0, 0.65), fontsize=10, title='Tissue')
    
    fig.legend(handles=style_handles, loc='center left', bbox_to_anchor=(1.0, 0.4), fontsize=10, title='Series')

    if title:
        fig.suptitle(title, fontsize=14, y=0.995)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    return fig

def plot_cluster_trajectory_profiles(
    grid,
    features,
    labels,
    features2=None,
    series_labels=("True", "Predicted"),
    cluster_order=None,
    color_by=None,
    cmap="tab20",
    title_by=None,
    n_draw=120,
    n_cols=4,
    seed=0,
    ylabel="SSE",
    xlabel="Timepoint",
    ylim=(-0.05, 1.05),
    figsize=None,
    suptitle=None,
):
    """Plot per-cluster trajectory profiles: thin individual trajectories + a
    mean ± SD band + a bold **plain-mean** line, one panel per cluster.

    Pure plotting — no smoothing is done here (that would over-smooth). Pass
    *already GP-smoothed* per-trajectory ``features`` (e.g. from
    :func:`smooth_all_trajectories`, or the ``_gp_features.npy`` saved by
    ``cluster_trajectories.py``); the bold line is then the plain average of those
    smooth curves, exactly as in the clustering pipeline's saved plots.

    Matches the cluster-profile style of ``splice_trajectory_clustering.ipynb`` /
    ``cluster_trajectories.py`` and is shared by ``splice_trajectory_type_eval.ipynb``
    (which passes a second, predicted series drawn dashed).

    Parameters
    ----------
    grid : array-like, shape (G,)
        Common x positions (e.g. developmental timepoints).
    features : ndarray, shape (N, G)
        Per-trajectory (smoothed) values on ``grid`` (may contain NaN).
    labels : array-like, shape (N,)
        Cluster id per trajectory. NaN entries are ignored.
    features2 : ndarray or None
        Optional second series (same shape), drawn dashed (e.g. predicted usage).
    series_labels : (str, str)
        Legend labels for ``features`` (solid) and ``features2`` (dashed).
    cluster_order : list or None
        Clusters (and their order); defaults to sorted unique labels.
    color_by : dict or None
        Maps cluster id -> colour. If None, colours come from ``cmap``.
    title_by : dict or None
        Maps cluster id -> string appended to the panel title (e.g. shape name).
    n_draw : int
        Max individual trajectories drawn per cluster (per series).
    """
    grid = np.asarray(grid, dtype=float)
    labels = np.asarray(labels)
    features = np.asarray(features, dtype=float)
    if features2 is not None:
        features2 = np.asarray(features2, dtype=float)

    valid = ~pd.isna(labels)
    if cluster_order is None:
        cluster_order = sorted(pd.unique(labels[valid]))

    n = len(cluster_order)
    n_cols = min(n_cols, max(1, n))
    n_rows = (n + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (n_cols * 2.7, n_rows * 2.3)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             squeeze=False, sharey=True, sharex=True)
    rng = np.random.default_rng(seed)
    default_colors = plt.get_cmap(cmap)

    for idx, cl in enumerate(cluster_order):
        ax = axes[idx // n_cols, idx % n_cols]
        color = (color_by.get(cl) if color_by is not None
                 else default_colors(idx % default_colors.N))
        rows = np.where(labels == cl)[0]

        for F, ls, alpha_band in ((features, "-", 0.25),
                                  (features2, "--", 0.10)):
            if F is None:
                continue
            Fc = F[rows]

            draw = rng.choice(len(rows), size=min(n_draw, len(rows)), replace=False)
            for i in draw:
                ax.plot(grid, Fc[i], color=color, alpha=0.08, lw=0.5, ls=ls)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean = np.nanmean(Fc, axis=0)
                sd = np.nanstd(Fc, axis=0)
            ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=alpha_band, lw=0)
            ax.plot(grid, mean, ls, color=color, lw=2.5)

        title = f"Cluster {int(cl)}"
        if title_by is not None and cl in title_by:
            title += f" \u00b7 {title_by[cl]}"
        title += f"\n(n={len(rows):,})"
        ax.set_title(title, fontsize=8, fontweight="bold")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlim(grid[0], grid[-1])
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        if idx % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel(xlabel, fontsize=8)

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    if features2 is not None:
        handles = [
            plt.Line2D([0], [0], color="black", lw=2.5, ls="-", label=series_labels[0]),
            plt.Line2D([0], [0], color="black", lw=2.5, ls="--", label=series_labels[1]),
        ]
        fig.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    if suptitle is not None:
        fig.suptitle(suptitle, y=1.0)
    plt.tight_layout()
    return fig


def plot_usage_density(
    df: pd.DataFrame,
    title: str = None,
    site_coords=None,
    true_col: str = "true",
    pred_col: str = "pred",
    gridsize: int = 25,
    mincnt: int = 1,
    cmap: str = "magma_r",
):
    """
    Hexbin density plot of predicted vs true splice-site usage from a DataFrame.

    Args:
        df: DataFrame containing at least true_col and pred_col, and optionally
                Species/Chromosome/Position/Strand for site filtering.
        title: Plot title.
        site_coords: None, str, or list[str], with formats:
            - "species chrom:pos[:strand]"
            - "species chrom:start-end[:strand]"
        true_col: Column with true usage values.
        pred_col: Column with predicted usage values.
    """
    if df is None or df.empty:
        return None

    if true_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f"Missing required columns: '{true_col}' and/or '{pred_col}'")

    # Normalize column casing
    rename_targets = ['species', 'chromosome', 'position', 'strand', 'tissue', 'timepoint']
    df.columns = [col.capitalize() if col.lower() in rename_targets else col for col in df.columns]
    
    df_plot = df.copy()

    # Optional filtering by site coordinates (same style as plot_splice_site_dynamics)
    if site_coords is not None:
        if isinstance(site_coords, str):
            site_coords = [site_coords]

        required_cols = {"Species", "Chromosome", "Position", "Strand"}
        if not required_cols.issubset(df_plot.columns):
            raise ValueError(f"site_coords filtering requires columns: {sorted(required_cols)}")

        masks = []
        for coord in site_coords:
            parts = coord.strip().split()
            if len(parts) != 2:
                continue

            species = parts[0].lower()
            chrom_pos_strand = parts[1].split(":")
            if len(chrom_pos_strand) < 2 or len(chrom_pos_strand) > 3:
                continue

            chrom = str(chrom_pos_strand[0])
            pos_or_range = chrom_pos_strand[1]
            strand = chrom_pos_strand[2] if len(chrom_pos_strand) == 3 else None

            base_mask = (
                (df_plot["Species"].str.lower() == species)
                & (df_plot["Chromosome"].astype(str) == chrom)
            )

            if "-" in pos_or_range:
                try:
                    start, end = map(int, pos_or_range.split("-"))
                except ValueError:
                    continue
                m = base_mask & (df_plot["Position"] >= start) & (df_plot["Position"] <= end)
            else:
                try:
                    pos = int(pos_or_range)
                except ValueError:
                    continue
                m = base_mask & (df_plot["Position"] == pos)

            if strand is not None:
                m = m & (df_plot["Strand"].astype(str) == strand)

            masks.append(m)

        if not masks:
            return None

        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask = combined_mask | m
        df_plot = df_plot[combined_mask].copy()

    # Keep valid points only
    df_plot = df_plot[[true_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(df_plot) < 2:
        raise ValueError("Not enough valid data points to plot after filtering")

    true_arr = df_plot[true_col].to_numpy(dtype=np.float32)
    pred_arr = df_plot[pred_col].to_numpy(dtype=np.float32)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    hb = ax.hexbin(true_arr, pred_arr, gridsize=gridsize, cmap=cmap, mincnt=mincnt)

    if true_arr.std() > 1e-8 and pred_arr.std() > 1e-8:
        corr = float(np.corrcoef(true_arr, pred_arr)[0, 1])
        ax.text(
            0.05, 0.95, f"r = {corr:.3f}\nn = {len(true_arr):,}",
            transform=ax.transAxes, fontsize=10, va="top"
        )

    # Marginals
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.2], sharex=ax)
    ax_histx.hist(true_arr, bins=30, color="gray", alpha=0.7)
    ax_histx.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    ax_histy = ax.inset_axes([1.05, 0, 0.2, 1], sharey=ax)
    ax_histy.hist(pred_arr, bins=30, orientation="horizontal", color="gray", alpha=0.7)
    ax_histy.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    cax = ax.inset_axes([1.28, 0, 0.04, 1])
    fig.colorbar(hb, cax=cax, label="Count")

    ax.set_xlabel(f"True Usage ({true_col})")
    ax.set_ylabel(f"Predicted Usage ({pred_col})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Genomic-overlap category plots
# (paired loading/computation functions live in alphagenome_pytorch.evaluation.splicing)
# ══════════════════════════════════════════════════════════════════════════════


def _short_feature_label(category):
    """Shorten a ';'-joined GTF feature-overlap category for axis labels."""
    return (category
        .replace('five_prime_utr', "5'UTR")
        .replace('three_prime_utr', "3'UTR")
        .replace("CDS;exon", "exon;CDS")
        .replace("CDS;5'UTR", "5'UTR;CDS")
        .replace("CDS;3'UTR", "3'UTR;CDS")
        .replace(';', ' | ')
    )


def plot_auprc_by_category_all_species(species_dfs, sorted_cats, min_positives=0, fig_size=(6, 5), out_dir=None):
    """Per-class AUPRC (top row) and n_positives (bottom row) bars, one column per species."""
    colors = CLASS_COLORS
    class_names = CLASS_LABELS
    bar_width = 0.2
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, 4)
    labels = [_short_feature_label(c) for c in sorted_cats]
    x = np.arange(len(sorted_cats))

    species_list = list(species_dfs.keys())
    n_species = len(species_list)
    fig, axes = plt.subplots(
        2, n_species,
        figsize=(fig_size[0] * n_species, fig_size[1]),
        gridspec_kw={'height_ratios': [1, 1]}
    )
    if n_species == 1:
        axes = axes[:, np.newaxis]
    fig.subplots_adjust(hspace=0.08, wspace=0.35)

    for col, sp in enumerate(species_list):
        auprc_df = species_dfs[sp]
        auprc_df = auprc_df[auprc_df['n_positives'] > min_positives]
        ax1, ax2 = axes[0, col], axes[1, col]

        for ci in range(4):
            sub = auprc_df[auprc_df['class_label'] == ci].set_index('category')
            y = [sub.loc[cat, 'auprc'] if cat in sub.index else np.nan for cat in sorted_cats]
            ax1.bar(x + offsets[ci], y, width=bar_width, color=colors[ci], label=class_names[ci])

        ax1.set_title(sp.capitalize(), fontsize=12, fontweight='normal', pad=6)
        ax1.set_ylim(0, 1.05)
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax1.grid(axis='y', linewidth=0.5, linestyle='--')
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.set_xticks(x)
        ax1.set_xticklabels([])
        ax1.set_xlim(-0.5, len(sorted_cats) - 0.5)
        if col == 0:
            ax1.set_ylabel('AUPRC', fontsize=11)
        else:
            ax1.set_yticklabels([])
        if col == n_species - 1:
            ax1.legend(fontsize=8, framealpha=0.7, loc='lower left',
                       bbox_to_anchor=(1.02, 0), borderaxespad=0)

        for ci in range(4):
            sub = auprc_df[auprc_df['class_label'] == ci].set_index('category')
            counts = [sub.loc[cat, 'n_positives'] if cat in sub.index else 0 for cat in sorted_cats]
            ax2.bar(x + offsets[ci], counts, width=bar_width, color=colors[ci])

        ax2.grid(axis='y', linewidth=0.5, linestyle='--')
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
        ax2.set_xlim(-0.5, len(sorted_cats) - 0.5)
        if col == 0:
            ax2.set_ylabel('N sites', fontsize=11)
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
        else:
            ax2.set_yticklabels([])

    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, 'auprc_by_category_all_species.png'), bbox_inches='tight', dpi=150)
    plt.show()


def plot_auprc_by_category_all_species_summary(species_dfs, sorted_cats, min_positives=0, fig_size=(6, 5), colors_cats=None, out_dir=None):
    """Median AUPRC across classes (top row) and total sites (bottom row), one column per species."""
    labels = [_short_feature_label(c) for c in sorted_cats]
    x = np.arange(len(sorted_cats))

    species_list = list(species_dfs.keys())
    n_species = len(species_list)
    fig, axes = plt.subplots(
        2, n_species,
        figsize=(fig_size[0] * n_species, fig_size[1]),
        gridspec_kw={'height_ratios': [1, 1]}
    )
    if n_species == 1:
        axes = axes[:, np.newaxis]
    fig.subplots_adjust(hspace=0.08, wspace=0.35)

    for col, sp in enumerate(species_list):
        auprc_df = species_dfs[sp]
        auprc_df = auprc_df[auprc_df['n_positives'] > min_positives]
        ax1, ax2 = axes[0, col], axes[1, col]

        summary = (auprc_df[auprc_df['category'].isin(sorted_cats)]
            .groupby('category')
            .agg(median_auprc=('auprc', 'median'), total_sites=('n_positives', 'sum'))
        )

        y       = [summary.loc[cat, 'median_auprc'] if cat in summary.index else np.nan for cat in sorted_cats]
        counts  = [summary.loc[cat, 'total_sites']  if cat in summary.index else 0       for cat in sorted_cats]

        if colors_cats is not None:
            bar_colors = [colors_cats.get(cat, '#378ADD') for cat in sorted_cats]
        else:
            bar_colors = ['#378ADD'] * len(sorted_cats)
        ax1.bar(x, y, color=bar_colors, width=0.6)
        ax1.set_title(sp.capitalize(), fontsize=12, fontweight='normal', pad=6)
        ax1.set_ylim(0, 1.05)
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax1.grid(axis='y', linewidth=0.5, linestyle='--')
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.set_xticks(x)
        ax1.set_xticklabels([])
        if col == 0:
            ax1.set_ylabel('AUPRC', fontsize=11)
        else:
            ax1.set_yticklabels([])

        if colors_cats is not None:
            bar_colors = [colors_cats.get(cat, '#AAAAAA') for cat in sorted_cats]
        else:
            bar_colors = ["#AAAAAA"] * len(sorted_cats)
        ax2.bar(x, counts, color=bar_colors, width=0.6)
        ax2.grid(axis='y', linewidth=0.5, linestyle='--')
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
        if col == 0:
            ax2.set_ylabel('Total sites', fontsize=11)
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
        else:
            ax2.set_yticklabels([])

    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, 'auprc_by_category_all_species_summary.png'), bbox_inches='tight', dpi=150)
    plt.show()


def plot_pearson_r_by_category_all_species(species_dfs, sorted_cats, min_positives=0, fig_size=(6, 5), colors_cats=None, out_dir=None):
    """Median usage Pearson r (top row) and total sites (bottom row), one column per species."""
    labels = [_short_feature_label(c) for c in sorted_cats]
    x = np.arange(len(sorted_cats))

    species_list = list(species_dfs.keys())
    n_species = len(species_list)
    fig, axes = plt.subplots(
        2, n_species,
        figsize=(fig_size[0] * n_species, fig_size[1]),
        gridspec_kw={'height_ratios': [1, 1]}
    )
    if n_species == 1:
        axes = axes[:, np.newaxis]
    fig.subplots_adjust(hspace=0.08, wspace=0.35)

    for col, sp in enumerate(species_list):
        pearson_r_df = species_dfs[sp]
        pearson_r_df = pearson_r_df[pearson_r_df['n_sites'] > min_positives]
        ax1, ax2 = axes[0, col], axes[1, col]

        summary = (pearson_r_df[pearson_r_df['category'].isin(sorted_cats)]
            .groupby('category')
            .agg(median_pearson_r=('pearson_r', 'median'), total_sites=('n_sites', 'sum'))
        )

        y       = [summary.loc[cat, 'median_pearson_r'] if cat in summary.index else np.nan for cat in sorted_cats]
        counts  = [summary.loc[cat, 'total_sites']  if cat in summary.index else 0       for cat in sorted_cats]

        if colors_cats is not None:
            bar_colors = [colors_cats.get(cat, '#378ADD') for cat in sorted_cats]
        else:
            bar_colors = ['#378ADD'] * len(sorted_cats)
        ax1.bar(x, y, color=bar_colors, width=0.6)
        ax1.set_title(sp.capitalize(), fontsize=12, fontweight='normal', pad=6)
        ax1.set_ylim(0, 1.05)
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax1.grid(axis='y', linewidth=0.5, linestyle='--')
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.set_xticks(x)
        ax1.set_xticklabels([])
        if col == 0:
            ax1.set_ylabel('Pearson r', fontsize=11)
        else:
            ax1.set_yticklabels([])

        if colors_cats is not None:
            bar_colors = [colors_cats.get(cat, '#AAAAAA') for cat in sorted_cats]
        else:
            bar_colors = ["#AAAAAA"] * len(sorted_cats)
        ax2.bar(x, counts, color=bar_colors, width=0.6)
        ax2.grid(axis='y', linewidth=0.5, linestyle='--')
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
        if col == 0:
            ax2.set_ylabel('Total sites', fontsize=11)
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))
        else:
            ax2.set_yticklabels([])

    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, 'pearson_r_by_category_all_species_summary.png'), bbox_inches='tight', dpi=150)
    plt.show()


def plot_splice_site_usage_by_tissue(
    df_all,
    site_coords,
    data_config,
    tissue_order=None,
    tissue_colors=None,
    figsize=None,
    title=None,
    verbose=False,
):
    """
    Plot splice site dynamics with one tissue per subplot for a single site.
    Displays both true (solid line) and predicted (dotted line) usage.

    Args:
        df_all: DataFrame with splice site data (species, chromosome, position, strand,
                condition columns, true_usage, pred_usage)
        site_coords: Site coordinate string. Format: "species chrom:pos[:strand]" (e.g., "mouse 1:15188:+")
                     Strand is optional; if not specified, all strands included.
        data_config: Config dict with condition_labels mapping for each species
        tissue_order: List of tissues in desired order
        tissue_colors: Dict mapping tissue names to colors
        figsize: Tuple of (width, height) or None for auto
        title: Figure title (auto-generated if None)
        verbose: If True, print condition information

    Returns:
        matplotlib Figure object
    """
    parts = site_coords.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate format '{site_coords}'. Expected 'species chrom:pos[:strand]'")

    species = parts[0].lower()
    chrom_pos_strand = parts[1].split(':')

    if len(chrom_pos_strand) < 2 or len(chrom_pos_strand) > 3:
        raise ValueError(f"Invalid chrom:pos[:strand] format in '{site_coords}'")

    chrom = chrom_pos_strand[0]
    try:
        pos = int(chrom_pos_strand[1])
    except ValueError:
        raise ValueError(f"Invalid position in '{site_coords}'")

    mask = (df_all['species'] == species) & (df_all['chromosome'] == str(chrom)) & (df_all['position'] == pos)
    df_site = df_all[mask].copy()

    if df_site.empty:
        raise ValueError(f"No data found for {species} {chrom}:{pos}")

    cond_col = next((c for c in df_site.columns if "cond" in c.lower()), None)
    if cond_col is None:
        raise ValueError(f"No condition column found for {species} {chrom}:{pos}")

    if tissue_order is None:
        tissue_order = TISSUE_ORDER
    if tissue_colors is None:
        tissue_colors = TISSUE_COLORS

    tissue_list = []
    timepoint_list = []

    for idx, row in df_site.iterrows():
        cond_idx = row[cond_col]
        cond_labels = data_config.get(species, {})
        idx_to_name = {v: k for k, v in cond_labels.items()}
        cond_name = idx_to_name.get(int(cond_idx), None)

        if cond_name:
            tissue = cond_name.rsplit('_', 1)[0] if '_' in cond_name else cond_name
            try:
                tp = int(cond_name.rsplit('_', 1)[1])
            except (IndexError, ValueError):
                tp = None
            tissue_list.append(tissue)
            timepoint_list.append(tp)
        else:
            tissue_list.append(None)
            timepoint_list.append(None)

    df_site['tissue'] = tissue_list
    df_site['timepoint'] = timepoint_list
    df_site = df_site.dropna(subset=['tissue', 'timepoint'])
    df_site['timepoint'] = df_site['timepoint'].astype(int)

    tissues = tissue_order
    all_tps = sorted(df_site['timepoint'].unique())

    if not tissues:
        raise ValueError(f"No valid tissue-timepoint combinations found for {species} {chrom}:{pos}")

    if verbose:
        print(f"\n{species} {chrom}:{pos}")
        print(f"  Total data points: {len(df_site)}")
        print(f"  Tissues present: {', '.join(tissues)}")
        print(f"  Timepoints: {', '.join(map(str, all_tps))}")

    n_tissues = len(tissues)
    n_cols = min(3, n_tissues)
    n_rows = (n_tissues + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (n_cols * 5, n_rows * 4)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for tissue_idx, tissue in enumerate(tissues):
        ax = axes[tissue_idx]

        tissue_data = df_site[df_site['tissue'] == tissue]
        color = tissue_colors.get(tissue, '#808080')

        if tissue_data.empty:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11, color='lightgrey')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_xlim(all_tps[0] - 0.5, all_tps[-1] + 0.5)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel('Timepoint', fontsize=10)
            ax.set_ylabel('Usage', fontsize=10)
            ax.set_title(f'{tissue}', fontsize=11)
            continue

        for metric_type, linestyle, marker, label in [
            ('true_usage', '-', 'o', 'True'),
            ('pred_usage', '--', 'x', 'Predicted')
        ]:
            grouped = tissue_data.groupby('timepoint')[metric_type].agg(['mean', 'std', 'count']).reset_index()
            grouped['std'] = grouped['std'].fillna(0)

            x_coords = grouped['timepoint']

            ax.fill_between(
                x_coords,
                grouped['mean'] - grouped['std'],
                grouped['mean'] + grouped['std'],
                color=color,
                alpha=0.12,
                linewidth=0
            )

            ax.plot(x_coords, grouped['mean'],
                    color=color, linewidth=1.5,
                    linestyle=linestyle,
                    marker=marker, markersize=4,
                    alpha=0.7, zorder=2, label=label)

        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xlim(all_tps[0] - 0.5, all_tps[-1] + 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel('Timepoint', fontsize=10)
        ax.set_ylabel('Usage', fontsize=10)
        ax.set_title(f'{tissue}', fontsize=11)

    for idx in range(n_tissues, len(axes)):
        axes[idx].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color='black', linewidth=1.5, linestyle='-', marker='o', markersize=4, label='True'),
        plt.Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', marker='x', markersize=4, label='Predicted')
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=10, framealpha=0.9)

    if title is None:
        title = f'{species} {chrom}:{pos}'
    fig.suptitle(title, fontsize=14, y=0.995)

    plt.tight_layout()

    return fig


def plot_splice_site_dynamics_from_usage_results(
    usage_results,
    site_coords,
    metric='both',  # 'both' plots true (solid) + pred (dotted)
    species_order=None,
    tissue_order=None,
    tissue_colors=None,
    figsize=None,
    title=None,
    jitter=0.0,
    verbose=False,
):
    """Plot true/predicted usage trajectories directly from the raw ``usage_results``
    dict produced by the usage-prediction parsing step (per-site chr_pos/tissues/
    timepoints/trues/preds arrays), rather than from a merged DataFrame."""
    if isinstance(site_coords, str):
        site_coords = [site_coords]

    if tissue_order is None:
        tissue_order = TISSUE_ORDER
    if tissue_colors is None:
        tissue_colors = TISSUE_COLORS
    if species_order is None:
        species_order = list(usage_results.keys())

    # Build/cache per-species dataframe from usage_results
    sp_df_cache = {}
    for sp, sp_data in usage_results.items():
        chr_pos = np.asarray(sp_data['chr_pos']).astype(str)
        cond_ids = np.asarray(sp_data['cond_ids'])
        tissues = np.asarray(sp_data['tissues'])
        timepoints = np.asarray(sp_data['timepoints'])
        trues = np.asarray(sp_data['trues'], dtype=float)
        preds = np.asarray(sp_data['preds'], dtype=float)

        split = np.char.partition(chr_pos, ':')
        chrom = split[:, 0]
        pos = split[:, 2].astype(int)

        df_sp = pd.DataFrame({
            'Species': sp,
            'Chromosome': chrom,
            'Position': pos,
            'condition': cond_ids,
            'tissue': tissues,
            'timepoint': pd.to_numeric(timepoints, errors='coerce'),
            'true': trues,
            'pred': preds,
        }).dropna(subset=['timepoint'])
        df_sp['timepoint'] = df_sp['timepoint'].astype(int)
        df_sp['SSE'] = (df_sp['pred'] - df_sp['true']) ** 2
        df_sp['abs_error'] = np.abs(df_sp['pred'] - df_sp['true'])

        sp_df_cache[sp] = df_sp

    # Parse coordinates
    site_filters = []
    for coord in site_coords:
        parts = coord.strip().split()
        if len(parts) != 2:
            print(f"Warning: Invalid coordinate format '{coord}'. Expected 'species chrom:pos' or 'species chrom:start-end'")
            continue

        species = parts[0].lower()
        if species not in sp_df_cache:
            print(f"Warning: Species '{species}' not found in usage_results")
            continue

        cparts = parts[1].split(':')
        if len(cparts) < 2:
            print(f"Warning: Invalid coordinate '{coord}'")
            continue

        chrom = cparts[0]
        pos_or_range = cparts[1]

        if '-' in pos_or_range:
            try:
                start, end = map(int, pos_or_range.split('-'))
            except ValueError:
                print(f"Warning: Invalid range in '{coord}'")
                continue

            df_sp = sp_df_cache[species]
            mask = (
                (df_sp['Chromosome'] == str(chrom)) &
                (df_sp['Position'] >= start) &
                (df_sp['Position'] <= end)
            )
            positions = sorted(df_sp.loc[mask, 'Position'].unique())
            if len(positions) == 0:
                print(f"Warning: No sites found in range {coord}")
                continue

            for p in positions:
                site_filters.append((species, str(chrom), int(p)))
            print(f"Found {len(positions)} sites in {coord}")
        else:
            try:
                pos = int(pos_or_range)
            except ValueError:
                print(f"Warning: Invalid position in '{coord}'")
                continue
            site_filters.append((species, str(chrom), pos))

    if not site_filters:
        print("No valid site coordinates provided")
        return None

    sp_rank = {sp: i for i, sp in enumerate(species_order)}
    site_filters = sorted(site_filters, key=lambda x: (sp_rank.get(x[0], 10**9), x[1], x[2]))

    n_sites = len(site_filters)
    n_cols = min(3, n_sites)
    n_rows = (n_sites + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (n_cols * 5, n_rows * 4)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    all_tissues_found = set()

    valid_metrics = {'SSE', 'true', 'pred', 'abs_error', 'both'}
    metric_col = metric if metric in valid_metrics else 'both'
    if metric not in valid_metrics:
        print(f"Warning: metric '{metric}' not recognized. Using 'both'.")

    for site_idx, (species, chrom, pos) in enumerate(site_filters):
        ax = axes[site_idx]
        df_site = sp_df_cache[species]
        df_site = df_site[(df_site['Chromosome'] == chrom) & (df_site['Position'] == pos)].copy()

        if df_site.empty:
            ax.text(0.5, 0.5, f'No data\n{species} {chrom}:{pos}', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue

        tissues_here = [t for t in tissue_order if t in df_site['tissue'].values]
        all_tissues_found.update(tissues_here)

        all_tps = sorted(df_site['timepoint'].unique())

        if verbose:
            print(f"\n{species} {chrom}:{pos}")
            print(f"  Total data points: {len(df_site)}")
            print(f"  Tissues present: {', '.join(tissues_here)}")
            print(f"  Timepoints: {', '.join(map(str, all_tps))}")

        n_tissues = len(tissues_here)
        if jitter > 0 and n_tissues > 1:
            tissue_offsets = {
                tissue: jitter * (i - (n_tissues - 1) / 2) / (n_tissues - 1)
                for i, tissue in enumerate(tissues_here)
            }
        else:
            tissue_offsets = {tissue: 0.0 for tissue in tissues_here}

        for tissue in tissues_here:
            tissue_data = df_site[df_site['tissue'] == tissue]
            color = tissue_colors.get(tissue, '#808080')

            if metric_col == 'both':
                grouped = tissue_data.groupby('timepoint')[['true', 'pred']].agg(['mean', 'std']).reset_index()
                grouped[('true', 'std')] = grouped[('true', 'std')].fillna(0)
                grouped[('pred', 'std')] = grouped[('pred', 'std')].fillna(0)
                x_coords = grouped['timepoint'] + tissue_offsets[tissue]

                # true: solid
                ax.plot(
                    x_coords, grouped[('true', 'mean')],
                    color=color, linewidth=2, linestyle='-',
                    marker='o', markersize=3, alpha=0.95, zorder=3
                )
                # pred: dotted
                ax.plot(
                    x_coords, grouped[('pred', 'mean')],
                    color=color, linewidth=2, linestyle=':',
                    marker='o', markersize=3, alpha=0.95, zorder=3
                )
            else:
                grouped = tissue_data.groupby('timepoint')[metric_col].agg(['mean', 'std']).reset_index()
                grouped['std'] = grouped['std'].fillna(0)
                x_coords = grouped['timepoint'] + tissue_offsets[tissue]

                ax.fill_between(
                    x_coords,
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color=color,
                    alpha=0.1,
                    linewidth=0
                )
                ax.plot(
                    x_coords, grouped['mean'],
                    color=color, linewidth=2, marker='o', markersize=4,
                    alpha=0.9, zorder=2
                )

        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if len(all_tps) > 0:
            ax.set_xlim(all_tps[0] - 0.5, all_tps[-1] + 0.5)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if metric_col in ('true', 'pred', 'abs_error', 'both'):
            ax.set_ylim(-0.05, 1.05)

        ax.set_xlabel('Timepoint', fontsize=10)
        ax.set_ylabel('Usage' if metric_col == 'both' else metric_col, fontsize=10)
        ax.set_title(f'{species} {chrom}:{pos}', fontsize=11)

    for idx in range(n_sites, len(axes)):
        axes[idx].set_visible(False)

    # Tissue color legend
    if all_tissues_found:
        legend_tissues = [t for t in tissue_order if t in all_tissues_found]
        tissue_handles = [
            plt.Line2D([0], [0], color=tissue_colors.get(t, '#808080'),
                       linewidth=2, marker='o', markersize=4, label=t)
            for t in legend_tissues
        ]
        fig.legend(
            handles=tissue_handles, loc='center left', bbox_to_anchor=(1.0, 0.6),
            fontsize=10, framealpha=0.9, title='Tissue'
        )

    # Line-style legend (only for true/pred mode)
    if metric_col == 'both':
        style_handles = [
            plt.Line2D([0], [0], color='black', linewidth=2, linestyle='-', label='True'),
            plt.Line2D([0], [0], color='black', linewidth=2, linestyle=':', label='Predicted'),
        ]
        fig.legend(
            handles=style_handles, loc='center left', bbox_to_anchor=(1.0, 0.35),
            fontsize=10, framealpha=0.9, title='Line type'
        )

    if title is None:
        if metric_col == 'both':
            title = 'True (solid) and Predicted (dotted) usage dynamics' if n_sites == 1 else f'True/Predicted dynamics for {n_sites} splice sites'
        else:
            title = f'{metric_col} dynamics across tissues' if n_sites == 1 else f'{metric_col} dynamics for {n_sites} splice sites'
    fig.suptitle(title, fontsize=14, y=0.995)

    plt.tight_layout(rect=[0, 0, 0.85, 1] if all_tissues_found else None)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Developmental splice-trajectory clustering pipeline
# (moved from scripts/cluster_trajectories.py so the script and the
#  splice_trajectory_clustering / splice_trajectory_type_eval notebooks share them)
# ══════════════════════════════════════════════════════════════════════════════


def save_cluster_plots(features_v, cluster_labels, cluster_shapes,
                       n_clusters, out_dir, prefix, label, random_seed=42,
                       heatmap_height=9.0, cluster_ids=None, color_denom=None,
                       site_shapes=None):
    """Save the clustering heatmap (``<prefix>_heatmap.png``) and the per-cluster
    shape-profile grid (``<prefix>_profiles.png``).

    ``site_shapes`` (optional, row-aligned with ``features_v``): per-site shape
    labels. When given, the heatmap's right-hand shape strip is coloured **per row**
    by each site's own shape (and the legend title reads "site shape") instead of one
    block per cluster — use it to show the per-site ``ShapeSite`` annotation rather
    than the cluster-mean ``ClusterShape``. The profiles grid is unaffected (still
    one panel per cluster).

    ``heatmap_height`` caps the heatmap figure height (inches). The heatmap shows one
    row per site via ``imshow`` (which rescales to the axes), so a fixed compact height
    is used instead of scaling with the number of sites — otherwise large datasets
    produce absurdly tall figures.

    ``cluster_ids`` overrides which cluster IDs to iterate over (default
    ``range(1, n_clusters + 1)``). Pass the actual reference-cluster IDs when
    plotting a subset of a larger *fixed* clustering (e.g. predictions assigned
    by nearest-centroid to a reference Ward clustering) so the ``C#`` labels
    line up with the reference plot instead of being renumbered 1..k.

    ``color_denom`` overrides the tab20 color denominator (default ``n_clusters``).
    Pass the *full* reference cluster count so a given cluster ID gets the
    identical color whether or not every reference cluster is populated in this
    particular plot (e.g. predictions covering only 65 of 80 reference clusters
    still color cluster 74 the same as the 80-cluster reference plot does)."""
    import os
    import matplotlib.gridspec as gridspec

    cluster_ids = list(cluster_ids) if cluster_ids is not None else list(range(1, n_clusters + 1))
    denom = color_denom or n_clusters
    color_of = lambda k: plt.cm.tab20(((k - 1) % denom) / denom)
    rng = np.random.default_rng(random_seed)

    # ── Heatmap ──────────────────────────────────────────────────────────────
    n_per = [(cluster_labels == k).sum() for k in cluster_ids]
    order = []
    for k in cluster_ids:
        idx_k = np.where(cluster_labels == k)[0]
        ctr   = features_v[cluster_labels == k].mean(axis=0)
        d     = np.linalg.norm(features_v[idx_k] - ctr, axis=1)
        order.extend(idx_k[np.argsort(d)].tolist())

    order  = np.array(order)
    fsort  = features_v[order]
    GAP    = max(3, int(len(fsort) * 0.004))
    blocks = []
    bk     = []
    s      = 0
    for i, (k, nc) in enumerate(zip(cluster_ids, n_per)):
        blocks.append(fsort[s:s + nc])
        bk.append(k)
        if i != len(cluster_ids) - 1:
            blocks.append(np.full((GAP, fsort.shape[1]), np.nan))
            bk.append(None)
        s += nc
    fsplit = np.vstack(blocks)

    row_ranges = []
    rc = 0
    for blk, k in zip(blocks, bk):
        nr = blk.shape[0]
        if k is not None:
            row_ranges.append((k, rc, rc + nr))
        rc += nr

    # Compact, capped height (imshow rescales the site rows to the axes regardless).
    fig = plt.figure(figsize=(10.5, float(heatmap_height)))
    gs  = gridspec.GridSpec(2, 4, width_ratios=[0.05, 0.02, 0.88, 0.03],
                            height_ratios=[0.05, 0.95], hspace=0.15, wspace=0.02)
    ax_cb    = fig.add_subplot(gs[0, 2])
    ax_lbl   = fig.add_subplot(gs[1, 0])
    ax_str   = fig.add_subplot(gs[1, 1])
    ax_ht    = fig.add_subplot(gs[1, 2])
    ax_shape = fig.add_subplot(gs[1, 3])   # right-side per-cluster shape strip

    masked = np.ma.masked_invalid(fsplit)
    cm_bad = plt.get_cmap("RdBu_r").copy()
    cm_bad.set_bad("white")
    im = ax_ht.imshow(masked, aspect="auto", cmap=cm_bad,
                      vmin=0, vmax=1, interpolation="nearest")
    ax_ht.set_xlabel("Developmental Timepoint")
    ax_ht.set_xticks(range(len(T_GRID)))
    ax_ht.set_xticklabels([str(int(t)) for t in T_GRID], fontsize=8)
    ax_ht.set_yticks([])
    k_lbl = f"k={len(cluster_ids)}" if denom == len(cluster_ids) else f"k={len(cluster_ids)}/{denom}"
    ax_ht.set_title(f"GP SSE  |  {label}  |  {len(fsort):,} sites  |  {k_lbl}",
                    fontsize=10)

    ax_str.set_xlim(0, 1); ax_str.set_ylim(len(fsplit), 0); ax_str.axis("off")
    ax_lbl.set_xlim(0, 1); ax_lbl.set_ylim(len(fsplit), 0); ax_lbl.axis("off")
    ax_shape.set_xlim(0, 1); ax_shape.set_ylim(len(fsplit), 0); ax_shape.axis("off")

    # Shape colour scheme (detect which classifier produced cluster_shapes, same as
    # the profiles section below).
    if set(cluster_shapes.values()) <= set(DYNAMIC_SHAPE_COLORS):
        _sc_order, _sc_colors = DYNAMIC_SHAPE_ORDER, DYNAMIC_SHAPE_COLORS
    else:
        _sc_order, _sc_colors = SHAPE_ORDER, SHAPE_COLORS

    # Right-side shape strip: per ROW (each site's own shape) when site_shapes is
    # given, else one colour block per CLUSTER.
    _per_site = site_shapes is not None and len(site_shapes) == len(features_v)
    if _per_site:
        import matplotlib.colors as _mcolors
        ss_sorted = np.asarray(site_shapes, dtype=object)[order]   # align to fsort rows
        rgba = np.ones((len(fsplit), 1, 4)); rgba[..., 3] = 0.0    # transparent → gaps blank
        s = 0
        for k, y0, y1 in row_ranges:
            nc = y1 - y0
            rgba[y0:y1, 0, :] = [_mcolors.to_rgba(_sc_colors.get(sh, "#7f7f7f"))
                                 for sh in ss_sorted[s:s + nc]]
            s += nc
        ax_shape.imshow(rgba, aspect="auto", interpolation="nearest",
                        origin="upper", extent=(0, 1, len(fsplit), 0))

    # Only label clusters thick enough to be legible at the compact height, so the
    # C# labels don't pile up over the many thin mid clusters.
    _min_label_rows = 0.012 * len(fsplit)
    for k, y0, y1 in row_ranges:
        ax_str.add_patch(plt.Rectangle((0, y0), 1, y1 - y0,
                                        color=color_of(k), ec="none"))
        if not _per_site:   # one shape colour block per cluster
            ax_shape.add_patch(plt.Rectangle((0, y0), 1, y1 - y0,
                                             color=_sc_colors.get(cluster_shapes[k], "#7f7f7f"),
                                             ec="none"))
        if (y1 - y0) >= _min_label_rows:
            ax_lbl.text(1.0, (y0+y1)/2, f"C{k}", ha="right", va="center",
                        fontsize=8, fontweight="bold", color=color_of(k))

    # Shape legend (only the shapes actually present), below the heatmap.
    _shape_vals = set(np.asarray(site_shapes)) if _per_site else set(cluster_shapes.values())
    _present = [s for s in _sc_order if s in _shape_vals]
    _handles = [plt.Rectangle((0, 0), 1, 1, color=_sc_colors.get(s, "#7f7f7f")) for s in _present]
    if _handles:
        ax_ht.legend(_handles, _present, loc="upper center",
                     bbox_to_anchor=(0.5, -0.06), ncol=min(len(_present), 7),
                     fontsize=7, frameon=False, handlelength=1.0, columnspacing=1.2,
                     title=("site shape" if _per_site else "cluster shape"), title_fontsize=7)

    cb = plt.colorbar(im, cax=ax_cb, orientation="horizontal")
    cb.set_label("SSE", fontsize=8)
    ax_cb.xaxis.set_ticks_position("top")
    ax_cb.xaxis.set_label_position("top")

    out_hm = os.path.join(out_dir, f"{prefix}_heatmap.png")
    fig.savefig(out_hm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Heatmap -> %s", out_hm)

    # ── Cluster profiles (ordered by shape, then descending mean SSE) ─────────
    # cluster_shapes may come from either shape-classification scheme (the fine-grained
    # classify_cluster_shape or the simpler 5-label classify_dynamic_direction) -- detect
    # which one from the labels actually present so ordering/coloring use the matching
    # SHAPE_ORDER/SHAPE_COLORS vs DYNAMIC_SHAPE_ORDER/DYNAMIC_SHAPE_COLORS pair. Using the
    # wrong pair leaves most labels unmatched, falling back to a uniform grey.
    if set(cluster_shapes.values()) <= set(DYNAMIC_SHAPE_COLORS):
        _shape_order, _shape_colors = DYNAMIC_SHAPE_ORDER, DYNAMIC_SHAPE_COLORS
    else:
        _shape_order, _shape_colors = SHAPE_ORDER, SHAPE_COLORS
    _shape_rank = {s: i for i, s in enumerate(_shape_order)}
    _sorted_clusters = sorted(
        cluster_ids,
        key=lambda k: (
            _shape_rank.get(cluster_shapes[k], len(_shape_order)),
            -float(features_v[cluster_labels == k].mean()),
        ),
    )

    K = len(cluster_ids)
    n_cols = min(5, K)
    n_rows = (K + n_cols - 1) // n_cols
    fig2, axes2 = plt.subplots(n_rows, n_cols,
                               figsize=(n_cols * 3, n_rows * 2.5),
                               squeeze=False, sharey=True)
    for plot_pos, k in enumerate(_sorted_clusters):
        r, c   = plot_pos // n_cols, plot_pos % n_cols
        ax     = axes2[r, c]
        mask_k = cluster_labels == k
        mean_k = features_v[mask_k].mean(axis=0)
        std_k  = features_v[mask_k].std(axis=0)
        shape_k = cluster_shapes[k]
        color  = _shape_colors.get(shape_k, "#7f7f7f")

        n_draw = min(80, mask_k.sum())
        for i in rng.choice(np.where(mask_k)[0], size=n_draw, replace=False):
            ax.plot(T_GRID, features_v[i], color=color, alpha=0.08, lw=0.5)
        ax.fill_between(T_GRID, mean_k - std_k, mean_k + std_k, alpha=0.3, color=color)
        ax.plot(T_GRID, mean_k, "-", color=color, lw=2.5)

        badge = shape_k.replace("_", "\n")
        ax.text(0.97, 0.97, badge, transform=ax.transAxes,
                ha="right", va="top", fontsize=6, fontweight="bold",
                color="white", linespacing=1.1,
                bbox=dict(facecolor=color, edgecolor="none",
                          boxstyle="round,pad=0.25", alpha=0.9))
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(T_GRID[0], T_GRID[-1])
        ax.set_title(f"C{k}  n={mask_k.sum():,}", fontsize=8, fontweight="bold")
        if c == 0:
            ax.set_ylabel("SSE", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    for idx in range(K, n_rows * n_cols):
        axes2[idx // n_cols, idx % n_cols].set_visible(False)

    fig2.suptitle(f"Cluster shapes (ordered by shape)  |  {label}", fontsize=11, y=1.01)
    plt.tight_layout()
    out_pr = os.path.join(out_dir, f"{prefix}_profiles.png")
    fig2.savefig(out_pr, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    log.info("Profiles -> %s", out_pr)
