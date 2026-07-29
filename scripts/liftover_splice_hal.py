import os
import tempfile
import subprocess
import pandas as pd

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
HAL_FILE = "240-mammals.hal"
REF_SPECIES = "Human"

# Paths to your pre-computed splice site Parquet files per species
# Target species keys must match the exact species names inside the .hal file
PARQUET_PATHS = {
    "Human": "/home/elek/sds/sd17d003/Anamaria/alphagenome_genomicsxai/data/Homo_sapiens/splice_sites_intersect_usage.parquet",
    "Mouse": "/home/elek/sds/sd17d003/Anamaria/alphagenome_genomicsxai/data/Mus_musculus/splice_sites_intersect_usage.parquet",
    "Rat": "/home/elek/sds/sd17d003/Anamaria/alphagenome_genomicsxai/data/Rattus_norvegicus/splice_sites_intersect_usage.parquet",
    "Rabbit": "/home/elek/sds/sd17d003/Anamaria/alphagenome_genomicsxai/data/Oryctolagus_cuniculus/splice_sites_intersect_usage.parquet",
    "Opossum": "/home/elek/sds/sd17d003/Anamaria/alphagenome_genomicsxai/data/Monodelphis_domestica/splice_sites_intersect_usage.parquet",
    "Macaque": "/home/elek/sds/sd17d003/Anamaria/alphagenome_genomicsxai/data/Macaca_mulatta/splice_sites_intersect_usage.parquet",
    "Chicken": "/home/elek/sds/sd17d003/Anamaria/alphagenome_genomicsxai/data/Gallus_gallus/splice_sites_intersect_usage.parquet"
}

# Max allowed distance (in bp) between lifted coordinate and target annotated site.
# Set to 0 for exact matches, or e.g., 2 to allow slight alignment shifts.
MAX_DISTANCE = 0 


def run_hal_liftover(hal_path, ref_sp, target_sp, input_bed, output_bed):
    """Executes halLiftover to project coordinates across species."""
    cmd = ["halLiftover", hal_path, ref_sp, input_bed, target_sp, output_bed]
    subprocess.run(cmd, check=True)


def main():
    # 1. Load Human Splice Sites Parquet
    # Expected columns: ['chrom', 'start', 'end', 'site_id', 'strand'] (0-based coordinates)
    human_df = pd.read_parquet(PARQUET_PATHS["Human"])
    
    # Write human coordinates to a temporary BED file for halLiftover
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as tmp_human_bed:
        human_bed_path = tmp_human_bed.name
        # Write BED6 format: chrom, start, end, name, score, strand
        for idx, row in human_df.iterrows():
            tmp_human_bed.write(f"{row['chrom']}\t{row['start']}\t{row['end']}\t{row['site_id']}\t0\t{row['strand']}\n")

    matched_results = []

    try:
        # 2. Iterate through target species and match
        for species, parquet_path in PARQUET_PATHS.items():
            if species == REF_SPECIES or not os.path.exists(parquet_path):
                continue

            print(f"[*] Processing {species}...")
            
            # Liftover human coordinates to target species
            with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as tmp_target_bed:
                target_bed_path = tmp_target_bed.name

            run_hal_liftover(HAL_FILE, REF_SPECIES, species, human_bed_path, target_bed_path)

            # Load lifted human coordinates in target space
            lifted_df = pd.read_csv(
                target_bed_path, 
                sep='\t', 
                header=None, 
                names=['chrom', 'start', 'end', 'human_site_id', 'score', 'strand']
            )

            # Load existing Target Species Splice Site Parquet
            target_df = pd.read_parquet(parquet_path)

            # 3. Match lifted coordinates to target species annotations
            if MAX_DISTANCE == 0:
                # Exact coordinate match
                merged = pd.merge(
                    lifted_df, 
                    target_df, 
                    on=['chrom', 'start', 'end', 'strand'], 
                    suffixes=('_lifted', '_target')
                )
            else:
                # Range-based / approximate match (within N bases)
                merged = pd.merge(lifted_df, target_df, on=['chrom', 'strand'], suffixes=('_lifted', '_target'))
                merged['dist'] = (merged['start_lifted'] - merged['start_target']).abs()
                merged = merged[merged['dist'] <= MAX_DISTANCE]

            merged['species'] = species
            matched_results.append(merged[['species', 'human_site_id', 'site_id', 'chrom', 'start', 'end', 'strand']])
            
            # Clean up target temp bed
            os.remove(target_bed_path)

        # 4. Concatenate all matched results and save
        final_matches = pd.concat(matched_results, ignore_index=True)
        final_matches.to_parquet("matched_cross_species_splice_sites.parquet")
        print("[+] Done! Saved matches to matched_cross_species_splice_sites.parquet")

    finally:
        # Clean up human temp bed
        if os.path.exists(human_bed_path):
            os.remove(human_bed_path)


if __name__ == "__main__":
    main()