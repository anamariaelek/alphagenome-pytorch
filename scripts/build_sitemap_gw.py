#!/usr/bin/env python
"""Build the genome-wide human-anchored ortholog site map.

Consumes the per-chromosome MAF lift-over caches written by build_liftover_gw.py
and snaps each aligned coordinate to the nearest clustered test site of the
partner species (tolerance 3 bp, as in the chr1-13 map it supersedes).

Assembly handling per species:
    mouse    mm10      MAF coordinate used directly
    rabbit   oryCun2   MAF coordinate used directly
    opossum  monDom5   MAF coordinate used directly
    chicken  galGal4   MAF coordinate used directly (predictions are on Galgal4)
    rat      rn6       chain-lifted rn6 -> rn5 (predictions are on Rnor_5.0)
    macaque  rheMac10  NOT from the MAF: the human site is chain-lifted directly
                       hg38 -> rheMac10, because predictions are on Mmul_10 while
                       the MAF carries rheMac3. The MAF rheMac3 rows are lifted
                       rheMac3 -> rheMac8 -> rheMac10 as an independent route and
                       the two are compared, reported as macaque route concordance.

Outputs (--out-dir):
    xspecies_site_map_genomewide.parquet   human_site species assembly aln_chrom
                                           aln_pos aln_strand aln_site aln_dist
    sitemap_coverage.csv                   per species x chromosome funnel
    macaque_route_concordance.csv          direct vs MAF+chain agreement

    python build_sitemap_gw.py [--lift-dir D] [--pred-dir D] [--out-dir D]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

CODE = "/home/elek/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code/src"
sys.path.insert(0, CODE)
from alphagenome_pytorch import xspecies as sx  # noqa: E402

B = "/home/elek/sds/sd17d003/Anamaria"
AG = f"{B}/alphagenome_genomicsxai"
PRED = f"{AG}/best_model/preds_intersect_protein_coding"
LIFT = f"{AG}/best_model/cross_species_alignment"
CHAINS = f"{B}/genomes/chains"
TISSUES = ["Brain", "Cerebellum", "Heart", "Kidney", "Liver", "Testis", "Ovary"]
SPECIES = ["mouse", "rat", "rabbit", "opossum", "macaque", "chicken"]
ALIGN_TOL = 3


def universe(pred_dir, sp):
    """Union over tissues of a species' clustered test sites."""
    fr = []
    for t in TISSUES:
        p = (f"{pred_dir}/{sp}/pred_gp_splice_usage/{t}/"
             f"{sp}_{t}_test_prediction_clusters.parquet")
        if os.path.exists(p):
            fr.append(pd.read_parquet(p, columns=["Chromosome", "Position", "Strand"]))
    if not fr:
        return None
    d = pd.concat(fr, ignore_index=True)
    d["Chromosome"] = d["Chromosome"].astype(str)
    u = d.drop_duplicates(["Chromosome", "Position"]).copy()
    u["site"] = sx._site_series(u["Chromosome"], u["Position"], u["Strand"])
    return u


def snap(sub, u, sp):
    """Snap aligned coordinates to that species' clustered sites."""
    pos_by = {ch: np.sort(g["Position"].to_numpy()) for ch, g in u.groupby("Chromosome")}
    site_by = {(r.Chromosome, int(r.Position)): r.site for r in u.itertuples()}
    rows = []
    for r in sub.itertuples():
        ch = str(r.aln_chrom)
        sn = sx.nearest(pos_by.get(ch, np.array([])), int(r.aln_pos), ALIGN_TOL)
        if sn is None:
            continue
        rows.append(dict(human_site=r.human_site, species=sp, assembly=r.assembly,
                         aln_chrom=ch, aln_pos=int(r.aln_pos), aln_strand=r.aln_strand,
                         aln_site=site_by[(ch, sn)], aln_dist=abs(sn - int(r.aln_pos))))
    return pd.DataFrame(rows)


def direct_macaque(hum, chain):
    """Chain-lift human sites hg38 -> rheMac10 (0-based, UCSC naming on input)."""
    from pyliftover import LiftOver
    lo = LiftOver(chain)
    rows = []
    for r in hum.itertuples():
        hit = lo.convert_coordinate("chr" + str(r.Chromosome), int(r.Position))
        if not hit:
            continue
        ch, pos, strand = hit[0][0], int(hit[0][1]), hit[0][2]
        rows.append(dict(human_site=r.site, species="macaque", assembly="rheMac10",
                         aln_chrom=str(ch).removeprefix("chr"), aln_pos=pos,
                         aln_strand=strand))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lift-dir", default=LIFT)
    p.add_argument("--pred-dir", default=PRED)
    p.add_argument("--out-dir", default=".")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    hum = universe(a.pred_dir, "human")
    uni = {sp: universe(a.pred_dir, sp) for sp in SPECIES}
    print("human clustered sites %d | partners %s" % (
        len(hum), {k: (0 if v is None else len(v)) for k, v in uni.items()}), flush=True)

    files = sorted(glob.glob(f"{a.lift_dir}/liftover_chr*.parquet"))
    lift = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    h_strand = hum.set_index(["Chromosome", "Position"])["Strand"]
    lift["human_site"] = [sx.ensure_site_strand(s, h_strand) for s in lift["human_site"]]
    print("lift cache: %d files, %d rows, %d human sites, species %s" % (
        len(files), len(lift), lift["human_site"].nunique(),
        lift["species"].value_counts().to_dict()), flush=True)

    # rat rn6 -> rn5
    lift = sx.chain_lift(lift, "rat", f"{CHAINS}/rn6ToRn5.over.chain.gz",
                         new_assembly_label="rn6->rn5")
    # macaque: MAF route rheMac3 -> rheMac8 -> rheMac10, kept for the concordance check
    maf_mac = lift[lift["species"] == "macaque"].copy()
    if len(maf_mac):
        maf_mac = sx.chain_lift(maf_mac, "macaque", f"{CHAINS}/rheMac3ToRheMac8.over.chain.gz",
                                new_assembly_label="rheMac8")
        maf_mac = sx.chain_lift(maf_mac, "macaque", f"{CHAINS}/rheMac8ToRheMac10.over.chain.gz",
                                new_assembly_label="rheMac3->rheMac10")
    # macaque primary: direct hg38 -> rheMac10 on the human sites themselves
    dir_mac = direct_macaque(hum, f"{CHAINS}/hg38ToRheMac10.over.chain.gz")
    print("macaque: MAF+chain route %d rows, direct route %d rows" % (
        len(maf_mac), len(dir_mac)), flush=True)

    lift = pd.concat([lift[lift["species"] != "macaque"], dir_mac], ignore_index=True)

    maps, cov = [], []
    for sp in SPECIES:
        sub = lift[lift["species"] == sp]
        u = uni[sp]
        if u is None or sub.empty:
            print("%s: no data, skipped" % sp, flush=True)
            continue
        mp = snap(sub, u, sp)
        maps.append(mp)
        for ch, g in sub.groupby(sub["human_site"].str.split(":").str[0]):
            g_snap = mp[mp["human_site"].isin(g["human_site"])]
            cov.append(dict(species=sp, ref_chrom=ch, n_aligned_rows=len(g),
                            n_aligned_sites=g["human_site"].nunique(),
                            n_snapped=len(g_snap),
                            n_snapped_sites=g_snap["human_site"].nunique()))
        print("%s: %d aligned -> %d snapped (%.1f%%)" % (
            sp, len(sub), len(mp), 100 * len(mp) / max(len(sub), 1)), flush=True)

    m = pd.concat(maps, ignore_index=True)
    m.to_parquet(f"{a.out_dir}/xspecies_site_map_genomewide.parquet", index=False)

    C = pd.DataFrame(cov)
    hum_by_chrom = hum["Chromosome"].value_counts().rename("n_human_sites")
    C = C.join(hum_by_chrom, on="ref_chrom")
    C.to_csv(f"{a.out_dir}/sitemap_coverage.csv", index=False)

    # macaque route concordance: same human site reachable by both routes?
    if len(maf_mac):
        mm = snap(maf_mac, uni["macaque"], "macaque")
        j = mm.merge(m[m["species"] == "macaque"], on="human_site",
                     suffixes=("_maf", "_direct"))
        conc = pd.DataFrame([dict(
            n_maf_route=mm["human_site"].nunique(),
            n_direct_route=int((m["species"] == "macaque").sum()),
            n_both=len(j),
            n_same_site=int((j["aln_site_maf"] == j["aln_site_direct"]).sum()),
            median_pos_diff=float(np.median(np.abs(
                j["aln_pos_maf"] - j["aln_pos_direct"]))) if len(j) else np.nan)])
        conc.to_csv(f"{a.out_dir}/macaque_route_concordance.csv", index=False)
        print("macaque route concordance:\n%s" % conc.to_string(index=False), flush=True)

    piv = m.pivot_table(index="human_site", columns="species", values="aln_site",
                        aggfunc="first").reindex(columns=SPECIES)
    print("\nMAP %s" % (m.shape,), flush=True)
    print("human sites with >=1 ortholog: %d / %d" % (len(piv), len(hum)), flush=True)
    print("partners per human site: %s"
          % piv.notna().sum(axis=1).value_counts().sort_index().to_dict(), flush=True)
    print(piv.notna().sum().to_string(), flush=True)
    print("aln_dist: %s" % m["aln_dist"].value_counts().sort_index().to_dict(), flush=True)


if __name__ == "__main__":
    main()
