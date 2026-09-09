#!/usr/bin/env python
"""Lift human clustered test sites onto other assemblies through the multiz100way MAF.

One chromosome per invocation, so the genome can be run in parallel. Writes a
durable per-chromosome cache under the prediction store (not a job workdir, so it
survives and can be reused):

    <out>/liftover_chr{C}.parquet   human_site species assembly aln_chrom aln_pos
                                    aln_strand ref_chrom
    <out>/status_chr{C}.csv         one row: sites, rows per species, zcat exit

The site universe is the union over tissues of the per-species clustered test
sites, i.e. exactly the sites the prediction store has trajectories for.

Assemblies extracted (MAF label -> species): mm10 mouse, rn6 rat, oryCun2 rabbit,
monDom5 opossum, galGal4 chicken, rheMac3 macaque. Chicken predictions are on
Galgal4 so galGal4 needs no chain; rat needs rn6->rn5 and macaque rheMac3->rheMac10
downstream (or, better for macaque, a direct hg38->rheMac10 chain lift instead of
these MAF rows).

A truncated MAF is detected through zcat's exit status and recorded in the status
row; the parquet is still written but must not be treated as complete.

    python build_liftover_gw.py --chrom 7 [--maf-dir D] [--pred-dir D] [--out D]
"""
import argparse
import os
import subprocess
import sys
import time

import pandas as pd

CODE = "/home/elek/sds/sd17d003/Anamaria/alphagenome_genomicsxai_code/src"
sys.path.insert(0, CODE)
from alphagenome_pytorch import xspecies as sx  # noqa: E402

B = "/home/elek/sds/sd17d003/Anamaria"
AG = f"{B}/alphagenome_genomicsxai"
MAF_DIR = f"{B}/genomes/multiz100way"
PRED = f"{AG}/best_model/preds_intersect_protein_coding"
OUT = f"{AG}/best_model/cross_species_alignment"
TISSUES = ["Brain", "Cerebellum", "Heart", "Kidney", "Liver", "Testis", "Ovary"]
ASM2SP = {"mm10": "mouse", "rn6": "rat", "oryCun2": "rabbit",
          "monDom5": "opossum", "galGal4": "chicken", "rheMac3": "macaque"}


def chrom_length(chrom, fai=f"{B}/genomes/mazin/fasta/Homo_sapiens.fa.fai"):
    for l in open(fai):
        f = l.split("\t")
        if f[0] == str(chrom):
            return int(f[1])
    raise SystemExit("chr%s not in %s" % (chrom, fai))


def human_sites(pred_dir):
    """Union over tissues of human clustered test sites."""
    fr = []
    for t in TISSUES:
        p = (f"{pred_dir}/human/pred_gp_splice_usage/{t}/"
             f"human_{t}_test_prediction_clusters.parquet")
        if os.path.exists(p):
            fr.append(pd.read_parquet(p, columns=["Chromosome", "Position", "Strand"]))
    if not fr:
        raise SystemExit("no human cluster tables under %s" % pred_dir)
    d = pd.concat(fr, ignore_index=True)
    d["Chromosome"] = d["Chromosome"].astype(str)
    return d.drop_duplicates(["Chromosome", "Position"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chrom", required=True)
    p.add_argument("--maf-dir", default=MAF_DIR)
    p.add_argument("--pred-dir", default=PRED)
    p.add_argument("--out", default=OUT)
    p.add_argument("--min-bytes-per-mb", type=float, default=10e6,
                   help="completeness floor; complete files sit near 25-30 MB per Mb "
                        "of chromosome, a failed download far below it")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    maf = f"{a.maf_dir}/chr{a.chrom}.maf.gz"
    if not os.path.exists(maf):
        raise SystemExit("no MAF for chr%s" % a.chrom)

    # ── validate the MAF before spending a scan on it ───────────────────────
    # The directory has been observed mid-replacement (an hg19 multiz100way copy
    # being overwritten by the hg38 one) and with a truncated download, either of
    # which yields a silently partial map. Three cheap guards, all recorded.
    ref_label = subprocess.run(
        "zcat %s 2>/dev/null | head -3 | sed -n '3p' | awk '{print $2}' | cut -d. -f1"
        % maf, shell=True, capture_output=True, text=True).stdout.strip()
    if ref_label != "hg38":
        raise SystemExit("chr%s: MAF reference is %r, expected hg38 — refusing to scan"
                         % (a.chrom, ref_label))
    size0, mtime0 = os.path.getsize(maf), os.path.getmtime(maf)
    bytes_per_mb = size0 / (chrom_length(a.chrom) / 1e6)
    if bytes_per_mb < a.min_bytes_per_mb:
        raise SystemExit(
            "chr%s: %.1f MAF bytes per Mb of chromosome, below the %.0f floor "
            "(incomplete download) — refusing to scan"
            % (a.chrom, bytes_per_mb, a.min_bytes_per_mb))

    u = human_sites(a.pred_dir)
    u = u[u["Chromosome"] == str(a.chrom)]
    if not len(u):
        print("chr%s: no human sites, nothing to do" % a.chrom, flush=True)
        return
    strand_by_pos = dict(zip(u["Position"].astype(int), u["Strand"]))
    print("chr%s: %d human sites" % (a.chrom, len(u)), flush=True)

    t0 = time.time()
    res = sx.maf_liftover(maf, f"hg38.chr{a.chrom}", list(strand_by_pos),
                          list(ASM2SP), progress_every=2_000_000)
    # maf_liftover streams through zcat; a truncated file leaves a nonzero status
    # that the helper does not raise on, so re-test the file cheaply here.
    stable = (os.path.getsize(maf) == size0) and (os.path.getmtime(maf) == mtime0)
    df = sx.liftover_to_frame(res, ASM2SP, str(a.chrom), ref_strand_by_pos=strand_by_pos)
    if df.empty:
        df = pd.DataFrame(columns=["human_site", "species", "assembly", "aln_chrom",
                                   "aln_pos", "aln_strand"])
    df["ref_chrom"] = str(a.chrom)
    df.to_parquet(f"{a.out}/liftover_chr{a.chrom}.parquet", index=False)

    per = df["species"].value_counts().to_dict()
    row = dict(chrom=str(a.chrom), n_human_sites=len(u),
               n_sites_lifted=df["human_site"].nunique(), n_rows=len(df),
               ref_label=ref_label, maf_bytes_per_mb=round(bytes_per_mb, 1),
               file_stable=int(stable), minutes=round((time.time() - t0) / 60, 1))
    row.update({f"n_{s}": per.get(s, 0) for s in sorted(set(ASM2SP.values()))})
    pd.DataFrame([row]).to_csv(f"{a.out}/status_chr{a.chrom}.csv", index=False)
    print("chr%s done: %s" % (a.chrom, row), flush=True)


if __name__ == "__main__":
    main()
