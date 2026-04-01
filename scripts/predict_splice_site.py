#!/usr/bin/env python
"""
Predict splice site classification for a genomic region using AlphaGenome-PyTorch.

Usage:
    python scripts/predict_splice_site.py \
        --coords 1:1000:1500 \
        --genome /path/to/genome.fa \
        --annotation /path/to/annotation.parquet \
        --checkpoint /path/to/model.pth \
        [--organism 0] [--device cuda]

Outputs classification probabilities for each position in the region.
"""
from sklearn.metrics import average_precision_score
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.utils.sequence import sequence_to_onehot

import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Splice site classification prediction")
    p.add_argument("--coords", required=True, help="Genomic coordinates (chrom:start-end or chrom:start:end) or path to BED file with chr, start, end columns")
    p.add_argument("--genome", required=True, help="Reference genome FASTA file")
    p.add_argument("--annotation", required=True, help="Parquet annotation file with splice site labels")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint (.pth)")
    p.add_argument("--organism", type=int, default=0, help="Organism index (default: 0 for human)")
    p.add_argument("--output", required=True, help="Output file path for predictions (TSV)")
    p.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    return p.parse_args()

def main():
    args = parse_args()
    import os

    all_probs = []
    all_true_classes = []
    all_pred_classes = []
    all_positions = []
    all_chroms = []

    SPLICE_CLASS_NAMES = ["Donor+", "Acceptor+", "Donor-", "Acceptor-"]
    BACKGROUND_CLASS = 4
    CLASS_LABELS = {0: 'donor +', 1: 'acceptor +', 2: 'donor -', 3: 'acceptor -', 4: 'no splice site'}
    CLASS_COLORS  = {0: '#ff7f00', 1: '#33a02c', 2: '#fdbf6f', 3: '#b2df8a', 4: '#1f78b4'}
    annotation = pd.read_parquet(args.annotation)

    region_auprcs = []
    region_ntrues = []
    region_npreds = []
    region_coords = []
    if os.path.isfile(args.coords):
        with open(args.coords) as bed:
            regions = [line.strip().split() for line in bed if line.strip() and not line.startswith('#')]
        for idx, (chrom, start, end, *_) in enumerate(regions):
            start, end = int(start), int(end)
            print(f"Processing region: {chrom}:{start}-{end}")
            probs, true_classes, pred_classes, positions, chroms = process_region(chrom, start, end, args, annotation, SPLICE_CLASS_NAMES, BACKGROUND_CLASS)
            all_probs.append(probs)
            all_true_classes.append(true_classes)
            all_pred_classes.append(pred_classes)
            all_positions.append(positions)
            all_chroms.append(chroms)
            # Calculate AUPRC for this region
            region_auprc = {}
            region_ntrue = {}
            region_npred = {}
            for cls in range(4):
                true_labels = (true_classes == cls).astype(int)
                pred_labels = (pred_classes == cls).astype(int)
                cls_probs = probs[:, cls]
                aupr = average_precision_score(true_labels, cls_probs)
                region_auprc[cls] = aupr
                region_ntrue[cls] = int(true_labels.sum())
                region_npred[cls] = int(pred_labels.sum())
            region_auprcs.append(region_auprc)
            region_ntrues.append(region_ntrue)
            region_npreds.append(region_npred)
            region_coords.append(f"{chrom}:{start}-{end}")
    else:
        if ":" in args.coords and "-" in args.coords:
            chrom, rest = args.coords.split(":", 1)
            start, end = rest.split("-")
        elif ":" in args.coords:
            chrom, start, end = args.coords.split(":")
        else:
            raise ValueError("Coordinates must be in 'chrom:start-end' or 'chrom:start:end' format or a BED file")
        start, end = int(start), int(end)
        probs, true_classes, pred_classes, positions, chroms = process_region(chrom, start, end, args, annotation, SPLICE_CLASS_NAMES, BACKGROUND_CLASS)
        all_probs.append(probs)
        all_true_classes.append(true_classes)
        all_pred_classes.append(pred_classes)
        all_positions.append(positions)
        all_chroms.append(chroms)
        region_auprc = {}
        region_ntrue = {}
        region_npred = {}
        for cls in range(4):
            true_labels = (true_classes == cls).astype(int)
            pred_labels = (pred_classes == cls).astype(int)
            cls_probs = probs[:, cls]
            aupr = average_precision_score(true_labels, cls_probs)
            region_auprc[cls] = aupr
            region_ntrue[cls] = int(true_labels.sum())
            region_npred[cls] = int(pred_labels.sum())
        region_auprcs.append(region_auprc)
        region_ntrues.append(region_ntrue)
        region_npreds.append(region_npred)
        region_coords.append(f"{chrom}:{start}-{end}")

    # Concatenate all results
    all_probs = np.concatenate(all_probs, axis=0)
    all_true_classes = np.concatenate(all_true_classes, axis=0)
    all_pred_classes = np.concatenate(all_pred_classes, axis=0)
    all_positions = np.concatenate(all_positions, axis=0)
    all_chroms = np.concatenate(all_chroms, axis=0)

    # Write only positions where either predicted or true class is not background 
    import gzip
    output_gz = args.output if args.output.endswith('.gz') else args.output + '.gz'
    with gzip.open(output_gz, "wt") as f:
        f.write("#chrom\tpos\tDonor+\tAcceptor+\tDonor-\tAcceptor-\tNotSplice\ttrue_class\tpred_class\n")
        for i in range(len(all_probs)):
            true_cls = all_true_classes[i]
            pred_cls = all_pred_classes[i]
            # Only save if either predicted or true class is not background
            if true_cls != BACKGROUND_CLASS or pred_cls != BACKGROUND_CLASS:
                p = all_probs[i]
                f.write(f"{all_chroms[i]}\t{all_positions[i]}\t" + "\t".join(f"{x:.4f}" for x in p) + f"\t{true_cls}\t{pred_cls}\n")
    print(f"Filtered predictions written to {output_gz}")

    # Save per-region AUPRCs, n_true, and n_pred counts
    auprc_path = args.output.replace('.tsv', '.auprc.tsv')
    with open(auprc_path, "w") as f:
        f.write("region\tDonor+\tAcceptor+\tDonor-\tAcceptor-\tDonor+_n_true\tAcceptor+_n_true\tDonor-_n_true\tAcceptor-_n_true\tDonor+_n_pred\tAcceptor+_n_pred\tDonor-_n_pred\tAcceptor-_n_pred\n")
        for region, auprc, ntrue, npred in zip(region_coords, region_auprcs, region_ntrues, region_npreds):
            f.write(region + "\t" + "\t".join(f"{auprc[cls]:.4f}" for cls in range(4)) + "\t" + "\t".join(str(ntrue[cls]) for cls in range(4)) + "\t" + "\t".join(str(npred[cls]) for cls in range(4)) + "\n")

    # Calculate total AUPRC for each class (except background) and plot PR curves
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    plt.figure(figsize=(7, 7))
    for cls in range(4):
        true_labels = (all_true_classes == cls).astype(int)
        cls_probs = all_probs[:, cls]
        aupr = average_precision_score(true_labels, cls_probs)
        precision, recall, _ = precision_recall_curve(true_labels, cls_probs)
        color = CLASS_COLORS[cls] if 'CLASS_COLORS' in locals() or 'CLASS_COLORS' in globals() else None
        plt.plot(recall, precision, label=f"{CLASS_LABELS[cls]} (AUPRC={aupr:.3f})", color=color)
        print(f"Total AUPR for class {cls} ({CLASS_LABELS[cls]}): {aupr:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Total)")
    plt.legend()
    prc_png = args.output.replace('.tsv', '_prc.png').replace('.gz', '')
    plt.tight_layout()
    plt.savefig(prc_png)
    print(f"Precision-recall curve saved to {prc_png}")

def process_region(chrom, start, end, args, annotation, SPLICE_CLASS_NAMES=["Donor+", "Acceptor+", "Donor-", "Acceptor-"], BACKGROUND_CLASS=4):
    SEQ_LEN = 131072
    half = SEQ_LEN // 2
    region_center = (start + end) // 2
    win_start = region_center - half
    win_end = region_center + half

    # Load genome sequence
    try:
        import pyfaidx
        fasta = pyfaidx.Fasta(args.genome, as_raw=True, sequence_always_upper=True)
    except ImportError:
        raise ImportError("pyfaidx required: pip install pyfaidx")

    seq_str = str(fasta[chrom][win_start:win_end])

    # One-hot encode sequence
    seq_np = sequence_to_onehot(seq_str).astype(np.float32)  # (S, 4)
    if seq_np.shape[0] != SEQ_LEN:
        pad_len = SEQ_LEN - seq_np.shape[0]
        seq_np = np.pad(seq_np, ((0, pad_len), (0, 0)), mode="constant")
    seq_t = torch.from_numpy(seq_np).unsqueeze(0).to(args.device)
    org_t = torch.full((1,), args.organism, dtype=torch.long, device=args.device)

    # Load model (pretrained or fine-tuned, seamless)
    if not hasattr(process_region, "model"):
        from alphagenome_pytorch.utils.model_loading import load_model_for_inference
        model, _ = load_model_for_inference(args.checkpoint, args.device)
        process_region.model = model
    model = process_region.model

    # Predict splice site classification
    with torch.no_grad():
        out = model.predict(seq_t, org_t, resolutions=(1,))
        probs = out["splice_sites_classification"]["probs"][0].cpu().numpy()

    # --- Calculate true classes for this region ---
    true_pq = annotation[(annotation['Chromosome'] == chrom) & (annotation['Position'] >= win_start) & (annotation['Position'] < win_end)]
    true_pq['WindowPosition'] = true_pq['Position'] - win_start
    pred_classes = np.argmax(probs, axis=-1)
    true_classes = np.full_like(pred_classes, fill_value=BACKGROUND_CLASS)
    true_pq['TrueClass'] = true_pq['SiteType'].map({name: i for i, name in enumerate(SPLICE_CLASS_NAMES)})
    true_classes[true_pq['WindowPosition'].values] = true_pq['TrueClass'].values

    positions = np.arange(win_start, win_end)
    chroms = np.array([chrom] * (win_end - win_start))
    return probs, true_classes, pred_classes, positions, chroms

if __name__ == "__main__":
    main()
