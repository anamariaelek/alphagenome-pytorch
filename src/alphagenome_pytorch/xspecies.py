"""Cross-species splice-usage comparison helpers.

Shared by the ``splice_cross_species_alignment`` (developmental-trajectory divergence)
and ``splice_cross_species_usage`` (usage-level divergence) notebooks. Everything here is
parameterised (paths / tissue / thresholds passed in) so it carries no notebook state.
"""

import os
import json
import bisect
import subprocess

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from alphagenome_pytorch.clustering import classify_cluster_shape

TPS = list(range(1, 16))  # developmental timepoints 1..15

DYNAMIC_SHAPES = {"up_early", "up_late", "up_mid",
                  "down_early", "down_late", "down_mid", "up-down", "down-up"}
FLAT_SHAPES = {"flat_low", "flat_mid_low", "flat_mid", "flat_mid_high", "flat_high"}


# ── Site ids (chrom:pos:strand) ───────────────────────────────────────────────────

def norm_strand(s):
    """Normalise a strand value to '+'/'-' or '?' (unknown)."""
    return s if s in ("+", "-") else "?"


def make_site(chrom, pos, strand="?"):
    """Build a canonical site id ``chrom:pos:strand`` ('?' if strand unknown)."""
    return f"{chrom}:{int(pos)}:{norm_strand(strand)}"


def parse_site(site):
    """Parse a site id into ``(chrom, pos:int, strand)``. Tolerates the legacy
    ``chrom:pos`` form (strand -> '?') as well as ``chrom:pos:strand``."""
    parts = str(site).split(":")
    chrom = parts[0]
    pos = int(parts[1]) if len(parts) > 1 and parts[1] != "" else None
    strand = norm_strand(parts[2]) if len(parts) > 2 else "?"
    return chrom, pos, strand


def _site_series(chrom, pos, strand=None):
    """Vectorised ``chrom:pos:strand`` site ids (strand '?' where None/unknown)."""
    ch = pd.Series(np.asarray(chrom)).astype(str).reset_index(drop=True)
    po = pd.Series(np.asarray(pos)).astype(int).astype(str).reset_index(drop=True)
    if strand is None:
        st = pd.Series(["?"] * len(ch))
    else:
        st = pd.Series(np.asarray(strand)).reset_index(drop=True)
        st = st.where(st.isin(["+", "-"]), "?")
    return (ch + ":" + po + ":" + st).to_numpy()


def add_site_col(df):
    """Add a ``site`` column (chrom:pos:strand) using ``Strand`` if present, else '?'."""
    strand = df["Strand"] if "Strand" in df.columns else None
    df["site"] = _site_series(df["Chromosome"], df["Position"], strand)
    return df


_add_site_col = add_site_col  # internal alias


_STRAND_CACHE = {}

def strand_series(usage_tmpl, sp):
    """Cached (Chromosome, Position) -> Strand mapping from a species' combined usage data."""
    key = (usage_tmpl, sp)
    if key not in _STRAND_CACHE:
        u = pd.read_parquet(usage_tmpl.format(species=sp),
                            columns=["Chromosome", "Position", "Strand"]).drop_duplicates()
        u["Chromosome"] = u["Chromosome"].astype(str)
        _STRAND_CACHE[key] = u.groupby(["Chromosome", "Position"])["Strand"].first()
    return _STRAND_CACHE[key]


def attach_strand(df, usage_tmpl, sp):
    """Add a ``Strand`` column to a prediction frame by joining the species' usage data
    on (Chromosome, Position). Sites with no usage match get '?'."""
    if df is None:
        return df
    s = strand_series(usage_tmpl, sp)
    df = df.copy()
    df["Chromosome"] = df["Chromosome"].astype(str)
    keys = list(zip(df["Chromosome"], df["Position"].astype(int)))
    df["Strand"] = pd.Series([norm_strand(s.get(k, "?")) for k in keys], index=df.index)
    return df


def ensure_site_strand(site, strand_ser):
    """Return ``site`` with a strand suffix, filling it from ``strand_ser`` (a
    (chrom,pos)->strand mapping) when the id lacks one. Idempotent."""
    chrom, pos, strand = parse_site(site)
    if strand == "?" and pos is not None:
        strand = norm_strand(strand_ser.get((chrom, pos), "?"))
    return make_site(chrom, pos, strand)


# ── Paths ───────────────────────────────────────────────────────────────────────

def pred_path(pred_dir, sp):
    return os.path.join(pred_dir, sp, f"usage_{sp}.parquet")

def pred_npz(pred_dir, sp):
    return os.path.join(pred_dir, sp, f"usage_{sp}.npz")


# ── Usage / prediction loading ──────────────────────────────────────────────────

def load_focus_usage(usage_tmpl, sp, tissue, positions=None):
    """True usage for one species/tissue (optionally only given Positions)."""
    filt = [("Tissue", "==", tissue)]
    if positions is not None:
        filt.append(("Position", "in", list({int(p) for p in positions})))
    return pd.read_parquet(
        usage_tmpl.format(species=sp),
        columns=["Chromosome", "Position", "Strand", "Timepoint", "SSE", "Reads"],
        filters=filt,
    )


def load_predictions(pred_dir, data_dir, species_sci, sp, tissue, positions=None):
    """Per-(site, timepoint) true & predicted usage for a species/tissue.

    Prefers the tidy ``usage_{sp}.parquet``; falls back to ``usage_{sp}.npz`` +
    ``usage.json`` condition labels. Returns None if no predictions exist.
    """
    ppath = pred_path(pred_dir, sp)
    if os.path.exists(ppath):
        filt = [("Tissue", "==", tissue)]
        if positions is not None:
            filt.append(("Position", "in", list({int(x) for x in positions})))
        try:
            p = pd.read_parquet(ppath, filters=filt)
        except Exception:
            p = pd.read_parquet(ppath)
            if "Tissue" in p.columns:
                p = p[p["Tissue"] == tissue]
            if positions is not None:
                p = p[p["Position"].isin({int(x) for x in positions})]
        p = p.rename(columns={"SSE_true": "true", "SSE_pred": "pred"})
        p["Chromosome"] = p["Chromosome"].astype(str)
        return p[["Chromosome", "Position", "Timepoint", "true", "pred"]]

    npz = pred_npz(pred_dir, sp)
    if not os.path.exists(npz):
        return None
    meta = json.load(open(os.path.join(data_dir, species_sci[sp], "usage.json")))
    idx_to_label = {int(v): k for k, v in meta["condition_labels"].items()}
    d = np.load(npz)
    cond = np.asarray(d["cond_ids"], int)
    tis  = np.array([idx_to_label[c].rsplit("_", 1)[0] for c in cond])
    tp   = np.array([int(idx_to_label[c].rsplit("_", 1)[1]) for c in cond])
    cp   = np.asarray(d["chr_pos"]).astype(str)
    chrom = np.array([s.split(":")[0] for s in cp]); pos = np.array([int(s.split(":")[1]) for s in cp])
    keep = tis == tissue
    if positions is not None:
        keep = keep & np.isin(pos, list({int(x) for x in positions}))
    return pd.DataFrame({"Chromosome": chrom[keep], "Position": pos[keep], "Timepoint": tp[keep],
                         "true": np.asarray(d["true"], float)[keep],
                         "pred": np.asarray(d["pred"], float)[keep]})


def load_all_tissue_preds(pred_dir, data_dir, species_sci, sp, positions):
    """All-tissue predictions at given positions, laid out for
    ``plot_splice_site_predictions`` (Species/Chromosome/Position/Tissue/Timepoint/
    SSE_true/SSE_pred). Returns None if the species has no predictions."""
    posset = {int(x) for x in positions}
    ppath = pred_path(pred_dir, sp)
    if os.path.exists(ppath):
        try:
            df = pd.read_parquet(ppath, filters=[("Position", "in", list(posset))])
        except Exception:
            df = pd.read_parquet(ppath); df = df[df["Position"].isin(posset)]
        df = df.rename(columns={"true": "SSE_true", "pred": "SSE_pred"})
        if "Species" not in df.columns:
            df["Species"] = sp
        df["Chromosome"] = df["Chromosome"].astype(str)
        return df
    npz = pred_npz(pred_dir, sp)
    if not os.path.exists(npz):
        return None
    meta = json.load(open(os.path.join(data_dir, species_sci[sp], "usage.json")))
    idx_to_label = {int(v): k for k, v in meta["condition_labels"].items()}
    d = np.load(npz)
    cond = np.asarray(d["cond_ids"], int)
    cp = np.asarray(d["chr_pos"]).astype(str)
    ch = np.array([s.split(":")[0] for s in cp]); po = np.array([int(s.split(":")[1]) for s in cp])
    keep = np.isin(po, list(posset))
    tis = np.array([idx_to_label[c].rsplit("_", 1)[0] for c in cond])
    tp  = np.array([int(idx_to_label[c].rsplit("_", 1)[1]) for c in cond])
    return pd.DataFrame({"Species": sp, "Chromosome": ch[keep], "Position": po[keep],
                         "Tissue": tis[keep], "Timepoint": tp[keep],
                         "SSE_true": np.asarray(d["true"], float)[keep],
                         "SSE_pred": np.asarray(d["pred"], float)[keep]})


# ── Per-site metrics ────────────────────────────────────────────────────────────

def trajectory_stats(df, tps=TPS):
    """Per-site trajectory table + wide per-timepoint mean-SSE matrix (keyed chrom:pos)."""
    df = df.copy()
    df["Chromosome"] = df["Chromosome"].astype(str)
    strand = df.groupby(["Chromosome", "Position"])["Strand"].first()
    agg = (df.groupby(["Chromosome", "Position", "Timepoint"])
             .agg(SSE=("SSE", "mean"), Reads=("Reads", "sum")).reset_index())
    agg.loc[agg["Reads"] <= 0, "SSE"] = np.nan
    wide = (agg.pivot_table(index=["Chromosome", "Position"], columns="Timepoint", values="SSE")
               .reindex(columns=tps))
    W = wide.to_numpy(); idx = wide.index
    with np.errstate(invalid="ignore"):
        amp = np.nanmax(W, axis=1) - np.nanmin(W, axis=1)
        mean_sse = np.nanmean(W, axis=1)
    stats = pd.DataFrame({"Chromosome": idx.get_level_values("Chromosome"),
                          "Position":   idx.get_level_values("Position"),
                          "n_obs": np.isfinite(W).sum(axis=1),
                          "amplitude": amp, "mean_sse": mean_sse})
    stats["Strand"] = strand.reindex(idx).to_numpy()
    stats["site"] = _site_series(stats["Chromosome"], stats["Position"], stats["Strand"])
    wide.index = stats["site"].to_numpy()
    return stats, wide


def pred_site_metrics(pred_df):
    """Per-site prediction metrics: Pearson r, RMSE, mean/amplitude of true & predicted."""
    pred_df = pred_df.copy()
    pred_df["Chromosome"] = pred_df["Chromosome"].astype(str)
    pred_df = _add_site_col(pred_df)

    def _m(g):
        t = g["true"].to_numpy(float); p = g["pred"].to_numpy(float)
        r = (np.corrcoef(t, p)[0, 1] if len(t) > 2 and t.std() > 1e-9 and p.std() > 1e-9 else np.nan)
        return pd.Series({"pred_r": r,
                          "pred_rmse": float(np.sqrt(np.mean((t - p) ** 2))),
                          "mean_true": float(np.mean(t)), "mean_pred": float(np.mean(p)),
                          "pred_amplitude": float(p.max() - p.min()),
                          "n_obs_pred": int((t > 0).sum())})

    return pred_df.groupby("site")[["true", "pred"]].apply(_m)


def site_usage_metrics(pred_df):
    """Per-site usage summary treating **each condition independently** (no trajectory).

    Unlike :func:`pred_site_metrics` (which builds a per-site trajectory and its Pearson r
    across timepoints), this pools the conditions of a site as independent observations:
    mean true/pred usage and the per-condition error (RMSE, MAE). Use this for usage-*level*
    comparisons where the timepoints are not treated as an ordered trajectory.
    """
    df = pred_df.copy()
    df["Chromosome"] = df["Chromosome"].astype(str)
    df = _add_site_col(df)
    df["_se"] = (df["true"] - df["pred"]) ** 2
    df["_ae"] = (df["true"] - df["pred"]).abs()
    out = df.groupby("site").agg(
        mean_true=("true", "mean"), mean_pred=("pred", "mean"),
        mse=("_se", "mean"), mae=("_ae", "mean"), n_obs=("true", "size"))
    out["rmse"] = np.sqrt(out["mse"])
    return out.drop(columns="mse").reset_index()


def usage_wide(pred_df, tps=TPS):
    """Per-(site x timepoint) true and predicted usage matrices (keyed chrom:pos).

    Returns (true_wide, pred_wide): DataFrames indexed by site, columns = ``tps``, so each
    timepoint (condition) can be compared independently — no averaging across timepoints.
    """
    df = pred_df.copy()
    df["Chromosome"] = df["Chromosome"].astype(str)
    df = _add_site_col(df)
    tw = df.pivot_table(index="site", columns="Timepoint", values="true", aggfunc="mean").reindex(columns=tps)
    pw = df.pivot_table(index="site", columns="Timepoint", values="pred", aggfunc="mean").reindex(columns=tps)
    return tw, pw


def classify_traj(vec, min_timepoints=5, tps=TPS):
    """Shape label for a per-timepoint SSE vector (gaps linearly interpolated first)."""
    v = np.asarray(vec, float); m = np.isfinite(v)
    if m.sum() < min_timepoints:
        return None
    x = np.arange(len(v))
    filled = interp1d(x[m], v[m], kind="linear", bounds_error=False,
                      fill_value=(v[m][0], v[m][-1]))(x)
    return classify_cluster_shape(filled)


def is_dynamic_shape(s):
    return bool(s) and not s.startswith("flat") and s != "complex"


def role(shape, amplitude, dynamic_shapes=DYNAMIC_SHAPES, flat_shapes=FLAT_SHAPES,
         amp_dynamic=0.40, amp_flat=0.10):
    """'dynamic' (up/down shape, amp>=amp_dynamic), 'flat' (flat shape, amp<=amp_flat), None."""
    if shape in flat_shapes and amplitude <= amp_flat:
        return "flat"
    if shape in dynamic_shapes and amplitude >= amp_dynamic:
        return "dynamic"
    return None


def xcorr(a, b):
    """Pearson r between two timepoint vectors on their shared observed timepoints."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.nanstd(a[m]) < 1e-9 or np.nanstd(b[m]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def is_well_predicted(pred_r, pred_rmse, true_dynamic, r_well=0.70, mae_well=0.20):
    """Low error always; plus shape correlation when the true trajectory actually varies."""
    if not np.isfinite(pred_rmse) or pred_rmse > mae_well:
        return False
    if true_dynamic:
        return np.isfinite(pred_r) and pred_r >= r_well
    return True


def nearest(sorted_pos, pos, tol):
    """Nearest value in sorted_pos to pos within +/- tol (or None)."""
    sorted_pos = np.asarray(sorted_pos)
    if len(sorted_pos) == 0:
        return None
    j = bisect.bisect_left(sorted_pos, pos)
    cand = [int(sorted_pos[k]) for k in (j - 1, j) if 0 <= k < len(sorted_pos)]
    if not cand:
        return None
    best = min(cand, key=lambda x: abs(x - pos))
    return int(best) if abs(best - pos) <= tol else None


# ── Multiz MAF whole-genome-alignment lift-over ─────────────────────────────────

def maf_liftover(maf_path, ref_prefix, positions, assemblies, progress_every=500_000):
    """Lift ref positions to other assemblies via a multiz MAF.

    ``ref_prefix`` e.g. ``"hg38.chr8"`` (the reference is +strand). Returns
    ``{ref_pos: {assembly: (chrom, pos, strand)}}`` — chrom has no ``chr`` prefix, pos
    is 0-based +strand genomic. Reads the (gzipped) MAF through ``zcat`` for speed.
    """
    pos_sorted = np.array(sorted(int(p) for p in positions))
    res = {int(p): {} for p in positions}
    want = set(assemblies)
    block, n_blocks = [], 0

    def process(block):
        ref = next((l.split() for l in block if l.startswith("s " + ref_prefix + " ")), None)
        if ref is None:
            return
        start, size, text = int(ref[2]), int(ref[3]), ref[6]
        lo = bisect.bisect_left(pos_sorted, start)
        hi = bisect.bisect_left(pos_sorted, start + size)
        need = set(int(x) for x in pos_sorted[lo:hi])
        if not need:
            return
        col_for, cur = {}, start
        for ci, ch in enumerate(text):
            if ch != "-":
                if cur in need:
                    col_for[cur] = ci
                cur += 1
        needed_cols = set(col_for.values())
        for l in block:
            p = l.split()
            asm = p[1].split(".", 1)[0]
            if asm not in want:
                continue
            tchrom = p[1].split(".", 1)[1].removeprefix("chr")
            tstart, tstrand, tsrc, ttext = int(p[2]), p[4], int(p[5]), p[6]
            cnt, col2 = 0, {}
            for ci, ch in enumerate(ttext):
                if ci in needed_cols and ch != "-":
                    g = tstart + cnt if tstrand == "+" else tsrc - (tstart + cnt) - 1
                    col2[ci] = (tchrom, g, tstrand)
                if ch != "-":
                    cnt += 1
            for rp, ci in col_for.items():
                if ci in col2:
                    res[rp][asm] = col2[ci]

    proc = subprocess.Popen(["bash", "-c", f"zcat {maf_path}"], stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if line.startswith("a"):
            process(block); block = []
            n_blocks += 1
            if progress_every and n_blocks % progress_every == 0:
                print(f"  ...{n_blocks:,} alignment blocks", flush=True)
        elif line.startswith("s "):
            block.append(line)
        elif not line.strip():
            process(block); block = []
    process(block)
    proc.stdout.close(); proc.wait()
    return res


def liftover_to_frame(res, assembly_to_species, ref_chrom, ref_strand_by_pos=None):
    """Convert a maf_liftover result dict to a tidy DataFrame:
    human_site / species / assembly / aln_chrom / aln_pos / aln_strand.

    ``ref_strand_by_pos`` (optional): a ``{ref_pos: strand}`` mapping used to build the
    reference ``human_site`` id as ``chrom:pos:strand`` (defaults to '?')."""
    smap = ref_strand_by_pos or {}
    rows = []
    for hpos, d in res.items():
        hsite = make_site(ref_chrom, hpos, smap.get(int(hpos), "?"))
        for asm, (tchrom, tpos, tstrand) in d.items():
            rows.append(dict(human_site=hsite,
                             species=assembly_to_species[asm], assembly=asm,
                             aln_chrom=str(tchrom), aln_pos=int(tpos), aln_strand=tstrand))
    return pd.DataFrame(rows)


def chain_lift(lift_df, species, chain_path, new_assembly_label=None):
    """Lift a species' aligned coordinates through a UCSC chain (e.g. rn6->rn5).

    Modifies rows where ``lift_df.species == species`` in place-ish (returns a new
    frame). Drops rows that don't lift. Needs ``pyliftover``.
    """
    from pyliftover import LiftOver
    lo = LiftOver(chain_path)
    mask = lift_df["species"] == species
    if not mask.any():
        return lift_df
    n_before = int(mask.sum())
    new_chrom, new_pos = [], []
    for _, r in lift_df[mask].iterrows():
        conv = lo.convert_coordinate("chr" + str(r["aln_chrom"]), int(r["aln_pos"]))
        if conv:
            new_chrom.append(conv[0][0].removeprefix("chr")); new_pos.append(int(conv[0][1]))
        else:
            new_chrom.append(None); new_pos.append(np.nan)
    lift_df = lift_df.copy()
    lift_df.loc[mask, "aln_chrom"] = new_chrom
    lift_df.loc[mask, "aln_pos"]   = new_pos
    if new_assembly_label:
        lift_df.loc[mask, "assembly"] = new_assembly_label
    lift_df = lift_df[lift_df["aln_pos"].notna()].copy()
    lift_df["aln_pos"] = lift_df["aln_pos"].astype(int)
    n_after = int((lift_df["species"] == species).sum())
    print(f"{species} chain lift: {n_after:,}/{n_before:,}")
    return lift_df
