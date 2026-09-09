# Evaluating predicted developmental splicing trajectories

> **Units.** The usage values throughout are **SSE** (splice site strength
> estimate) as computed by SpliSER — Dent et al. 2021,
> [doi:10.1093/nargab/lqab041](https://doi.org/10.1093/nargab/lqab041).
> They are compared against the PSI trajectories of Mazin et al. 2021 because
> both are bounded in [0, 1] and describe the same quantity in spirit, but they
> are not the same estimator and the two should never be pooled or subtracted.

Proposed end-to-end workflow, written after the devAS comparison and the GP
reconciliation. The organising principle: **every trajectory metric is reported
against an explicit null and against the noise ceiling, on one shared site set.**
Discrete pattern classes are demoted from scoring metric to descriptive summary.

Motivation for the rewrite, in one line each:

- per-class recall from a row-normalised concordance table tracks the predicted
  marginal, not skill (up-recall 0.83–0.87 in human brain against a 0.72 base rate);
- pooled correlation of mean usage (r = 0.70 human) is a
  constitutive-vs-alternative discrimination (AUC 0.89), and collapses to 0.16–0.24
  within any usage band;
- both effects inflate an evaluation of *trajectories* with information that is
  purely about the *level* of a site;
- and a correlation on centred trajectories is **scale-free** — a prediction with
  the correct shape at a third of the true amplitude scores r = 1.0 — so it is
  reported as the shape metric *alongside* RMSE in SSE units, never on its own.

Primary evaluation metrics: **Pearson r** for shape, **RMSE** for magnitude, both on the centred
trajectory, both per site–organ, both against the nulls and the noise ceiling.

---

## Stage 0 — Define one set of splice sites.

Gate: ≥ 60% of conditions covered, ≥ 4 conditions at intermediate usage, > 6
distinct covered conditions, per-condition depth ≥ `min_total`. Record `n` for
every reported number, and never compare a metric computed on one gate against
a metric computed on another.

Stratify sites by:

- **species** and **organ**;
- **observed mean-usage band** (`≤0.3`, `0.3–0.7`, `≥0.7`), because model
  discrimination differs several-fold between them;
- **coverage decile**, to separate model error from counting noise.

The intermediate band is the analysis set for anything downstream, including
attribution target selection. Near-constitutive sites stay in the tables as a
reported stratum but carry no conclusions.

## Stage 1 — Noise ceiling

Every metric below has an upper bound set by the data, not by the model. 
Every metric is reported as a fraction of its noise ceiling.

1. Split replicates within each condition into two halves; recompute the usage
   trajectory from each half.
2. Compute every metric of half A against half B, and vice versa.
3. Scale the halves back to the full data before using them as a ceiling. Each
   half carries twice the sampling variance of the full set, so:
    - RMSE:  `ceiling_RMSE = RMSE_AB  / 2`;  
    - correlation: `R = 2 r_AB / (1 + r_AB); ceiling_r = sqrt(R)`.  
  This is easy to get wrong for correlation, and getting it wrong makes the 
  ceiling look lower than the model. `r_AB` is the reliability of *one half*; 
  Spearman–Brown lifts it to the reliability of the pooled series, and the 
  attenuation bound on a correlation measured *against* a series of reliability 
  `R` is the **square root** of `R`.

   Worked example (human brain, sites with mean SSE ≥ 0.7): `r_AB` = 0.13, so
   `R` = 0.23 and `ceiling_r` = 0.48, against a model `r` of 0.42. Comparing the
   model to 0.13, or to 0.23, would have wrongly said it had exceeded the
   ceiling.
4. Report `metric / ceiling_metric`. A model correlation of 0.30 against a ceiling 
   of 0.45 is a very different result from 0.30 against a ceiling of 0.95.
5. The ceiling is empirical, so it absorbs biological replicate variability and
   not only counting noise — at high-usage sites `r_AB` is far below what
   binomial sampling alone would give. That is the right behaviour for a
   ceiling, but it means `ceiling_r` is a statement about *this* replicate
   structure, not an intrinsic property of the locus.

Where replicate counts are too low, substitute the quasi-binomial dispersion
already estimated in `devas_glm.py`: the expected attenuation of a correlation
under known binomial noise is analytic and gives a usable bound.

## Stage 2 — Three nulls, computed for every metric

Report every model number beside these. A model that does not beat null 2 has
not learned anything developmental.

| null | definition | what it isolates |
|---|---|---|
| **N1 flat, oracle** | predict each site's own *observed* mean at every timepoint | trajectory information over and above the level |
| **N3 organ mean trajectory** | predict the organ's average centred trajectory for every site | site-specific dynamics over and above a global developmental trend |
| **N2 flat, realisable** | predict the site's *predicted* mean at every timepoint | trajectory information the model adds beyond what it already gets right about level |

**N1 and N2 coincide in centred space.** Any flat prediction has zero centred
variance, so both reduce to the same thing once the level is removed: an error
equal to the observed trajectory itself, `RMSE_flat = σ_o`. They differ only in
absolute SSE units, where N1 is an oracle (it uses the observed mean, which the
model does not have) and N2 is realisable. Hence both spaces are evaluated:

- **centred** — the trajectory question. One flat null, `RMSE_flat = σ_o`,
  plus N3. The headline is the fraction of sites with `RMSE < RMSE_flat`.
- **absolute** — the deployable question. N1 and N2 are distinct; the fraction
  of sites where the full prediction beats its own flat N2 says whether modelling
  the trajectory buys anything over predicting one number per site.

## Stage 3 — Level and trajectory, separated; shape and magnitude scored jointly

Decompose each observed and predicted series into

```
usage[site, organ, t]  =  site_mean[site]  +  organ_offset[site, organ]  +  traj[site, organ, t]
```

with `traj` summing to zero over `t`. Then:

- **Level** — evaluate `site_mean` as a ranking problem (AUC for
  constitutive vs alternative, within-band Spearman, calibration slope). Already
  done: AUC 0.891, slope 0.66. This is a real result; it is just not a trajectory
  result.
- **Trajectory** — evaluate `traj` only, in SSE units, with the statistic below.

### The two primary statistics

For each site–organ, with `o[t]` and `p[t]` the observed and predicted centred
trajectories over `T` timepoints:

```
r     = pearson(o, p)                    ← shape
RMSE  = sqrt( mean_t (p[t] − o[t])² )    ← magnitude, in SSE units
```

`r` answers *does the model get the developmental pattern right*; `RMSE` answers
*is it right by the right amount*. Neither is interpretable alone: `r` ignores
scale entirely, and `RMSE` on its own cannot say whether a large error came from
mistimed dynamics or from correct dynamics at the wrong amplitude. We always report 
them as a pair.

RMSE is only meaningful against a reference, so every RMSE is reported next to
the null RMSEs and the noise-ceiling RMSE from Stages 1–2, and as the ratio
`RMSE / RMSE_flat` (below 1 = the prediction carries real trajectory
information; above 1 = worse than a flat line).

Two observed references are carried side by side, since they are not the same
series: `sse_true`, the stored observed usage the model was trained against, and
`counts`, pooled Alpha/(Alpha+Beta) over each condition's libraries. Their
construction is known to differ, so any conclusion should hold under both.

### How the two relate

The pair is not independent — they are linked exactly. Because both series are
centred, the mean-error term vanishes and

```
RMSE²  =  (σ_p − σ_o)²          ← amplitude mismatch
       +  2 σ_p σ_o (1 − r)     ← pattern disagreement
```

with `σ_o`, `σ_p` the standard deviations over `t`. This is worth computing
because it partitions the RMSE into the part `r` already explains and the part it
does not: a site can have a poor RMSE with `r` near 1 (right shape, wrong
amplitude) or with `σ_p ≈ σ_o` (right amplitude, wrong shape), and the two call
for different fixes. We report the split as percentages of RMSE².

Report per site–organ, then summarise as **distributions**, not pooled values
(pooling across sites would resurrect the level signal):

| quantity | reported as |
|---|---|
| `r` (shape) | median, IQR, fraction > 0, fraction > N3 95th percentile |
| `RMSE` (magnitude), SSE units | median, IQR |
| `RMSE / RMSE_N2` | median, IQR, **fraction below 1** — the decisive number |
| `RMSE / RMSE_ceiling` | median — how much of the achievable error is left |
| `σ_p / σ_o` | median and IQR — direct read on compression; 0.66 pooled on level today |
| error split | median % of RMSE² from amplitude vs pattern |

### Display

A **Taylor diagram per organ** puts both primary metrics on one plot by
construction: `r` is the polar angle, `σ_p/σ_o` the radius, and the distance to
the reference point *is* the RMSE — the geometry is the decomposition above. The
three nulls and the noise ceiling plot as RMSE arcs on the same axes, so "does
the model beat N2" is read off directly. One panel per organ, points coloured by
usage band.

Alongside it, a plain `r` versus `RMSE` scatter per organ, since that is the pair
being reported and the Taylor geometry is not universally read.

## Stage 4 — Event-level amplitude (secondary)

RMSE above is the primary magnitude statistic, with `σ_p/σ_o` as its
scale-free companion. ΔSSE
(max − min over conditions) is retained as a secondary readout because the devAS
gate is defined on it:

- ΔSSE observed vs predicted, Spearman and regression slope. Current pooled
  Pearson 0.24 with slope ≪ 1.
- Always within usage band; a ΔSSE of 0.2 is not the same event at mean usage
  0.5 as at 0.9.

## Stage 5 — Detection, threshold-matched

The devAS yes/no comparison stays as it is: MCC and the 2×2 confusion counts at
the pipeline's own thresholds (significance plus ΔSSE ≥ 0.2), reported beside the
observed and predicted positive rates so the base rate is always visible.
Current values: MCC 0.103 human brain, 0.039 human liver.

No ranking curve. Detection is a summary of the trajectory result, not a
substitute for it — the Stage 3 pair is what carries the evaluation.

## Stage 6 — Pattern classes, demoted

Keep the up / down / up-down / down-up / not-dynamic labels for figures and for
gene-level interpretation. When they are scored:

- always plot the predicted marginal alongside the recall (as in
  `recall_vs_marginal.png`);
- score with Cohen's κ or adjusted mutual information, never raw per-class
  recall;
- report the confusion matrix, never a single-class number in isolation.

## Stage 7 — Cross-species, only after Stages 3–5 pass

The evolutionary question is a difference-of-differences: does the model predict
that an orthologous site's trajectory differs between species in the way it
actually does?

- Match orthologous sites, then evaluate `traj_speciesA − traj_speciesB`
  (predicted vs observed) with the same `r` and RMSE pair as Stage 3, applied to
  the difference series.
- Null: shuffle species labels within orthologous groups. This null is
  essential — shared sequence between orthologues means a model that ignores
  species entirely still scores well on the raw cross-species comparison.
- Only sites clearing Stage 3 in **both** species are eligible.

## Reporting template

One row per (species, organ, usage band):

```
centred   n · r · RMSE · RMSE_flat · RMSE_N3 · RMSE_ceiling
          · RMSE/RMSE_flat · frac_beating_flat · frac_beating_N3
          · sigma_ratio · pct_RMSE2_amplitude
absolute  RMSE_abs · RMSE_abs_N1 · RMSE_abs_N2 · frac_beating_N2
other     dSSE_spearman · detection_MCC · obs_positive_rate
```

The (`r`, `RMSE`) pair and `frac_beating_flat` are the three numbers that carry
the result; everything else is supporting. A row where `r` is respectable but RMSE
sits near `RMSE_flat` means the model finds the right shape at the wrong
magnitude — a different scientific conclusion, and a different fix, from low `r`
at the right amplitude. Reporting either metric alone would hide that distinction,
which is why both are primary.

---

## What exists and what is missing

| stage | status |
|---|---|
| 0 gate and strata | exists — `classify_devas.py` coverage filters; needs the band and coverage-decile strata added |
| 1 noise ceiling | **missing** — needs a replicate-split pass; the dispersion route is already available in `devas_glm.py` |
| 2 nulls | **missing** — cheap, all three are one-liners on the existing matrices |
| 3 decomposition, per-site `r` and RMSE, RMSE² split, Taylor diagrams | **missing** — the main piece of new code |
| 4 event-level ΔSSE | exists (pooled); needs banding and slope |
| 5 detection MCC and confusion | exists — computed and saved in the devAS run |
| 6 class scoring | exists; needs κ/AMI and the marginal overlay made standard |
| 7 cross-species | site map exists (`xspecies_site_map.parquet`); the delta statistic and label-shuffle null are new |

Stages 1–3 are the ones that change conclusions and should be built first.
Everything they need is already in the per-timepoint matrices the devAS run
constructs; nothing requires re-running the model.
