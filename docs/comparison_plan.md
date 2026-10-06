# Agent ↔ Human Scanpath Comparison Plan

How we compare MLLM and human visual-search behaviour — the metric catalog, the comparison
structure, and the statistics. Spans **outcome** behaviour (did/how-fast it found the target) and
**intrinsic scanpath structure** (how the gaze moves). Implemented in `src/metrics.py`,
`src/scanpath_metrics.py`, `src/stats.py`; orchestrated by `scripts/compute_metrics.py` (outcome)
and `scripts/compare.py` (intrinsic + similarity + MDS), offline on saved scanpaths.

## 1. Metric catalog

### A. Outcome / task efficiency
| metric | definition | condition |
|---|---|---|
| existence accuracy | one-shot "is the {target} present?" yes/no on full sharp image | TP & TA |
| search success | model declares FOUND with a fixation inside the target bbox | TP |
| **NumFix2T** | fixations to the first target fixation | TP |
| **TFP** + **TFP-AUC** | cumulative target-fixation prob by saccade *n*; AUC = mean of the curve | TP |
| first-saccade targeting | TFP@1, TFP@2 | TP |
| NumFix | fixations at termination | TP & TA |
| stopping | declared-absent / false-present / hit-cap fractions | TA |

### B. Intrinsic spatial signature (per-scanpath → compare distributions)
saccade **amplitude** (dva), saccade **direction** (deg) + cardinal bias, **turning angle**
(successive-saccade change — meander vs directed), **scanpath length** (dva), spatial **coverage**
(convex-hull area, bbox area, 2-D spread), **gaze entropy** (bits, over a 14×9 grid), **center
bias** (mean fixation eccentricity), **refixation rate** (revisits to a visited cell — intrinsic
inhibition-of-return signature), saccade-amplitude-vs-ordinal-position. Fixation **duration** is
**human-only** (agents have no durations — flagged, never faked).

### C. Intrinsic scanpath similarity (pairwise, sequence-level)
**ScanMatch / Sequence Score** (dataset standard; Needleman–Wunsch on 14×9 bins, [0,1] higher =
more alike) — the robust primary (works at any length). **MultiMatch** (vector / direction /
length / position; **duration dim excluded** for agents; needs ≥3 fixations). **DTW, Hausdorff,
Fréchet, TDE, Levenshtein** (distances, lower = more alike). Lifted from
`src_legacy_project_for_ref/metrics/`, adapted to N×2 px / 1680×1050.

### D. Spatial density agreement
**NSS / CC / KL** between the model fixation-density map and the human map, with the **human↔human
density** as the ceiling.

## 2. Comparison structure (the scaffold that gives the numbers scale)
Per image, three sets — aggregated across images and across the bracket {sharp, geisler_perry,
gaussian-k (k∈{4,8,16,24,32,48,64,128}), crop}:
- **agent ↔ human** — each agent scanpath (seed) vs each of the 10 human scanpaths.
- **human ↔ human** — all 45 human pairs = the **agreement ceiling**. Reported for *every*
  similarity/agreement metric: without it, an agent↔human number has no scale.
- **agent ↔ agent** — across seeds = the model's **self-consistency**.
For distributional metrics (B): the agent per-scanpath distribution vs the pooled-human
distribution. **Per-image paired** comparison controls image difficulty. Search metrics are
computed only on images the model passes the existence baseline (filtering rule).

## 3. Statistics
Distributions first (box / ECDF), then **effect sizes** — **Cliff's δ** (nonparametric, vs human;
|δ|>0.33 non-trivial, >0.47 large) and Cohen's d. CIs use an **image-clustered bootstrap**
(`src/stats.cluster_bootstrap_ci` / `cliffs_delta_ci`): the resampling unit is the **image**, not
the individual scanpath, because each image yields many scanpaths (seeds × the 10 human subjects)
that share image-level variance — a flat row bootstrap would treat them as independent and understate
the CI. The δ-vs-human CI resamples images shared by both groups (the honest per-image paired
comparison; the 45 human↔human pairs and seed×subject cells are never treated as independent). Lead
with effect sizes + clustered CIs, not p-values. Fallback: paired Wilcoxon on per-image medians +
Holm. **Headline multivariate:** per-group signature vector → **MDS** of {human, each condition} —
"does the model cluster apart from humans?"

## 4. Outputs
One offline command — `scripts/regenerate_all.sh` — regenerates every table and figure from saved
scanpaths (no model calls). It is config-driven: `model.slug` + `analysis.bracket` in `config.yaml`
are the only things to change to add a model later (`src/conditions.py` resolves the bracket; the
temperature-sweep dirs `fov-*-t<T>` are excluded from it). It runs:
- `scripts/compute_metrics.py` → outcome (`results/metrics/pilot_metrics.json`) + figures
  `figures/pilot/{tfp,numfix,stopping_ta,density_agreement,existence,gist_sweep}.png` (the gist-k
  sweep plots TFP@1 and TFP-end vs k against the human reference — the "is there a human-like band?"
  figure).
- `scripts/compare.py` → intrinsic + similarity + `results/metrics/comparison.json` + figures
  `figures/pilot/{intrinsic_signature,scanpath_similarity,mds}.png` (clustered δ-vs-human CIs).
- `scripts/dropout_report.py` → `results/metrics/dropout.json` (per-episode completed/retry/dropped,
  by condition × TP/TA × category).
- `scripts/temp_sweep.py` → `results/metrics/temp_sweep.json` +
  `figures/pilot/selfconsistency_temp.png` (agent↔agent ScanMatch/DTW vs temperature, against the
  human↔human ceiling), when temperature-sweep runs are present.

## 5. Caveats
- Agents have **no fixation durations** → duration metrics & MultiMatch-duration are human-only.
- Short agent scanpaths (≤2 fixations under sharp/GP) make MultiMatch undefined (NaN) and inflate
  ScanMatch toward the ceiling — interpret similarity alongside NumFix.
- Pilot scale (20 TP / cond, 1 model, 3 seeds) → effect sizes indicative; same pipeline at full scale.
