# Dataset & Stimuli (paper-ready methods section)

> Draft prose for the workshop paper's data/methods section. Numbers trace to
> [`figures/analysis_stats.json`](../figures/analysis_stats.json); figures are in
> [`figures/`](../figures/). Citation style and claims follow the project's positioning: this
> is a behavioural *characterization* study, not a scanpath-prediction method.

## Dataset

We draw human reference behaviour from **COCO-Search18** [Yang et al., 2020], a
laboratory-quality eye-tracking dataset in which ten observers searched for each of eighteen
target-object categories in natural COCO scenes. The dataset spans two regimes that we treat as
distinct conditions: **target-present (TP)**, in which the searched-for object is present and
the observer fixates and reports it, and **target-absent (TA)**, in which the object is absent
and the observer must decide so and terminate the search. The target-absent regime is central
to our study because the decision of *when to stop* — rather than where to look — is the
dependent variable for the stopping analysis. All gaze was recorded on stimuli resized to a
common 1680 × 1050-pixel frame (zero-padded, aspect ratio preserved), and we adopt this frame
as the canonical coordinate space, normalizing all positions to [0, 1] as (x/1680, y/1050).

Because COCO-Search18's test-split fixations are withheld by the dataset authors for a separate
benchmark track, only the training and validation human scanpaths are publicly available. We
draw our stimuli from the **validation split**, which is the held-out official partition; we do
not re-partition the data. Since every model in our panel is evaluated **zero-shot** — no model
is trained or fine-tuned on any portion of the data — the train/validation/test boundary
carries no risk of information leakage, and the validation split serves purely as a fixed,
frozen evaluation set.

## Stimulus subset selection

To keep the downstream experiment tractable while preserving the factors that govern search
difficulty, we select a stratified subset of images with a fixed random seed. For
target-present images we balance across **category × target eccentricity {near, far} × scene
clutter {low, high}**; for target-absent images, where the target has no location, we balance
across **category × clutter** only. Eccentricity is the distance of the target's bounding-box
centre from the image centre in normalized coordinates, and clutter is the number of annotated
COCO object instances in the image, obtained from the COCO-2017 instance annotations (the
instance counts of which are identical to the 2014 release and cover every COCO-Search18
image). To avoid skew across categories of differing composition, near/far and low/high splits
are made at **per-category medians** rather than a global threshold.

We retain only images for which all ten human scanpaths are available (an exact-ten criterion
that yields a clean human–human agreement ceiling for later scanpath-similarity analysis) and
sample a target of eight images per category — two per cell for target-present (four cells) and
four per cell for target-absent (two cells). The resulting subset comprises **141
target-present and 144 target-absent images** (285 in total), each with its full complement of
ten human scanpaths (2,850 scanpaths in all). The target-absent subset is exactly balanced; the
target-present subset reaches the full eight images in fifteen of eighteen categories and seven
in the remaining three (bowl, laptop, oven), where a single eccentricity-by-clutter cell in the
validation pool contained only one eligible image. Such shortfalls are logged rather than
back-filled, preserving the integrity of the stratification (Figure 1). The frozen image-id
lists, per-category thresholds, and shortfall log are committed alongside the code, and the
subset is exported as a self-contained package of images and filtered scanpaths for
reproducibility.

## Human reference behaviour

The selected scanpaths reproduce the expected signatures of goal-directed human search and
establish the reference distribution against which agent behaviour will be compared. Observers
are accurate in both regimes (92.3% target-present, 93.2% target-absent), and no target-absent
trial registers a fixation on the target, as expected. Target-present search is efficient and
short — a median of three fixations (mean 3.8; 99th percentile 14) — whereas target-absent
search is longer and heavier-tailed — a median of five fixations (mean 6.0; 99th percentile 20)
— reflecting the greater effort of confirming absence (Figure 2). Targets are small (median 3.7%
of image area), consistent with a genuine search rather than a pop-out task. Spatially, initial
fixations cluster tightly near the image centre (mean ≈ (815, 510) px against the (840, 525)
centre), reflecting the forced central-fixation start of the recording protocol; saccade
amplitudes have a median of 0.12 (target-present) and 0.14 (target-absent) of the image diagonal
(Figures 3–4). The longest human scanpaths in the subset are 34 fixations (target-present) and
45 (target-absent).

## Agent-data generation (separate stage)

Presenting these stimuli to the agents under a foveated view — the foveation renderer, the
agentic search loop, inhibition of return, and the fixation cap — is a separate stage, **not part
of the dataset**; it is documented in [experiment_methods.md](experiment_methods.md). No agent
results appear in this dataset description.

---

**Figures.** (1) per-category subset balance by stratum cell —
[`fig_category_balance.png`](../figures/fig_category_balance.png),
[`fig_stratum_heatmap.png`](../figures/fig_stratum_heatmap.png);
(2) human scanpath-length distributions —
[`fig_scanpath_length.png`](../figures/fig_scanpath_length.png);
(3) eccentricity, clutter, target size —
[`fig_eccentricity.png`](../figures/fig_eccentricity.png),
[`fig_clutter.png`](../figures/fig_clutter.png),
[`fig_target_size.png`](../figures/fig_target_size.png);
(4) saccade amplitude and initial-fixation centre bias —
[`fig_saccade_amplitude.png`](../figures/fig_saccade_amplitude.png),
[`fig_initial_fixation.png`](../figures/fig_initial_fixation.png).

**Reference.** Yang, Z., Huang, L., Chen, Y., Wei, Z., Ahn, S., Zelinsky, G., Samaras, D., &
Hoai, M. (2020). Predicting Goal-directed Human Attention Using Inverse Reinforcement Learning.
In *Proc. IEEE/CVF CVPR* (pp. 193–202).
