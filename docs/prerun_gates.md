# Pre-run gate outcomes (HiCV_run.md §0)

Recorded before the full-subset run. These decide paper wording / confirm feasibility; none block
the run except where noted.

## §0.1 Resolution / no-resize (BLOCKING) — RESOLVED, with an important caveat
- The endpoint does **not** server-side downscale: a full 1680×1050 frame → **1719 vision tokens**
  (matches the ~1700 expectation; no `--mm-processor-kwargs` override).
- **But the harness downscales every glimpse to `model.image_max_side = 1024` before sending.** The
  model therefore actually sees a **1024×640 image ≈ 643 vision tokens** per glimpse — *not* the full
  1680×1050 render. This is intentional (bounds multi-image context over long episodes) and is applied
  **identically to every condition**, so it does not bias the comparison.
- **Paper wording:** foveation is *rendered* at 1680×1050, ppd=30 (the human rig), then downscaled to
  1024-max for the model input (~643 tokens/glimpse; effective ~18 px/° at the encoder). State this
  plainly; do not claim the model sees full 1680×1050.
- Kept at 1024 per §10 (byte-identical to pilot); raising it would change the experiment and risk the
  32k context cap on long multi-image TA episodes.

## §0.2 GP-vs-sharp at the model input (INFORMATIVE) — clean architectural reading
Mean absolute pixel difference (÷255) between `sharp` and each condition, computed on the
**actually-sent (1024-downscaled, JPEG q95)** glimpses, 5 images, gaze=center:

| condition | mean |Δ|/255 | % pixels |Δ|>2 |
|---|--:|--:|
| geisler_perry | 0.0065 | 24% |
| gaussian-k8 | 0.033 | 64% |
| gaussian-k16 | 0.050 | 77% |
| gaussian-k32 | 0.075 | 91% |
| crop | 0.329 | 99% |

GP **does** differ from sharp at the encoder input (mild, peripheral — 0.65% mean, ~a quarter of
pixels) and does *not* collapse onto sharp. So "GP ≈ sharp" behaviour is the **architectural** result
(the parallel encoder receives a genuinely foveated frame and still reads it like sharp), not an
artifact of identical inputs. The ladder scales monotonically; crop is near-total replacement.

## §0.3 Ramp test (BLOCKING — feasibility)
Run 30 min at the intended worker count before committing the full run; confirm throughput scales
~linearly and latency/error rate stays flat. [pending]

## §0.4 Prompt audit (BLOCKING — "no injected policy") — flagged
Verbatim system prompt is neutral on *where* to look (no "scan", "look carefully", "step by step", no
saliency/grid). One clause — "use that history to decide where to look next **and to avoid
re-checking the same spots**" — is a memory/efficiency nudge (not a spatial policy). Kept byte-identical
(§10); in the paper, state no spatial policy is injected and **caveat the refixation-rate result** as
partly prompt-shaped. Optional §8 prompt-variant mini-run closes the objection.

## §0.1 (correctness, found during coverage check) — FIXED
The id files have 8 images recurring under two target categories (1 TP, 7 TA). Episodes/metrics now
key on **(image, target)**, not image alone (commit "Phase 0.1 fix"); verified backward-compatible on
the pilot (all statistics identical; only numfix list ordering shifts).
