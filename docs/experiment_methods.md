# Agent-Data Generation — Foveated Visual-Search Method

**Scope.** This documents how **agent** scanpaths are *generated* — the foveation renderer (a
graded **constraint bracket**), the agentic **present/absent** search loop, the existence
baselines, and the batch runner. It is a **separate stage from the dataset**: the dataset
([dataset_report.md](dataset_report.md)) is the frozen COCO-Search18 image subset plus the human
reference scanpaths; this stage *consumes* it and *produces* agent scanpaths into `results/`.

**Reframe.** Human-faithful foveation (Geisler–Perry) does **not** constrain an MLLM: the human
search constraint is a *serial sampling bottleneck* (can't resolve the periphery without
fixating), a property of the visual architecture, not the acuity profile — a parallel encoder
reads a mostly-sharp GP frame in one pass and barely searches. We therefore *demonstrate* this
and *characterize* the divergence across a graded bracket {sharp, geisler_perry, gaussian-gist,
crop}, with GP as the faithful/mild under-constrained reference rather than the primary stimulus.

**Validity contract (CLAUDE.md).** The harness only renders, prompts, parses, records, and
loops. It injects **no search policy** — no saliency, detector, candidate proposal, grid, or
heuristic. Where to look and when to stop are 100% the model's. Foveation, prompt, and loop are
**byte-identical across models**.

---

## 1. Foveation renderer (`src/foveation.py`) — the constraint bracket

`render_foveated(image, fixation_xy, cfg, visited)` is pure and deterministic. The modes form a
**graded constraint bracket** (`config.yaml → foveation.mode`). **Only `geisler_perry` is
human-matched** — and the central finding is that it does *not* constrain a parallel vision
encoder (matching retinal input ≠ matching the human serial-sampling bottleneck). The others are
**synthetic manipulations / anchors**, never "human-matched."

- **sharp** — no foveation (image unmodified). Unconstrained anchor; in-harness it should
  terminate in ~1 step (the "GP ≈ sharp" check).
- **geisler_perry** — faithful Geisler & Perry (1998) acuity falloff: perspective eccentricity
  `ec = atan(eradius/view_dist_m)`; eye cutoff `fc(ec)=e2·ln(1/CT0)/(α·(ec+e2))` (`e2=2.3,
  α=0.106, CT0=1/64`); per-pixel pyramid level `L = log2(local_Nyquist/fc)` — the **canonical
  log2** form, which renders the local image cutoff equal to the eye's cutoff (sharp centre,
  since ratio<1 → log2<0 → level 0). Mild at the COCO-Search18 geometry (`px_per_degree≈30`,
  `view_dist_m≈0.6`, ~54°×35° field; Chen et al. 2021). The **under-constrained reference**.
- **gaussian** — the *usable-gist* manipulation: the GP falloff with peripheral acuity reduced by
  a factor **k = `gist_k`** (`L = log2(k·local_Nyquist/fc)`; k>1 shrinks the sharp region and
  k×-downsamples the periphery). Tuned to *detectable-but-not-confirmable* (report k + a sweep;
  pilot sweeps k∈{4,8,16}).
- **crop** — fovea-only disk (radius `foveal_radius_deg`·ppd), rest gray. No peripheral gist; the
  **over-constrained bound** (forces serial accumulation).
- **periphery** — legacy two-level control (sharp disk + single blurred surround), out of the
  pilot bracket.

Octave Gaussian pyramid (level k ≈ a 2^k mip, σ=0.3748·2^k px); byte-identical across models.
Examples: [figures/foveation_samples/](../figures/foveation_samples/) — `06_geisler_perry.png`
(mild GP), `gaussian_vs_gp.png` (GP vs gist k=8).

---

## 2. The agentic search loop (`src/harness.py`)

```
step 0: gaze forced to image center (0.5, 0.5)          # matches the human central-fixation start
loop until stop or the safety cap:
  render foveated glimpse at current gaze  ──►  add to a multi-turn chat
  model replies with ONE directive:
     LOOK: x=.., y=..   → append fixation, continue
     FOUND: x=.., y=..  → record it, STOP (found=true,  reason="found")
     ABSENT             → STOP            (found=false, reason="not_present")
  if #fixations reaches the cap → STOP (reason="max_fixations")
```

- **Safety cap = 50 glimpses, uniform across all conditions** (TP, TA, every K). The step-0
  center counts as the first glimpse, so search stops after 50 glimpses. The cap is a **bound
  only** — the model decides when to stop earlier; we **never** auto-stop on a target hit
  (the stopping decision is the data, especially for target-absent).
- **Output** is one record in the locked scanpath schema: `{image_id, condition, target_category,
  agent, seed, fixations:[{x,y,step}…], stopped, stop_reason, found}`.

---

## 3. Multi-turn glimpse history & Inhibition of Return (the memory window K)

Each turn carries past glimpse **images** so the model can integrate the scene across saccades.
The memory window **K = the total number of glimpses the model sees, including the current one**:

| K | what the model sees each step |
|---|---|
| **1** | only the current foveated glimpse — no prior images, no history. **Memoryless.** |
| 3 / 5 / 8 / 11 | current + the last 2 / 4 / 7 / 10 glimpses |
| **full** (`null`) | every glimpse so far |

There is **no coordinate trail and no stated gaze coordinate** — memory is exactly the kept
images (and the model's own `LOOK:` replies within those kept turns). Smaller K = shorter visual
memory → the operational inhibition-of-return manipulation. **The full trajectory is always
recorded** in `scanpaths.jsonl`/`responses.jsonl` regardless of K — the window only limits what
the *model* sees, never what is logged for analysis.

(A second, stimulus-side IoR — a visual mask that dims recently-visited regions — also exists
(`config.yaml → ior.enabled`) but is **off** for the memory-window sweep and for the primary
human-comparison run.)

---

## 4. Model adapter & directive parsing (`src/models/`)

- **`client.py`** — one thin OpenAI-compatible client drives every model (only `base_url` +
  served id change), so the harness is byte-identical across models. The served model
  (`Qwen/Qwen3.5-35B-A3B`) is a **reasoning** model: it thinks (hidden) and returns only the
  final answer in `content`, so `model.max_tokens` is set generously. Sampling:
  `temperature 0.6, top_p 0.95, top_k 20, min_p 0`.
- **`parse.py`** — extracts the normalized `(x,y)` and the stop decision from free text
  (`LOOK/FOUND/ABSENT`, JSON, or Molmo `<point>`); clamps out-of-range coords; **never raises**.
  Malformed output → one nudge retry → if still bad, the episode is dropped (`parse_error`,
  re-tried on the next resumable run) so the scanpath file stays schema-pure.

---

## 5. Batch runner (`scripts/run_agent.py`)

```
for condition in [TP, TA]:           # 141 TP + 144 TA = 285 stimuli
  for image in that condition:
    for seed in run.seeds:           # each seed = one independent repetition
        run one episode → append to results/raw/<model_slug>/<ior_slug>/scanpaths.jsonl
```

- **Preflight** verifies the served model is up and accepts an image.
- **Seeds = repetitions** (sampling at temperature > 0 gives a distribution of trajectories
  per image, comparable to the 10 human subjects).
- **Resumable**: skips any `(image_id, condition, seed)` already in `scanpaths.jsonl`.
- **Slug per condition** isolates results: `--memory-k 3` → `mem3/`, `--memory-k none` →
  `fullmem_nomask/`. Each condition has its own folder and resume state.
- **Outputs** per run dir: `scanpaths.jsonl` (one record/episode), `responses.jsonl` (per-step
  raw text + latency/tokens/parse-status), `run_config.json` (full config, prompts, code hash).

CLI overrides (auto-derive the slug): `--seeds 0,1,2`, `--memory-k K|none`, `--slug NAME`,
`--limit N`, `--smoke`, `--skip-preflight`.

---

## 6. Reproducibility

The harness is deterministic (fixed center start, deterministic parse, frozen iteration order);
**all randomness is confined to seeded model sampling**. Each run logs model id, prompts
verbatim, the full config, a code hash, and raw responses. Caveat: vLLM sampling is not
bit-reproducible even with a fixed seed (continuous batching / tensor-parallel reductions), so
we report across multiple seeds and pin the vLLM + model revision.

---

## 7. Running the experiment

```bash
export VLLM_API_KEY="<token>"                            # never committed; read from env
python3 scripts/run_agent.py --seeds 0,1,2 --limit 50    # no-IoR calibration -> fullmem_nomask/
python3 scripts/run_agent.py --seeds 0,1,2 --memory-k 1  # memoryless        -> mem1/
python3 scripts/run_agent.py --seeds 0,1,2 --memory-k 3  # -> mem3/
python3 scripts/run_agent.py --seeds 0,1,2 --memory-k 5  # -> mem5/
python3 scripts/run_agent.py --seeds 0,1,2 --memory-k 8  # -> mem8/
python3 scripts/run_agent.py --seeds 0,1,2 --memory-k 11 # -> mem11/
```

Each `mem<K>` run is the full subset (285 stimuli) × 3 repetitions = 855 agent scanpaths, in the
same schema as the human reference data — the input to the (forthcoming) metrics + comparison.
