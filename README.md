# MODG — Matched Outcomes, Divergent Gaze

Code, analysis pipeline and paper for **"Matched Outcomes, Divergent Gaze: How Foveated MLLMs Search
Compared to Humans."**

**Data:** [huggingface.co/datasets/Amine-CV/MODG](https://huggingface.co/datasets/Amine-CV/MODG)

## What this is

This is a behavioural study, not a new method. It asks whether multimodal LLMs search a natural scene the way
humans do when they see it through a human-matched foveated view.

Three off-the-shelf models search COCO-Search18 scenes one fixation at a time, zero-shot:
- Qwen3.5-35B-A3B
- GLM-4.6V-Flash
- Gemma-4-E4B

Each model drives the same minimal harness. Every step, the harness renders the scene foveated at the
current gaze point, the model replies `LOOK x,y`, `FOUND x,y` or `ABSENT`, and the loop repeats. The
harness injects no saliency, proposals or stopping rule, so where to look and when to stop are entirely
the model's choice. The model scanpaths are compared with the ten human scanpaths recorded for each scene.

**Result.** The models match or beat humans on two of three axes, but not on the third:

| Axis | Measure | Qwen / GLM / Gemma | Human |
|---|---|---|---|
| **Decision** (target present or absent?) | d′ | 3.84 / 4.23 / 3.14 | 2.91 |
| **Finding** (does gaze reach the target?) | first-saccade target hit rate (TFP@1) | 0.97 / 0.97 / 0.80 | 0.49 |
| **Gaze process** | cross-seed self-consistency (ScanMatch) | 0.84 / 0.91 / 0.71 | 0.53 (inter-observer ceiling) |

On the gaze process all three models share one non-human signature: low-entropy, large-amplitude
scanpaths that agree with themselves far more than two humans agree with each other.

Making peripheral acuity worse never produces a regime where the models search like humans *and* still
find the target.

## Repository layout

```
config.yaml   every experimental knob (model, foveation bracket, harness, seeds, analysis)
src/          foveation renderer, agent harness, model client + directive parser, metrics, stats
scripts/      subset building, agent runs, baselines, metrics, cross-model analysis, paper numbers/figures
tests/        unit tests (foveation golden images, parser, harness smoke, metrics, stats)
data/         frozen subset ID lists (image_ids_{tp,ta}.json) + sampling manifest
docs/         dataset report, experimental methods, comparison plan, pre-run gates
paper/        LaTeX sources and PDFs of the paper and supplement
```

## Setup

```bash
pip install -r requirements.txt
pip install huggingface_hub
# Download the data. It mirrors this repo's paths (data/subset/..., results/...), so it drops into place;
# the excludes keep the dataset card from overwriting this README:
hf download Amine-CV/MODG --repo-type dataset --local-dir . --exclude README.md --exclude .gitattributes
pytest -q
```

## Reproduce the paper numbers (offline, no model calls)

Analysis reads only the saved scanpaths. It never queries a model.

```bash
python3 scripts/build_paper_data.py                  # -> results/metrics/paper_data.json (single source of truth)
python3 scripts/verify_numbers.py --dir paper        # fails if any number in paper/*.tex disagrees with it
python3 scripts/emit_paper_tex.py --out <dir>        # regenerate numbers.tex + table fragments
python3 scripts/make_paper_figures.py --out <dir>/figures
```

`build_paper_data.py` takes a few minutes; all-pairs ScanMatch is the slow part. Pass `--fast` to skip
similarity and density.

## Re-run an agent

1. Serve an OpenAI-compatible vision model with vLLM:
   ```bash
   bash scripts/serve_model.sh <hf-model-id>
   ```
2. In `config.yaml`, set `model.served_id`, `model.slug` and `model.base_url`.
3. Run:
   ```bash
   VLLM_API_KEY=<token> WORKERS=16 bash scripts/run_full.sh
   ```

Runs are resumable: work already on disk is skipped. Outputs land in `results/raw/<slug>/fov-<condition>/`.

To include a new model in the comparison, add its slug to `analysis.compare_slugs` in `config.yaml`.

## Experimental design

### Stimuli and human reference

- **Source:** COCO-Search18, validation split. Its test-split fixations are withheld by the dataset authors.
- **Subset:** stratified and frozen: 141 target-present and 144 target-absent scenes, with all 10 human
  scanpaths per scene.
- **Coordinates:** the canonical frame is 1680×1050 px, about 30 px per degree of visual angle.

### Foveation bracket

| Condition | What it is |
|---|---|
| `sharp` | no foveation |
| `geisler_perry` | Geisler & Perry (1998) acuity falloff; the **only human-matched** condition |
| `gaussian-k{8,16,24,32,48,128}` | synthetic peripheral degradation, not tuned to human behaviour |
| `crop` | fovea-only disc |

### Runs

- 5 seeds at temperature 0.6.
- 50-glimpse safety cap.
- Each glimpse is rendered in the 1680×1050 frame, then downscaled to a 1024-px maximum side before it is
  sent to the model.

## Citation

If you use this code or data, please cite the paper (BibTeX will be added here on publication) and the
COCO-Search18 dataset:

```bibtex
@article{chen2021cocosearch18,
  author  = {Chen, Yupei and Yang, Zhibo and Ahn, Seoyoung and Samaras, Dimitris and Hoai, Minh and Zelinsky, Gregory},
  title   = {{COCO-Search18} fixation dataset for predicting goal-directed attention control},
  journal = {Scientific Reports}, volume = {11}, number = {1}, pages = {8776}, year = {2021},
  doi     = {10.1038/s41598-021-87715-9}
}
@inproceedings{yang2020irl,
  author    = {Yang, Zhibo and Huang, Lihan and Chen, Yupei and Wei, Zijun and Ahn, Seoyoung and Zelinsky, Gregory and Samaras, Dimitris and Hoai, Minh},
  title     = {Predicting Goal-directed Human Attention Using Inverse Reinforcement Learning},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, year = {2020}
}
```

## License

The code and the model-generated outputs are released under the MIT License.

The COCO-Search18 human scanpaths and the stimulus images (MS-COCO) are **not** covered by it. They remain
under the licenses and terms published by their original authors. See [LICENSE](LICENSE).
