#!/usr/bin/env bash
# Regenerate every table and figure from SAVED scanpaths — no model calls, fully offline.
#
# The whole analysis is config-driven: model identity (model.slug) and the condition bracket
# (analysis.bracket) live in config.yaml, so adding a model later is "run scripts/run_agent.py for
# the new slug, then re-run this" — zero code edits. Reads results/raw/<slug>/ + the existence mask;
# writes results/metrics/*.json and figures/pilot/*.png.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== [1/4] outcome metrics + TFP/NumFix/density/gist-sweep figures ==="
python3 scripts/compute_metrics.py

echo "=== [2/4] intrinsic signature + scanpath similarity + PCA (single model, clustered CIs) ==="
python3 scripts/compare.py

echo "=== [2b] cross-model comparison (PCA + similarity over analysis.compare_slugs) ==="
python3 scripts/compare_models.py

echo "=== [3/4] per-episode dropout accounting ==="
python3 scripts/dropout_report.py

echo "=== [3b] TFP robustness (hit-tolerance §6.3) + TFP@1 by ecc/size (§6.2) ==="
python3 scripts/tfp_robustness.py

echo "=== [3c] mixed-effects models (crossed image+rater REs, §6.1) ==="
python3 scripts/mixed_effects.py

echo "=== [4/4] temperature-sweep self-consistency (if fov-*-t* runs exist) ==="
if compgen -G "results/raw/$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['model']['slug'])")/fov-*-t*" > /dev/null; then
  python3 scripts/temp_sweep.py
else
  echo "    (no temperature-sweep dirs; skipping)"
fi

echo "=== done. tables -> results/metrics/  figures -> figures/pilot/ ==="
