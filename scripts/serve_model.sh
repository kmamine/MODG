#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for the configured vision-language model.
# Usage: bash scripts/serve_model.sh [MODEL_ID]   (defaults to config.yaml model.served_id)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:-$(python3 -c "import yaml;print(yaml.safe_load(open('$ROOT/config.yaml'))['model']['served_id'])")}"
PORT="${PORT:-8000}"
MAX_IMAGES="${MAX_IMAGES:-46}"     # >= TA fixation cap + 1 (multi-image glimpse history)

command -v vllm >/dev/null 2>&1 || { echo "vLLM not installed. Run: pip install vllm"; exit 1; }

echo "Serving $MODEL on :$PORT  (tensor-parallel=2, image limit=$MAX_IMAGES)"
exec vllm serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --limit-mm-per-prompt image="$MAX_IMAGES" \
  --max-model-len "${MAX_MODEL_LEN:-32768}"
# Notes:
#  - The served model MUST be a vision-language checkpoint (run_agent.py preflights this).
#  - --limit-mm-per-prompt / --max-model-len syntax can vary by vLLM version; adjust if it errors.
#  - Pin the vLLM version + model revision and record them for reproducibility.
