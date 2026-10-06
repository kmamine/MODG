#!/usr/bin/env bash
# Full-subset run (HiCV_run.md §3-5): 285 images, 5 seeds, 9 search conditions + baselines.
# Resumable job queue of (condition × seed) jobs over WORKERS concurrent workers; a slow job never
# blocks the rest. Order: baselines (build mask) -> anchors (submittable) -> gist ladder (high-k first).
# Usage: VLLM_API_KEY=... WORKERS=16 bash scripts/run_full.sh   (RAMP=1 stops after the anchor phase)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
: "${VLLM_API_KEY:?set VLLM_API_KEY}"
WORKERS="${WORKERS:-16}"
IDS=(--ids-tp data/image_ids_tp.json --ids-ta data/image_ids_ta.json)
SEEDS=(0 1 2 3 4)
mkdir -p /tmp/full_logs
export VLLM_API_KEY ROOT
echo "=== $(date) START full run | workers=$WORKERS | 5 seeds x 285 images ==="

# one worker: args "mode|k|seed" -> a single resumable run_agent invocation (one condition, one seed)
run_one() {
  IFS='|' read -r mode k seed <<< "$1"
  local tag="${mode}${k:+-k$k}_s${seed}"
  if [ -n "$k" ]; then
    python3 scripts/run_agent.py --foveation gaussian --gist-k "$k" --seeds "$seed" \
      --ids-tp data/image_ids_tp.json --ids-ta data/image_ids_ta.json --skip-preflight \
      > "/tmp/full_logs/${tag}.log" 2>&1
  else
    python3 scripts/run_agent.py --foveation "$mode" --seeds "$seed" \
      --ids-tp data/image_ids_tp.json --ids-ta data/image_ids_ta.json --skip-preflight \
      > "/tmp/full_logs/${tag}.log" 2>&1
  fi
}
export -f run_one

# 1. baselines (existence mask + localization) in the BACKGROUND — needed only at analysis time,
#    so they run concurrently with the search queue instead of blocking it (~85 min serial).
echo "=== $(date) baselines (background) ==="
python3 scripts/run_baselines.py "${IDS[@]}" --localize --skip-preflight > /tmp/full_logs/baselines.log 2>&1 &
BL_PID=$!

# 2. anchors x 5 seeds = 15 jobs (submittable result after this)
echo "=== $(date) anchors (sharp, geisler_perry, crop) ==="
for c in sharp geisler_perry crop; do for s in "${SEEDS[@]}"; do echo "${c}||${s}"; done; done \
  | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
echo "=== $(date) anchors done ==="
[ "${RAMP:-0}" = "1" ] && { echo "RAMP stop after anchors"; exit 0; }

# 3. gist ladder x 5 seeds = 30 jobs, high-k FIRST (slowest, front-loaded)
echo "=== $(date) gist ladder (k128,48,32,24,16,8) ==="
for k in 128 48 32 24 16 8; do for s in "${SEEDS[@]}"; do echo "gaussian|${k}|${s}"; done; done \
  | xargs -P "$WORKERS" -I{} bash -c 'run_one "$@"' _ {}
wait "$BL_PID" 2>/dev/null
echo "=== $(date) DONE full run ==="
