#!/usr/bin/env bash
# Pilot driver — foveation constraint bracket x {TP,TA} x 3 seeds + existence/localization
# baselines, on the frozen 20 TP + 20 TA pilot subset. Fully RESUMABLE: re-run to continue
# after an interruption (completed (image,condition,seed) episodes are skipped).
#
# Launch detached:
#   screen -S pilot
#   export VLLM_API_KEY="<your token>"
#   bash scripts/run_pilot.sh
#   # Ctrl-A D to detach;  screen -r pilot to reattach
#
# NOT set -e: a transient failure in one condition must not abort the rest (each run is
# independent and resumable). The harness also drops/ retries failed episodes on its own.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
: "${VLLM_API_KEY:?set VLLM_API_KEY before running (export VLLM_API_KEY=...)}"

IDS="--ids-tp data/pilot_ids_tp.json --ids-ta data/pilot_ids_ta.json"
S="--seeds 0,1,2"
LOG="results/pilot_run.log"; mkdir -p results
echo "### PILOT START $(date)" | tee -a "$LOG"

# first run does the vision preflight; the rest skip it
python3 scripts/run_agent.py --foveation sharp           $IDS $S                  2>&1 | tee -a "$LOG"
python3 scripts/run_agent.py --foveation geisler_perry   $IDS $S --skip-preflight 2>&1 | tee -a "$LOG"
python3 scripts/run_agent.py --foveation gaussian --gist-k 4  $IDS $S --skip-preflight 2>&1 | tee -a "$LOG"
python3 scripts/run_agent.py --foveation gaussian --gist-k 8  $IDS $S --skip-preflight 2>&1 | tee -a "$LOG"
python3 scripts/run_agent.py --foveation gaussian --gist-k 16 $IDS $S --skip-preflight 2>&1 | tee -a "$LOG"
python3 scripts/run_agent.py --foveation crop            $IDS $S --skip-preflight 2>&1 | tee -a "$LOG"
python3 scripts/run_baselines.py $IDS --localize --skip-preflight                  2>&1 | tee -a "$LOG"

echo "### PILOT DONE $(date)"            | tee -a "$LOG"
echo "Next: python3 scripts/compute_metrics.py  (offline; writes figures/pilot/ + results/metrics/)" | tee -a "$LOG"
