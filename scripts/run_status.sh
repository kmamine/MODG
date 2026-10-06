#!/usr/bin/env bash
# Progress snapshot for the full run — run anytime: `bash scripts/run_status.sh`
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
SLUG=$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['model']['slug'])")
RAW="results/raw/$SLUG"
echo "==== $(date) — full-run status (model=$SLUG) ===="
echo "supervisor alive : $(pgrep -fc supervise_run.sh 2>/dev/null || echo 0)"
echo "search workers   : $(pgrep -fc 'run_agent.py --foveation' 2>/dev/null || echo 0)"
echo "baselines proc   : $(pgrep -fc run_baselines.py 2>/dev/null || echo 0)"
echo
# expected per condition = 5 seeds x 285 images (search); reused pilot episodes count toward this
printf "%-22s %8s %8s\n" "condition(slug)" "scanpath" "of~1425"
total=0
for d in "$RAW"/fov-sharp "$RAW"/fov-geisler_perry "$RAW"/fov-crop \
         "$RAW"/fov-gaussian-k8 "$RAW"/fov-gaussian-k16 "$RAW"/fov-gaussian-k24 \
         "$RAW"/fov-gaussian-k32 "$RAW"/fov-gaussian-k48 "$RAW"/fov-gaussian-k128; do
  n=$(cat "$d/scanpaths.jsonl" 2>/dev/null | wc -l); total=$((total+n))
  printf "%-22s %8s %8s\n" "$(basename "$d")" "$n" "1425"
done
echo "------"
printf "%-22s %8s %8s\n" "TOTAL search" "$total" "12825"
echo "(target 12825 = 9 cond x 5 seeds x 285; pilot's 40 imgs at seeds 0-4 reused)"
echo
echo "baselines: existence=$(cat "results/baselines/$SLUG/existence.jsonl" 2>/dev/null | wc -l)/570  localization=$(cat "results/baselines/$SLUG/localization.jsonl" 2>/dev/null | wc -l)/141"
echo
echo "---- last supervisor/run lines ----"
tail -n 4 /tmp/supervisor.log 2>/dev/null | sed 's/token[^ ]*/<token>/g' || true
tail -n 2 /tmp/run_full.log 2>/dev/null | sed 's/token[^ ]*/<token>/g' || true
