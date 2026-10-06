#!/usr/bin/env bash
# Auto-restart supervisor for the full run. Survives:
#   - Claude/SSH disconnection: launch with `setsid nohup` (fully detached, own session).
#   - run_full / endpoint death: run_full is resumable; this re-runs it until a pass adds ~no new
#     episodes (loop-until-dry, which also absorbs the irreducible ~1% over-reasoner dropout instead
#     of retrying it forever).
# A flock guarantees only ONE supervised run is ever active (no double-running / duplicate episodes).
#
# Launch (token inherited, survives disconnection):
#   VLLM_API_KEY=... WORKERS=16 setsid nohup bash scripts/supervise_run.sh > /tmp/supervisor.log 2>&1 &
# After a machine reboot, just re-run that same line.  Check progress with scripts/run_status.sh.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
[ -f .run_env ] && . ./.run_env          # optional gitignored file with `export VLLM_API_KEY=...`
: "${VLLM_API_KEY:?set VLLM_API_KEY (env or .run_env)}"
export VLLM_API_KEY WORKERS="${WORKERS:-16}"
MINDELTA="${MINDELTA:-5}"                 # converged when a full pass adds fewer than this many episodes
MAXPASS="${MAXPASS:-12}"                  # hard safety cap on passes

exec 9>/tmp/gazeagent_run.lock
flock -n 9 || { echo "$(date) another supervised run holds the lock; exiting."; exit 0; }

SENTINEL="results/.run_complete"
[ -f "$SENTINEL" ] && { echo "$(date) run already complete ($SENTINEL present); nothing to do."; exit 0; }
SLUG=$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['model']['slug'])")
count(){ cat "results/raw/$SLUG"/fov-*/scanpaths.jsonl 2>/dev/null | wc -l; }

echo "=== $(date) SUPERVISOR START (workers=$WORKERS, slug=$SLUG, start_count=$(count)) ==="
pass=0
while [ "$pass" -lt "$MAXPASS" ]; do
  pass=$((pass+1)); before=$(count)
  echo "=== $(date) pass $pass START (episodes on disk: $before) ==="
  bash scripts/run_full.sh >> /tmp/run_full.log 2>&1 || echo "$(date) run_full exited nonzero — will resume next pass"
  after=$(count); delta=$((after-before))
  echo "=== $(date) pass $pass END (added $delta; total $after) ==="
  if [ "$delta" -lt "$MINDELTA" ]; then echo "$(date) CONVERGED (pass added <$MINDELTA new)."; break; fi
done
touch "$SENTINEL"   # mark complete: a @reboot relaunch becomes a clean no-op
echo "=== $(date) SUPERVISOR DONE after $pass pass(es); $(count) episodes. Now run scripts/regenerate_all.sh ==="
