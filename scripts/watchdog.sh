#!/usr/bin/env bash
# Relaunch the run supervisor if it has died and the run isn't complete. Idempotent: the supervisor's
# flock prevents double-runs, and the .run_complete sentinel makes this a no-op once finished. Safe to
# run manually, from a Claude cron watchdog, or from a login hook.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
ts=$(date '+%F %H:%M')
[ -f .run_env ] || { echo "watchdog $ts: no .run_env (token) — cannot relaunch"; exit 0; }
[ -f results/.run_complete ] && { echo "watchdog $ts: run COMPLETE; idle"; exit 0; }
sup=$(pgrep -fc "[s]upervise_run.sh" 2>/dev/null || echo 0)
wrk=$(pgrep -fc "[r]un_agent.py --foveation" 2>/dev/null || echo 0)
if [ "${sup:-0}" -gt 0 ] || [ "${wrk:-0}" -gt 0 ]; then
  echo "watchdog $ts: healthy (supervisor=$sup, workers=$wrk)"
else
  WORKERS="${WORKERS:-16}" setsid nohup bash scripts/supervise_run.sh >> /tmp/supervisor.log 2>&1 &
  echo "watchdog $ts: supervisor was DOWN -> relaunched"
fi
