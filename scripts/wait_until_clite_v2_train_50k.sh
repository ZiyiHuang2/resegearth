#!/usr/bin/env bash
# Sleep until 03:00 then launch formal 50k pipeline inside this shell (expected: tmux).
set -euo pipefail

TARGET_TIME="${TARGET_TIME:-2026-06-08 03:00:00}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_clite_v2_train_50k_scheduled.sh"
LOG_DIR="/root/rivermind-data/huangziyi/reseg/output/set/clite-v2-from-scratch-50k"
WAIT_LOG="${LOG_DIR}/wait_until_run.log"

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "${WAIT_LOG}"
}

target_epoch=$(date -d "${TARGET_TIME}" +%s)
now_epoch=$(date +%s)
sleep_sec=$((target_epoch - now_epoch))

log "TARGET_TIME=${TARGET_TIME}"
log "sleep_sec=${sleep_sec}"
log "tmux_session=${TMUX_PANE:-}${TMUX:-}"

if (( sleep_sec > 0 )); then
  log "Sleeping ${sleep_sec}s until scheduled start..."
  sleep "${sleep_sec}"
elif (( sleep_sec < -3600 )); then
  log "ERROR: target time is more than 1h in the past; abort."
  exit 1
else
  log "Target time passed recently; starting immediately."
fi

log "Launching ${RUN_SCRIPT}"
exec bash "${RUN_SCRIPT}"
