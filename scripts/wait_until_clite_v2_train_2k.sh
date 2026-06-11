#!/usr/bin/env bash
# Sleep until TARGET_TIME then run 2k train-only job (OOM -> wait 30min -> retry).
set -euo pipefail

TARGET_TIME="${TARGET_TIME:-2026-06-08 05:00:00}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_clite_v2_train_2k_scheduled.sh"
LOG_DIR="/root/rivermind-data/huangziyi/reseg/output/set/clite-v2-train-2k-bs4"
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
