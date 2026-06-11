#!/usr/bin/env bash
# Formal C-lite-v2: 50k steps (train -> merge -> eval -> diagnostic). OOM -> wait 30min -> retry.
set -euo pipefail

REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+set"
OUTPUT_DIR="${OUTPUT_DIR:-/root/rivermind-data/huangziyi/reseg/output/set/clite-v2-from-scratch-50k}"
RUNNER_LOG="${OUTPUT_DIR}/scheduled_runner.log"
OOM_RETRY_WAIT_SEC="${OOM_RETRY_WAIT_SEC:-1800}"
MAX_OOM_RETRIES="${MAX_OOM_RETRIES:-8}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "${RUNNER_LOG}"
}

is_oom_failure() {
  local log_file="$1"
  [[ -f "${log_file}" ]] || return 1
  grep -qiE \
    'out of memory|CUDA out of memory|CUDNN_STATUS_ALLOC_FAILED|torch\.cuda\.OutOfMemoryError|RuntimeError: CUDA error: out of memory' \
    "${log_file}"
}

run_once() {
  local attempt="$1"
  local run_log="${OUTPUT_DIR}/run_attempt_${attempt}_$(date +%Y%m%d_%H%M%S).log"

  log "=== attempt ${attempt} start ==="
  log "run_log=${run_log}"
  log "MAX_STEPS=50000 TRAIN_ONLY=0 OUTPUT_DIR=${OUTPUT_DIR}"

  set +e
  MAX_STEPS=50000 \
  SAVE_STEPS=5000 \
  SAVE_TOTAL_LIMIT=2 \
  PER_DEVICE_TRAIN_BATCH_SIZE=2 \
  GRADIENT_ACCUMULATION_STEPS=1 \
  TRAIN_ONLY=0 \
  RUN_DIAGNOSTIC=1 \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  WANDB_NAME="${WANDB_NAME:-clite-v2-from-scratch-50k}" \
  EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-clite-v2-from-scratch-50k}" \
  FRESH_START="${FRESH_START:-0}" \
  bash run_train_clite_v2_from_scratch.sh 2>&1 | tee "${run_log}"
  local exit_code=${PIPESTATUS[0]}
  set -e

  ln -sfn "$(basename "${run_log}")" "${OUTPUT_DIR}/latest_run_attempt.log"

  if [[ "${exit_code}" -eq 0 ]]; then
    log "=== attempt ${attempt} SUCCESS (exit 0) ==="
    return 0
  fi

  log "=== attempt ${attempt} FAILED (exit ${exit_code}) ==="
  if is_oom_failure "${run_log}"; then
    log "OOM detected in ${run_log}"
    return 2
  fi

  log "Non-OOM failure; see ${run_log}"
  return 1
}

main() {
  log "formal 50k job started in tmux (pid=$$, session=${TMUX:-none})"
  local attempt=1
  while (( attempt <= MAX_OOM_RETRIES + 1 )); do
    run_once "${attempt}"
    local rc=$?
    if [[ "${rc}" -eq 0 ]]; then
      log "Done. output=${OUTPUT_DIR}"
      exit 0
    fi
    if [[ "${rc}" -eq 2 ]]; then
      if (( attempt > MAX_OOM_RETRIES )); then
        log "OOM retries exhausted (${MAX_OOM_RETRIES})"
        exit 2
      fi
      log "Waiting ${OOM_RETRY_WAIT_SEC}s (30min) before OOM retry..."
      sleep "${OOM_RETRY_WAIT_SEC}"
      attempt=$((attempt + 1))
      continue
    fi
    exit 1
  done
}

main "$@"
