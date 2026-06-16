#!/usr/bin/env bash
set -euo pipefail

########################################
# Test 4 full pipeline
#   0) baseline from base lasers_test_table2_metrics.json (default; no identity eval)
#   1) train 5k (default: 2k fail-fast stop, else resume to 5k)
#   2) test eval at checkpoint 2k + 5k only
#   3) optional Probe E (RUN_PROBE_E=1)
########################################

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=test4_common.sh
source "${SCRIPT_DIR}/test4_common.sh"

REPO_DIR="${REPO_DIR}"
cd "${REPO_DIR}"

RUN_TAG="${RUN_TAG:-test4-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${TEST4_OUTPUT_ROOT}/${RUN_TAG}}"
MAX_STEPS="${MAX_STEPS:-5000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_IDENTITY="${SKIP_IDENTITY:-1}"
SKIP_CHECKPOINT_SWEEP="${SKIP_CHECKPOINT_SWEEP:-0}"
RUN_PROBE_E="${RUN_PROBE_E:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
EVAL_CHECKPOINT_STEPS="${EVAL_CHECKPOINT_STEPS:-2000,5000}"
FAILFAST_STOP_TRAIN="${FAILFAST_STOP_TRAIN:-1}"
BASELINE_METRICS_JSON="${BASELINE_METRICS_JSON:-}"

TRAIN_DIR="${RUN_ROOT}/train"
BASELINE_FILE="${BASELINE_FILE:-${RUN_ROOT}/test4_baseline_multi_cate_giou.txt}"
IDENTITY_EVAL_DIR="${RUN_ROOT}/baseline_identity_eval"
PIPELINE_LOG="${RUN_ROOT}/pipeline.log"

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

echo "========================================"
echo "[Test4] PIPELINE START"
echo "RUN_ROOT               : ${RUN_ROOT}"
echo "MAX_STEPS              : ${MAX_STEPS}"
echo "SAVE_STEPS             : ${SAVE_STEPS}"
echo "SKIP_IDENTITY          : ${SKIP_IDENTITY}"
echo "EVAL_CHECKPOINT_STEPS  : ${EVAL_CHECKPOINT_STEPS}"
echo "FAILFAST_STOP_TRAIN    : ${FAILFAST_STOP_TRAIN}"
echo "PIPELINE_LOG           : ${PIPELINE_LOG}"
echo "========================================"

test4_preflight
test4_resolve_base_model
echo "[INFO] training base: ${MODEL_PATH} (${MODEL_BASE_SOURCE})"

# --- Step 0: baseline ---
if [[ "${SKIP_IDENTITY}" != "1" ]]; then
  echo ""
  echo "========== STEP 0: identity eval (optional) =========="
  IDENTITY_EVAL_DIR="${IDENTITY_EVAL_DIR}" \
  BASELINE_FILE="${BASELINE_FILE}" \
  MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES}" \
  bash "${SCRIPT_DIR}/test4_identity_eval.sh"
else
  echo ""
  echo "========== STEP 0: baseline from base metrics (default) =========="
  if [[ -z "${BASELINE_METRICS_JSON}" ]]; then
    BASELINE_METRICS_JSON="$(test4_default_base_metrics_json)"
  fi
  echo "[INFO] BASELINE_METRICS_JSON=${BASELINE_METRICS_JSON}"
  test4_seed_baseline_from_metrics "${BASELINE_METRICS_JSON}" "${BASELINE_FILE}"
  cp -f "${BASELINE_METRICS_JSON}" "${RUN_ROOT}/baseline_lasers_test_table2_metrics.json"
fi
BASELINE_GIOU="$(test4_read_baseline_giou "${BASELINE_FILE}")"
echo "[INFO] baseline test_multi_cate gIoU = ${BASELINE_GIOU}%"

test4_eval_one_checkpoint() {
  local step="$1"
  local ckpt="${TRAIN_DIR}/checkpoint-${step}"
  if [[ ! -d "${ckpt}" ]]; then
    echo "[WARN] missing ${ckpt}; skip eval"
    return 1
  fi
  local eval_out="${TRAIN_DIR}/eval_ckpt-${step}"
  CHECKPOINT_PATH="${ckpt}" \
  MERGED_DIR="${TRAIN_DIR}/merged_ckpt-${step}" \
  EVAL_OUTPUT_DIR="${eval_out}" \
  BASELINE_FILE="${BASELINE_FILE}" \
  MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES}" \
  bash "${SCRIPT_DIR}/test4_smoke_eval.sh"

  local gio gate baseline
  gio="$(test4_parse_multi_cate_giou "$(test4_find_table2_json "${eval_out}")")"
  baseline="$(test4_read_baseline_giou "${BASELINE_FILE}")"
  gate="$(test4_gate_label "${baseline}" "${gio}")"
  echo "${step},${baseline},${gio},$(python3 -c "print(${gio}-${baseline})"),${gate}" >> "${SUMMARY_CSV}"
  echo "[INFO] checkpoint-${step} gate=${gate}"
  [[ "${gate}" == "FAIL-FAST" ]]
}

# --- Step 1: train ---
if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo ""
  if [[ "${FAILFAST_STOP_TRAIN}" == "1" && "${MAX_STEPS}" -gt 2000 ]]; then
    echo "========== STEP 1a: train to 2k (fail-fast phase) =========="
    OUTPUT_DIR="${TRAIN_DIR}" \
    MAX_STEPS=2000 \
    SAVE_STEPS="${SAVE_STEPS}" \
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
    MODEL_PATH="${MODEL_PATH}" \
    bash "${SCRIPT_DIR}/test4_smoke_train.sh"

    SUMMARY_CSV="${TRAIN_DIR}/test4_checkpoint_eval_summary.csv"
    echo "checkpoint_step,baseline_giou,result_giou,delta,gate" > "${SUMMARY_CSV}"
    echo ""
    echo "========== STEP 1b: eval checkpoint-2000 =========="
    if test4_eval_one_checkpoint 2000; then
      echo "[Test4] FAIL-FAST at 2k (delta < ${FAILFAST_DELTA} pt vs baseline ${BASELINE_GIOU}%)"
      echo "[Test4] Skipping resume to ${MAX_STEPS} and final eval."
      echo "Summary CSV: ${SUMMARY_CSV}"
      exit 0
    fi

    echo ""
    echo "========== STEP 1c: resume train to ${MAX_STEPS} =========="
    OUTPUT_DIR="${TRAIN_DIR}" \
    MAX_STEPS="${MAX_STEPS}" \
    SAVE_STEPS="${SAVE_STEPS}" \
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
    MODEL_PATH="${MODEL_PATH}" \
    bash "${SCRIPT_DIR}/test4_smoke_train.sh"
  else
    echo "========== STEP 1: train to ${MAX_STEPS} =========="
    OUTPUT_DIR="${TRAIN_DIR}" \
    MAX_STEPS="${MAX_STEPS}" \
    SAVE_STEPS="${SAVE_STEPS}" \
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
    MODEL_PATH="${MODEL_PATH}" \
    bash "${SCRIPT_DIR}/test4_smoke_train.sh"
  fi
else
  echo "[SKIP] train (SKIP_TRAIN=1)"
  TRAIN_DIR="${TRAIN_OUTPUT_DIR:-${TRAIN_DIR}}"
fi

# --- Step 2: checkpoint eval (2k + 5k by default) ---
SUMMARY_CSV="${SUMMARY_CSV:-${TRAIN_DIR}/test4_checkpoint_eval_summary.csv}"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "checkpoint_step,baseline_giou,result_giou,delta,gate" > "${SUMMARY_CSV}"
fi

if [[ "${SKIP_CHECKPOINT_SWEEP}" != "1" ]]; then
  echo ""
  echo "========== STEP 2: eval checkpoints (${EVAL_CHECKPOINT_STEPS}) =========="
  TRAIN_OUTPUT_DIR="${TRAIN_DIR}" \
  BASELINE_FILE="${BASELINE_FILE}" \
  EVAL_CHECKPOINT_STEPS="${EVAL_CHECKPOINT_STEPS}" \
  MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES}" \
  SUMMARY_CSV="${SUMMARY_CSV}" \
  bash "${SCRIPT_DIR}/test4_eval_checkpoints.sh"
else
  echo "[SKIP] checkpoint sweep (SKIP_CHECKPOINT_SWEEP=1)"
fi

# --- Step 3: optional Probe E on last checkpoint ---
if [[ "${RUN_PROBE_E}" == "1" ]]; then
  echo ""
  echo "========== STEP 3: Probe E =========="
  LAST_CKPT="$(find "${TRAIN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1)"
  STEP="$(basename "${LAST_CKPT}" | sed 's/checkpoint-//')"
  MERGED="${TRAIN_DIR}/merged_ckpt-${STEP}"
  EXPORTED="$(test4_export_checkpoint "${LAST_CKPT}" "${MERGED}" "${MODEL_PATH}")"
  MODEL_PATH="${EXPORTED}" \
  OUTPUT_DIR="${RUN_ROOT}/probe_e" \
  bash "${SCRIPT_DIR}/test4_probe_e.sh"
fi

echo ""
echo "========================================"
echo "[Test4] PIPELINE DONE"
echo "RUN_ROOT       : ${RUN_ROOT}"
echo "Baseline gIoU  : ${BASELINE_GIOU}% (file: ${BASELINE_FILE})"
echo "Train dir      : ${TRAIN_DIR}"
echo "Summary CSV    : ${SUMMARY_CSV}"
echo "PASS if any row GATE=PASS vs baseline ${BASELINE_GIOU}%"
echo "========================================"
