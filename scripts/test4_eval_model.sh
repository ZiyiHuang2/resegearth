#!/usr/bin/env bash
set -euo pipefail

# Eval any merged HF model dir on LaSeRS test (+ metrics + optional gate).

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=test4_common.sh
source "${SCRIPT_DIR}/test4_common.sh"

REPO_DIR="${REPO_DIR}"
cd "${REPO_DIR}"
export PATH="${CONDA_BIN}:${PATH}"

test4_preflight

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to merged model directory}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:?Set EVAL_OUTPUT_DIR}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"

if [[ -f "${BASELINE_FILE:-}" ]]; then
  BASELINE_GIOU="$(test4_read_baseline_giou "${BASELINE_FILE}")"
elif [[ -n "${BASELINE_GIOU:-}" ]]; then
  :
else
  BASELINE_GIOU="${BASELINE_GIOU:-42.45}"
  echo "[WARN] Using default BASELINE_GIOU=${BASELINE_GIOU}% (set BASELINE_FILE to base lasers_test_table2_metrics.json)"
fi

echo "========================================"
echo "[Test4] eval model"
echo "MODEL_PATH      : ${MODEL_PATH}"
echo "EVAL_OUTPUT_DIR : ${EVAL_OUTPUT_DIR}"
echo "BASELINE_GIOU   : ${BASELINE_GIOU}%"
echo "========================================"

rm -rf "${EVAL_OUTPUT_DIR}"
mkdir -p "${EVAL_OUTPUT_DIR}"

EVAL_EXTRA=()
if [[ "${MAX_EVAL_SAMPLES}" -gt 0 ]]; then
  EVAL_EXTRA+=(--max_eval_samples "${MAX_EVAL_SAMPLES}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON}" segearth_r2/eval/eval.py \
  --base_data_path "${DATA_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${EVAL_OUTPUT_DIR}" \
  --dataset_name lasers \
  --split test \
  --eval_batch_size 1 \
  --zip_results False \
  "${EVAL_EXTRA[@]}"

DATASET_TYPE=lasers \
BASE_DATA_PATH="${DATA_PATH}" \
SPLIT=test \
PRED_DIR="${EVAL_OUTPUT_DIR}" \
USE_WANDB="${USE_WANDB:-False}" \
WANDB_PROJECT="${WANDB_PROJECT:-segearth-eval-tcpd}" \
WANDB_RUN_NAME="${WANDB_RUN_NAME:-test4-eval}" \
"${PYTHON}" "${EVAL_METRICS_SCRIPT}"

TABLE2_JSON="$(test4_find_table2_json "${EVAL_OUTPUT_DIR}")"
GIOU="$(test4_parse_multi_cate_giou "${TABLE2_JSON}")"

echo "Metrics json : ${TABLE2_JSON}"
echo "test_multi_cate gIoU : ${GIOU}%"
test4_print_gate "${BASELINE_GIOU}" "${GIOU}"

SUMMARY_FILE="${SUMMARY_FILE:-${EVAL_OUTPUT_DIR}/test4_eval_summary.txt}"
{
  echo "model_path=${MODEL_PATH}"
  echo "eval_output_dir=${EVAL_OUTPUT_DIR}"
  echo "metrics_json=${TABLE2_JSON}"
  echo "baseline_giou=${BASELINE_GIOU}"
  echo "result_giou=${GIOU}"
  echo "delta=$(python3 - <<PY
b=float("${BASELINE_GIOU}"); g=float("${GIOU}"); print(f"{g-b:+.4f}")
PY
)"
} > "${SUMMARY_FILE}"
echo "Summary saved: ${SUMMARY_FILE}"
