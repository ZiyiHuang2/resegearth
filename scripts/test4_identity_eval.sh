#!/usr/bin/env bash
set -euo pipefail

# Eval the same merged base used for Test 4 training (identity / step-0 baseline).

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
test4_resolve_base_model

EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${IDENTITY_EVAL_DIR:-${TEST4_OUTPUT_ROOT}/test4-baseline-identity-eval}}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"

echo "========================================"
echo "[Test4] identity eval (baseline, no training)"
echo "MODEL_PATH        : ${MODEL_PATH}"
echo "MODEL_BASE_SOURCE : ${MODEL_BASE_SOURCE}"
echo "EVAL_OUTPUT_DIR   : ${EVAL_OUTPUT_DIR}"
echo "MAX_EVAL_SAMPLES  : ${MAX_EVAL_SAMPLES}"
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
USE_WANDB=False \
"${PYTHON}" "${EVAL_METRICS_SCRIPT}"

TABLE2_JSON="$(test4_find_table2_json "${EVAL_OUTPUT_DIR}")"
if [[ -z "${TABLE2_JSON}" || ! -f "${TABLE2_JSON}" ]]; then
  echo "[ERROR] lasers_test_table2_metrics.json not found after identity eval"
  exit 1
fi

GIOU="$(test4_parse_multi_cate_giou "${TABLE2_JSON}")"
BASELINE_FILE="${BASELINE_FILE:-${EVAL_OUTPUT_DIR}/../test4_baseline_multi_cate_giou.txt}"
mkdir -p "$(dirname "${BASELINE_FILE}")"
echo "${GIOU}" > "${BASELINE_FILE}"

echo "========================================"
echo "[Test4] identity baseline"
echo "Metrics json : ${TABLE2_JSON}"
echo "test_multi_cate gIoU : ${GIOU}%"
echo "Saved baseline to    : ${BASELINE_FILE}"
echo "========================================"
