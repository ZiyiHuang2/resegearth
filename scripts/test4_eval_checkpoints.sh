#!/usr/bin/env bash
set -euo pipefail

# Eval selected checkpoint-* steps under TRAIN_OUTPUT_DIR (default: 2000,5000).

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=test4_common.sh
source "${SCRIPT_DIR}/test4_common.sh"

REPO_DIR="${REPO_DIR}"
cd "${REPO_DIR}"
export PATH="${CONDA_BIN}:${PATH}"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:?Set TRAIN_OUTPUT_DIR}"
BASELINE_FILE="${BASELINE_FILE:-${TRAIN_OUTPUT_DIR}/../test4_baseline_multi_cate_giou.txt}"
EVAL_CHECKPOINT_STEPS="${EVAL_CHECKPOINT_STEPS:-2000,5000}"

if [[ ! -f "${BASELINE_FILE}" ]]; then
  echo "[WARN] baseline file missing: ${BASELINE_FILE}"
  echo "[WARN] Set BASELINE_FILE or run pipeline Step 0"
fi

SUMMARY_CSV="${SUMMARY_CSV:-${TRAIN_OUTPUT_DIR}/test4_checkpoint_eval_summary.csv}"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "checkpoint_step,baseline_giou,result_giou,delta,gate" > "${SUMMARY_CSV}"
fi

IFS=',' read -ra STEPS <<< "${EVAL_CHECKPOINT_STEPS}"
CHECKPOINTS=()
for STEP in "${STEPS[@]}"; do
  STEP="$(echo "${STEP}" | tr -d '[:space:]')"
  [[ -z "${STEP}" ]] && continue
  CKPT="${TRAIN_OUTPUT_DIR}/checkpoint-${STEP}"
  if [[ -d "${CKPT}" ]]; then
    CHECKPOINTS+=("${CKPT}")
  else
    echo "[WARN] checkpoint-${STEP} not found under ${TRAIN_OUTPUT_DIR}"
  fi
done

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "[ERROR] no matching checkpoints for EVAL_CHECKPOINT_STEPS=${EVAL_CHECKPOINT_STEPS}"
  exit 1
fi

echo "========================================"
echo "[Test4] eval checkpoints (n=${#CHECKPOINTS[@]}, steps=${EVAL_CHECKPOINT_STEPS})"
echo "TRAIN_OUTPUT_DIR : ${TRAIN_OUTPUT_DIR}"
echo "BASELINE_FILE    : ${BASELINE_FILE}"
echo "SUMMARY_CSV      : ${SUMMARY_CSV}"
echo "========================================"

for CKPT in "${CHECKPOINTS[@]}"; do
  STEP="$(basename "${CKPT}" | sed 's/checkpoint-//')"
  MERGED="${TRAIN_OUTPUT_DIR}/merged_ckpt-${STEP}"
  EVAL_OUT="${TRAIN_OUTPUT_DIR}/eval_ckpt-${STEP}"

  if grep -q "^${STEP}," "${SUMMARY_CSV}" 2>/dev/null; then
    echo "[SKIP] checkpoint-${STEP} already in ${SUMMARY_CSV}"
    continue
  fi

  echo ""
  echo "--- checkpoint step ${STEP} ---"
  CHECKPOINT_PATH="${CKPT}" \
  MERGED_DIR="${MERGED}" \
  EVAL_OUTPUT_DIR="${EVAL_OUT}" \
  BASELINE_FILE="${BASELINE_FILE}" \
  MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}" \
  bash "${SCRIPT_DIR}/test4_smoke_eval.sh" || {
    echo "[WARN] eval failed for ${CKPT}"
    continue
  }

  GIOU="$(test4_parse_multi_cate_giou "$(test4_find_table2_json "${EVAL_OUT}")")"
  BASELINE="$(test4_read_baseline_giou "${BASELINE_FILE}")"
  GATE="$(test4_gate_label "${BASELINE}" "${GIOU}")"
  echo "${STEP},${BASELINE},${GIOU},$(python3 -c "print(${GIOU}-${BASELINE})"),${GATE}" >> "${SUMMARY_CSV}"
done

echo ""
echo "========================================"
echo "[Test4] checkpoint eval done"
cat "${SUMMARY_CSV}"
echo "========================================"
