#!/usr/bin/env bash
set -euo pipefail

# Merge a DeepSpeed checkpoint and eval on LaSeRS test.

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=test4_common.sh
source "${SCRIPT_DIR}/test4_common.sh"

REPO_DIR="${REPO_DIR}"
cd "${REPO_DIR}"
export PATH="${CONDA_BIN}:${PATH}"

test4_preflight

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${1:-}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${2:-}}"

if [[ -z "${TRAIN_OUTPUT_DIR}" ]]; then
  echo "Usage: TRAIN_OUTPUT_DIR=/path/to/run [CHECKPOINT_PATH=/path/to/checkpoint-N] bash scripts/test4_smoke_eval.sh"
  exit 1
fi

MERGED_DIR="${MERGED_DIR:-${TRAIN_OUTPUT_DIR}/merged_model}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${TRAIN_OUTPUT_DIR}/lasers_test_results}"
LORA_ENABLE="${LORA_ENABLE:-False}"

if [[ -f "${BASELINE_FILE:-}" ]]; then
  BASELINE_GIOU="$(test4_read_baseline_giou "${BASELINE_FILE}")"
elif [[ -n "${BASELINE_GIOU:-}" ]]; then
  :
else
  BASELINE_GIOU="${BASELINE_GIOU:-42.45}"
fi

test4_resolve_base_model
SPOT_CHECK_BASE="${SPOT_CHECK_BASE:-${MODEL_PATH}}"

read_last_checkpoint() {
  TRAIN_OUTPUT_DIR="${1}" "${PYTHON}" - <<'PY'
import os, re, sys
output_dir = os.environ["TRAIN_OUTPUT_DIR"]
candidates = []
for name in os.listdir(output_dir):
    m = re.match(r"checkpoint-(\d+)$", name)
    if m:
        candidates.append((int(m.group(1)), os.path.join(output_dir, name)))
if not candidates:
    sys.exit(1)
candidates.sort()
print(candidates[-1][1])
PY
}

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  if ! CHECKPOINT_PATH="$(read_last_checkpoint "${TRAIN_OUTPUT_DIR}")"; then
    echo "[ERROR] no checkpoint under ${TRAIN_OUTPUT_DIR}"
    exit 1
  fi
  echo "[INFO] using last checkpoint: ${CHECKPOINT_PATH}"
fi

echo "========================================"
echo "[Test4] merge + eval"
echo "CHECKPOINT     : ${CHECKPOINT_PATH}"
echo "MERGED_DIR     : ${MERGED_DIR}"
echo "BASELINE_GIOU  : ${BASELINE_GIOU}%"
echo "SPOT_CHECK_BASE: ${SPOT_CHECK_BASE}"
echo "========================================"

echo "[1/2] export checkpoint"
EXPORTED="$(test4_export_checkpoint "${CHECKPOINT_PATH}" "${MERGED_DIR}" "${SPOT_CHECK_BASE}")"
if [[ ! -d "${EXPORTED}" ]]; then
  echo "[ERROR] export failed"
  exit 1
fi
# Use export dir (HF ckpt in-place, or merged output)
EVAL_MODEL_PATH="${EXPORTED}"

echo "[2/2] test eval"
MODEL_PATH="${EVAL_MODEL_PATH}" \
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
BASELINE_GIOU="${BASELINE_GIOU}" \
BASELINE_FILE="${BASELINE_FILE:-}" \
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}" \
SUMMARY_FILE="${TRAIN_OUTPUT_DIR}/test4_eval_summary.txt" \
bash "${SCRIPT_DIR}/test4_eval_model.sh"
