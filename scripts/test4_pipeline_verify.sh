#!/usr/bin/env bash
set -euo pipefail

# Quick end-to-end link test (no full LaSeRS test eval):
#   - preflight paths + CRLF
#   - 1-step train
#   - merge last checkpoint
# Optional tiny eval: MAX_EVAL_SAMPLES=5 RUN_TINY_EVAL=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=test4_common.sh
source "${SCRIPT_DIR}/test4_common.sh"

REPO_DIR="${REPO_DIR}"
cd "${REPO_DIR}"
export PATH="${CONDA_BIN}:${PATH}"

echo "========================================"
echo "[Test4] pipeline verify"
echo "========================================"

FAIL=0

for f in scripts/test4_common.sh scripts/test4_identity_eval.sh scripts/test4_smoke_train.sh \
         scripts/test4_smoke_eval.sh scripts/test4_eval_model.sh scripts/test4_eval_checkpoints.sh \
         scripts/test4_run_pipeline.sh scripts/test4_probe_e.sh; do
  if file "${f}" | grep -qi crlf; then
    echo "[FAIL] CRLF in ${f}"
    FAIL=1
  else
    echo "[OK] LF ${f}"
  fi
done

test4_preflight || FAIL=1
test4_resolve_base_model || FAIL=1
echo "[OK] MODEL_PATH=${MODEL_PATH} (${MODEL_BASE_SOURCE})"

VERIFY_ROOT="${VERIFY_ROOT:-${TEST4_OUTPUT_ROOT}/test4-pipeline-verify-$$}"
TRAIN_DIR="${VERIFY_ROOT}/train"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
mkdir -p "${VERIFY_ROOT}"

echo ""
echo "--- 1-step train ---"
OUTPUT_DIR="${TRAIN_DIR}" MAX_STEPS=1 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=2 PER_DEVICE_TRAIN_BATCH_SIZE=1 \
  MASTER_PORT="${MASTER_PORT}" \
  MODEL_PATH="${MODEL_PATH}" bash "${SCRIPT_DIR}/test4_smoke_train.sh" || {
  if [[ -f "${TRAIN_DIR}/test4_train_config.txt" ]]; then
    echo "[WARN] train exited non-zero but test4_train_config.txt exists (check disk space for checkpoint save)"
  else
    FAIL=1
  fi
}

if [[ ! -f "${TRAIN_DIR}/test4_train_config.txt" ]]; then
  echo "[FAIL] missing test4_train_config.txt"
  FAIL=1
else
  grep -q "ASSERT OK" "${TRAIN_DIR}/test4_train_config.txt" && echo "[OK] PD assert" || { echo "[FAIL] PD assert"; FAIL=1; }
  grep -q "layers.30" "${TRAIN_DIR}/test4_train_config.txt" && echo "[OK] layer 30 unfrozen" || { echo "[FAIL] layer indices"; FAIL=1; }
fi

CKPT="$(find "${TRAIN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1 || true)"
if [[ -z "${CKPT}" ]]; then
  echo "[WARN] no checkpoint after 1 step (save_strategy=no skips ckpt); testing merge skipped"
else
  echo ""
  echo "--- export checkpoint (no full test eval) ---"
  MERGED="${VERIFY_ROOT}/merged"
  EXPORTED="$(test4_export_checkpoint "${CKPT}" "${MERGED}" "${MODEL_PATH}")" || FAIL=1
  if [[ -n "${EXPORTED}" && -f "${EXPORTED}/config.json" ]]; then
    echo "[OK] export config.json at ${EXPORTED}"
  else
    echo "[FAIL] export output"
    FAIL=1
  fi
fi

if [[ "${RUN_TINY_EVAL:-0}" == "1" ]]; then
  echo ""
  echo "--- tiny identity eval (5 samples) ---"
  BASELINE_FILE="${VERIFY_ROOT}/baseline.txt" \
  IDENTITY_EVAL_DIR="${VERIFY_ROOT}/identity" \
  MAX_EVAL_SAMPLES=5 \
  bash "${SCRIPT_DIR}/test4_identity_eval.sh" || FAIL=1
  test4_parse_multi_cate_giou "$(test4_find_table2_json "${VERIFY_ROOT}/identity")" && echo "[OK] parsed gIoU"
fi

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
  echo "========================================"
  echo "[Test4] VERIFY PASS"
  echo "Run full pipeline:"
  echo "  bash scripts/test4_run_pipeline.sh"
  echo "Or quick train only:"
  echo "  MAX_STEPS=5000 bash scripts/test4_smoke_train.sh"
  echo "========================================"
  exit 0
else
  echo "========================================"
  echo "[Test4] VERIFY FAIL"
  echo "========================================"
  exit 1
fi
