#!/usr/bin/env bash
set -euo pipefail

########################################
# Test 4 optional Probe E wrapper
########################################

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=test4_common.sh
source "${SCRIPT_DIR}/test4_common.sh"

export PATH="${CONDA_BIN}:${PATH}"

MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${TEST4_OUTPUT_ROOT}/test4-probe-e}"
SAMPLE_LIST_DEFAULT="${RESEG_ROOT}/output/base/standard-base-lasers-siglip1-8w-gd4/feature_probe_stage3/sample_list.json"
SAMPLE_LIST_FALLBACK="${RESEG_ROOT}/output/base/standard-base-lasers-siglip1-8w-gd4/feature_probe_v1/sample_list.json"
SAMPLE_LIST="${SAMPLE_LIST:-${SAMPLE_LIST_DEFAULT}}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "[ERROR] Set MODEL_PATH to a Test 4 checkpoint or export dir under ${TEST4_OUTPUT_ROOT}/"
  exit 1
fi

if [[ ! -f "${SAMPLE_LIST}" && -f "${SAMPLE_LIST_FALLBACK}" ]]; then
  echo "[WARN] sample_list not found at ${SAMPLE_LIST}; using ${SAMPLE_LIST_FALLBACK}"
  SAMPLE_LIST="${SAMPLE_LIST_FALLBACK}"
fi

if [[ ! -f "${SAMPLE_LIST}" ]]; then
  echo "[ERROR] sample_list.json not found."
  echo "  tried: ${SAMPLE_LIST_DEFAULT}"
  echo "  tried: ${SAMPLE_LIST_FALLBACK}"
  echo "Set SAMPLE_LIST=/path/to/sample_list.json"
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi
if [[ ! -d "${DATA_PATH}" ]]; then
  echo "[ERROR] DATA_PATH not found: ${DATA_PATH}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[Probe E] MODEL_PATH  : ${MODEL_PATH}"
echo "[Probe E] OUTPUT_DIR  : ${OUTPUT_DIR}"
echo "[Probe E] SAMPLE_LIST : ${SAMPLE_LIST}"
echo "[Probe E] feature dump -> ${OUTPUT_DIR}/features"
"${PYTHON}" "${RESEG_ROOT}/temp_feature_dump_stage3.py" \
  --dump_features \
  --data_path "${DATA_PATH}" \
  --model_path "${MODEL_PATH}" \
  --sample_list "${SAMPLE_LIST}" \
  --output_dir "${OUTPUT_DIR}/features"

echo "[Probe E] analyze -> ${OUTPUT_DIR}/analysis"
"${PYTHON}" "${RESEG_ROOT}/temp_feature_analyze_probe_e.py" \
  --feature_dir "${OUTPUT_DIR}/features" \
  --output_dir "${OUTPUT_DIR}/analysis"

echo "[Probe E] done. See ${OUTPUT_DIR}/analysis"
