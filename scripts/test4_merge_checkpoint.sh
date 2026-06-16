#!/usr/bin/env bash
set -euo pipefail

# Export DeepSpeed checkpoint to merged HF dir (no eval).

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=test4_common.sh
source "${SCRIPT_DIR}/test4_common.sh"

REPO_DIR="${REPO_DIR}"
cd "${REPO_DIR}"
export PATH="${CONDA_BIN}:${PATH}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH}"
MERGED_DIR="${MERGED_DIR:?Set MERGED_DIR}"
LORA_ENABLE="${LORA_ENABLE:-False}"

test4_preflight
test4_resolve_base_model
SPOT_CHECK_BASE="${SPOT_CHECK_BASE:-${MODEL_PATH}}"

mkdir -p "${MERGED_DIR}"

MERGE_ARGS=(
  --model_path "${CHECKPOINT_PATH}"
  --vision_tower "${VISION_TOWER}"
  --vision_tower_mask "${VISION_TOWER_MASK}"
  --mask_config "${MASK_CONFIG}"
  --save_path "${MERGED_DIR}"
  --spot_check_key "${SPOT_CHECK_KEY}"
  --spot_check_base "${SPOT_CHECK_BASE}"
)
if [[ "${LORA_ENABLE}" == "True" || "${LORA_ENABLE}" == "true" ]]; then
  MERGE_ARGS+=(--lora_enable)
else
  MERGE_ARGS+=(--no-lora_enable)
fi
if [[ -n "${USE_TCPD:-}" ]]; then
  if [[ "${USE_TCPD}" == "True" || "${USE_TCPD}" == "true" || "${USE_TCPD}" == "1" ]]; then
    MERGE_ARGS+=(--use_tcpd)
  else
    MERGE_ARGS+=(--no-use_tcpd)
  fi
fi
if [[ -n "${TCPD_CONDITION_SOURCE:-}" ]]; then
  MERGE_ARGS+=(--tcpd_condition_source "${TCPD_CONDITION_SOURCE}")
fi
if [[ -n "${TCPD_SPATIAL_MODE:-}" ]]; then
  MERGE_ARGS+=(--tcpd_spatial_mode "${TCPD_SPATIAL_MODE}")
fi
if [[ -n "${TCPD_CONDITION_MSDEFORM:-}" ]]; then
  if [[ "${TCPD_CONDITION_MSDEFORM}" == "True" || "${TCPD_CONDITION_MSDEFORM}" == "true" || "${TCPD_CONDITION_MSDEFORM}" == "1" ]]; then
    MERGE_ARGS+=(--tcpd_condition_msdeform)
  else
    MERGE_ARGS+=(--no-tcpd_condition_msdeform)
  fi
fi
if [[ -n "${TCPD_CONDITION_FPN:-}" ]]; then
  if [[ "${TCPD_CONDITION_FPN}" == "True" || "${TCPD_CONDITION_FPN}" == "true" || "${TCPD_CONDITION_FPN}" == "1" ]]; then
    MERGE_ARGS+=(--tcpd_condition_fpn)
  else
    MERGE_ARGS+=(--no-tcpd_condition_fpn)
  fi
fi
if [[ -n "${TCPD_CONDITION_OUTPUT_SCALE:-}" ]]; then
  if [[ "${TCPD_CONDITION_OUTPUT_SCALE}" == "True" || "${TCPD_CONDITION_OUTPUT_SCALE}" == "true" || "${TCPD_CONDITION_OUTPUT_SCALE}" == "1" ]]; then
    MERGE_ARGS+=(--tcpd_condition_output_scale)
  else
    MERGE_ARGS+=(--no-tcpd_condition_output_scale)
  fi
fi
if [[ -n "${SPOT_CHECK_BASE:-}" && -d "${SPOT_CHECK_BASE}" ]]; then
  MERGE_ARGS+=(--init_model_path "${SPOT_CHECK_BASE}")
fi

echo "[Test4] merge ${CHECKPOINT_PATH} -> ${MERGED_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON}" \
  segearth_r2/train/merge_lora_weights_and_save_hf_model.py "${MERGE_ARGS[@]}"

if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  echo "[ERROR] merge failed: missing ${MERGED_DIR}/config.json"
  exit 1
fi
echo "[Test4] merge OK: ${MERGED_DIR}"
