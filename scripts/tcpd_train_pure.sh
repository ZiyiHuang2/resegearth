#!/usr/bin/env bash
# Pure TCPD-only train: freeze LLM, predictor, SEG_projector, lm_head, base pixel_decoder.
# Trains only *tcpd* params inside pixel_decoder.
set -euo pipefail

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
export PATH="${CONDA_BIN}:${PATH}"

test4_preflight
if [[ -z "${MODEL_PATH:-}" ]]; then
  test4_resolve_base_model
fi

MAX_STEPS="${MAX_STEPS:-500}"
OUTPUT_DIR="${OUTPUT_DIR:-${TEST4_OUTPUT_ROOT}/tcpd-only-seg-${MAX_STEPS}steps}"
# Smoke (MAX_STEPS=1): skip ZeRO checkpoint gather — training step is ~4s, save can take 10+ min.
if [[ "${MAX_STEPS}" == "1" && -z "${SAVE_STRATEGY:-}" ]]; then
  SAVE_STRATEGY="no"
else
  SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
fi

echo "========================================"
echo "[TCPD] pure TCPD-only train"
echo "OUTPUT_DIR : ${OUTPUT_DIR}"
echo "use_tcpd   : True"
echo "tcpd_train : True (unfreeze *tcpd* inside frozen PD)"
echo "freeze     : LLM, predictor, SEG_projector, lm_head, base PD"
echo "========================================"

"${DEEPSPEED}" --master_port="${MASTER_PORT:-29520}" --include "${GPU_SLOT:-localhost:0}" \
  segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${DATA_PATH}" \
  --dataset_name lasers \
  --output_dir "${OUTPUT_DIR}" \
  --use_tcpd True \
  --tcpd_train True \
  --tcpd_spatial_mode "${TCPD_SPATIAL_MODE:-spatial}" \
  --tcpd_condition_msdeform "${TCPD_CONDITION_MSDEFORM:-True}" \
  --tcpd_condition_fpn "${TCPD_CONDITION_FPN:-True}" \
  --tcpd_condition_output_scale "${TCPD_CONDITION_OUTPUT_SCALE:-True}" \
  --freeze_pixel_decoder True \
  --freeze_predictor True \
  --freeze_seg_projector True \
  --freeze_lm_head True \
  --unfreeze_llm_last_n 0 \
  --no_lora_enable \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 2 \
  --save_strategy "${SAVE_STRATEGY}" \
  --save_steps "${MAX_STEPS}" \
  --save_total_limit 1 \
  --bf16 True \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio '1' \
  --switch_bs 4 \
  --logging_steps 10 \
  --report_to "${REPORT_TO:-none}"

echo "[TCPD] finished -> ${OUTPUT_DIR}"
