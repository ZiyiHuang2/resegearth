#!/usr/bin/env bash
# Test4 + TCPD: intentionally unfreezes LLM last-2, predictor, SEG_projector (Test4 settings) plus TCPD.
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
OUTPUT_DIR="${OUTPUT_DIR:-${TEST4_OUTPUT_ROOT}/test4-plus-tcpd-${MAX_STEPS}steps}"

echo "========================================"
echo "[TCPD] Test4 + TCPD train"
echo "OUTPUT_DIR : ${OUTPUT_DIR}"
echo "test4_mode : True"
echo "use_tcpd   : True"
echo "tcpd_train : True"
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
  --test4_mode True \
  --no_lora_enable \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 2 \
  --save_strategy steps \
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
