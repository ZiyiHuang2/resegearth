#!/usr/bin/env bash
set -euo pipefail

########################################
# Test 4 train (segearth+pd)
# Query-side: unfreeze LLM last 2 layers + SEG_projector/predictor,
# freeze pixel_decoder + lm_head, no LoRA, differential lr.
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
export PATH="${CONDA_BIN}:${PATH}"

test4_preflight
if [[ -z "${MODEL_PATH:-}" ]]; then
  test4_resolve_base_model
fi

MAX_STEPS="${MAX_STEPS:-5000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-6}"
OUTPUT_DIR="${OUTPUT_DIR:-${TEST4_OUTPUT_ROOT}/test4-unfreeze-llm2-pd-freeze-${MAX_STEPS}steps}"
MASTER_PORT="${MASTER_PORT:-29510}"
GPU_SLOT="${GPU_SLOT:-localhost:0}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi

echo "========================================"
echo "[Test4] train"
echo "REPO_DIR           : ${REPO_DIR}"
echo "MODEL_PATH         : ${MODEL_PATH}"
echo "MODEL_BASE_SOURCE  : ${MODEL_BASE_SOURCE:-explicit}"
echo "DATA_PATH          : ${DATA_PATH}"
echo "OUTPUT_DIR         : ${OUTPUT_DIR}"
echo "MAX_STEPS          : ${MAX_STEPS}"
echo "SAVE_STEPS         : ${SAVE_STEPS}"
echo "SAVE_TOTAL_LIMIT   : ${SAVE_TOTAL_LIMIT}"
echo "WANDB_MODE         : ${WANDB_MODE}"
echo "========================================"

RESUME_FLAG=()
if compgen -G "${OUTPUT_DIR}/checkpoint-*" > /dev/null; then
  echo "[INFO] existing checkpoints in ${OUTPUT_DIR}; trainer will resume"
fi

"${DEEPSPEED}" --master_port="${MASTER_PORT}" --include "${GPU_SLOT}" \
  segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${DATA_PATH}" \
  --dataset_name lasers \
  --output_dir "${OUTPUT_DIR}" \
  --test4_mode True \
  --no_lora_enable \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --llm_lr 1e-5 \
  --mask_lr 1e-4 \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --bf16 True \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio '1' \
  --switch_bs 4 \
  --logging_steps 10 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --model_max_length 2048 \
  --dataloader_num_workers 4 \
  --report_to none

echo "========================================"
echo "[Test4] training finished"
echo "OUTPUT_DIR : ${OUTPUT_DIR}"
if [[ -f "${OUTPUT_DIR}/test4_train_config.txt" ]]; then
  cat "${OUTPUT_DIR}/test4_train_config.txt"
else
  echo "[WARN] missing test4_train_config.txt"
fi
echo "========================================"
