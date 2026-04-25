#!/usr/bin/env bash
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT=segearth-standard
export WANDB_NAME=base-siglip1
export WANDB_INIT_TIMEOUT=300
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="localhost:1"
GPU_ID="1"
MASTER_PORT="29500"

########################################
# Project dir
########################################
REPO_DIR="/home/wangchengjun/huangziyi/reseg/SegearthR2"
cd "${REPO_DIR}"

########################################
# Common paths
########################################
MODEL_NAME_OR_PATH="/home/wangchengjun/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
VISION_TOWER="/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"

########################################
# Dataset config
########################################
BASE_DATA_PATH="/home/wangchengjun/huangziyi/data/RRSISD"
DATASET_NAME="rrsisd"
TEST_SPLIT="test"

########################################
# Output
########################################
OUTPUT_DIR="/home/wangchengjun/huangziyi/reseg/output/standard-base-siglip1"
MERGED_DIR="${OUTPUT_DIR}/merged_model"
TEST_OUTPUT_DIR="${OUTPUT_DIR}/test_results"

########################################
# Train config
########################################
MAX_STEPS="50000"
PER_DEVICE_TRAIN_BATCH_SIZE="1"
GRADIENT_ACCUMULATION_STEPS="1"

SAVE_STEPS="10000"
SAVE_TOTAL_LIMIT="2"

LEARNING_RATE="1e-4"
WEIGHT_DECAY="0.0"
WARMUP_RATIO="0.03"
LR_SCHEDULER_TYPE="cosine"

LOGGING_STEPS="10"
BF16="True"
TF32="False"
MODEL_MAX_LENGTH="2048"
GRADIENT_CHECKPOINTING="False"
DATALOADER_NUM_WORKERS="4"

LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"

DATA_RATIO="1"
SWITCH_BS="4"

SEED="42"
DATA_SEED="42"

########################################
# 1) Train
########################################
echo "========================================"
echo "[1/4] Training (includes val + best ckpt)"
echo "========================================"

deepspeed --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --bf16 "${BF16}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --logging_steps "${LOGGING_STEPS}" \
  --tf32 "${TF32}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --lora_r "${LORA_R}" \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio "${DATA_RATIO}" \
  --switch_bs "${SWITCH_BS}" \
  --seed "${SEED}" \
  --data_seed "${DATA_SEED}" \
  --report_to wandb

########################################
# 2) Read best checkpoint from trainer_state.json
########################################
echo "========================================"
echo "[2/4] Reading best checkpoint"
echo "========================================"

export OUTPUT_DIR
BEST_CHECKPOINT=$(python - <<'PY'
import json
import os
import sys

output_dir = os.environ["OUTPUT_DIR"]
trainer_state = os.path.join(output_dir, "trainer_state.json")

if not os.path.exists(trainer_state):
    print("")
    sys.exit(0)

with open(trainer_state, "r", encoding="utf-8") as f:
    state = json.load(f)

print(state.get("best_model_checkpoint", ""))
PY
)

if [[ -z "${BEST_CHECKPOINT}" ]]; then
  echo "[ERROR] best_model_checkpoint not found in ${OUTPUT_DIR}/trainer_state.json"
  exit 1
fi

echo "[OK] BEST_CHECKPOINT=${BEST_CHECKPOINT}"

########################################
# 3) Merge best checkpoint only
########################################
echo "========================================"
echo "[3/4] Merging best checkpoint"
echo "========================================"

rm -rf "${MERGED_DIR}"
mkdir -p "${MERGED_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
  --model_path "${BEST_CHECKPOINT}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --save_path "${MERGED_DIR}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}"

########################################
# 4) Test merged best model
########################################
echo "========================================"
echo "[4/4] Testing merged best model"
echo "========================================"

mkdir -p "${TEST_OUTPUT_DIR}"

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python segearth_r2/eval/eval.py \
  --base_data_path "${BASE_DATA_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --model_path "${MERGED_DIR}" \
  --output_dir "${TEST_OUTPUT_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --split "${TEST_SPLIT}" \
  --eval_batch_size 1 \
  --zip_results False

echo "========================================"
echo "[DONE]"
echo "Best checkpoint : ${BEST_CHECKPOINT}"
echo "Merged model    : ${MERGED_DIR}"
echo "Test outputs    : ${TEST_OUTPUT_DIR}"
echo "========================================"