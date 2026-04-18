#!/usr/bin/env bash
set -euo pipefail

# =========================
# 基础环境
# =========================
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
unset CUDA_VISIBLE_DEVICES

# =========================
# 你需要改的核心实验名
# =========================
EXP_NAME="base+lgce+r4+15k"
WANDB_PROJECT="segearth-train-base-RRSISD"
WANDB_NAME="${EXP_NAME}"

# =========================
# 路径配置
# =========================
ROOT_DIR="/home/wangchengjun/huangziyi/reseg"
CODE_DIR="${ROOT_DIR}/reseagearth-lgce"
DATA_DIR="/home/wangchengjun/huangziyi/data/RRSISD"

MODEL_NAME_OR_PATH="${ROOT_DIR}/pretrained_model/mllm/Mipha-3B"
VISION_TOWER="${ROOT_DIR}/pretrained_model/CLIP/siglip2-so400m-patch14-384"
VISION_TOWER_MASK="${ROOT_DIR}/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"

OUTPUT_ROOT="${ROOT_DIR}/output/RRSISD"
TRAIN_OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
FINAL_CKPT="${TRAIN_OUTPUT_DIR}/checkpoint-15000"
MERGED_MODEL_DIR="${FINAL_CKPT}/merged_model"
EVAL_OUTPUT_DIR="${FINAL_CKPT}/test_results"

# =========================
# 训练配置
# =========================
TRAIN_MASTER_PORT=29000
TRAIN_INCLUDE="localhost:0"

MAX_STEPS=15000
SAVE_STEPS=1500
SAVE_TOTAL_LIMIT=5
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS=10
MODEL_MAX_LENGTH=2048
DATALOADER_NUM_WORKERS=4
LORA_R=4
DATA_RATIO=1
SWITCH_BS=4

USE_LGCE_BRIDGE=True
LGCE_DEBUG=False

# =========================
# 评测配置
# =========================
EVAL_CUDA_VISIBLE_DEVICES=1
EVAL_BATCH_SIZE=1
EVAL_SPLIT="test"
ZIP_RESULTS=False
EVAL_DATASET_NAME="rrsisd"

# =========================
# 进入代码目录
# =========================
cd "${CODE_DIR}"

echo "========================================"
echo "Stage 1/3: Train"
echo "========================================"
export WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_NAME="${WANDB_NAME}"

deepspeed --master_port="${TRAIN_MASTER_PORT}" --include="${TRAIN_INCLUDE}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${DATA_DIR}" \
  --output_dir "${TRAIN_OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --bf16 True \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --logging_steps "${LOGGING_STEPS}" \
  --tf32 False \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing False \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --lora_r "${LORA_R}" \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio "${DATA_RATIO}" \
  --switch_bs "${SWITCH_BS}" \
  --use_lgce_bridge "${USE_LGCE_BRIDGE}" \
  --lgce_debug "${LGCE_DEBUG}" \
  --report_to wandb

echo "========================================"
echo "Stage 2/3: Merge LoRA weights"
echo "========================================"
python segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
  --model_path "${FINAL_CKPT}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --save_path "${MERGED_MODEL_DIR}" \
  --lora_r "${LORA_R}"

echo "========================================"
echo "Stage 3/3: Eval"
echo "========================================"
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
python segearth_r2/eval/eval.py \
  --base_data_path "${DATA_DIR}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --model_path "${MERGED_MODEL_DIR}" \
  --output_dir "${EVAL_OUTPUT_DIR}" \
  --dataset_name "${EVAL_DATASET_NAME}" \
  --split "${EVAL_SPLIT}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --zip_results "${ZIP_RESULTS}"

echo "========================================"
echo "All done."
echo "Train output: ${TRAIN_OUTPUT_DIR}"
echo "Merged model: ${MERGED_MODEL_DIR}"
echo "Eval output:  ${EVAL_OUTPUT_DIR}"
echo "========================================"