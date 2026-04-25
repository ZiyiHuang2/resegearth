#!/usr/bin/env bash
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT=segearth-bseg
export WANDB_NAME=bseg-refined-structured-7w-resume
export WANDB_INIT_TIMEOUT=300
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="localhost:2"
MASTER_PORT="29302"

########################################
# Project dir
########################################
REPO_DIR="/home/wangchengjun/huangziyi/reseg/resegearth+bseg"
cd "${REPO_DIR}"

########################################
# Common paths
########################################
MODEL_NAME_OR_PATH="/home/wangchengjun/huangziyi/reseg/output/standard-base/merged_model"
VISION_TOWER="/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384"
VISION_TOWER_MASK="/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"

########################################
# Dataset config
########################################
BASE_DATA_PATH="/home/wangchengjun/huangziyi/data/RRSISD"
DATASET_NAME="rrsisd"

########################################
# Resume target
########################################
OUTPUT_DIR="/home/wangchengjun/huangziyi/reseg/output/bseg/refined_structured_standard-base_5w"
RESUME_CHECKPOINT="${OUTPUT_DIR}/checkpoint-50000"

########################################
# Refined structured config
########################################
TARGET_LAYERS="24,25,26,27,28"
TARGET_HEADS_CONFIG_PATH="/home/wangchengjun/huangziyi/reseg/resegearth+bseg/configs/target_heads_rrsisd.json"
TOP_K_HEADS="4"

ATTENTION_LOSS_WEIGHT="0.005"
STRUCTURED_FG_BG_WEIGHT="0.8"
STRUCTURED_BOUNDARY_OUTER_WEIGHT="0.2"
STRUCTURED_ATTENTION_MARGIN="0.05"

SMALL_WEIGHT="1.5"
SMALL_AREA_RATIO_THRESHOLD="0.03"

########################################
# Train config
########################################
MAX_STEPS="70000"
PER_DEVICE_TRAIN_BATCH_SIZE="1"
GRADIENT_ACCUMULATION_STEPS="1"

SAVE_STEPS="5000"
SAVE_TOTAL_LIMIT="6"

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
DATA_RATIO="1"
SWITCH_BS="4"

SEED="42"
DATA_SEED="42"
MAX_GRAD_NORM="1.0"

########################################
# Preflight
########################################
echo "========================================"
echo "[0/2] Preflight checks"
echo "========================================"

if [[ ! -d "${RESUME_CHECKPOINT}" ]]; then
  echo "[ERROR] resume checkpoint not found:"
  echo "        ${RESUME_CHECKPOINT}"
  exit 1
fi

if [[ ! -f "${TARGET_HEADS_CONFIG_PATH}" ]]; then
  echo "[ERROR] target heads config not found:"
  echo "        ${TARGET_HEADS_CONFIG_PATH}"
  exit 1
fi

echo "[OK] resume checkpoint: ${RESUME_CHECKPOINT}"
echo "[OK] output dir       : ${OUTPUT_DIR}"
echo "[OK] target layers    : ${TARGET_LAYERS}"
echo "[OK] target heads cfg : ${TARGET_HEADS_CONFIG_PATH}"
echo "[OK] max_steps        : ${MAX_STEPS}"

########################################
# Resume training
########################################
echo "========================================"
echo "[1/2] Resume refined structured from 50k to 70k"
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
  --use_attention_loss True \
  --use_structured_attention_loss True \
  --use_precomputed_structured_maps False \
  --target_layers "${TARGET_LAYERS}" \
  --target_heads_config_path "${TARGET_HEADS_CONFIG_PATH}" \
  --top_k_heads "${TOP_K_HEADS}" \
  --attention_loss_weight "${ATTENTION_LOSS_WEIGHT}" \
  --structured_fg_bg_weight "${STRUCTURED_FG_BG_WEIGHT}" \
  --structured_boundary_outer_weight "${STRUCTURED_BOUNDARY_OUTER_WEIGHT}" \
  --structured_attention_margin "${STRUCTURED_ATTENTION_MARGIN}" \
  --small_weight "${SMALL_WEIGHT}" \
  --small_area_ratio_threshold "${SMALL_AREA_RATIO_THRESHOLD}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --resume_from_checkpoint "${RESUME_CHECKPOINT}" \
  --report_to wandb

########################################
# Done
########################################
echo "========================================"
echo "[2/2] DONE"
echo "Resumed from : ${RESUME_CHECKPOINT}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "Target steps : ${MAX_STEPS}"
echo "Now evaluate checkpoint-55000 / 60000 / 65000 / 70000 on test"
echo "========================================"