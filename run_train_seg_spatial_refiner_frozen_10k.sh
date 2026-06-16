#!/usr/bin/env bash
# Stage 1: Frozen-base Single-Query SEG Spatial Refiner (10k steps).
# Do NOT run automatically — use smoke commands first.
set -euo pipefail

REPO_DIR="/root/rivermind-data/huangziyi/reseg/resegearth+tgi"
cd "${REPO_DIR}"
PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
export PYTHONPATH="detectron2:${PYTHONPATH:-}"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-tgi}"
export WANDB_NAME="${WANDB_NAME:-seg-refiner-frozen-10k}"
export WANDB_INIT_TIMEOUT=300
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="${GPU_SLOT:-localhost:0}"
GPU_ID="${GPU_ID:-0}"
MASTER_PORT="${MASTER_PORT:-29551}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/root/rivermind-data/huangziyi/reseg/output/base/standard-base-siglip1-28w-gd4/merged_model}"
VISION_TOWER="${VISION_TOWER:-/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"

BASE_DATA_PATH="${BASE_DATA_PATH:-/root/rivermind-data/huangziyi/data/RRSISD}"
DATASET_NAME="${DATASET_NAME:-rrsisd}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/rivermind-data/huangziyi/reseg/output/tgi/seg_refiner_frozen_10k}"

MAX_STEPS="${MAX_STEPS:-10000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LOGGING_STEPS="${LOGGING_STEPS:-50}"
BF16="${BF16:-True}"
FP16="${FP16:-False}"
TF32="${TF32:-False}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_ENABLE="${LORA_ENABLE:-False}"
DATA_RATIO="${DATA_RATIO:-1}"
SWITCH_BS="${SWITCH_BS:-4}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"

USE_SEG_SPATIAL_REFINER="${USE_SEG_SPATIAL_REFINER:-True}"
TRAIN_SEG_SPATIAL_REFINER_ONLY="${TRAIN_SEG_SPATIAL_REFINER_ONLY:-True}"
SEG_SPATIAL_REFINER_ALPHA="${SEG_SPATIAL_REFINER_ALPHA:-0.1}"
SEG_SPATIAL_REFINER_LOSS_WEIGHT="${SEG_SPATIAL_REFINER_LOSS_WEIGHT:-0.1}"
SEG_SPATIAL_REFINER_DICE_WEIGHT="${SEG_SPATIAL_REFINER_DICE_WEIGHT:-1.0}"
SEG_SPATIAL_REFINER_BCE_WEIGHT="${SEG_SPATIAL_REFINER_BCE_WEIGHT:-1.0}"

USE_QUERY_AWARE_DECODER_BIAS="${USE_QUERY_AWARE_DECODER_BIAS:-False}"
USE_DECODER_ATTN_BIAS="${USE_DECODER_ATTN_BIAS:-False}"
USE_MSTVA="${USE_MSTVA:-False}"
USE_MSTVA_LOSS="${USE_MSTVA_LOSS:-False}"
USE_TEXT_FILM="${USE_TEXT_FILM:-False}"
USE_ATTENTION_LOSS="${USE_ATTENTION_LOSS:-False}"
EVALUATION_STRATEGY="${EVALUATION_STRATEGY:-no}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-False}"

mkdir -p "${OUTPUT_DIR}"

echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] MAX_STEPS=${MAX_STEPS}"
echo "[INFO] USE_SEG_SPATIAL_REFINER=${USE_SEG_SPATIAL_REFINER}"
echo "[INFO] TRAIN_SEG_SPATIAL_REFINER_ONLY=${TRAIN_SEG_SPATIAL_REFINER_ONLY}"
echo "[INFO] SEG_SPATIAL_REFINER_ALPHA=${SEG_SPATIAL_REFINER_ALPHA}"

deepspeed --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" "${PYTHON}" segearth_r2/train/train.py \
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
  --fp16 "${FP16}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --logging_steps "${LOGGING_STEPS}" \
  --tf32 "${TF32}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --lora_enable "${LORA_ENABLE}" \
  --lora_r "${LORA_R}" \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio "${DATA_RATIO}" \
  --switch_bs "${SWITCH_BS}" \
  --seed "${SEED}" \
  --data_seed "${DATA_SEED}" \
  --report_to wandb \
  --use_seg_spatial_refiner "${USE_SEG_SPATIAL_REFINER}" \
  --train_seg_spatial_refiner_only "${TRAIN_SEG_SPATIAL_REFINER_ONLY}" \
  --seg_spatial_refiner_alpha "${SEG_SPATIAL_REFINER_ALPHA}" \
  --seg_spatial_refiner_loss_weight "${SEG_SPATIAL_REFINER_LOSS_WEIGHT}" \
  --seg_spatial_refiner_dice_weight "${SEG_SPATIAL_REFINER_DICE_WEIGHT}" \
  --seg_spatial_refiner_bce_weight "${SEG_SPATIAL_REFINER_BCE_WEIGHT}" \
  --use_query_aware_decoder_bias "${USE_QUERY_AWARE_DECODER_BIAS}" \
  --use_decoder_attn_bias "${USE_DECODER_ATTN_BIAS}" \
  --use_mstva "${USE_MSTVA}" \
  --use_mstva_loss "${USE_MSTVA_LOSS}" \
  --use_text_film "${USE_TEXT_FILM}" \
  --use_attention_loss "${USE_ATTENTION_LOSS}" \
  --evaluation_strategy "${EVALUATION_STRATEGY}" \
  --load_best_model_at_end "${LOAD_BEST_MODEL_AT_END}" \
  --train_midstage_recalibration False

echo "[DONE] Seg spatial refiner frozen training finished: ${OUTPUT_DIR}"
