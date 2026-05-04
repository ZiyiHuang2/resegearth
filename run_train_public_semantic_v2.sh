#!/usr/bin/env bash
# RRSIS-D train — public_semantic_v2 (adds --concept_public_semantic_library only).
# All other CLI hyperparameters match run_train_baseline_raw.sh (same MASTER_PORT default).
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

REPO_DIR="/home/wangchengjun/huangziyi/reseg/resegearth+tgi"
cd "${REPO_DIR}"

CONCEPT_LIB="/home/wangchengjun/huangziyi/reseg/resegearth+source/configs/concept_public_semantic_library_v2.json"

MODEL_NAME_OR_PATH="/home/wangchengjun/huangziyi/reseg/output/bseg/baseline_standard-base_5w/merged_model"
VISION_TOWER="/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384"
VISION_TOWER_MASK="/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
BASE_DATA_PATH="/home/wangchengjun/huangziyi/data/RRSISD"
DATASET_NAME="rrsisd"

OUTPUT_DIR="/home/wangchengjun/huangziyi/reseg/resegearth+tgi/checkpoints/rrsisd_public_semantic_v2"

MAX_STEPS="70000"
PER_DEVICE_TRAIN_BATCH_SIZE="1"
GRADIENT_ACCUMULATION_STEPS="1"
SAVE_STEPS="2000"
SAVE_TOTAL_LIMIT="3"
LEARNING_RATE="3e-5"
WEIGHT_DECAY="0.0"
WARMUP_RATIO="0.03"
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS="10"
BF16="False"
FP16="True"
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

TRAIN_MIDSTAGE_RECALIBRATION="True"
STAGE3_NORM_ONLY="False"
USE_ATTENTION_LOSS="False"
USE_MIDSTAGE_GATE_LOSS="False"
MIDSTAGE_GATE_LOSS_WEIGHT="0.0"
USE_MSTVA="True"
MSTVA_ALIGN_DIM="256"
USE_MSTVA_LOSS="True"
MSTVA_LOSS_WEIGHT="0.01"
MSTVA_SCALE_WEIGHTS="0.5,0.3,0.2"

GPU_SLOT="${GPU_SLOT:-localhost:0}"
MASTER_PORT="${MASTER_PORT:-29531}"

mkdir -p "${OUTPUT_DIR}"

deepspeed --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --concept_public_semantic_library "${CONCEPT_LIB}" \
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
  --lora_r "${LORA_R}" \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio "${DATA_RATIO}" \
  --switch_bs "${SWITCH_BS}" \
  --seed "${SEED}" \
  --data_seed "${DATA_SEED}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --train_midstage_recalibration "${TRAIN_MIDSTAGE_RECALIBRATION}" \
  --stage3_norm_only "${STAGE3_NORM_ONLY}" \
  --use_attention_loss "${USE_ATTENTION_LOSS}" \
  --use_midstage_gate_loss "${USE_MIDSTAGE_GATE_LOSS}" \
  --midstage_gate_loss_weight "${MIDSTAGE_GATE_LOSS_WEIGHT}" \
  --use_mstva "${USE_MSTVA}" \
  --mstva_align_dim "${MSTVA_ALIGN_DIM}" \
  --use_mstva_loss "${USE_MSTVA_LOSS}" \
  --mstva_loss_weight "${MSTVA_LOSS_WEIGHT}" \
  --mstva_scale_weights "${MSTVA_SCALE_WEIGHTS}"
