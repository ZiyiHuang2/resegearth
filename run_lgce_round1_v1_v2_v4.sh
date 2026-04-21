#!/usr/bin/env bash
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_INIT_TIMEOUT=300
export WANDB_PROJECT=segearth-lgce-round1
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="localhost:2"
GPU_ID="2"
MASTER_PORT_BASE="29222"

########################################
# Project dir
########################################
REPO_DIR="/home/wangchengjun/huangziyi/reseg/reseagearth-lgce"
cd "${REPO_DIR}"

########################################
# Common paths
########################################
MODEL_NAME_OR_PATH="/home/wangchengjun/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
VISION_TOWER="/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384"
VISION_TOWER_MASK="/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"

########################################
# Dataset config
########################################
BASE_DATA_PATH="/home/wangchengjun/huangziyi/data/RRSISD"
DATASET_NAME="rrsisd"
TEST_SPLIT="test"

########################################
# Train config
########################################
MAX_STEPS="15000"
PER_DEVICE_TRAIN_BATCH_SIZE="1"
GRADIENT_ACCUMULATION_STEPS="1"

SAVE_STEPS="1500"
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

LORA_R="4"
DATA_RATIO="1"
SWITCH_BS="4"

SEED="42"
DATA_SEED="42"

########################################
# LGCE common config
########################################
USE_LGCE_BRIDGE="True"
LGCE_DEBUG="False"
LGCE_GUIDANCE_MODE="sentence_mean"

########################################
# Output root
########################################
EXP_ROOT="/home/wangchengjun/huangziyi/reseg/output/lgce"

########################################
# Run one experiment
########################################
run_exp() {
  local EXP_TAG="$1"
  local PORT="$2"
  local LGCE_SCALES="$3"
  local LGCE_RESIDUAL_INIT="$4"
  local LGCE_CROSS_SCALE_INIT="$5"
  local LGCE_ENABLE_LONG_SKIP="$6"

  local EXP_NAME="lgce_${EXP_TAG}"
  local OUTPUT_DIR="${EXP_ROOT}/${EXP_NAME}"
  local MERGED_DIR="${OUTPUT_DIR}/merged_model"
  local TEST_OUTPUT_DIR="${OUTPUT_DIR}/test_results"
  local FINAL_CKPT="${OUTPUT_DIR}/checkpoint-${MAX_STEPS}"

  export WANDB_NAME="${EXP_NAME}"

  echo "========================================"
  echo "[EXP] ${EXP_NAME}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "guidance_mode=${LGCE_GUIDANCE_MODE}"
  echo "lgce_scales=${LGCE_SCALES}"
  echo "lgce_residual_init=${LGCE_RESIDUAL_INIT}"
  echo "lgce_cross_scale_init=${LGCE_CROSS_SCALE_INIT}"
  echo "lgce_enable_long_skip=${LGCE_ENABLE_LONG_SKIP}"
  echo "lora_r=${LORA_R}"
  echo "========================================"

  ########################################
  # 1) Train
  ########################################
  deepspeed --master_port="${PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
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
    --use_lgce_bridge "${USE_LGCE_BRIDGE}" \
    --lgce_debug "${LGCE_DEBUG}" \
    --lgce_guidance_mode "${LGCE_GUIDANCE_MODE}" \
    --lgce_scales "${LGCE_SCALES}" \
    --lgce_residual_init "${LGCE_RESIDUAL_INIT}" \
    --lgce_cross_scale_init "${LGCE_CROSS_SCALE_INIT}" \
    --lgce_enable_long_skip "${LGCE_ENABLE_LONG_SKIP}" \
    --report_to wandb

  ########################################
  # 2) Merge final checkpoint
  ########################################
  if [[ ! -d "${FINAL_CKPT}" ]]; then
    echo "[ERROR] ${FINAL_CKPT} not found"
    exit 1
  fi

  rm -rf "${MERGED_DIR}"
  mkdir -p "${MERGED_DIR}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
    --model_path "${FINAL_CKPT}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --save_path "${MERGED_DIR}" \
    --lora_r "${LORA_R}"

  ########################################
  # 3) Eval merged model
  ########################################
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
  echo "[DONE] ${EXP_NAME}"
  echo "Final checkpoint : ${FINAL_CKPT}"
  echo "Merged model     : ${MERGED_DIR}"
  echo "Test outputs     : ${TEST_OUTPUT_DIR}"
  echo "========================================"
}

########################################
# v1: 双层桥复现版
########################################
run_exp \
  "v1_r34_base_r4_15k" \
  "$((MASTER_PORT_BASE + 1))" \
  "res3,res4" \
  "0.05" \
  "0.10" \
  "False"

########################################
# v2: 三层桥主实验
########################################
run_exp \
  "v2_r345_main_r4_15k" \
  "$((MASTER_PORT_BASE + 2))" \
  "res3,res4,res5" \
  "0.05" \
  "0.10" \
  "True"

########################################
# v4: 三层桥弱注入
########################################
run_exp \
  "v4_r345_weak_r4_15k" \
  "$((MASTER_PORT_BASE + 4))" \
  "res3,res4,res5" \
  "0.03" \
  "0.05" \
  "True"

echo "========================================"
echo "[ALL DONE] v1 + v2 + v4 finished"
echo "========================================"