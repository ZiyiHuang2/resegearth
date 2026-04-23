#!/usr/bin/env bash
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT=segearth-bseg
export WANDB_INIT_TIMEOUT=300
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="localhost:2"
GPU_ID="2"

MASTER_PORT_BASELINE="29300"
MASTER_PORT_STRUCTURED="29301"

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
TEST_SPLIT="test"

########################################
# Train config
########################################
MAX_STEPS="15000"
PER_DEVICE_TRAIN_BATCH_SIZE="1"
GRADIENT_ACCUMULATION_STEPS="1"

SAVE_STEPS="5000"
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
MAX_GRAD_NORM="1.0"

########################################
# Baseline output
########################################
BASELINE_OUTPUT_DIR="/home/wangchengjun/huangziyi/reseg/output/bseg/standard-base"
BASELINE_MERGED_DIR="${BASELINE_OUTPUT_DIR}/merged_model"
BASELINE_TEST_OUTPUT_DIR="${BASELINE_OUTPUT_DIR}/test_results"

########################################
# Online structured output
########################################
STRUCTURED_OUTPUT_DIR="/home/wangchengjun/huangziyi/reseg/output/bseg/online_structured_standard-base_k8_15k"
STRUCTURED_MERGED_DIR="${STRUCTURED_OUTPUT_DIR}/merged_model"
STRUCTURED_TEST_OUTPUT_DIR="${STRUCTURED_OUTPUT_DIR}/test_results"

########################################
# Structured switches
########################################
ATTENTION_LOSS_WEIGHT="0.005"
STRUCTURED_FG_BG_WEIGHT="0.8"
STRUCTURED_BOUNDARY_OUTER_WEIGHT="0.2"
STRUCTURED_ATTENTION_MARGIN="0.05"
ATTENTION_LOSS_LAST_K_LAYERS="8"

########################################
# Helpers
########################################
read_best_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR="${out_dir}" python - <<'PY'
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
}

merge_ckpt () {
  local ckpt="$1"
  local save_dir="$2"

  rm -rf "${save_dir}"
  mkdir -p "${save_dir}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
    --model_path "${ckpt}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --save_path "${save_dir}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}"
}

eval_model () {
  local model_dir="$1"
  local out_dir="$2"

  mkdir -p "${out_dir}"

  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python segearth_r2/eval/eval.py \
    --base_data_path "${BASE_DATA_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --model_path "${model_dir}" \
    --output_dir "${out_dir}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${TEST_SPLIT}" \
    --eval_batch_size 1 \
    --zip_results False
}

########################################
# 1) Train baseline
########################################
echo "========================================"
echo "[1/6] Training baseline (15k)"
echo "========================================"

export WANDB_NAME="bseg-base-15k"

deepspeed --master_port="${MASTER_PORT_BASELINE}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${BASELINE_OUTPUT_DIR}" \
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
  --use_structured_attention_loss False \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --report_to wandb

########################################
# 2) Merge + eval baseline
########################################
echo "========================================"
echo "[2/6] Merge + eval baseline"
echo "========================================"

BASELINE_BEST_CHECKPOINT=$(read_best_checkpoint "${BASELINE_OUTPUT_DIR}")

if [[ -z "${BASELINE_BEST_CHECKPOINT}" ]]; then
  echo "[ERROR] baseline best_model_checkpoint not found"
  exit 1
fi

echo "[OK] BASELINE_BEST_CHECKPOINT=${BASELINE_BEST_CHECKPOINT}"

merge_ckpt "${BASELINE_BEST_CHECKPOINT}" "${BASELINE_MERGED_DIR}"
eval_model "${BASELINE_MERGED_DIR}" "${BASELINE_TEST_OUTPUT_DIR}"

########################################
# 3) Train online structured
########################################
echo "========================================"
echo "[3/6] Training online structured (15k)"
echo "========================================"

export WANDB_NAME="online-structured-standard-base-k8-15k"

deepspeed --master_port="${MASTER_PORT_STRUCTURED}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${STRUCTURED_OUTPUT_DIR}" \
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
  --attention_loss_weight "${ATTENTION_LOSS_WEIGHT}" \
  --structured_fg_bg_weight "${STRUCTURED_FG_BG_WEIGHT}" \
  --structured_boundary_outer_weight "${STRUCTURED_BOUNDARY_OUTER_WEIGHT}" \
  --structured_attention_margin "${STRUCTURED_ATTENTION_MARGIN}" \
  --attention_loss_last_k_layers "${ATTENTION_LOSS_LAST_K_LAYERS}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --report_to wandb

########################################
# 4) Merge + eval online structured
########################################
echo "========================================"
echo "[4/6] Merge + eval online structured"
echo "========================================"

STRUCTURED_BEST_CHECKPOINT=$(read_best_checkpoint "${STRUCTURED_OUTPUT_DIR}")

if [[ -z "${STRUCTURED_BEST_CHECKPOINT}" ]]; then
  echo "[ERROR] structured best_model_checkpoint not found"
  exit 1
fi

echo "[OK] STRUCTURED_BEST_CHECKPOINT=${STRUCTURED_BEST_CHECKPOINT}"

merge_ckpt "${STRUCTURED_BEST_CHECKPOINT}" "${STRUCTURED_MERGED_DIR}"
eval_model "${STRUCTURED_MERGED_DIR}" "${STRUCTURED_TEST_OUTPUT_DIR}"

########################################
# 5) Final compare hint
########################################
echo "========================================"
echo "[5/6] Compare hints"
echo "========================================"
echo "Check these during/after training:"
echo "- loss_llm / loss_mask / loss_attention"
echo "- loss_attn_fg_bg / loss_attn_boundary_outer"
echo "- data_time / iter_time / throughput"
echo "- overall metric"
echo "- small-object / boundary-related subset"
echo "- whether R-subset drops"

########################################
# 6) Done
########################################
echo "========================================"
echo "[6/6] DONE"
echo "Baseline best ckpt   : ${BASELINE_BEST_CHECKPOINT}"
echo "Baseline merged      : ${BASELINE_MERGED_DIR}"
echo "Baseline test output : ${BASELINE_TEST_OUTPUT_DIR}"
echo "Structured best ckpt : ${STRUCTURED_BEST_CHECKPOINT}"
echo "Structured merged    : ${STRUCTURED_MERGED_DIR}"
echo "Structured test out  : ${STRUCTURED_TEST_OUTPUT_DIR}"
echo "========================================"