#!/usr/bin/env bash
# Seg Spatial Refiner (frozen-base): train -> merge -> eval -> metrics
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT=segearth-tgi
export WANDB_NAME=seg-refiner-frozen-v2
export WANDB_INIT_TIMEOUT=300
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="localhost:0"
GPU_ID="0"
MASTER_PORT="29551"

########################################
# Project dir
########################################
REPO_DIR="/root/rivermind-data/huangziyi/reseg/resegearth+tgi"
cd "${REPO_DIR}"

########################################
# Common paths
########################################
MODEL_NAME_OR_PATH="/root/rivermind-data/huangziyi/reseg/output/base/standard-base-siglip1-28w-gd4/merged_model"
VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"

########################################
# Dataset config
########################################
BASE_DATA_PATH="/root/rivermind-data/huangziyi/data/RRSISD"
DATASET_NAME="rrsisd"
TEST_SPLIT="test"

########################################
# Output
########################################
RUN_NAME="seg_refiner_frozen_v2"
OUTPUT_DIR="/root/rivermind-data/huangziyi/reseg/output/tgi/${RUN_NAME}"
MERGED_DIR="${OUTPUT_DIR}/merged_model"
TEST_OUTPUT_DIR="${OUTPUT_DIR}/test_results"

########################################
# Eval metrics config (auto upload to W&B)
########################################
EVAL_METRICS_SCRIPT="/root/rivermind-data/huangziyi/reseg/eval_val_metrics.py"
EVAL_USE_WANDB="True"
EVAL_WANDB_PROJECT="segearth-eval-tgi-val"
EVAL_WANDB_RUN_NAME="${RUN_NAME}"
METRICS_TAG="${RUN_NAME}"
BASELINE_PRED_DIR="/root/rivermind-data/huangziyi/reseg/output/base/standard-base-siglip1-28w-gd4/test_results"

########################################
# Train config
########################################
MAX_STEPS="30000"
PER_DEVICE_TRAIN_BATCH_SIZE="4"
GRADIENT_ACCUMULATION_STEPS="1"

SAVE_STEPS="2000"
SAVE_TOTAL_LIMIT="2"

LEARNING_RATE="1e-4"
WEIGHT_DECAY="0.0"
WARMUP_RATIO="0.03"
LR_SCHEDULER_TYPE="cosine"

LOGGING_STEPS="50"
BF16="True"
FP16="False"
TF32="False"
MODEL_MAX_LENGTH="2048"
GRADIENT_CHECKPOINTING="False"
DATALOADER_NUM_WORKERS="4"

LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"
LORA_ENABLE="False"

DATA_RATIO="1"
SWITCH_BS="4"

SEED="42"
DATA_SEED="42"

########################################
# Seg Spatial Refiner (Stage 1 frozen-base)
########################################
USE_SEG_SPATIAL_REFINER="True"
TRAIN_SEG_SPATIAL_REFINER_ONLY="True"

SEG_SPATIAL_REFINER_ALPHA="0.1"
SEG_SPATIAL_REFINER_LOSS_WEIGHT="0.1"
SEG_SPATIAL_REFINER_DICE_WEIGHT="1.0"
SEG_SPATIAL_REFINER_BCE_WEIGHT="1.0"

USE_QUERY_AWARE_DECODER_BIAS="False"
USE_DECODER_ATTN_BIAS="False"
USE_MSTVA="False"
USE_MSTVA_LOSS="False"
USE_TEXT_FILM="False"
USE_ATTENTION_LOSS="False"
EVALUATION_STRATEGY="no"
LOAD_BEST_MODEL_AT_END="False"

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

read_last_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR="${out_dir}" python - <<'PY'
import os
import re

output_dir = os.environ["OUTPUT_DIR"]
if not os.path.isdir(output_dir):
    print("")
    raise SystemExit

candidates = []
for name in os.listdir(output_dir):
    m = re.match(r"checkpoint-(\d+)$", name)
    if m:
        candidates.append((int(m.group(1)), os.path.join(output_dir, name)))

if not candidates:
    print("")
else:
    candidates.sort()
    print(candidates[-1][1])
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
    --lora_dropout "${LORA_DROPOUT}" \
    --use_seg_spatial_refiner "${USE_SEG_SPATIAL_REFINER}" \
    --seg_spatial_refiner_alpha "${SEG_SPATIAL_REFINER_ALPHA}" \
    --seg_spatial_refiner_loss_weight "${SEG_SPATIAL_REFINER_LOSS_WEIGHT}" \
    --seg_spatial_refiner_dice_weight "${SEG_SPATIAL_REFINER_DICE_WEIGHT}" \
    --seg_spatial_refiner_bce_weight "${SEG_SPATIAL_REFINER_BCE_WEIGHT}" \
    --use_query_aware_decoder_bias False \
    --use_decoder_attn_bias False \
    --use_mstva False \
    --use_mstva_loss False \
    --use_text_film False
}

check_merged_refiner_config () {
  local merged_dir="$1"

  python - <<PY
import json
import sys
from pathlib import Path

merged = Path("${merged_dir}") / "config.json"
if not merged.is_file():
    print("[ERROR] merged config.json not found")
    sys.exit(1)

cfg = json.load(open(merged, encoding="utf-8"))
use_refiner = bool(cfg.get("use_seg_spatial_refiner", False))
alpha = cfg.get("seg_spatial_refiner_alpha", None)
print(f"[INFO] use_seg_spatial_refiner={use_refiner}")
print(f"[INFO] seg_spatial_refiner_alpha={alpha}")
if not use_refiner:
    print("[ERROR] merged config missing use_seg_spatial_refiner=True")
    sys.exit(1)

keys = []
idx = Path("${merged_dir}") / "model.safetensors.index.json"
if idx.is_file():
    wm = json.load(open(idx, encoding="utf-8")).get("weight_map", {})
    keys = sorted(k for k in wm if "seg_spatial_refiner" in k)

single = Path("${merged_dir}") / "model.safetensors"
if not keys and single.is_file():
    from safetensors import safe_open
    with safe_open(str(single), framework="pt") as f:
        keys = sorted(k for k in f.keys() if "seg_spatial_refiner" in k)

binf = Path("${merged_dir}") / "pytorch_model.bin"
if not keys and binf.is_file():
    import torch
    sd = torch.load(binf, map_location="cpu")
    keys = sorted(k for k in sd if "seg_spatial_refiner" in k)

print(f"[INFO] merged seg_spatial_refiner weight keys={len(keys)}")
if keys:
    print("[OK] sample:", keys[:3])
else:
    print("[WARN] no seg_spatial_refiner keys in merged weights")
PY
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
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --zip_results False
}

run_eval_metrics () {
  local pred_dir="$1"
  local run_name="$2"

  if [[ ! -f "${EVAL_METRICS_SCRIPT}" ]]; then
    echo "[ERROR] eval metrics script not found: ${EVAL_METRICS_SCRIPT}"
    exit 1
  fi

  USE_WANDB="${EVAL_USE_WANDB}" \
  WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
  WANDB_RUN_NAME="${run_name}" \
  DATASET_TYPE="${DATASET_NAME}" \
  BASE_DATA_PATH="${BASE_DATA_PATH}" \
  SPLIT="${TEST_SPLIT}" \
  PRED_DIR="${pred_dir}" \
  METRICS_TAG="${METRICS_TAG}" \
  BASELINE_PRED_DIR="${BASELINE_PRED_DIR}" \
  python "${EVAL_METRICS_SCRIPT}"
}

########################################
# Preflight
########################################
echo "========================================"
echo "[0/6] Preflight checks"
echo "========================================"

echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "[INFO] VISION_TOWER=${VISION_TOWER}"
echo "[INFO] VISION_TOWER_MASK=${VISION_TOWER_MASK}"
echo "[INFO] MASK_CONFIG=${MASK_CONFIG}"
echo "[INFO] BASE_DATA_PATH=${BASE_DATA_PATH}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] GPU_SLOT=${GPU_SLOT}"
echo "[INFO] MAX_STEPS=${MAX_STEPS}"
echo "[INFO] USE_SEG_SPATIAL_REFINER=${USE_SEG_SPATIAL_REFINER}"
echo "[INFO] TRAIN_SEG_SPATIAL_REFINER_ONLY=${TRAIN_SEG_SPATIAL_REFINER_ONLY}"
echo "[INFO] EVAL_WANDB_PROJECT=${EVAL_WANDB_PROJECT}"
echo "[INFO] EVAL_WANDB_RUN_NAME=${EVAL_WANDB_RUN_NAME}"
echo "[INFO] BASELINE_PRED_DIR=${BASELINE_PRED_DIR}"

if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "[ERROR] model path not found: ${MODEL_NAME_OR_PATH}"
  exit 1
fi

if [[ ! -d "${VISION_TOWER}" ]]; then
  echo "[ERROR] vision tower not found: ${VISION_TOWER}"
  exit 1
fi

if [[ ! -f "${VISION_TOWER_MASK}" ]]; then
  echo "[ERROR] vision tower mask not found: ${VISION_TOWER_MASK}"
  exit 1
fi

if [[ ! -f "${MASK_CONFIG}" ]]; then
  echo "[ERROR] mask config not found: ${MASK_CONFIG}"
  exit 1
fi

if [[ ! -f "${BASE_DATA_PATH}/rrsisd/refs(unc).p" ]]; then
  echo "[ERROR] refs file not found: ${BASE_DATA_PATH}/rrsisd/refs(unc).p"
  exit 1
fi

if [[ ! -f "${BASE_DATA_PATH}/rrsisd/instances.json" ]]; then
  echo "[ERROR] instances file not found: ${BASE_DATA_PATH}/rrsisd/instances.json"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/images" ]]; then
  echo "[ERROR] images dir not found: ${BASE_DATA_PATH}/images"
  exit 1
fi

if [[ ! -f "scripts/zero1.json" ]]; then
  echo "[ERROR] DeepSpeed config not found: ${REPO_DIR}/scripts/zero1.json"
  exit 1
fi

if [[ ! -f "${EVAL_METRICS_SCRIPT}" ]]; then
  echo "[ERROR] eval metrics script not found: ${EVAL_METRICS_SCRIPT}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ "${LORA_ENABLE}" == "False" || "${LORA_ENABLE}" == "false" ]]; then
  if compgen -G "${OUTPUT_DIR}/checkpoint-"*/adapter_config.json > /dev/null; then
    echo "[WARN] Found LoRA checkpoints under ${OUTPUT_DIR} but LORA_ENABLE=False."
    echo "[WARN] Training will NOT resume those checkpoints (incompatible). Starting fresh from merged_model."
    echo "[WARN] To avoid confusion, remove old checkpoints or set a new OUTPUT_DIR/RUN_NAME."
  fi
fi

echo "[OK] preflight passed"

########################################
# 1) Train
########################################
echo "========================================"
echo "[1/6] Training frozen-base Seg Spatial Refiner"
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

########################################
# 2) Select checkpoint
########################################
echo "========================================"
echo "[2/6] Select checkpoint"
echo "========================================"

BEST_CHECKPOINT=$(read_best_checkpoint "${OUTPUT_DIR}")

if [[ -z "${BEST_CHECKPOINT}" ]]; then
  echo "[WARN] best_model_checkpoint not found; fallback to last checkpoint"
  BEST_CHECKPOINT=$(read_last_checkpoint "${OUTPUT_DIR}")
fi

if [[ -z "${BEST_CHECKPOINT}" ]]; then
  echo "[ERROR] no checkpoint found under ${OUTPUT_DIR}"
  exit 1
fi

echo "[OK] SELECTED_CHECKPOINT=${BEST_CHECKPOINT}"

########################################
# 3) Merge
########################################
echo "========================================"
echo "[3/6] Merge selected checkpoint"
echo "========================================"

merge_ckpt "${BEST_CHECKPOINT}" "${MERGED_DIR}"

########################################
# 4) Check merged config
########################################
echo "========================================"
echo "[4/6] Check merged refiner config/weights"
echo "========================================"

check_merged_refiner_config "${MERGED_DIR}"

########################################
# 5) Eval merged model + upload eval metrics
########################################
echo "========================================"
echo "[5/6] Eval merged model + upload eval metrics"
echo "========================================"

eval_model "${MERGED_DIR}" "${TEST_OUTPUT_DIR}"
run_eval_metrics "${TEST_OUTPUT_DIR}" "${EVAL_WANDB_RUN_NAME}"

########################################
# 6) Done
########################################
echo "========================================"
echo "[6/6] DONE"
echo "Output dir         : ${OUTPUT_DIR}"
echo "Selected ckpt      : ${BEST_CHECKPOINT}"
echo "Merged model       : ${MERGED_DIR}"
echo "Test outputs       : ${TEST_OUTPUT_DIR}"
echo "Eval W&B project   : ${EVAL_WANDB_PROJECT}"
echo "Eval W&B run name  : ${EVAL_WANDB_RUN_NAME}"
echo "Baseline pred dir  : ${BASELINE_PRED_DIR}"
echo "========================================"
