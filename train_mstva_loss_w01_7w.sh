#!/usr/bin/env bash
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT=segearth-tgi
export WANDB_INIT_TIMEOUT=300
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="localhost:0"
GPU_ID="0"

MASTER_PORT_MSTVA="29530"

########################################
# Project dir
########################################
REPO_DIR="/home/wangchengjun/huangziyi/reseg/resegearth+tgi"
cd "${REPO_DIR}"

########################################
# Common paths
########################################
MODEL_NAME_OR_PATH="/home/wangchengjun/huangziyi/reseg/output/bseg/baseline_standard-base_5w/merged_model"

VISION_TOWER="/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384"
VISION_TOWER_MASK="/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl"
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
LORA_ALPHA="16"
LORA_DROPOUT="0.05"

DATA_RATIO="1"
SWITCH_BS="4"

SEED="42"
DATA_SEED="42"
MAX_GRAD_NORM="1.0"

########################################
# Mid-stage / MSTVA module train config
########################################
TRAIN_MIDSTAGE_RECALIBRATION="True"
STAGE3_NORM_ONLY="False"

########################################
# Loss config
########################################
USE_ATTENTION_LOSS="False"
USE_MIDSTAGE_GATE_LOSS="False"
MIDSTAGE_GATE_LOSS_WEIGHT="0.0"

USE_MSTVA="True"
MSTVA_ALIGN_DIM="256"
USE_MSTVA_LOSS="True"
MSTVA_LOSS_WEIGHT="0.01"
MSTVA_SCALE_WEIGHTS="0.5,0.3,0.2"

########################################
# Output
########################################
MSTVA_OUTPUT_DIR="/home/wangchengjun/huangziyi/reseg/output/tgi/mstva_loss_w01_7w"
MSTVA_MERGED_DIR="${MSTVA_OUTPUT_DIR}/merged_model"
MSTVA_TEST_OUTPUT_DIR="${MSTVA_OUTPUT_DIR}/test_results"
MSTVA_DIAG_OUTPUT_DIR="${MSTVA_OUTPUT_DIR}/mstva_diagnosis"

########################################
# Eval metrics config (auto upload)
########################################
EVAL_METRICS_SCRIPT="/home/wangchengjun/huangziyi/reseg/eval_val_metrics.py"
EVAL_USE_WANDB="True"
EVAL_WANDB_PROJECT="segearth-eval-tgi-val"
EVAL_WANDB_RUN_NAME="mstva_loss_w01_7w"

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
    --use_mstva "${USE_MSTVA}" \
    --mstva_align_dim "${MSTVA_ALIGN_DIM}" \
    --use_mstva_loss "${USE_MSTVA_LOSS}" \
    --mstva_loss_weight "${MSTVA_LOSS_WEIGHT}" \
    --mstva_scale_weights "${MSTVA_SCALE_WEIGHTS}"
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
  python "${EVAL_METRICS_SCRIPT}"
}

diagnose_mstva () {
  local model_dir="$1"
  local out_dir="$2"

  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python tools/diagnose_mstva_maps.py \
    --model_path "${model_dir}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --base_data_path "${BASE_DATA_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${TEST_SPLIT}" \
    --num_samples 100 \
    --device cuda \
    --dtype float16 \
    --output_dir "${out_dir}"
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
echo "[INFO] OUTPUT_DIR=${MSTVA_OUTPUT_DIR}"
echo "[INFO] GPU_SLOT=${GPU_SLOT}"
echo "[INFO] LORA_R=${LORA_R}"
echo "[INFO] LEARNING_RATE=${LEARNING_RATE}"
echo "[INFO] USE_ATTENTION_LOSS=${USE_ATTENTION_LOSS}"
echo "[INFO] USE_MSTVA=${USE_MSTVA}"
echo "[INFO] MSTVA_ALIGN_DIM=${MSTVA_ALIGN_DIM}"
echo "[INFO] USE_MSTVA_LOSS=${USE_MSTVA_LOSS}"
echo "[INFO] MSTVA_LOSS_WEIGHT=${MSTVA_LOSS_WEIGHT}"
echo "[INFO] MSTVA_SCALE_WEIGHTS=${MSTVA_SCALE_WEIGHTS}"

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

if ! python segearth_r2/train/train.py --help 2>/dev/null | grep -q "use_mstva"; then
  echo "[ERROR] train.py does not expose --use_mstva."
  exit 1
fi

if ! python segearth_r2/train/train.py --help 2>/dev/null | grep -q "mstva_align_dim"; then
  echo "[ERROR] train.py does not expose --mstva_align_dim."
  exit 1
fi

if ! python segearth_r2/train/train.py --help 2>/dev/null | grep -q "use_mstva_loss"; then
  echo "[ERROR] train.py does not expose --use_mstva_loss."
  exit 1
fi

if ! python segearth_r2/train/train.py --help 2>/dev/null | grep -q "mstva_loss_weight"; then
  echo "[ERROR] train.py does not expose --mstva_loss_weight."
  exit 1
fi

if ! python segearth_r2/train/train.py --help 2>/dev/null | grep -q "mstva_scale_weights"; then
  echo "[ERROR] train.py does not expose --mstva_scale_weights."
  exit 1
fi

mkdir -p "${MSTVA_OUTPUT_DIR}"

echo "[OK] preflight passed"

########################################
# 1) Train
########################################
echo "========================================"
echo "[1/6] Training MSTVA experiment: w=0.01, 2k"
echo "========================================"

export WANDB_NAME="tgi-mstva-loss-w01-7w"

deepspeed --master_port="${MASTER_PORT_MSTVA}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${MSTVA_OUTPUT_DIR}" \
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
  --mstva_scale_weights "${MSTVA_SCALE_WEIGHTS}" \
  --report_to wandb

########################################
# 2) Select checkpoint
########################################
echo "========================================"
echo "[2/6] Select checkpoint"
echo "========================================"

MSTVA_BEST_CHECKPOINT=$(read_best_checkpoint "${MSTVA_OUTPUT_DIR}")

if [[ -z "${MSTVA_BEST_CHECKPOINT}" ]]; then
  echo "[WARN] best_model_checkpoint not found; fallback to last checkpoint"
  MSTVA_BEST_CHECKPOINT=$(read_last_checkpoint "${MSTVA_OUTPUT_DIR}")
fi

if [[ -z "${MSTVA_BEST_CHECKPOINT}" ]]; then
  echo "[ERROR] no checkpoint found under ${MSTVA_OUTPUT_DIR}"
  exit 1
fi

echo "[OK] MSTVA_CHECKPOINT=${MSTVA_BEST_CHECKPOINT}"

########################################
# 3) Merge
########################################
echo "========================================"
echo "[3/6] Merge selected checkpoint"
echo "========================================"

merge_ckpt "${MSTVA_BEST_CHECKPOINT}" "${MSTVA_MERGED_DIR}"

########################################
# 4) Check merged config
########################################
echo "========================================"
echo "[4/6] Check merged config for MSTVA fields"
echo "========================================"

if [[ ! -f "${MSTVA_MERGED_DIR}/config.json" ]]; then
  echo "[ERROR] merged config.json not found: ${MSTVA_MERGED_DIR}/config.json"
  exit 1
fi

grep -n '"use_mstva"\|"mstva_align_dim"\|"use_mstva_loss"\|"mstva_loss_weight"\|"mstva_scale_weights"' "${MSTVA_MERGED_DIR}/config.json" || {
  echo "[ERROR] merged config.json missing MSTVA keys"
  exit 1
}

########################################
# 5) Eval + Metrics + Diagnose
########################################
echo "========================================"
echo "[5/6] Eval merged model + upload eval metrics + diagnose MSTVA maps"
echo "========================================"

eval_model "${MSTVA_MERGED_DIR}" "${MSTVA_TEST_OUTPUT_DIR}"
run_eval_metrics "${MSTVA_TEST_OUTPUT_DIR}" "mstva_loss_w01_7w"
diagnose_mstva "${MSTVA_MERGED_DIR}" "${MSTVA_DIAG_OUTPUT_DIR}"

########################################
# 6) Done
########################################
echo "========================================"
echo "[6/6] DONE"
echo "Output dir       : ${MSTVA_OUTPUT_DIR}"
echo "Selected ckpt    : ${MSTVA_BEST_CHECKPOINT}"
echo "Merged model     : ${MSTVA_MERGED_DIR}"
echo "Test output      : ${MSTVA_TEST_OUTPUT_DIR}"
echo "Diag output      : ${MSTVA_DIAG_OUTPUT_DIR}"
echo "========================================"

