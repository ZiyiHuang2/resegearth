#!/usr/bin/env bash
set -euo pipefail

########################################
# CS-DEG++ 全流程：Warm-start Train → Merge → LaSeRS Test Eval → W&B Metrics
#
# 数据集：LaSeRS（dataset_name=lasers）
# 起点：LaSeRS base merged_model（standard-base-lasers-siglip1-8w-gd4）
# 模型：CS-DEG++ decoder（maskformer2_csdeg_full.yaml，默认 ENABLED=true）
#
# 官方 LaSeRS 只有 train + test（无 val）：
#   - 训练：train_data.json；训练期 validation 从 train holdout 5%
#   - 最终评估：test/annotations/*.json（9 个 benchmark）
#
# Override examples:
#   GPU_ID=0 MAX_STEPS=50000 bash run_train_merge_test.sh
#   MASK_CONFIG=segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml bash run_train_merge_test.sh  # baseline ablation (CS_DEG off)
#   WARM_START_MODEL=/path/to/merged_model RUN_EVAL=0 bash run_train_merge_test.sh
########################################

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-csdeg}"
export WANDB_NAME="${WANDB_NAME:-csdeg-lasers-warmstart-50k-gd4}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

RESEG_ROOT="/root/rivermind-data/huangziyi/reseg"
CONDA_ENV_DIR="/root/rivermind-data/miniconda3/envs/reseg"
PYTHON="${CONDA_ENV_DIR}/bin/python"
DEEPSPEED="${CONDA_ENV_DIR}/bin/deepspeed"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"

GPU_ID="${GPU_ID:-0}"
GPU_SLOT="${GPU_SLOT:-localhost:${GPU_ID}}"
MASTER_PORT="${MASTER_PORT:-29631}"

########################################
# Project dir
########################################
REPO_DIR="${REPO_DIR:-${RESEG_ROOT}/segearth+csdeg}"
cd "${REPO_DIR}"

########################################
# Common paths
########################################
WARM_START_MODEL="${WARM_START_MODEL:-${RESEG_ROOT}/output/base/standard-base-lasers-siglip1-8w-gd4/merged_model}"
VISION_TOWER="${RESEG_ROOT}/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="${RESEG_ROOT}/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_csdeg_full.yaml}"

########################################
# LaSeRS 数据集配置
########################################
BASE_DATA_PATH="${BASE_DATA_PATH:-/root/rivermind-data/huangziyi/data/LaSeRS}"
DATASET_NAME="lasers"
EVAL_SPLIT="test"
LASERS_BENCHMARK="${LASERS_BENCHMARK:-all}"
LASERS_HOLDOUT_RATIO="${LASERS_HOLDOUT_RATIO:-0.05}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"

########################################
# Output
########################################
OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/csdeg/csdeg-lasers-warmstart-8w-gd4}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}/merged_model}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/test_results}"

########################################
# Eval metrics config (auto upload to W&B)
########################################
EVAL_METRICS_SCRIPT="${RESEG_ROOT}/eval_val_metrics.py"
EVAL_USE_WANDB="True"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-csdeg}"
EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-csdeg-lasers-warmstart-8w-gd4}"

########################################
# Train config（对齐 base/set++ LaSeRS 8w；CS-DEG++ 更重，默认 bs=1×4）
########################################
MAX_STEPS="${MAX_STEPS:-50000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="0.0"
WARMUP_RATIO="0.03"
LR_SCHEDULER_TYPE="cosine"

LOGGING_STEPS="10"
BF16="True"
TF32="False"
MODEL_MAX_LENGTH="2048"
GRADIENT_CHECKPOINTING="False"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"

DATA_RATIO="1"
SWITCH_BS="4"

SEED="42"
DATA_SEED="42"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_MERGE="${RUN_MERGE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
MERGE_CHECKPOINT="${MERGE_CHECKPOINT:-}"

########################################
# Helpers
########################################
read_best_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR_FOR_READ="${out_dir}" "${PYTHON}" - <<'PY'
import json
import os
import sys

output_dir = os.environ["OUTPUT_DIR_FOR_READ"]
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
  OUTPUT_DIR_FOR_READ="${out_dir}" "${PYTHON}" - <<'PY'
import os
import re

output_dir = os.environ["OUTPUT_DIR_FOR_READ"]
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

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
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
  "${PYTHON}" segearth_r2/eval/eval.py \
    --base_data_path "${BASE_DATA_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --model_path "${model_dir}" \
    --output_dir "${out_dir}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${EVAL_SPLIT}" \
    --eval_batch_size 1 \
    --max_eval_samples "${EVAL_MAX_SAMPLES}" \
    --dataloader_num_workers 0 \
    --skip_existing True \
    --zip_results False
}

check_merged_csdeg_model () {
  local model_dir="$1"
  MERGED_MODEL_DIR="${model_dir}" MASK_CONFIG_PATH="${MASK_CONFIG}" "${PYTHON}" - <<'PY'
import json
import os
import sys

model_dir = os.environ["MERGED_MODEL_DIR"]
mask_config = os.environ.get("MASK_CONFIG_PATH", "")

index_path = os.path.join(model_dir, "model.safetensors.index.json")
if not os.path.isfile(index_path):
    print(f"[WARN] missing weight index: {index_path} (skip CS-DEG weight check)")
    sys.exit(0)

with open(index_path, "r", encoding="utf-8") as f:
    weight_map = json.load(f).get("weight_map", {})

keys = list(weight_map.keys())
if "csdeg" in mask_config.lower() or "cs_deg" in mask_config.lower():
    required = ("ret.", "depg.", "bhef.", "cs_alpha", "mask_gamma")
    missing = [name for name in required if not any(name in k for k in keys)]
    if missing:
        print(f"[WARN] merged model may miss CS-DEG++ weights (new modules train from init): {missing}")
    else:
        for name in required:
            matched = [k for k in keys if name in k][0]
            print(f"[OK] CS-DEG weight: {matched}")
else:
    print("[OK] baseline mask config — skip CS-DEG weight check")
PY
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
  SPLIT="${EVAL_SPLIT}" \
  LASERS_BENCHMARK="${LASERS_BENCHMARK}" \
  PRED_DIR="${pred_dir}" \
  "${PYTHON}" "${EVAL_METRICS_SCRIPT}"
}

########################################
# Preflight（LaSeRS train + test，无 val）
########################################
echo "========================================"
echo "[0/6] Preflight checks (LaSeRS + CS-DEG++)"
echo "========================================"

echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] WARM_START_MODEL=${WARM_START_MODEL}"
echo "[INFO] MASK_CONFIG=${MASK_CONFIG}"
echo "[INFO] BASE_DATA_PATH=${BASE_DATA_PATH}"
echo "[INFO] DATASET_NAME=${DATASET_NAME}"
echo "[INFO] EVAL_SPLIT=${EVAL_SPLIT} (LaSeRS 无 val，评估应对 test 子集)"
echo "[INFO] LASERS_BENCHMARK=${LASERS_BENCHMARK}"
echo "[INFO] LASERS_HOLDOUT_RATIO=${LASERS_HOLDOUT_RATIO}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] GPU_SLOT=${GPU_SLOT}"
echo "[INFO] GPU_ID=${GPU_ID}"
echo "[INFO] LEARNING_RATE=${LEARNING_RATE}"
echo "[INFO] MAX_STEPS=${MAX_STEPS}"
echo "[INFO] PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "[INFO] GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "[INFO] RUN_TRAIN=${RUN_TRAIN} RUN_MERGE=${RUN_MERGE} RUN_EVAL=${RUN_EVAL}"
echo "[INFO] MERGE_CHECKPOINT=${MERGE_CHECKPOINT:-<auto>}"
echo "[INFO] EVAL_WANDB_PROJECT=${EVAL_WANDB_PROJECT}"
echo "[INFO] EVAL_WANDB_RUN_NAME=${EVAL_WANDB_RUN_NAME}"

if [[ ! -d "${WARM_START_MODEL}" ]]; then
  echo "[ERROR] warm-start model not found: ${WARM_START_MODEL}"
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

if [[ ! -f "${BASE_DATA_PATH}/train/annotations/train_data.json" ]]; then
  echo "[ERROR] train annotation not found: ${BASE_DATA_PATH}/train/annotations/train_data.json"
  echo "[HINT] 解压: tar -xzf ${BASE_DATA_PATH}/train.tar.gz -C ${BASE_DATA_PATH}"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/train/images" ]]; then
  echo "[ERROR] train images not found: ${BASE_DATA_PATH}/train/images"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/test/annotations" ]]; then
  echo "[ERROR] test annotations not found: ${BASE_DATA_PATH}/test/annotations"
  echo "[HINT] 解压: tar -xzf ${BASE_DATA_PATH}/test.tar.gz -C ${BASE_DATA_PATH}"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/test/images" ]]; then
  echo "[ERROR] test images not found: ${BASE_DATA_PATH}/test/images"
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

WARM_START_MODEL="${WARM_START_MODEL}" "${PYTHON}" - <<'PY'
from transformers import AutoTokenizer
import os

model_path = os.environ["WARM_START_MODEL"]
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
tokenizer.add_tokens(["[SEG]"])
ids = tokenizer("[SEG]", add_special_tokens=False)["input_ids"]
if max(ids) >= 51200:
    raise SystemExit(f"[ERROR] SEG token id exceeds historical lm_head size 51200: ids={ids}")
print(f"[OK] tokenizer len after [SEG]={len(tokenizer)}, SEG id={ids}")
PY

mkdir -p "${OUTPUT_DIR}"

echo "[OK] preflight passed (train + test present, no val required)"

########################################
# 1) Train
#    train_data.json；eval 用同文件 holdout 5%（非 test，避免泄漏）
########################################
if [[ "${RUN_TRAIN}" != "1" ]]; then
  echo "========================================"
  echo "[1/6] SKIP training (RUN_TRAIN=${RUN_TRAIN})"
  echo "========================================"
else
echo "========================================"
echo "[1/6] Training CS-DEG++ on LaSeRS train (holdout eval)"
echo "========================================"

"${DEEPSPEED}" --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${WARM_START_MODEL}" \
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
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio "${DATA_RATIO}" \
  --switch_bs "${SWITCH_BS}" \
  --seed "${SEED}" \
  --data_seed "${DATA_SEED}" \
  --lasers_holdout_ratio "${LASERS_HOLDOUT_RATIO}" \
  --lasers_holdout_seed "${DATA_SEED}" \
  --report_to wandb
fi

########################################
# 2) Select checkpoint
########################################
echo "========================================"
echo "[2/6] Select checkpoint"
echo "========================================"

BEST_CHECKPOINT="${MERGE_CHECKPOINT}"
if [[ -z "${BEST_CHECKPOINT}" ]]; then
  BEST_CHECKPOINT=$(read_best_checkpoint "${OUTPUT_DIR}")
fi

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
if [[ "${RUN_MERGE}" != "1" ]]; then
  echo "========================================"
  echo "[3/6] SKIP merge (RUN_MERGE=${RUN_MERGE})"
  echo "========================================"
else
echo "========================================"
echo "[3/6] Merge selected checkpoint"
echo "========================================"

merge_ckpt "${BEST_CHECKPOINT}" "${MERGED_DIR}"
fi

########################################
# 4) Check merged model
########################################
echo "========================================"
echo "[4/6] Check merged model"
echo "========================================"

if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  echo "[ERROR] merged config.json not found: ${MERGED_DIR}/config.json"
  exit 1
fi

check_merged_csdeg_model "${MERGED_DIR}"

########################################
# 5) Eval on LaSeRS test benchmarks + metrics
########################################
if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "========================================"
  echo "[5/6] SKIP eval (RUN_EVAL=${RUN_EVAL})"
  echo "========================================"
else
echo "========================================"
echo "[5/6] Eval on LaSeRS test benchmarks + upload metrics"
echo "========================================"

eval_model "${MERGED_DIR}" "${EVAL_OUTPUT_DIR}"
run_eval_metrics "${EVAL_OUTPUT_DIR}" "${EVAL_WANDB_RUN_NAME}"
fi

########################################
# 6) Done
########################################
echo "========================================"
echo "[6/6] DONE"
echo "Dataset           : LaSeRS (train + test, no val)"
echo "Mask config       : ${MASK_CONFIG}"
echo "Output dir        : ${OUTPUT_DIR}"
echo "Selected ckpt     : ${BEST_CHECKPOINT}"
echo "Merged model      : ${MERGED_DIR}"
echo "Test outputs      : ${EVAL_OUTPUT_DIR}"
echo "Metrics summary   : ${OUTPUT_DIR}/lasers_${EVAL_SPLIT}_metrics_summary.json"
echo "Eval W&B project  : ${EVAL_WANDB_PROJECT}"
echo "Eval W&B run name : ${EVAL_WANDB_RUN_NAME}"
echo "========================================"
