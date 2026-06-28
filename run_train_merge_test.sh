#!/usr/bin/env bash
# =============================================================================
# SegEarth-R2 Base 全流程：Train → Merge → Eval → Metrics
# 数据集：LaSeRS（dataset_name=lasers）
#
# ⚠️ 官方 LaSeRS **没有 val 划分**，只有：
#   ${BASE_DATA_PATH}/
#     train/
#       images/
#       annotations/train_data.json          # 训练集（有 GT mask）
#     test/
#       images/
#       annotations/
#         test_short_query.json              # 9 个 benchmark 子集，均有 GT mask
#         test_long_query.json
#         test_explicit.json / test_implicit.json
#         test_single_cate.json / test_multi_cate.json
#         test_instance_level.json / test_part_level.json / test_sematic_level.json
#
# 数据逻辑（已对齐官方 release）：
#   - 训练：train_data.json；训练期 validation 从 train 固定 seed holdout 5%
#   - 最终评估：test/annotations/*.json（9 个 benchmark，eval.py 全部跑一遍）
#   - 指标：eval_val_metrics.py LASERS_BENCHMARK=all → 每个 benchmark 单独 json + summary
# =============================================================================
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT=segearth-standard
export WANDB_NAME=base-lasers-siglip1-5w-gd4
export WANDB_INIT_TIMEOUT=300
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="localhost:0"
GPU_ID="0"
MASTER_PORT="29500"

########################################
# Project dir
########################################
REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+joint"
cd "${REPO_DIR}"

########################################
# Common paths
########################################
MODEL_NAME_OR_PATH="/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"


########################################
# LaSeRS 数据集配置
#   官方划分：train + test（无 val）
#   EVAL_SPLIT=test；LASERS_BENCHMARK=all 表示 9 个 test benchmark 都算指标
########################################
BASE_DATA_PATH="/root/rivermind-data/huangziyi/data/LaSeRS"
DATASET_NAME="lasers"
EVAL_SPLIT="test"
LASERS_BENCHMARK="${LASERS_BENCHMARK:-all}"
LASERS_HOLDOUT_RATIO="${LASERS_HOLDOUT_RATIO:-0.05}"

########################################
# Output
########################################
OUTPUT_DIR="/root/rivermind-data/huangziyi/reseg/output/base/standard-base-lasers-siglip1-5w-gd4"
MERGED_DIR="${OUTPUT_DIR}/merged_model"
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/test_results"

########################################
# Eval metrics
########################################
EVAL_METRICS_SCRIPT="/root/rivermind-data/huangziyi/reseg/eval_val_metrics.py"
EVAL_USE_WANDB="True"
EVAL_WANDB_PROJECT="segearth-eval-standard-val"
EVAL_WANDB_RUN_NAME="base-lasers-siglip1-5w-gd4"

########################################
# Train config
########################################
MAX_STEPS="50000"
PER_DEVICE_TRAIN_BATCH_SIZE="4"
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
DATALOADER_NUM_WORKERS="8"

LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"

DATA_RATIO="1"
SWITCH_BS="4"

SEED="42"
DATA_SEED="42"

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
    --split "${EVAL_SPLIT}" \
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
  SPLIT="${EVAL_SPLIT}" \
  LASERS_BENCHMARK="${LASERS_BENCHMARK}" \
  PRED_DIR="${pred_dir}" \
  python "${EVAL_METRICS_SCRIPT}"
}

########################################
# Preflight（LaSeRS train + test，无 val）
########################################
echo "========================================"
echo "[0/6] Preflight checks (LaSeRS train+test)"
echo "========================================"

echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] BASE_DATA_PATH=${BASE_DATA_PATH}"
echo "[INFO] DATASET_NAME=${DATASET_NAME}"
echo "[INFO] EVAL_SPLIT=${EVAL_SPLIT} (LaSeRS 无 val，评估应对 test 子集)"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"

if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "[ERROR] model path not found: ${MODEL_NAME_OR_PATH}"
  exit 1
fi

if [[ ! -f "${BASE_DATA_PATH}/train/annotations/train_data.json" ]]; then
  echo "[ERROR] train annotation not found"
  echo "[HINT] 解压: tar -xzf ${BASE_DATA_PATH}/train.tar.gz -C ${BASE_DATA_PATH}"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/train/images" ]]; then
  echo "[ERROR] train images not found: ${BASE_DATA_PATH}/train/images"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/test/annotations" ]]; then
  echo "[ERROR] test annotations not found"
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

mkdir -p "${OUTPUT_DIR}"
echo "[OK] preflight passed (train + test present, no val required)"

########################################
# 1) Train
#    train_data.json；eval 用同文件 holdout 5%（非 test，避免泄漏）
########################################
echo "========================================"
echo "[1/6] Training on LaSeRS train (holdout eval)"
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
  --lasers_holdout_ratio "${LASERS_HOLDOUT_RATIO}" \
  --lasers_holdout_seed "${DATA_SEED}" \
  --report_to wandb

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
echo "[4/6] Check merged config"
echo "========================================"

if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  echo "[ERROR] merged config.json not found: ${MERGED_DIR}/config.json"
  exit 1
fi

echo "[OK] merged config.json present"

########################################
# 5) Eval on LaSeRS test benchmarks + metrics
########################################
echo "========================================"
echo "[5/6] Eval on LaSeRS test benchmarks + metrics"
echo "========================================"

eval_model "${MERGED_DIR}" "${EVAL_OUTPUT_DIR}"
run_eval_metrics "${EVAL_OUTPUT_DIR}" "${EVAL_WANDB_RUN_NAME}"

########################################
# 6) Done
########################################
echo "========================================"
echo "[6/6] DONE"
echo "Dataset           : LaSeRS (train + test, no val)"
echo "Output dir        : ${OUTPUT_DIR}"
echo "Selected ckpt     : ${BEST_CHECKPOINT}"
echo "Merged model      : ${MERGED_DIR}"
echo "Metrics summary   : ${OUTPUT_DIR}/lasers_${EVAL_SPLIT}_metrics_summary.json"
echo "Metrics per bench : ${OUTPUT_DIR}/lasers_${EVAL_SPLIT}_test_*_metrics.json  (e.g. lasers_test_test_short_query_metrics.json)"
