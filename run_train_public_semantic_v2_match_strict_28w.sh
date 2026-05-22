#!/usr/bin/env bash
# 7w：public_semantic_v2 + --concept_match_strict；流程与 run_train_baseline_raw_vs_public_semantic_v2_7w.sh 中 public v2 段一致：
# train → best checkpoint → merge LoRA → eval → eval metrics。
# 依赖：conda run -n reseg bash 本脚本（或已激活 reseg 且含 deepspeed / transformers）
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-source}"
export WANDB_INIT_TIMEOUT=300
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="${GPU_SLOT:-localhost:1}"
GPU_ID="${GPU_ID:-1}"
MASTER_PORT="${MASTER_PORT:-29111}"

REPO_DIR="/home/wangchengjun/huangziyi/reseg/resegearth+source"
RESEG_ROOT="/home/wangchengjun/huangziyi/reseg"
cd "${REPO_DIR}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/wangchengjun/huangziyi/reseg/output/bseg/baseline_standard-base_5w/merged_model}"
VISION_TOWER="${VISION_TOWER:-/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"

BASE_DATA_PATH="${BASE_DATA_PATH:-/home/wangchengjun/huangziyi/data/RRSISD}"
DATASET_NAME="rrsisd"
TEST_SPLIT="test"
CONCEPT_PUBLIC_SEMANTIC_LIBRARY="${CONCEPT_PUBLIC_SEMANTIC_LIBRARY:-configs/concept_public_semantic_library_v2.json}"

EVAL_METRICS_SCRIPT="${EVAL_METRICS_SCRIPT:-${RESEG_ROOT}/eval_val_metrics.py}"
EVAL_USE_WANDB="${EVAL_USE_WANDB:-True}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-source-val}"

MAX_STEPS="${MAX_STEPS:-280000}"
PER_DEVICE_TRAIN_BATCH_SIZE="1"
GRADIENT_ACCUMULATION_STEPS="1"
SAVE_STEPS="2000"
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

OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/source/rrsisd_public_semantic_v2_match_strict_28w}"
MERGED_DIR="${OUTPUT_DIR}/merged_model"
TEST_OUTPUT_DIR="${OUTPUT_DIR}/test_results"
export WANDB_NAME="match_strict_28w"

########################################
# Helpers（与 run_train_baseline_raw_vs_public_semantic_v2_7w.sh 一致）
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

########################################
# Preflight
########################################
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] MERGED_DIR=${MERGED_DIR}"
echo "[INFO] TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}"

if ! python segearth_r2/train/train.py --help 2>&1 | grep -q "concept_match_strict"; then
  echo "[ERROR] train.py --help 未包含 concept_match_strict，请在正确 conda 环境中运行。"
  exit 1
fi

if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "[ERROR] model path not found: ${MODEL_NAME_OR_PATH}"
  exit 1
fi

if [[ ! -f "${EVAL_METRICS_SCRIPT}" ]]; then
  echo "[ERROR] eval metrics script not found: ${EVAL_METRICS_SCRIPT}"
  exit 1
fi

CONCEPT_ABS="${CONCEPT_PUBLIC_SEMANTIC_LIBRARY}"
if [[ ! "${CONCEPT_ABS}" = /* ]]; then
  CONCEPT_ABS="${REPO_DIR}/${CONCEPT_PUBLIC_SEMANTIC_LIBRARY}"
fi
if [[ ! -f "${CONCEPT_ABS}" ]]; then
  echo "[ERROR] concept library not found: ${CONCEPT_ABS}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

########################################
# [1/6] Train
########################################
echo "========================================"
echo "[1/6] Training rrsisd_public_semantic_v2_match_strict_7w"
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
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio "${DATA_RATIO}" \
  --switch_bs "${SWITCH_BS}" \
  --seed "${SEED}" \
  --data_seed "${DATA_SEED}" \
  --report_to wandb \
  --concept_public_semantic_library "${CONCEPT_PUBLIC_SEMANTIC_LIBRARY}" \
  --concept_match_strict True

########################################
# [2/6] Best checkpoint
########################################
echo "========================================"
echo "[2/6] Select checkpoint (match_strict)"
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

echo "[OK] BEST_CHECKPOINT=${BEST_CHECKPOINT}"

########################################
# [3/6] Merge
########################################
echo "========================================"
echo "[3/6] Merge LoRA → merged_model"
echo "========================================"

merge_ckpt "${BEST_CHECKPOINT}" "${MERGED_DIR}"

########################################
# [4/6] Check merged
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
# [5/6] Eval + metrics
########################################
echo "========================================"
echo "[5/6] Eval + metrics (test)"
echo "========================================"

eval_model "${MERGED_DIR}" "${TEST_OUTPUT_DIR}"
run_eval_metrics "${TEST_OUTPUT_DIR}" "rrsisd_public_semantic_v2_match_strict_7w"

########################################
# [6/6] Done
########################################
echo "========================================"
echo "[6/6] DONE rrsisd_public_semantic_v2_match_strict_7w"
echo "Output dir       : ${OUTPUT_DIR}"
echo "Selected ckpt    : ${BEST_CHECKPOINT}"
echo "Merged model     : ${MERGED_DIR}"
echo "Test output      : ${TEST_OUTPUT_DIR}"
echo "========================================"
