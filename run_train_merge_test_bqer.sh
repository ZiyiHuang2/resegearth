#!/usr/bin/env bash
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-bqer}"
export WANDB_NAME="${WANDB_NAME:-bqer-5w}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="${GPU_SLOT:-localhost:0}"
GPU_ID="${GPU_ID:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"

# A100 reseg env binaries (avoid PATH issues)
PYTHON_BIN="${PYTHON_BIN:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-/root/rivermind-data/miniconda3/envs/reseg/bin/deepspeed}"

########################################
# Project dir (A100)
########################################
REPO_DIR="${REPO_DIR:-/root/rivermind-data/huangziyi/reseg/segearth+bqer}"
cd "${REPO_DIR}"

########################################
# Common paths (A100)
########################################
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B}"
VISION_TOWER="${VISION_TOWER:-/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"

########################################
# Dataset config (A100)
########################################
BASE_DATA_PATH="${BASE_DATA_PATH:-/root/rivermind-data/huangziyi/data/RRSISD}"
DATASET_NAME="${DATASET_NAME:-rrsisd}"
TEST_SPLIT="${TEST_SPLIT:-test}"

########################################
# Output
########################################
OUTPUT_DIR="${OUTPUT_DIR:-/root/rivermind-data/huangziyi/reseg/output/bqer/bqer-5w}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}/merged_model}"
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-${OUTPUT_DIR}/test_results}"

########################################
# Eval metrics config
########################################
EVAL_METRICS_SCRIPT="${EVAL_METRICS_SCRIPT:-/root/rivermind-data/huangziyi/reseg/eval_val_metrics.py}"
EVAL_USE_WANDB="${EVAL_USE_WANDB:-True}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-bqer}"
EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-bqer-5w}"

########################################
# Train config
########################################
MAX_STEPS="${MAX_STEPS:-50000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"

LOGGING_STEPS="${LOGGING_STEPS:-20}"
BF16="${BF16:-True}"
TF32="${TF32:-False}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

DATA_RATIO="${DATA_RATIO:-1}"
SWITCH_BS="${SWITCH_BS:-4}"

SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"

########################################
# BQER / Bi-Lite config
########################################
BQER_ENABLE="${BQER_ENABLE:-True}"
BQER_K_LAYERS="${BQER_K_LAYERS:-2}"
BQER_BOUNDARY_WEIGHT="${BQER_BOUNDARY_WEIGHT:-0.4}"
BQER_QUERY_CONSISTENCY_WEIGHT="${BQER_QUERY_CONSISTENCY_WEIGHT:-0.2}"
BQER_SMALL_OBJECT_WEIGHT="${BQER_SMALL_OBJECT_WEIGHT:-1.8}"
BQER_SMALL_OBJECT_PERCENTILE="${BQER_SMALL_OBJECT_PERCENTILE:-30.0}"
BQER_MOD_ALPHA="${BQER_MOD_ALPHA:-0.1}"
BQER_TOKEN_DRIFT_WEIGHT="${BQER_TOKEN_DRIFT_WEIGHT:-0.02}"
BQER_Q2B_DETACH_QUERY="${BQER_Q2B_DETACH_QUERY:-False}"

is_true () {
  [[ "$1" == "True" || "$1" == "true" || "$1" == "1" ]]
}

read_best_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR="${out_dir}" "${PYTHON_BIN}" - <<'PY'
import json, os, sys
p = os.path.join(os.environ["OUTPUT_DIR"], "trainer_state.json")
if not os.path.exists(p):
    print(""); sys.exit(0)
with open(p, "r", encoding="utf-8") as f:
    s = json.load(f)
print(s.get("best_model_checkpoint", ""))
PY
}

read_last_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR="${out_dir}" "${PYTHON_BIN}" - <<'PY'
import os, re
d = os.environ["OUTPUT_DIR"]
if not os.path.isdir(d):
    print(""); raise SystemExit
c = []
for n in os.listdir(d):
    m = re.match(r"checkpoint-(\d+)$", n)
    if m:
        c.append((int(m.group(1)), os.path.join(d, n)))
print("" if not c else sorted(c)[-1][1])
PY
}

merge_ckpt () {
  local ckpt="$1"
  local save_dir="$2"

  rm -rf "${save_dir}"
  mkdir -p "${save_dir}"

  local bqer_enable_flag=()
  local q2b_detach_flag=()

  if is_true "${BQER_ENABLE}"; then
    bqer_enable_flag+=(--bqer_enable)
  fi
  if is_true "${BQER_Q2B_DETACH_QUERY}"; then
    q2b_detach_flag+=(--bqer_q2b_detach_query)
  fi

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
    --model_path "${ckpt}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --save_path "${save_dir}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    "${bqer_enable_flag[@]}" \
    "${q2b_detach_flag[@]}" \
    --bqer_k_layers "${BQER_K_LAYERS}" \
    --bqer_boundary_weight "${BQER_BOUNDARY_WEIGHT}" \
    --bqer_query_consistency_weight "${BQER_QUERY_CONSISTENCY_WEIGHT}" \
    --bqer_small_object_weight "${BQER_SMALL_OBJECT_WEIGHT}" \
    --bqer_small_object_percentile "${BQER_SMALL_OBJECT_PERCENTILE}" \
    --bqer_mod_alpha "${BQER_MOD_ALPHA}" \
    --bqer_token_drift_weight "${BQER_TOKEN_DRIFT_WEIGHT}"
}

eval_model () {
  local model_dir="$1"
  local out_dir="$2"

  mkdir -p "${out_dir}"

  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON_BIN}" segearth_r2/eval/eval.py \
    --base_data_path "${BASE_DATA_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --model_path "${model_dir}" \
    --output_dir "${out_dir}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${TEST_SPLIT}" \
    --eval_batch_size 1 \
    --zip_results False \
    --bqer_enable "${BQER_ENABLE}" \
    --bqer_k_layers "${BQER_K_LAYERS}" \
    --bqer_boundary_weight "${BQER_BOUNDARY_WEIGHT}" \
    --bqer_query_consistency_weight "${BQER_QUERY_CONSISTENCY_WEIGHT}" \
    --bqer_small_object_weight "${BQER_SMALL_OBJECT_WEIGHT}" \
    --bqer_small_object_percentile "${BQER_SMALL_OBJECT_PERCENTILE}" \
    --bqer_mod_alpha "${BQER_MOD_ALPHA}" \
    --bqer_token_drift_weight "${BQER_TOKEN_DRIFT_WEIGHT}" \
    --bqer_q2b_detach_query "${BQER_Q2B_DETACH_QUERY}"
}

run_eval_metrics () {
  local pred_dir="$1"
  local run_name="$2"
  USE_WANDB="${EVAL_USE_WANDB}" \
  WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
  WANDB_RUN_NAME="${run_name}" \
  DATASET_TYPE="${DATASET_NAME}" \
  BASE_DATA_PATH="${BASE_DATA_PATH}" \
  SPLIT="${TEST_SPLIT}" \
  PRED_DIR="${pred_dir}" \
  "${PYTHON_BIN}" "${EVAL_METRICS_SCRIPT}"
}

echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "[1/6] train"
"${DEEPSPEED_BIN}" --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
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
  --bqer_enable "${BQER_ENABLE}" \
  --bqer_k_layers "${BQER_K_LAYERS}" \
  --bqer_boundary_weight "${BQER_BOUNDARY_WEIGHT}" \
  --bqer_query_consistency_weight "${BQER_QUERY_CONSISTENCY_WEIGHT}" \
  --bqer_small_object_weight "${BQER_SMALL_OBJECT_WEIGHT}" \
  --bqer_small_object_percentile "${BQER_SMALL_OBJECT_PERCENTILE}" \
  --bqer_mod_alpha "${BQER_MOD_ALPHA}" \
  --bqer_token_drift_weight "${BQER_TOKEN_DRIFT_WEIGHT}" \
  --bqer_q2b_detach_query "${BQER_Q2B_DETACH_QUERY}" \
  --seed "${SEED}" \
  --data_seed "${DATA_SEED}" \
  --report_to wandb

echo "[2/6] select checkpoint"
BEST_CHECKPOINT="$(read_best_checkpoint "${OUTPUT_DIR}")"
if [[ -z "${BEST_CHECKPOINT}" ]]; then
  BEST_CHECKPOINT="$(read_last_checkpoint "${OUTPUT_DIR}")"
fi
if [[ -z "${BEST_CHECKPOINT}" ]]; then
  echo "[ERROR] no checkpoint found in ${OUTPUT_DIR}"
  exit 1
fi
echo "[OK] BEST_CHECKPOINT=${BEST_CHECKPOINT}"

echo "[3/6] merge"
merge_ckpt "${BEST_CHECKPOINT}" "${MERGED_DIR}"

echo "[4/6] check merged config"
[[ -f "${MERGED_DIR}/config.json" ]] || { echo "[ERROR] missing ${MERGED_DIR}/config.json"; exit 1; }

echo "[5/6] eval + metrics"
eval_model "${MERGED_DIR}" "${TEST_OUTPUT_DIR}"
run_eval_metrics "${TEST_OUTPUT_DIR}" "${EVAL_WANDB_RUN_NAME}"

echo "[6/6] done"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MERGED_DIR=${MERGED_DIR}"
echo "TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}"