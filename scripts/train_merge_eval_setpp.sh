#!/usr/bin/env bash
set -euo pipefail

########################################
# SET++ warm-start train -> merge -> eval (LaSeRS)
#
# Usage:
#   bash scripts/train_merge_eval_setpp.sh
#
# Common overrides:
#   GPU_ID=0 MAX_STEPS=500 SAVE_STEPS=250 bash scripts/train_merge_eval_setpp.sh
#   ABLATION=A0 bash scripts/train_merge_eval_setpp.sh
#   RUN_TRAIN=0 MERGE_CHECKPOINT=/path/to/checkpoint-250 bash scripts/train_merge_eval_setpp.sh
#   RUN_MERGE=0 RUN_EVAL=1 EVAL_MODEL=/path/to/merged_model bash scripts/train_merge_eval_setpp.sh
########################################

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-setpp}"

########################################
# Ablation matrix (A0–A3)
########################################
ABLATION="${ABLATION:-A3}"

case "${ABLATION}" in
  A0)
    SETPP_CLOSED_LOOP=False
    SETPP_CSQR_ENABLE=False
    ;;
  A1)
    SETPP_CLOSED_LOOP=True
    SETPP_CSQR_ENABLE=False
    ;;
  A2)
    SETPP_CLOSED_LOOP=False
    SETPP_CSQR_ENABLE=True
    ;;
  A3)
    SETPP_CLOSED_LOOP=True
    SETPP_CSQR_ENABLE=True
    ;;
  *)
    echo "[ERROR] unknown ABLATION=${ABLATION}, expected A0/A1/A2/A3"
    exit 1
    ;;
esac

RUN_TAG="${RUN_TAG:-setpp-${ABLATION}-lasers-warmstart-5w-gd4}"
export WANDB_NAME="${WANDB_NAME:-${RUN_TAG}}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_MERGE="${RUN_MERGE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

REPO_DIR="${REPO_DIR:-/root/rivermind-data/huangziyi/reseg/segearth+set++}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/root/rivermind-data/miniconda3/envs/reseg}"
PYTHON="${PYTHON:-${CONDA_ENV_DIR}/bin/python}"
DEEPSPEED="${DEEPSPEED:-${CONDA_ENV_DIR}/bin/deepspeed}"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"

GPU_ID="${GPU_ID:-0}"
GPU_SLOT="${GPU_SLOT:-localhost:${GPU_ID}}"
MASTER_PORT="${MASTER_PORT:-29621}"

WARM_START_MODEL="${WARM_START_MODEL:-/root/rivermind-data/huangziyi/reseg/output/base/standard-base-lasers-siglip1-5w-gd4/merged_model}"
VISION_TOWER="${VISION_TOWER:-/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"

TRAIN_DATASET="${TRAIN_DATASET:-lasers}"
TRAIN_BASE_DATA_PATH="${TRAIN_BASE_DATA_PATH:-/root/rivermind-data/huangziyi/data/LaSeRS}"
EVAL_DATASET="${EVAL_DATASET:-lasers}"
EVAL_BASE_DATA_PATH="${EVAL_BASE_DATA_PATH:-${TRAIN_BASE_DATA_PATH}}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
RUN_CROSS_DATASET_EVAL="${RUN_CROSS_DATASET_EVAL:-1}"
LASERS_HOLDOUT_RATIO="${LASERS_HOLDOUT_RATIO:-0.05}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/rivermind-data/huangziyi/reseg/output/setpp}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_TAG}}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}/merged_model}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/test_results}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train_merge_eval.log}"

MAX_STEPS="${MAX_STEPS:-50000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
BF16="${BF16:-True}"
TF32="${TF32:-False}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero1.json}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
DATA_RATIO="${DATA_RATIO:-1}"
SWITCH_BS="${SWITCH_BS:-4}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"

MERGE_CHECKPOINT="${MERGE_CHECKPOINT:-}"
EVAL_MODEL="${EVAL_MODEL:-}"

read_best_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR_FOR_READ="${out_dir}" "${PYTHON}" - <<'PY'
import json
import os
output_dir = os.environ["OUTPUT_DIR_FOR_READ"]
state_path = os.path.join(output_dir, "trainer_state.json")
if not os.path.exists(state_path):
    print("")
    raise SystemExit
with open(state_path, "r", encoding="utf-8") as f:
    state = json.load(f)
print(state.get("best_model_checkpoint") or "")
PY
}

read_last_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR_FOR_READ="${out_dir}" "${PYTHON}" - <<'PY'
import os
import re
output_dir = os.environ["OUTPUT_DIR_FOR_READ"]
candidates = []
if os.path.isdir(output_dir):
    for name in os.listdir(output_dir):
        m = re.match(r"checkpoint-(\d+)$", name)
        if m:
            candidates.append((int(m.group(1)), os.path.join(output_dir, name)))
if candidates:
    candidates.sort()
    print(candidates[-1][1])
else:
    print("")
PY
}

require_path () {
  local path="$1"
  local kind="$2"
  if [[ "${kind}" == "dir" && ! -d "${path}" ]]; then
    echo "[ERROR] missing directory: ${path}"
    exit 1
  fi
  if [[ "${kind}" == "file" && ! -f "${path}" ]]; then
    echo "[ERROR] missing file: ${path}"
    exit 1
  fi
}

check_merged_setpp_model () {
  local model_dir="$1"
  local expect_csqr="${2:-0}"
  MERGED_MODEL_DIR="${model_dir}" EXPECT_CSQR="${expect_csqr}" "${PYTHON}" - <<'PY'
import json
import os
import sys

model_dir = os.environ["MERGED_MODEL_DIR"]
expect_csqr = os.environ.get("EXPECT_CSQR", "0") == "1"
index_path = os.path.join(model_dir, "model.safetensors.index.json")
required_substrings = ("SET_token_projector", "SET_query_embed")

if not os.path.isfile(index_path):
    print(f"[ERROR] missing weight index: {index_path}")
    sys.exit(1)

with open(index_path, "r", encoding="utf-8") as f:
    weight_map = json.load(f).get("weight_map", {})

keys = list(weight_map.keys())
missing = [name for name in required_substrings if not any(name in k for k in keys)]
if missing:
    print(f"[ERROR] merged model missing SET++ weights: {missing}")
    sys.exit(1)

for name in required_substrings:
    matched = [k for k in keys if name in k]
    print(f"[OK] {name}: {matched[0]}")

csqr_keys = [k for k in keys if "csqr_block" in k]
if expect_csqr and not csqr_keys:
    print("[ERROR] setpp_csqr_enable=True but merged model has no predictor.csqr_block weights")
    sys.exit(1)
if csqr_keys:
    print(f"[OK] csqr_block: {len(csqr_keys)} tensors (e.g. {csqr_keys[0]})")
else:
    print("[WARN] no csqr_block weights (setpp_csqr_enable=False or legacy run)")
PY
}

preflight () {
  mkdir -p "${OUTPUT_DIR}"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  cd "${REPO_DIR}"

  echo "========================================"
  echo "[0/4] Preflight (LaSeRS)"
  echo "========================================"
  echo "[INFO] ABLATION=${ABLATION} closed_loop=${SETPP_CLOSED_LOOP} csqr=${SETPP_CSQR_ENABLE}"
  echo "[INFO] RUN_TAG=${RUN_TAG}"
  echo "[INFO] WARM_START_MODEL=${WARM_START_MODEL}"
  echo "[INFO] TRAIN_DATASET=${TRAIN_DATASET}"
  echo "[INFO] TRAIN_BASE_DATA_PATH=${TRAIN_BASE_DATA_PATH}"
  echo "[INFO] EVAL_DATASET=${EVAL_DATASET}"
  echo "[INFO] EVAL_SPLIT=${EVAL_SPLIT}"
  echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
  echo "[INFO] MERGED_DIR=${MERGED_DIR}"
  echo "[INFO] EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}"
  echo "[INFO] GPU_SLOT=${GPU_SLOT}"
  echo "[INFO] MAX_STEPS=${MAX_STEPS}"
  echo "[INFO] LASERS_HOLDOUT_RATIO=${LASERS_HOLDOUT_RATIO}"
  echo "[INFO] RUN_TRAIN=${RUN_TRAIN} RUN_MERGE=${RUN_MERGE} RUN_EVAL=${RUN_EVAL}"

  require_path "${WARM_START_MODEL}" dir
  require_path "${VISION_TOWER}" dir
  require_path "${VISION_TOWER_MASK}" file
  require_path "${MASK_CONFIG}" file
  require_path "${TRAIN_BASE_DATA_PATH}/train/annotations/train_data.json" file
  require_path "${TRAIN_BASE_DATA_PATH}/test/annotations" dir
  require_path "${DEEPSPEED_CONFIG}" file

  WARM_START_MODEL="${WARM_START_MODEL}" "${PYTHON}" - <<'PY'
from transformers import AutoTokenizer
import os
model_path = os.environ["WARM_START_MODEL"]
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
tokenizer.add_tokens(["[SEG]", "[SET]"])
ids = tokenizer("[SET][SEG]", add_special_tokens=False)["input_ids"]
if max(ids) >= 51200:
    raise SystemExit(f"[ERROR] SET/SEG token id exceeds historical lm_head size 51200: ids={ids}")
print(f"[INFO] tokenizer len after SET/SEG={len(tokenizer)}, adjacent ids={ids}")
PY
}

train_model () {
  if [[ "${RUN_TRAIN}" != "1" ]]; then
    echo "[SKIP] training disabled"
    return
  fi
  echo "========================================"
  echo "[1/4] Train SET++ on LaSeRS (holdout eval)"
  echo "========================================"
  "${DEEPSPEED}" --include "${GPU_SLOT}" --master_port "${MASTER_PORT}" \
    segearth_r2/train/train.py \
      --model_name_or_path "${WARM_START_MODEL}" \
      --vision_tower "${VISION_TOWER}" \
      --vision_tower_mask "${VISION_TOWER_MASK}" \
      --mask_config "${MASK_CONFIG}" \
      --base_data_path "${TRAIN_BASE_DATA_PATH}" \
      --dataset_name "${TRAIN_DATASET}" \
      --output_dir "${OUTPUT_DIR}" \
      --max_steps "${MAX_STEPS}" \
      --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --save_strategy steps \
      --save_steps "${SAVE_STEPS}" \
      --save_total_limit "${SAVE_TOTAL_LIMIT}" \
      --learning_rate "${LEARNING_RATE}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --warmup_ratio "${WARMUP_RATIO}" \
      --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
      --logging_steps "${LOGGING_STEPS}" \
      --bf16 "${BF16}" \
      --tf32 "${TF32}" \
      --model_max_length "${MODEL_MAX_LENGTH}" \
      --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
      --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
      --lora_r "${LORA_R}" \
      --lora_alpha "${LORA_ALPHA}" \
      --lora_dropout "${LORA_DROPOUT}" \
      --data_ratio "${DATA_RATIO}" \
      --switch_bs "${SWITCH_BS}" \
      --seed "${SEED}" \
      --data_seed "${DATA_SEED}" \
      --lasers_holdout_ratio "${LASERS_HOLDOUT_RATIO}" \
      --lasers_holdout_seed "${DATA_SEED}" \
      --setpp_closed_loop "${SETPP_CLOSED_LOOP}" \
      --setpp_csqr_enable "${SETPP_CSQR_ENABLE}" \
      --deepspeed "${DEEPSPEED_CONFIG}" \
      --report_to wandb
}

merge_model () {
  if [[ "${RUN_MERGE}" != "1" ]]; then
    echo "[SKIP] merge disabled"
    return
  fi
  echo "========================================"
  echo "[2/4] Merge LoRA checkpoint"
  echo "========================================"
  local ckpt="${MERGE_CHECKPOINT}"
  if [[ -z "${ckpt}" ]]; then
    ckpt="$(read_best_checkpoint "${OUTPUT_DIR}")"
  fi
  if [[ -z "${ckpt}" ]]; then
    ckpt="$(read_last_checkpoint "${OUTPUT_DIR}")"
  fi
  if [[ -z "${ckpt}" || ! -d "${ckpt}" ]]; then
    echo "[ERROR] no checkpoint found. Set MERGE_CHECKPOINT=/path/to/checkpoint-N or run training first."
    exit 1
  fi
  echo "[INFO] merging checkpoint: ${ckpt}"
  rm -rf "${MERGED_DIR}"
  mkdir -p "${MERGED_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
    --model_path "${ckpt}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --save_path "${MERGED_DIR}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --setpp_closed_loop "${SETPP_CLOSED_LOOP}" \
    --setpp_csqr_enable "${SETPP_CSQR_ENABLE}"

  if [[ "${SETPP_CSQR_ENABLE}" == "True" ]]; then
    EXPECT_CSQR=1
  else
    EXPECT_CSQR=0
  fi
  check_merged_setpp_model "${MERGED_DIR}" "${EXPECT_CSQR}"
}

eval_model () {
  if [[ "${RUN_EVAL}" != "1" ]]; then
    echo "[SKIP] eval disabled"
    return
  fi
  local model_dir="${EVAL_MODEL}"
  if [[ -z "${model_dir}" ]]; then
    model_dir="${MERGED_DIR}"
  fi
  require_path "${model_dir}" dir

  if [[ "${RUN_CROSS_DATASET_EVAL}" == "1" ]]; then
    echo "========================================"
    echo "[3/4] Eval merged model on 5 datasets (test, parallel)"
    echo "========================================"
    MODEL_PATH="${model_dir}" \
    OUT_DIR="${OUTPUT_DIR}" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    EVAL_SPLIT="${EVAL_SPLIT}" \
    PYTHON="${PYTHON}" \
    REPO_DIR="${REPO_DIR}" \
    bash "${REPO_DIR}/run_cross_dataset_test_eval.sh"
    return
  fi

  echo "========================================"
  echo "[3/4] Eval merged model on ${EVAL_DATASET} ${EVAL_SPLIT}"
  echo "========================================"
  mkdir -p "${EVAL_OUTPUT_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" segearth_r2/eval/eval.py \
    --base_data_path "${EVAL_BASE_DATA_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --model_path "${model_dir}" \
    --output_dir "${EVAL_OUTPUT_DIR}" \
    --dataset_name "${EVAL_DATASET}" \
    --split "${EVAL_SPLIT}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --max_eval_samples "${EVAL_MAX_SAMPLES}" \
    --dataloader_num_workers 0 \
    --skip_existing True \
    --zip_results False
}

finish () {
  echo "========================================"
  echo "[4/4] Done"
  echo "========================================"
  echo "[INFO] output: ${OUTPUT_DIR}"
  echo "[INFO] merged: ${MERGED_DIR}"
  echo "[INFO] eval: ${OUTPUT_DIR}/*_test_results (5 datasets when RUN_CROSS_DATASET_EVAL=1)"
  echo "[INFO] log: ${LOG_FILE}"
}

preflight
train_model
merge_model
eval_model
finish
