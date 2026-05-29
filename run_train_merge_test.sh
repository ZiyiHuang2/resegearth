#!/usr/bin/env bash
# SegEarth-R2 Stage 3 DGP-QDTI: train -> merge -> eval -> metrics
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT=segearth-dgp-stage3
export WANDB_NAME=dgp-qdti-stage3-v5
export WANDB_INIT_TIMEOUT=300
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="localhost:0"
GPU_ID="0"
MASTER_PORT="29800"

########################################
# Project dir
########################################
REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+DGP"
cd "${REPO_DIR}"

########################################
# Common paths
########################################
MODEL_NAME_OR_PATH="/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
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
OUTPUT_DIR="/root/rivermind-data/huangziyi/reseg/output/dgp/dgp-qdti-stage3-v5"
MERGED_DIR="${OUTPUT_DIR}/merged_model"
TEST_OUTPUT_DIR="${OUTPUT_DIR}/test_results"

########################################
# Eval metrics config (auto upload to W&B)
########################################
EVAL_METRICS_SCRIPT="/root/rivermind-data/huangziyi/reseg/eval_val_metrics.py"
EVAL_USE_WANDB="True"
EVAL_WANDB_PROJECT="segearth-eval-dgp-stage3"
EVAL_WANDB_RUN_NAME="dgp-qdti-stage3-v5"

########################################
# Stage 3 DGP-QDTI config (train / merge / eval must stay consistent)
########################################
USE_DGP_QDTI="True"
USE_QDTI_BIAS="True"
DGP_FUSE_DIM="256"
DGP_REFINER_HIDDEN_DIM="512"
DGP_PG_TOKENS="1"
QDTI_BIAS_DIM="128"
QDTI_INIT_STD="1e-3"
QDTI_MAX_ABS="0.01"
QDTI_APPLY_LAYERS="last3"
QDTI_SCALE_INIT="0.0"
SCALE_HARD_LOSS_WEIGHT="0.0"

########################################
# Train config
########################################
MAX_STEPS="80000"
PER_DEVICE_TRAIN_BATCH_SIZE="4"
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
    --use_dgp_qdti "${USE_DGP_QDTI}" \
    --use_qdti_bias "${USE_QDTI_BIAS}" \
    --dgp_fuse_dim "${DGP_FUSE_DIM}" \
    --dgp_refiner_hidden_dim "${DGP_REFINER_HIDDEN_DIM}" \
    --dgp_pg_tokens "${DGP_PG_TOKENS}" \
    --qdti_bias_dim "${QDTI_BIAS_DIM}" \
    --qdti_init_std "${QDTI_INIT_STD}" \
    --qdti_max_abs "${QDTI_MAX_ABS}" \
    --qdti_apply_layers "${QDTI_APPLY_LAYERS}" \
    --qdti_scale_init "${QDTI_SCALE_INIT}" \
    --scale_hard_loss_weight "${SCALE_HARD_LOSS_WEIGHT}"
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
    --use_dgp_qdti "${USE_DGP_QDTI}" \
    --use_qdti_bias "${USE_QDTI_BIAS}" \
    --dgp_fuse_dim "${DGP_FUSE_DIM}" \
    --dgp_refiner_hidden_dim "${DGP_REFINER_HIDDEN_DIM}" \
    --dgp_pg_tokens "${DGP_PG_TOKENS}" \
    --qdti_bias_dim "${QDTI_BIAS_DIM}" \
    --qdti_init_std "${QDTI_INIT_STD}" \
    --qdti_max_abs "${QDTI_MAX_ABS}" \
    --qdti_apply_layers "${QDTI_APPLY_LAYERS}" \
    --qdti_scale_init "${QDTI_SCALE_INIT}" \
    --scale_hard_loss_weight "${SCALE_HARD_LOSS_WEIGHT}" \
    --zip_results False
}

check_merged_dgp_config () {
  local merged_dir="$1"
  MERGED_DIR="${merged_dir}" python - <<'PY'
import json
import os
import sys

merged_dir = os.environ["MERGED_DIR"]
cfg_path = os.path.join(merged_dir, "config.json")
required = (
    "use_dgp_qdti",
    "use_qdti_bias",
    "dgp_pg_tokens",
    "qdti_apply_layers",
    "qdti_scale_init",
    "qdti_max_abs",
    "scale_hard_loss_weight",
)

if not os.path.isfile(cfg_path):
    print(f"[ERROR] missing {cfg_path}", file=sys.stderr)
    sys.exit(1)

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

missing = [k for k in required if k not in cfg]
if missing:
    print(f"[ERROR] merged config.json missing DGP keys: {missing}", file=sys.stderr)
    sys.exit(1)

if not cfg.get("use_dgp_qdti", False):
    print("[ERROR] merged config.json use_dgp_qdti is not True", file=sys.stderr)
    sys.exit(1)

for key in required:
    print(f"  config.{key}={cfg.get(key)!r}")
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
  SPLIT="${TEST_SPLIT}" \
  PRED_DIR="${pred_dir}" \
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
echo "[INFO] USE_DGP_QDTI=${USE_DGP_QDTI} USE_QDTI_BIAS=${USE_QDTI_BIAS}"
echo "[INFO] QDTI_APPLY_LAYERS=${QDTI_APPLY_LAYERS} QDTI_SCALE_INIT=${QDTI_SCALE_INIT}"
echo "[INFO] LORA_R=${LORA_R} LEARNING_RATE=${LEARNING_RATE}"
echo "[INFO] EVAL_METRICS_SCRIPT=${EVAL_METRICS_SCRIPT}"
echo "[INFO] EVAL_WANDB_PROJECT=${EVAL_WANDB_PROJECT}"
echo "[INFO] EVAL_WANDB_RUN_NAME=${EVAL_WANDB_RUN_NAME}"

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

echo "[OK] preflight passed"

########################################
# 1) Train
########################################
echo "========================================"
echo "[1/6] Training (includes val + best ckpt)"
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
  --use_dgp_qdti "${USE_DGP_QDTI}" \
  --use_qdti_bias "${USE_QDTI_BIAS}" \
  --dgp_fuse_dim "${DGP_FUSE_DIM}" \
  --dgp_refiner_hidden_dim "${DGP_REFINER_HIDDEN_DIM}" \
  --dgp_pg_tokens "${DGP_PG_TOKENS}" \
  --qdti_bias_dim "${QDTI_BIAS_DIM}" \
  --qdti_init_std "${QDTI_INIT_STD}" \
  --qdti_max_abs "${QDTI_MAX_ABS}" \
  --qdti_apply_layers "${QDTI_APPLY_LAYERS}" \
  --qdti_scale_init "${QDTI_SCALE_INIT}" \
  --scale_hard_loss_weight "${SCALE_HARD_LOSS_WEIGHT}" \
  --dgp_monitor_wandb True \
  --dgp_monitor_steps 10 \
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
echo "[4/6] Check merged DGP config"
echo "========================================"

check_merged_dgp_config "${MERGED_DIR}"
echo "[OK] merged config.json DGP fields verified"

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
echo "Output dir        : ${OUTPUT_DIR}"
echo "Selected ckpt     : ${BEST_CHECKPOINT}"
echo "Merged model      : ${MERGED_DIR}"
echo "Test outputs      : ${TEST_OUTPUT_DIR}"
echo "DGP-QDTI          : use_dgp_qdti=${USE_DGP_QDTI} use_qdti_bias=${USE_QDTI_BIAS}"
echo "Eval W&B project  : ${EVAL_WANDB_PROJECT}"
echo "Eval W&B run name : ${EVAL_WANDB_RUN_NAME}"
echo "========================================"
