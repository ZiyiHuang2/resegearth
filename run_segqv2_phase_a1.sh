#!/usr/bin/env bash
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_INIT_TIMEOUT=300
export WANDB_PROJECT=segearth-seg-train
unset CUDA_VISIBLE_DEVICES

# DeepSpeed 训练使用物理 0、2 号卡（双卡）
GPU_SLOT="localhost:0,2"
# merge / eval 单卡，固定用 0 号卡（避免与双进程争用）
MERGE_EVAL_GPU_ID="0"

########################################
# Project dir
########################################
REPO_DIR="/home/wangchengjun/huangziyi/reseg/resegearth+seg"
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

LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"

DATA_RATIO="1"
SWITCH_BS="4"

SEED="42"
DATA_SEED="42"

########################################
# Output root
########################################
EXP_ROOT="/home/wangchengjun/huangziyi/reseg/output/segqv2_phase_a1"

########################################
# Eval metrics config (auto upload to W&B)
########################################
EVAL_METRICS_SCRIPT="/home/wangchengjun/huangziyi/reseg/eval_val_metrics.py"
EVAL_USE_WANDB="True"
EVAL_WANDB_PROJECT="segearth-eval-seg-val"

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

  CUDA_VISIBLE_DEVICES="${MERGE_EVAL_GPU_ID}" python segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
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

  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${MERGE_EVAL_GPU_ID}" \
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
# Preflight (once, before all exps)
########################################
echo "========================================"
echo "[0/6] Preflight checks"
echo "========================================"

echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] GPU_SLOT=${GPU_SLOT} (train)"
echo "[INFO] MERGE_EVAL_GPU_ID=${MERGE_EVAL_GPU_ID} (merge/eval)"
echo "[INFO] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "[INFO] VISION_TOWER=${VISION_TOWER}"
echo "[INFO] VISION_TOWER_MASK=${VISION_TOWER_MASK}"
echo "[INFO] MASK_CONFIG=${MASK_CONFIG}"
echo "[INFO] BASE_DATA_PATH=${BASE_DATA_PATH}"
echo "[INFO] EXP_ROOT=${EXP_ROOT}"
echo "[INFO] EVAL_METRICS_SCRIPT=${EVAL_METRICS_SCRIPT}"
echo "[INFO] EVAL_WANDB_PROJECT=${EVAL_WANDB_PROJECT}"

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

mkdir -p "${EXP_ROOT}"

echo "[OK] preflight passed"

########################################
# Per-experiment pipeline
########################################
run_one () {
  local exp_name="$1"
  local master_port="$2"
  local num_local_queries="$3"
  local use_query_score_supervision="$4"

  export WANDB_NAME="${exp_name}"

  local output_dir="${EXP_ROOT}/${exp_name}"
  local merged_dir="${output_dir}/merged_model"
  local test_output_dir="${output_dir}/test_results"

  echo "========================================"
  echo "[EXP] ${exp_name}"
  echo "output_dir=${output_dir}"
  echo "num_local_queries=${num_local_queries}"
  echo "use_query_score_supervision=${use_query_score_supervision}"
  echo "========================================"

  ########################################
  # 1) Train
  ########################################
  echo "========================================"
  echo "[1/6] [${exp_name}] Training"
  echo "========================================"

  deepspeed --master_port="${master_port}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --base_data_path "${BASE_DATA_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --output_dir "${output_dir}" \
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
    --report_to wandb \
    --num_local_queries "${num_local_queries}" \
    --use_text_cross_attn_refine False \
    --use_query_score_supervision "${use_query_score_supervision}" \
    --use_query_margin_loss False \
    --query_score_loss_weight 0.25 \
    --query_div_loss_weight 0.0 \
    --query_margin_loss_weight 0.0 \
    --inference_query_select_mode top1

  ########################################
  # 2) Select checkpoint
  ########################################
  echo "========================================"
  echo "[2/6] [${exp_name}] Select checkpoint"
  echo "========================================"

  local best_checkpoint
  best_checkpoint=$(read_best_checkpoint "${output_dir}")

  if [[ -z "${best_checkpoint}" ]]; then
    echo "[WARN] best_model_checkpoint not found; fallback to last checkpoint"
    best_checkpoint=$(read_last_checkpoint "${output_dir}")
  fi

  if [[ -z "${best_checkpoint}" ]]; then
    echo "[ERROR] no checkpoint found under ${output_dir}"
    exit 1
  fi

  echo "[OK] SELECTED_CHECKPOINT=${best_checkpoint}"

  ########################################
  # 3) Merge
  ########################################
  echo "========================================"
  echo "[3/6] [${exp_name}] Merge selected checkpoint"
  echo "========================================"

  merge_ckpt "${best_checkpoint}" "${merged_dir}"

  ########################################
  # 4) Check merged config
  ########################################
  echo "========================================"
  echo "[4/6] [${exp_name}] Check merged config"
  echo "========================================"

  if [[ ! -f "${merged_dir}/config.json" ]]; then
    echo "[ERROR] merged config.json not found: ${merged_dir}/config.json"
    exit 1
  fi

  echo "[OK] merged config.json present"

  ########################################
  # 5) Eval + upload eval metrics
  ########################################
  echo "========================================"
  echo "[5/6] [${exp_name}] Eval + upload eval metrics"
  echo "========================================"

  eval_model "${merged_dir}" "${test_output_dir}"
  run_eval_metrics "${test_output_dir}" "segqv2_phase_a1__${exp_name}"

  ########################################
  # 6) Done (this exp)
  ########################################
  echo "========================================"
  echo "[6/6] [${exp_name}] DONE"
  echo "Best checkpoint : ${best_checkpoint}"
  echo "Merged model    : ${merged_dir}"
  echo "Test outputs    : ${test_output_dir}"
  echo "Eval W&B project: ${EVAL_WANDB_PROJECT}"
  echo "Eval W&B run    : segqv2_phase_a1__${exp_name}"
  echo "========================================"
}

########################################
# Experiments
########################################

# 1) Phase A1 主实验：K=16, score=off
run_one "a1_k16_top1_scoreoff" 29601 16 False

# 2) Phase A1 主实验：K=32, score=off
run_one "a1_k32_top1_scoreoff" 29602 32 False

# 3) Score supervision 对照：K=32, score=on
run_one "a1_k32_top1_scoreon" 29603 32 True

# 4) Query 数量继续探：K=64, score=off
run_one "a1_k64_top1_scoreoff" 29604 64 False

echo "========================================"
echo "All Phase A1 experiments completed."
echo "========================================"
