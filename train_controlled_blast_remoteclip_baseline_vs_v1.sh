#!/usr/bin/env bash
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-clip-remoteclip}"
export WANDB_INIT_TIMEOUT=300
unset CUDA_VISIBLE_DEVICES

# 让 deepspeed 与 PYTHON_BIN 同属一个 conda env（若 deepspeed 不在 PATH，可先安装或改 DEEPSPEED_BIN）
PYTHON_BIN="${PYTHON_BIN:-/home/wangchengjun/miniconda3/envs/reseg/bin/python}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"

# 训练：DeepSpeed --include，默认单卡物理 0 号（多卡示例：GPU_SLOT=localhost:0,2）
GPU_SLOT="${GPU_SLOT:-localhost:0}"
# merge / eval 单进程；默认与训练同卡 0（改训练卡时请同步 MERGE_EVAL_GPU_ID）
MERGE_EVAL_GPU_ID="${MERGE_EVAL_GPU_ID:-${GPU_ID:-0}}"
MASTER_PORT="${MASTER_PORT:-29511}"

########################################
# Project dir (segearth+clip)
########################################
REPO_DIR="/home/wangchengjun/huangziyi/reseg/segearth+clip"
cd "${REPO_DIR}"

########################################
# Paths
########################################
TRAIN_ENTRY="${REPO_DIR}/segearth_r2/train/train.py"
DEEPSPEED_CONFIG="${REPO_DIR}/scripts/zero1.json"

BASELINE_CKPT="${BASELINE_CKPT:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mllm/Mipha-3B}"
BASE_DATA_PATH="${BASE_DATA_PATH:-/home/wangchengjun/huangziyi/data/RRSISD}"
DATASET_NAME="${DATASET_NAME:-rrsisd}"
TEST_SPLIT="${TEST_SPLIT:-test}"

VISION_TOWER="${VISION_TOWER:-/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"

REMOTECLIP_WEIGHT="${REMOTECLIP_WEIGHT:-/home/wangchengjun/huangziyi/reseg/pretrained_model/RemoteCLIP/RemoteCLIP-ViT-B-32.pt}"

OUTPUT_DIR="${OUTPUT_DIR:-/home/wangchengjun/huangziyi/reseg/output/segearth+clip/controlled_blast_remoteclip_prior}"

EVAL_METRICS_SCRIPT="${EVAL_METRICS_SCRIPT:-/home/wangchengjun/huangziyi/reseg/eval_val_metrics.py}"
EVAL_USE_WANDB="${EVAL_USE_WANDB:-True}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-clip-remoteclip}"

########################################
# Train config（RemoteCLIP prior 单路）
# 显存：ZeRO-1 下每卡仍是整模型 + RemoteCLIP；OOM 时优先再降 BATCH_SIZE / 提 GRAD_ACC / 开 checkpointing。
########################################
SEED=42
DATA_SEED=42
MAX_STEPS=70000
SAVE_STEPS=2000
LOGGING_STEPS=50
# 每卡 batch；RemoteCLIP+大模型在 24GB 上 batch=2 易 OOM，默认 1
BATCH_SIZE="${BATCH_SIZE:-1}"
# 梯度累积步数；多卡时可提高以维持等效全局 batch
GRAD_ACC="${GRAD_ACC:-2}"
LR="1e-4"

LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

WANDB_RUN_NAME="${WANDB_RUN_NAME:-controlled_blast_remoteclip_prior}"

BF16="True"
TF32="False"
WEIGHT_DECAY="0.0"
WARMUP_RATIO="0.03"
LR_SCHEDULER_TYPE="cosine"
MODEL_MAX_LENGTH="2048"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
DATA_RATIO="1"
SWITCH_BS="4"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"

########################################
# Helpers
########################################
read_best_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR="${out_dir}" "${PYTHON_BIN}" - <<'PY'
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

print(state.get("best_model_checkpoint", "") or "")
PY
}

read_last_checkpoint () {
  local out_dir="$1"
  OUTPUT_DIR="${out_dir}" "${PYTHON_BIN}" - <<'PY'
import os
import re
import sys

output_dir = os.environ["OUTPUT_DIR"]
if not os.path.isdir(output_dir):
    print("")
    sys.exit(0)

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

pick_checkpoint_for_merge () {
  local out_dir="$1"
  local best_ckpt
  best_ckpt=$(read_best_checkpoint "${out_dir}")
  if [[ -n "${best_ckpt}" && -d "${best_ckpt}" ]]; then
    echo "${best_ckpt}"
    return 0
  fi
  echo "[WARN] best_model_checkpoint 为空或目录不存在，回退到最新 checkpoint-*" >&2
  read_last_checkpoint "${out_dir}"
}

merge_ckpt () {
  local ckpt="$1"
  local save_dir="$2"

  rm -rf "${save_dir}"
  mkdir -p "${save_dir}"

  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${MERGE_EVAL_GPU_ID}" \
    "${PYTHON_BIN}" segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
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
    "${PYTHON_BIN}" "${EVAL_METRICS_SCRIPT}"
}

run_train_merge_eval_metrics () {
  local group_tag="$1"
  local output_dir="$2"
  local wandb_run_name="$3"
  local master_port="$4"
  shift 4
  # "$@" 为该组 train 额外参数（本脚本仅 RemoteCLIP prior 一路）

  local merged_dir="${output_dir}/merged_model"
  local test_dir="${output_dir}/test_results"

  echo "========================================"
  echo "[TRAIN] ${group_tag}"
  echo "[INFO] output_dir=${output_dir}"
  echo "[INFO] --run_name=${wandb_run_name}"
  echo "========================================"

  export WANDB_NAME="${wandb_run_name}"

  "${DEEPSPEED_BIN}" --master_port="${master_port}" --include="${GPU_SLOT}" "${TRAIN_ENTRY}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --model_name_or_path "${BASELINE_CKPT}" \
    --base_data_path "${BASE_DATA_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --output_dir "${output_dir}" \
    --run_name "${wandb_run_name}" \
    --report_to wandb \
    --seed "${SEED}" \
    --data_seed "${DATA_SEED}" \
    --max_steps "${MAX_STEPS}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --logging_steps "${LOGGING_STEPS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --learning_rate "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --bf16 "${BF16}" \
    --tf32 "${TF32}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --data_ratio "${DATA_RATIO}" \
    --switch_bs "${SWITCH_BS}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    "$@"

  echo "========================================"
  echo "[BEST] ${group_tag} — 读取 checkpoint"
  echo "========================================"

  local ckpt_for_merge
  ckpt_for_merge=$(pick_checkpoint_for_merge "${output_dir}")
  if [[ -z "${ckpt_for_merge}" || ! -d "${ckpt_for_merge}" ]]; then
    echo "[ERROR] 未找到可用于 merge 的 checkpoint，目录: ${output_dir}"
    exit 1
  fi
  echo "[OK] MERGE_SOURCE_CHECKPOINT=${ckpt_for_merge}"

  echo "========================================"
  echo "[MERGE] ${group_tag}"
  echo "========================================"
  merge_ckpt "${ckpt_for_merge}" "${merged_dir}"

  echo "========================================"
  echo "[EVAL] ${group_tag} — merged -> test tif"
  echo "========================================"
  eval_model "${merged_dir}" "${test_dir}"

  echo "========================================"
  echo "[METRICS] ${group_tag} — ${DATASET_NAME}_${TEST_SPLIT}_metrics.json"
  echo "========================================"
  run_eval_metrics "${test_dir}" "${wandb_run_name}"

  echo "========================================"
  echo "[DONE] ${group_tag}"
  echo "  best/used ckpt : ${ckpt_for_merge}"
  echo "  merged_model   : ${merged_dir}"
  echo "  test_results   : ${test_dir}"
  echo "  metrics json   : $(dirname "${test_dir}")/${DATASET_NAME}_${TEST_SPLIT}_metrics.json"
  echo "========================================"
}

########################################
# Preflight
########################################
echo "========================================"
echo "[0/1] Preflight（仅 RemoteCLIP / clip prior）"
echo "========================================"
echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"
echo "[INFO] DEEPSPEED_BIN=${DEEPSPEED_BIN}"
echo "[INFO] GPU_SLOT (train)=${GPU_SLOT}"
echo "[INFO] MERGE_EVAL_GPU_ID=${MERGE_EVAL_GPU_ID}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] WANDB_RUN_NAME=${WANDB_RUN_NAME}"
echo "[INFO] per_device_train_batch_size=${BATCH_SIZE} gradient_accumulation_steps=${GRAD_ACC} gradient_checkpointing=${GRADIENT_CHECKPOINTING}"

if [[ ! -f "${REMOTECLIP_WEIGHT}" ]]; then
  echo "[ERROR] RemoteCLIP weight not found: ${REMOTECLIP_WEIGHT}"
  exit 1
fi
if [[ ! -f "${EVAL_METRICS_SCRIPT}" ]]; then
  echo "[ERROR] eval metrics script not found: ${EVAL_METRICS_SCRIPT}"
  exit 1
fi
echo "[OK] RemoteCLIP weight exists"
echo "[OK] eval_val_metrics exists"

########################################
# RemoteCLIP prior（use_remoteclip_prior=True）
########################################
run_train_merge_eval_metrics \
  "remoteclip_prior" \
  "${OUTPUT_DIR}" \
  "${WANDB_RUN_NAME}" \
  "${MASTER_PORT}" \
  --use_remoteclip_prior True \
  --use_confidence_scaling True \
  --use_weak_residual True \
  --remoteclip_weight_path "${REMOTECLIP_WEIGHT}"

echo "========================================"
echo "[ALL DONE] RemoteCLIP prior 全流程结束"
echo "========================================"
