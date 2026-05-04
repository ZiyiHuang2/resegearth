#!/usr/bin/env bash
set -euo pipefail

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT=segearth-bseg
export WANDB_INIT_TIMEOUT=300
# HF Trainer 使用 --run_name 作为 W&B run 名；WANDB_NAME 仅作部分旧版 wandb 的补充
WANDB_EXPERIMENT_NAME="contrast_7w_heads8"
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="localhost:2"
GPU_ID="2"

# 与 run_refined_structured_7w.sh 的 29331 错开，避免与其它 deepspeed 任务抢端口
MASTER_PORT="29332"

########################################
# Project dir
########################################
REPO_DIR="/home/wangchengjun/huangziyi/reseg/resegearth+bseg"
cd "${REPO_DIR}"

########################################
# Common paths
########################################
MODEL_NAME_OR_PATH="/home/wangchengjun/huangziyi/reseg/output/standard-base/merged_model"
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
MAX_STEPS="70000"
PER_DEVICE_TRAIN_BATCH_SIZE="1"
GRADIENT_ACCUMULATION_STEPS="1"

# 与 run_refined_structured_7w.sh 一致：过小 save_total_limit 会频繁删 ckpt，best 可能异常或丢步
SAVE_STEPS="5000"
SAVE_TOTAL_LIMIT="6"

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
MAX_GRAD_NORM="1.0"

########################################
# Eval metrics config (auto upload)
########################################
EVAL_METRICS_SCRIPT="/home/wangchengjun/huangziyi/reseg/eval_val_metrics.py"
EVAL_USE_WANDB="True"
EVAL_WANDB_PROJECT="segearth-eval-bseg-val"

########################################
# Refined structured config (top_k_heads = 8)
########################################
TOP_K_HEADS="8"

TARGET_LAYERS="24,25,26,27,28"
TARGET_HEADS_CONFIG_PATH="/home/wangchengjun/huangziyi/reseg/resegearth+bseg/configs/target_heads_rrsisd_top8.json"

ATTENTION_LOSS_WEIGHT="0.005"
STRUCTURED_FG_BG_WEIGHT="0.8"
STRUCTURED_BOUNDARY_OUTER_WEIGHT="0.2"
STRUCTURED_ATTENTION_MARGIN="0.05"

SMALL_WEIGHT="1.5"
SMALL_AREA_RATIO_THRESHOLD="0.03"

########################################
# Structured schedule / safety
########################################
STRUCTURED_WARMUP_STEPS="5000"
STRUCTURED_DECAY_START_STEP="60000"
STRUCTURED_DECAY_END_STEP="70000"
STRICT_ATTENTION_SELECTION="False"
STRUCTURED_LOG_INTERVAL="50"

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

run_heads8 () {
  # 勿与 refined_structured_standard-base_7w_heads8 共用目录：否则易误 resume / 覆盖旧 trainer_state 与 checkpoint
  local output_dir="/home/wangchengjun/huangziyi/reseg/output/bseg/contrast_7w_heads8"
  local merged_dir="${output_dir}/merged_model"
  local test_dir="${output_dir}/test_results"

  echo "========================================"
  echo "[RUN] ${WANDB_EXPERIMENT_NAME}"
  echo "========================================"
  echo "[OK] output_dir: ${output_dir}"
  echo "[OK] top_k_heads: ${TOP_K_HEADS}"
  echo "[OK] target_heads_config: ${TARGET_HEADS_CONFIG_PATH}"
  echo "[OK] wandb run name (--run_name): ${WANDB_EXPERIMENT_NAME}"

  export WANDB_NAME="${WANDB_EXPERIMENT_NAME}"

  deepspeed --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --base_data_path "${BASE_DATA_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --output_dir "${output_dir}" \
    --run_name "${WANDB_EXPERIMENT_NAME}" \
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
    --use_attention_loss True \
    --use_structured_attention_loss True \
    --use_precomputed_structured_maps False \
    --target_layers "${TARGET_LAYERS}" \
    --target_heads_config_path "${TARGET_HEADS_CONFIG_PATH}" \
    --top_k_heads "${TOP_K_HEADS}" \
    --attention_loss_weight "${ATTENTION_LOSS_WEIGHT}" \
    --structured_fg_bg_weight "${STRUCTURED_FG_BG_WEIGHT}" \
    --structured_boundary_outer_weight "${STRUCTURED_BOUNDARY_OUTER_WEIGHT}" \
    --structured_attention_margin "${STRUCTURED_ATTENTION_MARGIN}" \
    --small_weight "${SMALL_WEIGHT}" \
    --small_area_ratio_threshold "${SMALL_AREA_RATIO_THRESHOLD}" \
    --structured_warmup_steps "${STRUCTURED_WARMUP_STEPS}" \
    --structured_decay_start_step "${STRUCTURED_DECAY_START_STEP}" \
    --structured_decay_end_step "${STRUCTURED_DECAY_END_STEP}" \
    --strict_attention_selection "${STRICT_ATTENTION_SELECTION}" \
    --structured_log_interval "${STRUCTURED_LOG_INTERVAL}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --report_to wandb

  echo "========================================"
  echo "[INFO] Reading best checkpoint for heads${TOP_K_HEADS}"
  echo "========================================"

  local best_ckpt
  best_ckpt=$(read_best_checkpoint "${output_dir}")

  if [[ -z "${best_ckpt}" ]]; then
    echo "[ERROR] best_model_checkpoint not found for heads${TOP_K_HEADS}"
    exit 1
  fi

  echo "[OK] BEST_CHECKPOINT=${best_ckpt}"

  echo "========================================"
  echo "[INFO] Merge + eval for heads${TOP_K_HEADS}"
  echo "========================================"

  merge_ckpt "${best_ckpt}" "${merged_dir}"
  eval_model "${merged_dir}" "${test_dir}"

  echo "========================================"
  echo "[INFO] Eval metrics + upload for heads${TOP_K_HEADS}"
  echo "========================================"

  run_eval_metrics "${test_dir}" "${WANDB_EXPERIMENT_NAME}"

  echo "========================================"
  echo "[DONE] ${WANDB_EXPERIMENT_NAME}"
  echo "best ckpt : ${best_ckpt}"
  echo "merged    : ${merged_dir}"
  echo "test out  : ${test_dir}"
  echo "========================================"
}

########################################
# Preflight
########################################
echo "========================================"
echo "[0/2] Preflight checks (heads=8 only)"
echo "========================================"

if [[ ! -f "${TARGET_HEADS_CONFIG_PATH}" ]]; then
  echo "[ERROR] target heads config not found:"
  echo "        ${TARGET_HEADS_CONFIG_PATH}"
  exit 1
fi

echo "[OK] target heads config found: ${TARGET_HEADS_CONFIG_PATH}"
echo "[OK] target layers: ${TARGET_LAYERS}"
echo "[OK] top_k_heads: ${TOP_K_HEADS}"
echo "[OK] warmup steps: ${STRUCTURED_WARMUP_STEPS}"
echo "[OK] decay: ${STRUCTURED_DECAY_START_STEP} -> ${STRUCTURED_DECAY_END_STEP}"

########################################
# 1) Train + merge + eval + metrics (heads=8)
########################################
run_heads8

########################################
# 2) Done
########################################
echo "========================================"
echo "[2/2] ALL DONE"
echo "Finished heads=8 experiment only."
echo "========================================"
