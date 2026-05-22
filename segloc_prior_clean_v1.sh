#!/usr/bin/env bash
set -euo pipefail

########################################
# 实验版本：segloc_prior_clean_v1
########################################
#
# 假设（prior 质量验收）
#   若 [SEG] 被约束到「能指定目标」的方向，在关掉旧 attention structured 支路后，
#   自生成的 coarse prior 应至少满足：
#   - fg_mass 明显高于 bg_mass
#   - top1_in_fg_ratio 明显高于随机水平
#   - coarse_iou_mean 随训练稳定上升
#   且上述现象在分布上可见，而非极少数样本。
#
# 本脚本只做一条干净配置，便于归因：
#   必开：use_seg_loc_prior + prior 导出/诊断日志
#   必关：use_attention_loss / use_structured_attention_loss / use_seg_query_refiner
#
# 其余训练/数据/merge/eval 排版与 contrast_7w_heads8.sh 对齐。
########################################

########################################
# Environment
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT=segearth-bseg
export WANDB_INIT_TIMEOUT=300
WANDB_EXPERIMENT_NAME="segloc_prior_clean_v1"
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="localhost:2"
GPU_ID="2"

# 与 contrast_7w_heads8.sh (29332) 错开
MASTER_PORT="29334"

########################################
# Project dir
########################################
REPO_DIR="/home/wangchengjun/huangziyi/reseg/resegearth+bseg"
cd "${REPO_DIR}"

########################################
# Common paths
########################################
MODEL_NAME_OR_PATH="/home/wangchengjun/huangziyi/reseg/output/tgi/mstva_loss_w01_7w/merged_model"
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
# Train config（与 contrast_7w_heads8 一致）
########################################
MAX_STEPS="70000"
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
MAX_GRAD_NORM="1.0"

########################################
# Eval metrics config (auto upload)
########################################
EVAL_METRICS_SCRIPT="/home/wangchengjun/huangziyi/reseg/eval_val_metrics.py"
EVAL_USE_WANDB="True"
EVAL_WANDB_PROJECT="segearth-eval-bseg-val"

########################################
# Mask / small 对象权重（与 contrast 一致；与 attention 支路无关）
########################################
SMALL_WEIGHT="1.5"
SMALL_AREA_RATIO_THRESHOLD="0.03"

########################################
# --- segloc prior：固定一组，不扫参 ---
########################################
SEG_LOC_PRIOR_GRID_SIZE="14"
SEG_LOC_HIDDEN_DIM="512"
SEG_LOC_PRIOR_WEIGHT="0.1"
SEG_LOC_PRIOR_ENTROPY_HIGH_THRESH="0.65"
SEG_LOC_PRIOR_LOW_FG_MASS_THRESH="0.30"

########################################
# Helpers（与 contrast_7w_heads8.sh 相同）
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

run_segloc_prior_clean_v1 () {
  local output_dir="/home/wangchengjun/huangziyi/reseg/output/bseg/segloc_prior_clean_v1"
  local merged_dir="${output_dir}/merged_model"
  local test_dir="${output_dir}/test_results"

  echo "========================================"
  echo "[RUN] ${WANDB_EXPERIMENT_NAME}"
  echo "========================================"
  echo "[OK] output_dir: ${output_dir}"
  echo "[OK] isolation: attention_loss=off structured=off refiner=off | seg_loc_prior=on"
  echo "[OK] prior export: metrics+jsonl + samples+jsonl | diagnose_seg_loc_prior_only log"
  echo "[OK] wandb run name (--run_name): ${WANDB_EXPERIMENT_NAME}"
  echo "[OK] prior jsonl -> ${output_dir}/seg_loc_prior_metrics.jsonl"
  echo "                -> ${output_dir}/seg_loc_prior_samples.jsonl"

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
    --small_weight "${SMALL_WEIGHT}" \
    --small_area_ratio_threshold "${SMALL_AREA_RATIO_THRESHOLD}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --report_to wandb \
    --use_attention_loss False \
    --use_structured_attention_loss False \
    --use_seg_query_refiner False \
    --use_seg_loc_prior True \
    --seg_loc_prior_grid_size "${SEG_LOC_PRIOR_GRID_SIZE}" \
    --seg_loc_hidden_dim "${SEG_LOC_HIDDEN_DIM}" \
    --seg_loc_prior_weight "${SEG_LOC_PRIOR_WEIGHT}" \
    --seg_loc_prior_entropy_high_thresh "${SEG_LOC_PRIOR_ENTROPY_HIGH_THRESH}" \
    --seg_loc_prior_low_fg_mass_thresh "${SEG_LOC_PRIOR_LOW_FG_MASS_THRESH}" \
    --export_seg_loc_prior_metrics True \
    --export_seg_loc_prior_samples True \
    --diagnose_seg_loc_prior_only True

  echo "========================================"
  echo "[INFO] Reading best checkpoint"
  echo "========================================"

  local best_ckpt
  best_ckpt=$(read_best_checkpoint "${output_dir}")

  if [[ -z "${best_ckpt}" ]]; then
    echo "[ERROR] best_model_checkpoint not found under ${output_dir}"
    exit 1
  fi

  echo "[OK] BEST_CHECKPOINT=${best_ckpt}"

  echo "========================================"
  echo "[INFO] Merge + eval"
  echo "========================================"

  merge_ckpt "${best_ckpt}" "${merged_dir}"
  eval_model "${merged_dir}" "${test_dir}"

  echo "========================================"
  echo "[INFO] Eval metrics + upload"
  echo "========================================"

  run_eval_metrics "${test_dir}" "${WANDB_EXPERIMENT_NAME}"

  echo "========================================"
  echo "[DONE] ${WANDB_EXPERIMENT_NAME}"
  echo "best ckpt : ${best_ckpt}"
  echo "merged    : ${merged_dir}"
  echo "test out  : ${test_dir}"
  echo "prior logs: ${output_dir}/seg_loc_prior_*.jsonl"
  echo "========================================"
}

########################################
# Preflight
########################################
echo "========================================"
echo "[0/2] Preflight (segloc_prior_clean_v1)"
echo "========================================"

if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "[ERROR] model path not found: ${MODEL_NAME_OR_PATH}"
  exit 1
fi

echo "[OK] model: ${MODEL_NAME_OR_PATH}"
echo "[OK] seg loc prior: grid=${SEG_LOC_PRIOR_GRID_SIZE} hidden=${SEG_LOC_HIDDEN_DIM} w=${SEG_LOC_PRIOR_WEIGHT}"
echo "[OK] log ratios: entropy_high>${SEG_LOC_PRIOR_ENTROPY_HIGH_THRESH} fg_mass_low<${SEG_LOC_PRIOR_LOW_FG_MASS_THRESH}"

########################################
# 1) Train + merge + eval + metrics
########################################
run_segloc_prior_clean_v1

########################################
# 2) Done
########################################
echo "========================================"
echo "[2/2] ALL DONE (segloc_prior_clean_v1)"
echo "========================================"
