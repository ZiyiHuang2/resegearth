#!/usr/bin/env bash
# Eval + metrics for seg_refiner_frozen_v2 (RRSISD test).
set -euo pipefail

REPO_DIR="/root/rivermind-data/huangziyi/reseg/resegearth+tgi"
cd "${REPO_DIR}"

PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
GPU_ID="${GPU_ID:-0}"

OUTPUT_DIR="/root/rivermind-data/huangziyi/reseg/output/tgi/seg_refiner_frozen_v2"
MERGED_DIR="${OUTPUT_DIR}/merged_model"
# Always eval merged_model (sidecar refiner weights folded in at merge time).
MODEL_DIR="${MODEL_DIR:-${MERGED_DIR}}"
TEST_OUTPUT_DIR="${OUTPUT_DIR}/test_results"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint-30000"

BASELINE_MERGED="/root/rivermind-data/huangziyi/reseg/output/base/standard-base-siglip1-28w-gd4/merged_model"
VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
BASE_DATA_PATH="/root/rivermind-data/huangziyi/data/RRSISD"
BASELINE_PRED_DIR="/root/rivermind-data/huangziyi/reseg/output/base/standard-base-siglip1-28w-gd4/test_results"
EVAL_METRICS_SCRIPT="/root/rivermind-data/huangziyi/reseg/eval_val_metrics.py"

EVAL_LOG="${OUTPUT_DIR}/test_eval.log"
METRICS_LOG="${OUTPUT_DIR}/test_metrics.log"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  if [[ -d "${CHECKPOINT_DIR}" ]]; then
    echo "[INFO] merged_model missing; merging ${CHECKPOINT_DIR} -> ${MERGED_DIR}"
    rm -rf "${MERGED_DIR}"
    mkdir -p "${MERGED_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
      --model_path "${CHECKPOINT_DIR}" \
      --vision_tower "${VISION_TOWER}" \
      --vision_tower_mask "${VISION_TOWER_MASK}" \
      --mask_config "${MASK_CONFIG}" \
      --save_path "${MERGED_DIR}" \
      --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
      --use_seg_spatial_refiner True \
      --seg_spatial_refiner_alpha 0.1 \
      --seg_spatial_refiner_loss_weight 0.1 \
      --seg_spatial_refiner_dice_weight 1.0 \
      --seg_spatial_refiner_bce_weight 1.0 \
      --use_query_aware_decoder_bias False --use_decoder_attn_bias False \
      --use_mstva False --use_mstva_loss False --use_text_film False
  else
    echo "[ERROR] model config not found: ${MODEL_DIR}/config.json"
    exit 1
  fi
fi

if [[ ! -f "${EVAL_METRICS_SCRIPT}" ]]; then
  echo "[ERROR] metrics script not found: ${EVAL_METRICS_SCRIPT}"
  exit 1
fi

echo "[1/2] RRSISD test eval -> ${TEST_OUTPUT_DIR}"
rm -rf "${TEST_OUTPUT_DIR}"
mkdir -p "${TEST_OUTPUT_DIR}"

{
  echo "=== seg_refiner_frozen_v2 eval $(date -Iseconds) ==="
  echo "model=${MODEL_DIR}"
  echo "out=${TEST_OUTPUT_DIR}"
} | tee "${EVAL_LOG}"

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON}" segearth_r2/eval/eval.py \
    --base_data_path "${BASE_DATA_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --model_path "${MODEL_DIR}" \
    --output_dir "${TEST_OUTPUT_DIR}" \
    --dataset_name rrsisd \
    --split test \
    --eval_batch_size 1 \
    --dataloader_num_workers 4 \
    --zip_results False 2>&1 | tee -a "${EVAL_LOG}"

echo "[2/2] metrics (W&B + json)"
{
  echo "=== metrics $(date -Iseconds) ==="
} | tee "${METRICS_LOG}"

USE_WANDB="${USE_WANDB:-True}" \
WANDB_PROJECT="${WANDB_PROJECT:-segearth-eval-tgi-val}" \
WANDB_RUN_NAME="${WANDB_RUN_NAME:-seg_refiner_frozen_v2}" \
DATASET_TYPE=rrsisd \
BASE_DATA_PATH="${BASE_DATA_PATH}" \
SPLIT=test \
PRED_DIR="${TEST_OUTPUT_DIR}" \
METRICS_TAG=seg_refiner_frozen_v2 \
BASELINE_PRED_DIR="${BASELINE_PRED_DIR}" \
  "${PYTHON}" "${EVAL_METRICS_SCRIPT}" 2>&1 | tee -a "${METRICS_LOG}"

echo "[DONE] test_results=${TEST_OUTPUT_DIR}"
echo "[DONE] metrics log=${METRICS_LOG}"
ls -la "${OUTPUT_DIR}"/*metrics*.json 2>/dev/null || true
