#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/wangchengjun/huangziyi/reseg/segearth+text"
cd "${REPO_DIR}"

CONDA_ENV="${CONDA_ENV:-reseg}"
PYTHON=(conda run -n "${CONDA_ENV}" --no-capture-output python)

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

MODEL_PATH="/home/wangchengjun/huangziyi/reseg/output/base/standard-base-siglip1-28w/merged_model"
BASE_DATA_PATH="/home/wangchengjun/huangziyi/data/RRSISD"
OUTPUT_BASE="${REPO_DIR}/output/standard-base-siglip1-28w_prompt_ensemble"
PRED_DIR="${OUTPUT_BASE}/test_results"
LOG_FILE="${OUTPUT_BASE}/eval.log"

VISION_TOWER="/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
EVAL_METRICS_SCRIPT="/home/wangchengjun/huangziyi/reseg/eval_val_metrics.py"

mkdir -p "${OUTPUT_BASE}" "${PRED_DIR}"

echo "[1/2] RRSIS-D test inference (prompt ensemble K=8) ..."
"${PYTHON[@]}" segearth_r2/eval/eval.py \
  --base_data_path "${BASE_DATA_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${PRED_DIR}" \
  --dataset_name rrsisd \
  --split test \
  --eval_batch_size 1 \
  --dataloader_num_workers 4 \
  --prompt_ensemble \
  --zip_results False \
  2>&1 | tee "${LOG_FILE}"

echo "[2/2] Compute metrics (no wandb) ..."
USE_WANDB=False \
DATASET_TYPE=rrsisd \
BASE_DATA_PATH="${BASE_DATA_PATH}" \
SPLIT=test \
PRED_DIR="${PRED_DIR}" \
"${PYTHON[@]}" "${EVAL_METRICS_SCRIPT}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "Done. Predictions: ${PRED_DIR}"
echo "Metrics: ${OUTPUT_BASE}/rrsisd_test_metrics.json"
