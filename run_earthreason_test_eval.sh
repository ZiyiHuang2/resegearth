#!/usr/bin/env bash
# EarthReason test eval + metrics for one model (set OUT_DIR / MODEL_PATH).
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=disabled
export TMPDIR="/root/rivermind-data/tmp"
mkdir -p "${TMPDIR}"

OUT="${OUT_DIR:-/root/rivermind-data/huangziyi/reseg/output/base/standard-base-lasers-siglip1-5w-gd4}"
MODEL="${MODEL_PATH:-${OUT}/merged_model}"
DATA="/root/rivermind-data/huangziyi/data/EarthReason"
REPO="/root/rivermind-data/huangziyi/reseg/segearth+base"
PY="/root/rivermind-data/miniconda3/envs/reseg/bin/python"
METRICS="/root/rivermind-data/huangziyi/reseg/eval_val_metrics.py"
VT="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VTM="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MCFG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
NAME="earthreason"
SPLIT="test"

PRED_DIR="${OUT}/${NAME}_test_results"
LOG_EVAL="${OUT}/${NAME}_test_eval.log"
LOG_METRICS="${OUT}/${NAME}_test_metrics.log"

mkdir -p "${PRED_DIR}"
echo "[${NAME}] eval start OUT=${OUT} $(date -Is)" | tee "${LOG_EVAL}"

cd "${REPO}"
"${PY}" segearth_r2/eval/eval.py \
  --base_data_path "${DATA}" \
  --vision_tower "${VT}" \
  --vision_tower_mask "${VTM}" \
  --mask_config "${MCFG}" \
  --model_path "${MODEL}" \
  --output_dir "${PRED_DIR}" \
  --dataset_name "${NAME}" \
  --split "${SPLIT}" \
  --eval_batch_size 1 \
  --dataloader_num_workers 2 \
  --skip_existing True \
  --zip_results False \
  2>&1 | tee -a "${LOG_EVAL}"

USE_WANDB=false \
DATASET_TYPE="${NAME}" \
BASE_DATA_PATH="${DATA}" \
SPLIT="${SPLIT}" \
PRED_DIR="${PRED_DIR}" \
"${PY}" "${METRICS}" 2>&1 | tee "${LOG_METRICS}"

echo "[${NAME}] done $(date -Is)" | tee -a "${LOG_EVAL}"
echo "[OK] metrics: ${OUT}/${NAME}_test_metrics.json"
