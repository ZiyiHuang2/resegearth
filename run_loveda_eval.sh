#!/usr/bin/env bash
# LoveDA test eval for LaSeRS merged model (class-wise referring, inference-only on Test).
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=disabled
export TMPDIR="/root/rivermind-data/tmp"
mkdir -p "${TMPDIR}"

SPLIT="${LOVEDA_SPLIT:-test}"
OUT="/root/rivermind-data/huangziyi/reseg/output/base/standard-base-lasers-siglip1-28w-gd4"
LOVEDA="/root/rivermind-data/huangziyi/data/LoveDa"
REPO="/root/rivermind-data/huangziyi/reseg/segearth+base"
PY="/root/rivermind-data/miniconda3/envs/reseg/bin/python"
METRICS="/root/rivermind-data/huangziyi/reseg/eval_val_metrics.py"
MODEL="${OUT}/merged_model"
PRED="${OUT}/loveda_${SPLIT}_results"

cd "${REPO}"
mkdir -p "${PRED}"

"${PY}" segearth_r2/eval/eval.py \
  --base_data_path "${LOVEDA}" \
  --vision_tower /root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384 \
  --vision_tower_mask /root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl \
  --mask_config segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
  --model_path "${MODEL}" \
  --output_dir "${PRED}" \
  --dataset_name loveda \
  --split "${SPLIT}" \
  --eval_batch_size 1 \
  --skip_existing True \
  --zip_results False \
  2>&1 | tee "${OUT}/loveda_${SPLIT}_eval.log"

if [[ "${SPLIT}" == "test" ]]; then
  echo "[INFO] LoveDA Test has no public GT masks; skip IoU metrics."
  echo "[OK] preds: ${PRED}"
  exit 0
fi

USE_WANDB=false \
DATASET_TYPE=loveda \
BASE_DATA_PATH="${LOVEDA}" \
SPLIT="${SPLIT}" \
PRED_DIR="${PRED}" \
"${PY}" "${METRICS}" 2>&1 | tee "${OUT}/loveda_${SPLIT}_metrics.log"

echo "[OK] metrics: ${OUT}/loveda_${SPLIT}_metrics.json"
