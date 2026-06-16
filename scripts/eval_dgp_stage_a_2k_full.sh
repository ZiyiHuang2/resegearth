#!/usr/bin/env bash
# Full RRSISD test eval + class group / per-class / flip metrics for a Stage A checkpoint.
set -euo pipefail

REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+DGP"
cd "${REPO_DIR}"

PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
EVAL_METRICS="/root/rivermind-data/huangziyi/reseg/eval_val_metrics.py"

CKPT_DIR="${1:?usage: $0 /path/to/checkpoint-STEP [tag]}"
TAG="${2:-step$(basename "${CKPT_DIR}" | sed 's/checkpoint-//')}"
BASE_DATA_PATH="${BASE_DATA_PATH:-/root/rivermind-data/huangziyi/data/RRSISD}"
VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"

OUT_ROOT="$(dirname "${CKPT_DIR}")/eval_2k/${TAG}"
MODEL_PRED="${OUT_ROOT}/test_results_model"
BASE_PRED="${OUT_ROOT}/test_results_base_qseg"
METRICS_JSON="${OUT_ROOT}/rrsisd_test_metrics_class_group_with_base.json"

mkdir -p "${OUT_ROOT}"

echo "[eval] ckpt=${CKPT_DIR} model=${MODEL_PATH:-${BASE_MERGED:-/root/rivermind-data/huangziyi/reseg/output/base/standard-base-siglip1-28w-gd4/merged_model}} -> ${OUT_ROOT}"
MODEL_PATH="${MODEL_PATH:-/root/rivermind-data/huangziyi/reseg/output/base/standard-base-siglip1-28w-gd4/merged_model}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON}" tools/run_dgp_stage_a_test_eval.py \
  --model-path "${MODEL_PATH}" \
  --ckpt-dir "${CKPT_DIR}" \
  --out-root "${OUT_ROOT}" \
  --split test

USE_WANDB=false \
DATASET_TYPE=rrsisd \
BASE_DATA_PATH="${BASE_DATA_PATH}" \
SPLIT=test \
PRED_DIR="${MODEL_PRED}" \
BASELINE_PRED_DIR="${BASE_PRED}" \
METRICS_TAG="${TAG}_with_base" \
"${PYTHON}" "${EVAL_METRICS}" > "${OUT_ROOT}/eval_val_metrics.log" 2>&1

LATEST="${OUT_ROOT}/rrsisd_test_metrics_${TAG}_with_base.json"
if [[ ! -f "${LATEST}" ]]; then
  LATEST=$(ls -t "${OUT_ROOT}"/rrsisd_test_metrics_*.json 2>/dev/null | head -1 || true)
fi
if [[ -n "${LATEST}" ]]; then
  cp -f "${LATEST}" "${METRICS_JSON}"
  echo "[OK] metrics -> ${METRICS_JSON}"
else
  echo "[WARN] metrics json not found; see ${OUT_ROOT}/eval_val_metrics.log"
fi

echo "[OK] full eval done tag=${TAG}"
