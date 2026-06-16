#!/usr/bin/env bash
# Post-500 eval + health + report for DGP v6.1 Stage A from-base.
set -euo pipefail

REPO="/root/rivermind-data/huangziyi/reseg/segearth+DGP"
OUT="/root/rivermind-data/huangziyi/reseg/output/dgp/v6.1-stage-a-from-base"
PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
BASE_MERGED="/root/rivermind-data/huangziyi/reseg/output/base/standard-base-siglip1-28w-gd4/merged_model"
EVAL_METRICS="/root/rivermind-data/huangziyi/reseg/eval_val_metrics.py"
BASE_DATA_PATH="/root/rivermind-data/huangziyi/data/RRSISD"

CKPT="${1:-${OUT}/checkpoint-500}"
TAG="${2:-step500}"

cd "${REPO}"
mkdir -p "${OUT}/health_500" "${OUT}/post_train_qseg_sanity"

if [[ ! -f "${CKPT}/dgp_stage_a_weights.bin" ]]; then
  echo "[ERROR] missing ${CKPT}/dgp_stage_a_weights.bin"
  exit 1
fi

echo "[1/4] Q_ref + Q_seg full test eval..."
bash scripts/eval_dgp_stage_a_2k_full.sh "${CKPT}" "${TAG}"

echo "[2/4] Q_seg-only post-train sanity (trained DGP weights, dgp_use_refined_query=False)..."
QSEG_DIR="${OUT}/post_train_qseg_sanity/test_results_qseg"
rm -rf "${QSEG_DIR}"
mkdir -p "${QSEG_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" tools/run_dgp_stage_a_test_eval.py \
  --model-path "${BASE_MERGED}" \
  --ckpt-dir "${CKPT}" \
  --out-root "${OUT}/post_train_qseg_sanity/_tmp" \
  --mode base \
  --split test
mv "${OUT}/post_train_qseg_sanity/_tmp/test_results_base_qseg" "${QSEG_DIR}"
rmdir "${OUT}/post_train_qseg_sanity/_tmp" 2>/dev/null || true

USE_WANDB=false DATASET_TYPE=rrsisd SPLIT=test \
  BASE_DATA_PATH="${BASE_DATA_PATH}" \
  PRED_DIR="${QSEG_DIR}" \
  METRICS_TAG="qseg_post500" \
  "${PYTHON}" "${EVAL_METRICS}" > "${OUT}/post_train_qseg_sanity/eval_val_metrics.log" 2>&1
LATEST=$(ls -t "${OUT}/post_train_qseg_sanity"/rrsisd_test_metrics_*.json 2>/dev/null | head -1 || true)
if [[ -n "${LATEST}" ]]; then
  cp -f "${LATEST}" "${OUT}/post_train_qseg_sanity/rrsisd_test_metrics_qseg_post500.json"
fi

echo "[3/4] health probe..."
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" tools/probe_dgp_health.py \
  --weights "${CKPT}/dgp_stage_a_weights.bin" \
  --model-path "${BASE_MERGED}" \
  --out "${OUT}/health_500/${TAG}.json"

echo "[4/4] report..."
"${PYTHON}" tools/generate_stage_a_500_from_base_report.py --out-dir "${OUT}" --tag "${TAG}"

echo "[OK] post-500 done -> ${OUT}/report_500_from_base.md"
