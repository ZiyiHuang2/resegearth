#!/usr/bin/env bash
# Resume post-process after partial eval (OOM-safe: one pass at a time).
set -euo pipefail

REPO="/root/rivermind-data/huangziyi/reseg/segearth+DGP"
OUT="/root/rivermind-data/huangziyi/reseg/output/dgp/v6.1-stage-a-500diag-full"
PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

cd "${REPO}"
mkdir -p "${OUT}/health_2k" "${OUT}/eval_2k"

run_ckpt () {
  local ckpt="$1"
  local tag="$2"
  local root="${OUT}/eval_2k/${tag}"
  mkdir -p "${root}"

  echo "[eval] ${tag} model preds..."
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" tools/run_dgp_stage_a_test_eval.py \
    --ckpt-dir "${ckpt}" --out-root "${root}" --split test --mode model

  echo "[eval] ${tag} base preds..."
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" tools/run_dgp_stage_a_test_eval.py \
    --ckpt-dir "${ckpt}" --out-root "${root}" --split test --mode base

  echo "[metrics] ${tag}..."
  USE_WANDB=false DATASET_TYPE=rrsisd SPLIT=test \
    BASE_DATA_PATH="/root/rivermind-data/huangziyi/data/RRSISD" \
    PRED_DIR="${root}/test_results_model" \
    BASELINE_PRED_DIR="${root}/test_results_base_qseg" \
    METRICS_TAG="${tag}_with_base" \
    "${PYTHON}" /root/rivermind-data/huangziyi/reseg/eval_val_metrics.py \
    > "${root}/eval_val_metrics.log" 2>&1

  latest=$(ls -t "${root}"/rrsisd_test_metrics_*.json 2>/dev/null | head -1 || true)
  if [[ -n "${latest}" ]]; then
    cp -f "${latest}" "${root}/rrsisd_test_metrics_class_group_with_base.json"
    echo "[OK] ${tag} metrics -> ${root}/rrsisd_test_metrics_class_group_with_base.json"
  fi
}

# step1000: skip if metrics already computed
if [[ -f "${OUT}/eval_2k/step1000/rrsisd_test_metrics_class_group_with_base.json" ]]; then
  echo "[skip] step1000 metrics already exist"
elif [[ -d "${OUT}/eval_2k/step1000/test_results_model" ]] && [[ $(ls "${OUT}/eval_2k/step1000/test_results_model" | wc -l) -ge 3400 ]]; then
  echo "[skip] step1000 model preds exist; running base + metrics only"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" tools/run_dgp_stage_a_test_eval.py \
    --ckpt-dir "${OUT}/checkpoint-1000" --out-root "${OUT}/eval_2k/step1000" --split test --mode base
  USE_WANDB=false DATASET_TYPE=rrsisd SPLIT=test \
    BASE_DATA_PATH="/root/rivermind-data/huangziyi/data/RRSISD" \
    PRED_DIR="${OUT}/eval_2k/step1000/test_results_model" \
    BASELINE_PRED_DIR="${OUT}/eval_2k/step1000/test_results_base_qseg" \
    METRICS_TAG="step1000_with_base" \
    "${PYTHON}" /root/rivermind-data/huangziyi/reseg/eval_val_metrics.py \
    > "${OUT}/eval_2k/step1000/eval_val_metrics.log" 2>&1
  latest=$(ls -t "${OUT}/eval_2k/step1000"/rrsisd_test_metrics_*.json 2>/dev/null | head -1 || true)
  [[ -n "${latest}" ]] && cp -f "${latest}" "${OUT}/eval_2k/step1000/rrsisd_test_metrics_class_group_with_base.json"
else
  run_ckpt "${OUT}/checkpoint-1000" step1000
fi

run_ckpt "${OUT}/checkpoint-1300" step1300

for spec in \
  "1000:${OUT}/checkpoint-1000/dgp_stage_a_weights.bin" \
  "1300:${OUT}/checkpoint-1300/dgp_stage_a_weights.bin"; do
  step="${spec%%:*}"
  w="${spec#*:}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" tools/probe_dgp_health.py \
    --weights "${w}" --out "${OUT}/health_2k/step${step}.json" || true
  sleep 3
done

"${PYTHON}" tools/generate_dgp_2k_report.py --out-dir "${OUT}"
echo "[OK] report -> ${OUT}/report_2k.md"
