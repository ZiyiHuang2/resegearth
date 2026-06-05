#!/usr/bin/env bash
# Parallel test eval on all datasets (max MAX_PARALLEL jobs on one GPU).
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TMPDIR="/root/rivermind-data/tmp"
mkdir -p "${TMPDIR}"

REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+set"
OUT_DIR="${OUT_DIR:-/root/rivermind-data/huangziyi/reseg/output/set/a3-frozen-lasers-set-5w}"
MODEL="${MODEL_PATH:-${OUT_DIR}/merged_model}"
PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
METRICS="${METRICS:-/root/rivermind-data/huangziyi/reseg/eval_val_metrics.py}"
VT="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VTM="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MCFG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
GPU_ID="${GPU_ID:-0}"
SPLIT="test"
LASERS_BENCHMARK="${LASERS_BENCHMARK:-all}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/all_datasets_test_eval_parallel.log}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================"
echo "[PARALLEL-TEST] start $(date -Is)"
echo "OUT_DIR=${OUT_DIR}"
echo "MODEL=${MODEL}"
echo "MAX_PARALLEL=${MAX_PARALLEL}"
echo "========================================"

if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "[ERROR] merged model not found: ${MODEL}/config.json"
  exit 1
fi

cd "${REPO_DIR}"

run_one () {
  local name="$1"
  local data_path="$2"
  local pred_dir="${OUT_DIR}/${name}_test_results"
  local log_eval="${OUT_DIR}/${name}_test_eval.log"
  local log_metrics="${OUT_DIR}/${name}_test_metrics.log"

  mkdir -p "${pred_dir}"
  echo "[${name}] eval start $(date -Is)"

  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON}" segearth_r2/eval/eval.py \
    --base_data_path "${data_path}" \
    --vision_tower "${VT}" \
    --vision_tower_mask "${VTM}" \
    --mask_config "${MCFG}" \
    --model_path "${MODEL}" \
    --output_dir "${pred_dir}" \
    --dataset_name "${name}" \
    --split "${SPLIT}" \
    --eval_batch_size 1 \
    --dataloader_num_workers 2 \
    --skip_existing True \
    --zip_results False \
    > "${log_eval}" 2>&1

  if [[ "${name}" == "lasers" ]]; then
    USE_WANDB="${USE_WANDB:-false}" \
    DATASET_TYPE="${name}" \
    BASE_DATA_PATH="${data_path}" \
    SPLIT="${SPLIT}" \
    LASERS_BENCHMARK="${LASERS_BENCHMARK}" \
    PRED_DIR="${pred_dir}" \
    "${PYTHON}" "${METRICS}" > "${log_metrics}" 2>&1
  else
    USE_WANDB="${USE_WANDB:-false}" \
    DATASET_TYPE="${name}" \
    BASE_DATA_PATH="${data_path}" \
    SPLIT="${SPLIT}" \
    PRED_DIR="${pred_dir}" \
    "${PYTHON}" "${METRICS}" > "${log_metrics}" 2>&1
  fi

  echo "[${name}] done $(date -Is)"
}

launch () {
  local name="$1"
  local data_path="$2"
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
    wait -n || true
  done
  run_one "${name}" "${data_path}" &
  echo "[launcher] started ${name} pid=$! ($(date -Is))"
}

launch lasers "/root/rivermind-data/huangziyi/data/LaSeRS"
launch rrsisd "/root/rivermind-data/huangziyi/data/RRSISD"
launch refsegrs "/root/rivermind-data/huangziyi/data/RefSegRS"
launch risbench "/root/rivermind-data/huangziyi/data/RISBench_dataset"
launch earthreason "/root/rivermind-data/huangziyi/data/EarthReason"

wait || true

echo "========================================"
echo "[PARALLEL-TEST] finished $(date -Is)"
echo "LaSeRS metrics     : ${OUT_DIR}/lasers_test_metrics_summary.json"
echo "RRSISD metrics     : ${OUT_DIR}/rrsisd_test_metrics.json"
echo "RefSegRS metrics   : ${OUT_DIR}/refsegrs_test_metrics.json"
echo "RISBench metrics   : ${OUT_DIR}/risbench_test_metrics.json"
echo "EarthReason metrics: ${OUT_DIR}/earthreason_test_metrics.json"
echo "Master log         : ${LOG_FILE}"
echo "========================================"
