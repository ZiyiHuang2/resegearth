#!/usr/bin/env bash
# Parallel eval for clite-v2 merged model (5 datasets, 1 GPU).
# Wall time ≈ slowest dataset (risbench ~16k), not sum of all.
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
unset CUDA_VISIBLE_DEVICES

REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+set"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
GPU_ID="${GPU_ID:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-/root/rivermind-data/huangziyi/reseg/output/set/clite-v2-from-scratch-50k}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}/merged_model}"
PARALLEL_EVAL_LOG="${OUTPUT_DIR}/all_datasets_test_eval_parallel.log"

VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
EVAL_METRICS_SCRIPT="/root/rivermind-data/huangziyi/reseg/eval_val_metrics.py"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_DATALOADER_NUM_WORKERS="${EVAL_DATALOADER_NUM_WORKERS:-2}"
EVAL_USE_WANDB="${EVAL_USE_WANDB:-True}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-set-eval}"
EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-clite-v2-from-scratch-50k}"
LASERS_BENCHMARK="${LASERS_BENCHMARK:-all}"

[[ -f "${MERGED_DIR}/config.json" ]] || { echo "[ERROR] merged model missing: ${MERGED_DIR}/config.json"; exit 1; }

run_one () {
  local name="$1"
  local data_path="$2"
  local pred_dir="${OUTPUT_DIR}/${name}_test_results"
  local log_eval="${OUTPUT_DIR}/${name}_test_eval.log"
  local log_metrics="${OUTPUT_DIR}/${name}_test_metrics.log"

  mkdir -p "${pred_dir}"
  echo "[${name}] start $(date -Iseconds)" | tee -a "${PARALLEL_EVAL_LOG}"

  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON}" segearth_r2/eval/eval.py \
    --base_data_path "${data_path}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --model_path "${MERGED_DIR}" \
    --output_dir "${pred_dir}" \
    --dataset_name "${name}" \
    --split "${EVAL_SPLIT}" \
    --eval_batch_size 1 \
    --dataloader_num_workers "${EVAL_DATALOADER_NUM_WORKERS}" \
    --skip_existing True \
    --zip_results False \
    > "${log_eval}" 2>&1

  if [[ "${name}" == "lasers" ]]; then
    USE_WANDB="${EVAL_USE_WANDB}" WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
    WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME}" DATASET_TYPE="${name}" \
    BASE_DATA_PATH="${data_path}" SPLIT="${EVAL_SPLIT}" \
    LASERS_BENCHMARK="${LASERS_BENCHMARK}" PRED_DIR="${pred_dir}" \
    "${PYTHON}" "${EVAL_METRICS_SCRIPT}" > "${log_metrics}" 2>&1
  else
    USE_WANDB="${EVAL_USE_WANDB}" WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
    WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME}" DATASET_TYPE="${name}" \
    BASE_DATA_PATH="${data_path}" SPLIT="${EVAL_SPLIT}" PRED_DIR="${pred_dir}" \
    "${PYTHON}" "${EVAL_METRICS_SCRIPT}" > "${log_metrics}" 2>&1
  fi

  echo "[${name}] done $(date -Iseconds)" | tee -a "${PARALLEL_EVAL_LOG}"
}

{
  echo "========================================"
  echo "[PARALLEL-TEST] start $(date -Iseconds)"
  echo "MODEL=${MERGED_DIR}"
  echo "GPU=${GPU_ID}, MAX_PARALLEL=5"
  echo "samples: lasers=1337 refsegrs=1817 rrsisd=3481 earthreason=11568 risbench=16159"
  echo "========================================"
} | tee "${PARALLEL_EVAL_LOG}"

# Launch all 5; wall time dominated by risbench.
run_one lasers "/root/rivermind-data/huangziyi/data/LaSeRS" &
run_one refsegrs "/root/rivermind-data/huangziyi/data/RefSegRS" &
run_one rrsisd "/root/rivermind-data/huangziyi/data/RRSISD" &
run_one earthreason "/root/rivermind-data/huangziyi/data/EarthReason" &
run_one risbench "/root/rivermind-data/huangziyi/data/RISBench_dataset" &
wait || true

{
  echo "========================================"
  echo "[PARALLEL-TEST] finished $(date -Iseconds)"
  echo "LaSeRS summary     : ${OUTPUT_DIR}/lasers_${EVAL_SPLIT}_metrics_summary.json"
  echo "RRSISD metrics     : ${OUTPUT_DIR}/rrsisd_test_metrics.json"
  echo "RefSegRS metrics   : ${OUTPUT_DIR}/refsegrs_test_metrics.json"
  echo "RISBench metrics   : ${OUTPUT_DIR}/risbench_test_metrics.json"
  echo "EarthReason metrics: ${OUTPUT_DIR}/earthreason_test_metrics.json"
  echo "========================================"
} | tee -a "${PARALLEL_EVAL_LOG}"
