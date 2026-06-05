#!/usr/bin/env bash
# Wait for SET merged_model, then run test eval on all datasets.
# Does NOT merge checkpoints — merge is handled by the training pipeline.
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TMPDIR="/root/rivermind-data/tmp"
mkdir -p "${TMPDIR}"

REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+set"
OUT_DIR="${OUT_DIR:-/root/rivermind-data/huangziyi/reseg/output/set/a3-frozen-lasers-set-5w}"
MODEL="${MODEL_PATH:-${OUT_DIR}/merged_model}"
PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
METRICS="${METRICS:-/root/rivermind-data/huangziyi/reseg/eval_val_metrics.py}"
VT="${VT:-/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384}"
VTM="${VTM:-/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl}"
MCFG="${MCFG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"
GPU_ID="${GPU_ID:-0}"
SPLIT="test"
LASERS_BENCHMARK="${LASERS_BENCHMARK:-all}"
WAIT_FOR_MERGED="${WAIT_FOR_MERGED:-1}"
POLL_SEC="${POLL_SEC:-300}"
LOG_FILE="${LOG_FILE:-${OUT_DIR}/all_datasets_test_eval.log}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================"
echo "[ALL-DS-TEST] start $(date -Is)"
echo "OUT_DIR=${OUT_DIR}"
echo "MODEL=${MODEL}"
echo "WAIT_FOR_MERGED=${WAIT_FOR_MERGED} (no auto-merge)"
echo "========================================"

cd "${REPO_DIR}"

merged_model_ready () {
  [[ -f "${MODEL}/config.json" ]]
}

wait_for_merged_model () {
  if [[ "${WAIT_FOR_MERGED}" != "1" ]]; then
    if ! merged_model_ready; then
      echo "[ERROR] merged model not found: ${MODEL}/config.json"
      exit 1
    fi
    return 0
  fi

  while ! merged_model_ready; do
    echo "[WAIT] merged_model not ready yet (${MODEL}/config.json); sleep ${POLL_SEC}s ($(date -Is))"
    sleep "${POLL_SEC}"
  done
  echo "[OK] merged_model ready: ${MODEL} ($(date -Is))"
}

verify_merged_model () {
  PYTHONPATH="${REPO_DIR}" "${PYTHON}" - <<PY
from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2

mask_cfg = get_mask_config("${MCFG}")
model = SegEarthR2.from_pretrained("${MODEL}", mask_decoder_cfg=mask_cfg, add_cross_attn=True, device_map="cpu")
model.init_set_conditioning_modules(model.config)
assert getattr(model.config, "use_set_conditioner", False), "use_set_conditioner missing from merged config"
shape = tuple(model.category_set_head.head.weight.shape)
assert shape == (191, 256), f"unexpected category_set_head shape: {shape}"
print("[OK] merged model loads; category_set_head shape =", shape)
PY
}

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
    2>&1 | tee "${log_eval}"

  if [[ "${name}" == "lasers" ]]; then
    USE_WANDB="${USE_WANDB:-false}" \
    DATASET_TYPE="${name}" \
    BASE_DATA_PATH="${data_path}" \
    SPLIT="${SPLIT}" \
    LASERS_BENCHMARK="${LASERS_BENCHMARK}" \
    PRED_DIR="${pred_dir}" \
    "${PYTHON}" "${METRICS}" 2>&1 | tee "${log_metrics}"
  else
    USE_WANDB="${USE_WANDB:-false}" \
    DATASET_TYPE="${name}" \
    BASE_DATA_PATH="${data_path}" \
    SPLIT="${SPLIT}" \
    PRED_DIR="${pred_dir}" \
    "${PYTHON}" "${METRICS}" 2>&1 | tee "${log_metrics}"
  fi

  echo "[${name}] done $(date -Is)"
}

wait_for_merged_model
verify_merged_model

run_one lasers "/root/rivermind-data/huangziyi/data/LaSeRS"
run_one rrsisd "/root/rivermind-data/huangziyi/data/RRSISD"
run_one refsegrs "/root/rivermind-data/huangziyi/data/RefSegRS"
run_one risbench "/root/rivermind-data/huangziyi/data/RISBench_dataset"
run_one earthreason "/root/rivermind-data/huangziyi/data/EarthReason"

echo "========================================"
echo "[ALL-DS-TEST] finished $(date -Is)"
echo "LaSeRS metrics     : ${OUT_DIR}/lasers_test_metrics_summary.json"
echo "RRSISD metrics     : ${OUT_DIR}/rrsisd_test_metrics.json"
echo "RefSegRS metrics   : ${OUT_DIR}/refsegrs_test_metrics.json"
echo "RISBench metrics   : ${OUT_DIR}/risbench_test_metrics.json"
echo "EarthReason metrics: ${OUT_DIR}/earthreason_test_metrics.json"
echo "Master log         : ${LOG_FILE}"
echo "========================================"
