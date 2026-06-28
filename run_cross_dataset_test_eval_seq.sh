#!/usr/bin/env bash
# TG-Swin: sequential cross-dataset test eval (skip LaSeRS if already done).
# Datasets: RRSISD → RefSegRS → RISBench → EarthReason
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TMPDIR="${TMPDIR:-/root/rivermind-data/tmp}"
mkdir -p "${TMPDIR}"

RESEG_ROOT="${RESEG_ROOT:-/root/rivermind-data/huangziyi/reseg}"
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO_DIR:-${_SCRIPT_DIR}}"
OUT="${OUT_DIR:-${RESEG_ROOT}/output/tgswin/tgswin-wti-v15-lasers-warmstart-8w-gd4}"
MODEL="${MODEL_PATH:-${OUT}/merged_model}"
PY="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
METRICS="${EVAL_METRICS_SCRIPT:-${RESEG_ROOT}/eval_val_metrics.py}"
VT="${VISION_TOWER:-${RESEG_ROOT}/pretrained_model/CLIP/siglip-so400m-patch14-384}"
VTM="${VISION_TOWER_MASK:-${RESEG_ROOT}/pretrained_model/mask2former/model_final_54b88a.pkl}"
MCFG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin_v15.yaml}"
SPLIT="${EVAL_SPLIT:-test}"
EVAL_USE_WANDB="${EVAL_USE_WANDB:-false}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-tgswin}"
EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-tgswin-wti-v15-lasers-warmstart-8w-gd4}"

DATA_RRSISD="${DATA_RRSISD:-/root/rivermind-data/huangziyi/data/RRSISD}"
DATA_REFSEGRS="${DATA_REFSEGRS:-/root/rivermind-data/huangziyi/data/RefSegRS}"
DATA_RISBENCH="${DATA_RISBENCH:-/root/rivermind-data/huangziyi/data/RISBench_dataset}"
DATA_EARTHREASON="${DATA_EARTHREASON:-/root/rivermind-data/huangziyi/data/EarthReason}"

DATASETS=(
  "rrsisd:${DATA_RRSISD}"
  "refsegrs:${DATA_REFSEGRS}"
  "risbench:${DATA_RISBENCH}"
  "earthreason:${DATA_EARTHREASON}"
)

run_one () {
  local name="$1"
  local data_path="$2"
  local pred_dir="${OUT}/${name}_test_results"
  local log_eval="${OUT}/${name}_test_eval.log"
  local metrics_json="${OUT}/${name}_${SPLIT}_metrics.json"

  if [[ -f "${metrics_json}" ]]; then
    echo "[SKIP] ${name}: metrics already exist (${metrics_json})"
    return 0
  fi

  mkdir -p "${pred_dir}"
  {
    echo "========================================"
    echo "[${name}] eval start $(date -Is)"
    echo "[${name}] model=${MODEL}"
    echo "[${name}] data=${data_path}"
    echo "[${name}] pred_dir=${pred_dir}"
    echo "[${name}] GPU=${CUDA_VISIBLE_DEVICES}"

    cd "${REPO}"
    "${PY}" segearth_r2/eval/eval.py \
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
      --zip_results False

    USE_WANDB="${EVAL_USE_WANDB}" \
    WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
    WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME}-${name}" \
    DATASET_TYPE="${name}" \
    BASE_DATA_PATH="${data_path}" \
    SPLIT="${SPLIT}" \
    PRED_DIR="${pred_dir}" \
    "${PY}" "${METRICS}"

    echo "[${name}] done $(date -Is)"
  } 2>&1 | tee "${log_eval}"
}

echo "[INFO] OUT=${OUT}"
echo "[INFO] MODEL=${MODEL}"
echo "[INFO] Sequential 4-dataset test eval on GPU ${CUDA_VISIBLE_DEVICES}"

[[ -d "${MODEL}" ]] || { echo "[ERROR] merged model not found: ${MODEL}"; exit 1; }

FAIL=0
for entry in "${DATASETS[@]}"; do
  name="${entry%%:*}"
  data_path="${entry#*:}"
  echo "----------------------------------------"
  echo "[QUEUE] next dataset: ${name}"
  if run_one "${name}" "${data_path}"; then
    echo "[OK] ${name} finished"
  else
    echo "[FAIL] ${name} exit=$?"
    FAIL=1
    break
  fi
done

echo "========================================"
if [[ "${FAIL}" -eq 0 ]]; then
  echo "[ALL DONE] cross-dataset metrics under ${OUT}:"
  for entry in "${DATASETS[@]}"; do
    name="${entry%%:*}"
    echo "  ${OUT}/${name}_${SPLIT}_metrics.json"
  done
else
  echo "[ERROR] eval stopped; check *_test_eval.log under ${OUT}"
  exit 1
fi
echo "========================================"
