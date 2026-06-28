#!/usr/bin/env bash
# Resume cross-dataset test eval sequentially on one GPU (skip_existing=True).
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TMPDIR="${TMPDIR:-/root/rivermind-data/tmp}"
mkdir -p "${TMPDIR}"

RESEG_ROOT="${RESEG_ROOT:-/root/rivermind-data/huangziyi/reseg}"
REPO="${REPO_DIR:-${RESEG_ROOT}/segearth+set++}"
OUT="${OUT_DIR:-${RESEG_ROOT}/output/setpp/setpp-A3-lasers-warmstart-5w-gd4}"
MODEL="${MODEL_PATH:-${OUT}/merged_model}"
PY="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
METRICS="${EVAL_METRICS_SCRIPT:-${RESEG_ROOT}/eval_val_metrics.py}"
VT="${VISION_TOWER:-${RESEG_ROOT}/pretrained_model/CLIP/siglip-so400m-patch14-384}"
VTM="${VISION_TOWER_MASK:-${RESEG_ROOT}/pretrained_model/mask2former/model_final_54b88a.pkl}"
MCFG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"
SPLIT="${EVAL_SPLIT:-test}"
EVAL_USE_WANDB="${EVAL_USE_WANDB:-True}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-setpp}"
EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-setpp-A3-lasers-warmstart-5w-gd4}"
LASERS_BENCHMARK="${LASERS_BENCHMARK:-all}"
MASTER_LOG="${OUT}/cross_dataset_eval_sequential.log"

declare -A DATA_PATHS=(
  [rrsisd]="${DATA_RRSISD:-/root/rivermind-data/huangziyi/data/RRSISD}"
  [refsegrs]="${DATA_REFSEGRS:-/root/rivermind-data/huangziyi/data/RefSegRS}"
  [risbench]="${DATA_RISBENCH:-/root/rivermind-data/huangziyi/data/RISBench_dataset}"
  [earthreason]="${DATA_EARTHREASON:-/root/rivermind-data/huangziyi/data/EarthReason}"
)

DATASETS=(${DATASETS:-rrsisd refsegrs risbench earthreason})

run_one () {
  local name="$1"
  local data_path="$2"
  local pred_dir="${OUT}/${name}_test_results"
  local log_eval="${OUT}/${name}_test_eval_resume.log"

  mkdir -p "${pred_dir}"
  {
    echo "========================================"
    echo "[${name}] sequential eval start $(date -Is)"
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

    METRICS_ENV=(
      USE_WANDB="${EVAL_USE_WANDB}"
      WANDB_PROJECT="${EVAL_WANDB_PROJECT}"
      WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME}-${name}"
      DATASET_TYPE="${name}"
      BASE_DATA_PATH="${data_path}"
      SPLIT="${SPLIT}"
      PRED_DIR="${pred_dir}"
    )
    if [[ "${name}" == "lasers" ]]; then
      METRICS_ENV+=(LASERS_BENCHMARK="${LASERS_BENCHMARK}")
    fi
    env "${METRICS_ENV[@]}" "${PY}" "${METRICS}"

    echo "[${name}] done $(date -Is)"
  } > >(tee -a "${log_eval}") 2>&1
}

{
  echo "========================================"
  echo "[MASTER] sequential cross-dataset eval start $(date -Is)"
  echo "[MASTER] OUT=${OUT}"
  echo "[MASTER] MODEL=${MODEL}"
  echo "[MASTER] GPU=${CUDA_VISIBLE_DEVICES}"
  echo "[MASTER] datasets: ${DATASETS[*]}"
} | tee "${MASTER_LOG}"

FAIL=0
for name in "${DATASETS[@]}"; do
  data_path="${DATA_PATHS[$name]}"
  if [[ -z "${data_path}" ]]; then
    echo "[ERROR] unknown dataset: ${name}" | tee -a "${MASTER_LOG}"
    FAIL=1
    continue
  fi
  if run_one "${name}" "${data_path}"; then
    echo "[OK] ${name} finished" | tee -a "${MASTER_LOG}"
  else
    echo "[FAIL] ${name} exit=$?" | tee -a "${MASTER_LOG}"
    FAIL=1
  fi
done

{
  echo "========================================"
  if [[ "${FAIL}" -eq 0 ]]; then
    echo "[ALL DONE] sequential eval finished $(date -Is)"
  else
    echo "[ERROR] one or more datasets failed; see *_test_eval_resume.log"
  fi
} | tee -a "${MASTER_LOG}"

exit "${FAIL}"
