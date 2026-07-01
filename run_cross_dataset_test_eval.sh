#!/usr/bin/env bash
# Cross-dataset test eval + metrics for SET++ merged model (5 datasets serial).
# Datasets: LaSeRS / RRSISD / RefSegRS / RISBench / EarthReason
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TMPDIR="${TMPDIR:-/root/rivermind-data/tmp}"
mkdir -p "${TMPDIR}"

RESEG_ROOT="${RESEG_ROOT:-/root/rivermind-data/huangziyi/reseg}"
REPO="${REPO_DIR:-${RESEG_ROOT}/segearth+set++}"
OUT="${OUT_DIR:-${RESEG_ROOT}/output/setpp/setpp-lasers-warmstart-5w-gd4}"
MODEL="${MODEL_PATH:-${OUT}/merged_model}"
PY="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
METRICS="${EVAL_METRICS_SCRIPT:-${RESEG_ROOT}/eval_val_metrics.py}"
VT="${VISION_TOWER:-${RESEG_ROOT}/pretrained_model/CLIP/siglip-so400m-patch14-384}"
VTM="${VISION_TOWER_MASK:-${RESEG_ROOT}/pretrained_model/mask2former/model_final_54b88a.pkl}"
MCFG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"
SPLIT="${EVAL_SPLIT:-test}"
EVAL_USE_WANDB="${EVAL_USE_WANDB:-false}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-setpp}"
EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-setpp-cross-5datasets}"
LASERS_BENCHMARK="${LASERS_BENCHMARK:-all}"

DATA_LASERS="${DATA_LASERS:-/root/rivermind-data/huangziyi/data/LaSeRS}"
DATA_RRSISD="${DATA_RRSISD:-/root/rivermind-data/huangziyi/data/RRSISD}"
DATA_REFSEGRS="${DATA_REFSEGRS:-/root/rivermind-data/huangziyi/data/RefSegRS}"
DATA_RISBENCH="${DATA_RISBENCH:-/root/rivermind-data/huangziyi/data/RISBench_dataset}"
DATA_EARTHREASON="${DATA_EARTHREASON:-/root/rivermind-data/huangziyi/data/EarthReason}"

run_one () {
  local name="$1"
  local data_path="$2"
  local pred_dir="${OUT}/${name}_test_results"
  local log_eval="${OUT}/${name}_test_eval.log"

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
  } > >(tee "${log_eval}") 2>&1
}

echo "[INFO] OUT=${OUT}"
echo "[INFO] MODEL=${MODEL}"
echo "[INFO] Serial 5-dataset test eval on GPU ${CUDA_VISIBLE_DEVICES}"

FAIL=0
run_one lasers "${DATA_LASERS}" && echo "[OK] lasers finished" || { echo "[FAIL] lasers exit=$?"; FAIL=1; }
run_one rrsisd "${DATA_RRSISD}" && echo "[OK] rrsisd finished" || { echo "[FAIL] rrsisd exit=$?"; FAIL=1; }
run_one refsegrs "${DATA_REFSEGRS}" && echo "[OK] refsegrs finished" || { echo "[FAIL] refsegrs exit=$?"; FAIL=1; }
run_one risbench "${DATA_RISBENCH}" && echo "[OK] risbench finished" || { echo "[FAIL] risbench exit=$?"; FAIL=1; }
run_one earthreason "${DATA_EARTHREASON}" && echo "[OK] earthreason finished" || { echo "[FAIL] earthreason exit=$?"; FAIL=1; }

echo "========================================"
if [[ "${FAIL}" -eq 0 ]]; then
  echo "[ALL DONE] 5-dataset test metrics:"
  echo "  ${OUT}/lasers_test_results  (+ ${OUT}/lasers_test_metrics.json if generated)"
  echo "  ${OUT}/rrsisd_test_metrics.json"
  echo "  ${OUT}/refsegrs_test_metrics.json"
  echo "  ${OUT}/risbench_test_metrics.json"
  echo "  ${OUT}/earthreason_test_metrics.json"
else
  echo "[ERROR] one or more datasets failed; check *_test_eval.log under ${OUT}"
  exit 1
fi
echo "========================================"
