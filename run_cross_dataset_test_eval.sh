#!/usr/bin/env bash
# Cross-dataset test eval + metrics (RRSISD / RefSegRS / RISBench) in parallel.
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=disabled
export TMPDIR="/root/rivermind-data/tmp"
mkdir -p "${TMPDIR}"

OUT="${OUT_DIR:-/root/rivermind-data/huangziyi/reseg/output/base/standard-base-lasers-siglip1-5w-gd4}"
MODEL="${MODEL_PATH:-${OUT}/merged_model}"
REPO="/root/rivermind-data/huangziyi/reseg/segearth+base"
PY="/root/rivermind-data/miniconda3/envs/reseg/bin/python"
METRICS="/root/rivermind-data/huangziyi/reseg/eval_val_metrics.py"
VT="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VTM="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MCFG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
SPLIT="test"

run_one () {
  local name="$1"
  local data_path="$2"
  local pred_dir="${OUT}/${name}_test_results"
  local log_eval="${OUT}/${name}_test_eval.log"
  local log_metrics="${OUT}/${name}_test_metrics.log"

  mkdir -p "${pred_dir}"
  echo "[${name}] eval start $(date -Is)" | tee "${log_eval}"

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
    --zip_results False \
    2>&1 | tee -a "${log_eval}"

  USE_WANDB=false \
  DATASET_TYPE="${name}" \
  BASE_DATA_PATH="${data_path}" \
  SPLIT="${SPLIT}" \
  PRED_DIR="${pred_dir}" \
  "${PY}" "${METRICS}" 2>&1 | tee "${log_metrics}"

  echo "[${name}] done $(date -Is)" | tee -a "${log_eval}"
}

run_one rrsisd "/root/rivermind-data/huangziyi/data/RRSISD" &
PID_RRSISD=$!

run_one refsegrs "/root/rivermind-data/huangziyi/data/RefSegRS" &
PID_REF=$!

run_one risbench "/root/rivermind-data/huangziyi/data/RISBench_dataset" &
PID_RIS=$!

echo "[INFO] parallel PIDs: rrsisd=${PID_RRSISD} refsegrs=${PID_REF} risbench=${PID_RIS}"
wait "${PID_RRSISD}" && echo "[OK] rrsisd finished" || echo "[FAIL] rrsisd exit=$?"
wait "${PID_REF}" && echo "[OK] refsegrs finished" || echo "[FAIL] refsegrs exit=$?"
wait "${PID_RIS}" && echo "[OK] risbench finished" || echo "[FAIL] risbench exit=$?"

echo "[ALL DONE] metrics:"
echo "  ${OUT}/rrsisd_test_metrics.json"
echo "  ${OUT}/refsegrs_test_metrics.json"
echo "  ${OUT}/risbench_test_metrics.json"
