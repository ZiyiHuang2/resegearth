#!/usr/bin/env bash
set -euo pipefail

########################################
# TG-Swin: Train → Merge → LaSeRS Test Eval
#
# 默认 VARIANT=dr-ewti（代码在 segearth+tgswin-dr-ewti worktree）：
#   - 2w / bs2 / gd4，warmstart=base-8w（与 v1.5 的 2w/5w 实验一致）
#   - output/tgswin/dr-ewti-lasers-warmstart-2w-bs2-gd4
#
# ── Quick start — DR-EWTI 2w (default) ──
#   GPU_ID=0 bash run_train_merge_test_tgswin.sh
#
# ── Full DR-EWTI + SET++ (segearth+dr-ewti-setpp) ──
#   GPU_ID=0 VARIANT=full bash run_train_merge_test_tgswin.sh
#
# Override examples:
#   GPU_ID=0 RUN_TRAIN=0 RUN_MERGE=1 RUN_EVAL=1 bash run_train_merge_test_tgswin.sh
#   GPU_ID=0 MAX_STEPS=100 RUN_PREFLIGHT=0 bash run_train_merge_test_tgswin.sh
#   RUN_CROSS_DATASET_EVAL=0 bash run_train_merge_test_tgswin.sh
########################################

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-tgswin}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
# wandb 凭证在数据盘；tmux 未 source bashrc 时仍需能找到 .netrc
export NETRC="${NETRC:-/root/rivermind-data/.netrc}"
export WANDB_DIR="${WANDB_DIR:-/root/rivermind-data/.wandb}"

source /root/rivermind-data/huangziyi/contrast/paths.env
source /root/rivermind-data/huangziyi/contrast/lasers_baseline_a100.env
source /root/rivermind-data/huangziyi/contrast/shared/lasers_checkpoint.sh

RESEG_ROOT="/root/rivermind-data/huangziyi/reseg"
CONDA_ENV_DIR="/root/rivermind-data/miniconda3/envs/reseg"
PYTHON="${CONDA_ENV_DIR}/bin/python"
DEEPSPEED="${CONDA_ENV_DIR}/bin/deepspeed"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"

VARIANT="${VARIANT:-dr-ewti}"
GPU_ID="${GPU_ID:-0}"
GPU_SLOT="${GPU_SLOT:-localhost:${GPU_ID}}"
MASTER_PORT="${MASTER_PORT:-29641}"
# 避免 tmux/merge 残留 CUDA_VISIBLE_DEVICES="" 导致 DeepSpeed 看不到 GPU
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# 训练初始化权重（均为 merge 后的 HuggingFace 目录，子目录名固定叫 merged_model）
# - INIT_BASE: SegEarth base 8w LaSeRS（无 TG-Swin）— v1.5 / DR-EWTI 公平对比的默认起点
# - INIT_V15:  已训好的 v1.5 TG-Swin 10w — 仅增量 ablation 时用 MODEL_PATH=${INIT_V15} 覆盖
INIT_BASE="${INIT_BASE:-${RESEG_ROOT}/output/base/standard-base-lasers-siglip1-8w-gd4/merged_model}"
INIT_V15="${INIT_V15:-${RESEG_ROOT}/output/tgswin/tgswin-wti-v15-lasers-warmstart-10w-bs2-gd4/merged_model}"
# 兼容旧环境变量名（warmstart 在本项目里指 base 8w，不是 v1.5 checkpoint）
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${INIT_BASE}}"
V15_WARMSTART="${V15_WARMSTART:-${INIT_V15}}"
VISION_TOWER="${RESEG_ROOT}/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="${RESEG_ROOT}/pretrained_model/mask2former/model_final_54b88a.pkl"

case "${VARIANT}" in
  dr-ewti|DR-EWTI|dr_ewti)
    REPO_DIR="${REPO_DIR:-${RESEG_ROOT}/segearth+tgswin-dr-ewti}"
    MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin_dr_ewti.yaml}"
    MODEL_PATH="${MODEL_PATH:-${INIT_BASE}}"
    MAX_STEPS="${MAX_STEPS:-30000}"
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
    SAVE_STEPS="${SAVE_STEPS:-5000}"
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
    OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/tgswin/dr-ewti-lasers-warmstart-2w-bs2-gd4}"
    export WANDB_NAME="${WANDB_NAME:-dr-ewti-lasers-warmstart-2w-bs2-gd4}"
    EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-dr-ewti-lasers-warmstart-2w-bs2-gd4}"
    RUN_TAG="${RUN_TAG:-dr-ewti-lasers-warmstart-2w-bs2-gd4}"
    ;;
  dr-ewti-wo-coarse|DR-WO-COARSE|wo_coarse)
    REPO_DIR="${REPO_DIR:-${RESEG_ROOT}/segearth+tgswin-dr-ewti}"
    MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_dr_ewti_no_coarse.yaml}"
    MODEL_PATH="${MODEL_PATH:-${INIT_BASE}}"
    MAX_STEPS="${MAX_STEPS:-80000}"
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
    SAVE_STEPS="${SAVE_STEPS:-5000}"
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
    OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/tgswin/dr-wo-coarse-lasers-warmstart-80w-bs2-gd4}"
    export WANDB_NAME="${WANDB_NAME:-dr-wo-coarse-lasers-warmstart-80w-bs2-gd4}"
    EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-dr-wo-coarse-lasers-warmstart-80w-bs2-gd4}"
    RUN_TAG="${RUN_TAG:-dr-wo-coarse-lasers-warmstart-80w-bs2-gd4}"
    ;;
  v15|v1.5|V15)
    REPO_DIR="${REPO_DIR:-${RESEG_ROOT}/segearth+tgswin}"
    MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin_v15.yaml}"
    MODEL_PATH="${MODEL_PATH:-${INIT_BASE}}"
    MAX_STEPS="${MAX_STEPS:-50000}"
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
    SAVE_STEPS="${SAVE_STEPS:-5000}"
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
    OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/tgswin/tgswin-wti-v15-lasers-50k-bs4-gd4}"
    export WANDB_NAME="${WANDB_NAME:-tgswin-wti-v15-lasers-50k-bs4-gd4}"
    EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-tgswin-wti-v15-lasers-50k-bs4-gd4}"
    RUN_TAG="${RUN_TAG:-v15-50k-bs4-gd4}"
    ;;
  full|FULL|dr-ewti-setpp|dr_ewti_setpp)
    REPO_DIR="${REPO_DIR:-${RESEG_ROOT}/segearth+dr-ewti-setpp}"
    MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_dr_ewti_setpp.yaml}"
    MODEL_PATH="${MODEL_PATH:-${INIT_BASE}}"
    MAX_STEPS="${MAX_STEPS:-30000}"
    PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
    SAVE_STEPS="${SAVE_STEPS:-5000}"
    SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
    ABLATION="${ABLATION:-A3}"
    OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/full/dr-ewti-setpp-${ABLATION}-lasers-warmstart-2w-bs2-gd4}"
    export WANDB_NAME="${WANDB_NAME:-dr-ewti-setpp-${ABLATION}-lasers-warmstart-2w-bs2-gd4}"
    EVAL_WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME:-dr-ewti-setpp-${ABLATION}-lasers-warmstart-2w-bs2-gd4}"
    RUN_TAG="${RUN_TAG:-dr-ewti-setpp-${ABLATION}-lasers-warmstart-2w-bs2-gd4}"
    SETPP_CLOSED_LOOP="${SETPP_CLOSED_LOOP:-True}"
    SETPP_CSQR_ENABLE="${SETPP_CSQR_ENABLE:-True}"
    ;;
  *)
    echo "[ERROR] Unknown VARIANT=${VARIANT} (use dr-ewti, dr-ewti-wo-coarse, v15, or full)"
    exit 1
    ;;
esac

if [[ "${VARIANT}" == "full" || "${VARIANT}" == "FULL" || "${VARIANT}" == "dr-ewti-setpp" || "${VARIANT}" == "dr_ewti_setpp" ]]; then
  export GPU_ID GPU_SLOT MASTER_PORT RUN_TRAIN RUN_MERGE RUN_EVAL RUN_CROSS_DATASET_EVAL
  export MAX_STEPS PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS OUTPUT_DIR MERGE_CHECKPOINT
  export ABLATION SETPP_CLOSED_LOOP SETPP_CSQR_ENABLE WANDB_NAME RUN_TAG RUN_PREFLIGHT
  exec bash "${RESEG_ROOT}/segearth+dr-ewti-setpp/run_train_merge_test.sh"
fi

cd "${REPO_DIR}"
if [[ "${MASK_CONFIG}" != /* ]]; then
  MASK_CONFIG="${REPO_DIR}/${MASK_CONFIG}"
fi

BASE_DATA_PATH="${BASE_DATA_PATH:-/root/rivermind-data/huangziyi/data/LaSeRS}"
DATASET_NAME="lasers"
EVAL_SPLIT="test"
LASERS_BENCHMARK="${LASERS_BENCHMARK:-all}"
LASERS_HOLDOUT_RATIO="${LASERS_HOLDOUT_RATIO:-0}"

MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}/merged_model}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/test_results}"

EVAL_METRICS_SCRIPT="${RESEG_ROOT}/eval_val_metrics.py"
EVAL_USE_WANDB="True"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-tgswin}"

RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
MIN_DISK_GB="${MIN_DISK_GB:-25}"
LEARNING_RATE="${LEARNING_RATE:-${OURS_LR:-${R2_LR:-1e-4}}}"
WEIGHT_DECAY="${WEIGHT_DECAY:-${R2_WD:-0.0}}"
WARMUP_RATIO="${WARMUP_RATIO:-${R2_WARMUP_RATIO:-0.03}}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-${R2_LR_SCHEDULE:-cosine}}"
LOGGING_STEPS="10"
BF16="True"
TF32="False"
MODEL_MAX_LENGTH="2048"
GRADIENT_CHECKPOINTING="False"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"
DATA_RATIO="1"
SWITCH_BS="4"
SEED="42"
DATA_SEED="42"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_MERGE="${RUN_MERGE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_TAXONOMY="${RUN_TAXONOMY:-1}"
RUN_CROSS_DATASET_EVAL="${RUN_CROSS_DATASET_EVAL:-1}"
MERGE_CHECKPOINT="${MERGE_CHECKPOINT:-}"

DATA_RRSISD="${DATA_RRSISD:-/root/rivermind-data/huangziyi/data/RRSISD}"
DATA_REFSEGRS="${DATA_REFSEGRS:-/root/rivermind-data/huangziyi/data/RefSegRS}"
DATA_RISBENCH="${DATA_RISBENCH:-/root/rivermind-data/huangziyi/data/RISBench_dataset}"
DATA_EARTHREASON="${DATA_EARTHREASON:-/root/rivermind-data/huangziyi/data/EarthReason}"

CROSS_DATASET_EVALS=(
  "rrsisd:${DATA_RRSISD}"
  "refsegrs:${DATA_REFSEGRS}"
  "risbench:${DATA_RISBENCH}"
  "earthreason:${DATA_EARTHREASON}"
)

TAXONOMY_SCRIPT="${RESEG_ROOT}/segearth+base/tools/diagnostics/audit_lasers_error_taxonomy.py"
TAXONOMY_ANN_FILE="${BASE_DATA_PATH}/test/annotations/test_multi_cate.json"
TAXONOMY_OUT_JSON="${TAXONOMY_OUT_JSON:-${RESEG_ROOT}/plans/baseline_diagnostics/${RUN_TAG}_multi_cate_error_taxonomy.json}"
TAXONOMY_OUT_MD="${TAXONOMY_OUT_MD:-${RESEG_ROOT}/plans/baseline_diagnostics/${RUN_TAG}_multi_cate_error_taxonomy.md}"

assert_output_dir_safe () {
  local out_dir="$1"
  if [[ "${out_dir}" != "${RESEG_ROOT}"/* ]]; then
    echo "[ERROR] OUTPUT_DIR must be under ${RESEG_ROOT}/output/"
    echo "        Got: ${out_dir}"
    exit 1
  fi
  mkdir -p "${out_dir}"
  local avail_kb avail_gb
  avail_kb="$(df -Pk "${out_dir}" | awk 'NR==2 {print $4}')"
  avail_gb=$(( avail_kb / 1024 / 1024 ))
  echo "[INFO] Disk free at output: ${avail_gb} GB (need >= ${MIN_DISK_GB} GB)"
  if [[ "${avail_gb}" -lt "${MIN_DISK_GB}" ]]; then
    echo "[ERROR] Not enough disk space at ${out_dir}"
    exit 1
  fi
}

cleanup_failed_checkpoints () {
  local out_dir="$1"
  find "${out_dir}" -maxdepth 1 -type d -name 'tmp-checkpoint-*' -print -exec rm -rf {} + 2>/dev/null || true
}

assert_gpu_available () {
  if ! "${PYTHON}" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    print("[ERROR] PyTorch cannot see a CUDA GPU.", file=sys.stderr)
    print(f"        CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", file=sys.stderr)
    print("        Check: nvidia-smi; unset empty CUDA_VISIBLE_DEVICES; retry in a fresh shell.", file=sys.stderr)
    sys.exit(1)
print(f"[OK] GPU visible: {torch.cuda.get_device_name(0)} (CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')})")
PY
  then
    exit 1
  fi
}

run_dr_ewti_preflight () {
  echo "[INFO] DR-EWTI preflight: warmstart load + LaSeRS single-batch probe"
  "${PYTHON}" tools/diagnostics/probe_tgswin_dr_ewti_warmstart_load.py \
    --model-path "${MODEL_PATH}"
  "${PYTHON}" tools/diagnostics/probe_tgswin_dr_ewti_lasers_preflight.py \
    --model-path "${MODEL_PATH}"
}

merge_ckpt () {
  local ckpt="$1"
  local save_dir="$2"
  rm -rf "${save_dir}"
  mkdir -p "${save_dir}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
    --model_path "${ckpt}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --save_path "${save_dir}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}"
}

eval_model () {
  local model_dir="$1"
  local out_dir="$2"
  mkdir -p "${out_dir}"
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON}" segearth_r2/eval/eval.py \
    --base_data_path "${BASE_DATA_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --model_path "${model_dir}" \
    --output_dir "${out_dir}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${EVAL_SPLIT}" \
    --eval_batch_size 1 \
    --dataloader_num_workers 0 \
    --skip_existing True \
    --zip_results False
}

run_eval_metrics () {
  local pred_dir="$1"
  local run_name="$2"

  USE_WANDB="${EVAL_USE_WANDB}" \
  WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
  WANDB_RUN_NAME="${run_name}" \
  DATASET_TYPE="${DATASET_NAME}" \
  BASE_DATA_PATH="${BASE_DATA_PATH}" \
  SPLIT="${EVAL_SPLIT}" \
  LASERS_BENCHMARK="${LASERS_BENCHMARK}" \
  PRED_DIR="${pred_dir}" \
  "${PYTHON}" "${EVAL_METRICS_SCRIPT}"
}

run_taxonomy () {
  local pred_dir="$1"

  if [[ ! -f "${TAXONOMY_SCRIPT}" ]]; then
    echo "[ERROR] taxonomy script not found: ${TAXONOMY_SCRIPT}"
    exit 1
  fi
  if [[ ! -d "${pred_dir}" ]]; then
    echo "[ERROR] pred-dir not found for taxonomy: ${pred_dir}"
    exit 1
  fi
  if [[ ! -f "${TAXONOMY_ANN_FILE}" ]]; then
    echo "[ERROR] ann-file not found for taxonomy: ${TAXONOMY_ANN_FILE}"
    exit 1
  fi

  mkdir -p "$(dirname "${TAXONOMY_OUT_JSON}")"

  echo "[INFO] Running multi_cate error taxonomy..."
  "${PYTHON}" "${TAXONOMY_SCRIPT}" \
    --pred-dir "${pred_dir}" \
    --ann-file "${TAXONOMY_ANN_FILE}" \
    --split-name test_multi_cate \
    --out-json "${TAXONOMY_OUT_JSON}" \
    --out-md "${TAXONOMY_OUT_MD}"
  echo "[OK] taxonomy written: ${TAXONOMY_OUT_JSON}"
}

eval_cross_dataset_one () {
  local name="$1"
  local data_path="$2"
  local pred_dir="${OUTPUT_DIR}/${name}_test_results"
  local log_eval="${OUTPUT_DIR}/${name}_test_eval.log"
  local metrics_json="${OUTPUT_DIR}/${name}_${EVAL_SPLIT}_metrics.json"

  if [[ -f "${metrics_json}" ]]; then
    echo "[SKIP] ${name}: metrics already exist (${metrics_json})"
    return 0
  fi

  mkdir -p "${pred_dir}"
  {
    echo "========================================"
    echo "[${name}] eval start $(date -Is)"
    echo "[${name}] model=${MERGED_DIR}"
    echo "[${name}] data=${data_path}"
    echo "[${name}] pred_dir=${pred_dir}"
    echo "[${name}] GPU=${GPU_ID}"

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
      --dataloader_num_workers 2 \
      --skip_existing True \
      --zip_results False

    USE_WANDB="${EVAL_USE_WANDB}" \
    WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
    WANDB_RUN_NAME="${EVAL_WANDB_RUN_NAME}-${name}" \
    DATASET_TYPE="${name}" \
    BASE_DATA_PATH="${data_path}" \
    SPLIT="${EVAL_SPLIT}" \
    PRED_DIR="${pred_dir}" \
    "${PYTHON}" "${EVAL_METRICS_SCRIPT}"

    echo "[${name}] done $(date -Is)"
  } 2>&1 | tee "${log_eval}"
}

run_cross_dataset_eval () {
  [[ -d "${MERGED_DIR}" ]] || { echo "[ERROR] merged model not found: ${MERGED_DIR}"; exit 1; }

  echo "========================================"
  echo "[7/8] Cross-dataset test eval (sequential)"
  echo "========================================"
  echo "[INFO] model=${MERGED_DIR}"
  echo "[INFO] output=${OUTPUT_DIR}"
  echo "[INFO] datasets: rrsisd → refsegrs → risbench → earthreason"

  local fail=0
  for entry in "${CROSS_DATASET_EVALS[@]}"; do
    local name="${entry%%:*}"
    local data_path="${entry#*:}"
    echo "----------------------------------------"
    echo "[QUEUE] next dataset: ${name}"
    if eval_cross_dataset_one "${name}" "${data_path}"; then
      echo "[OK] ${name} finished"
    else
      echo "[FAIL] ${name} exit=$?"
      fail=1
      break
    fi
  done

  if [[ "${fail}" -ne 0 ]]; then
    echo "[ERROR] cross-dataset eval stopped; check *_test_eval.log under ${OUTPUT_DIR}"
    exit 1
  fi

  echo "[OK] cross-dataset metrics:"
  for entry in "${CROSS_DATASET_EVALS[@]}"; do
    local name="${entry%%:*}"
    echo "  ${OUTPUT_DIR}/${name}_${EVAL_SPLIT}_metrics.json"
  done
}

check_merged_tgswin_model () {
  MERGED_MODEL_DIR="${MERGED_DIR}" VARIANT_FOR_CHECK="${VARIANT}" "${PYTHON}" - <<'PY'
import json, os
model_dir = os.environ["MERGED_MODEL_DIR"]
variant = os.environ.get("VARIANT_FOR_CHECK", "")
index_path = os.path.join(model_dir, "model.safetensors.index.json")
if not os.path.isfile(index_path):
    print(f"[WARN] missing weight index: {index_path}")
    raise SystemExit(0)
with open(index_path) as f:
    keys = list(json.load(f).get("weight_map", {}).keys())
required = ["tg_swin_tcf", "tg_swin_controller"]
if "dr-ewti" in variant.lower():
    required.append("dr_wti_blocks")
missing = [r for r in required if not any(r in k for k in keys)]
if missing:
    print(f"[WARN] merged model may miss weights: {missing}")
else:
    for r in required:
        matched = [k for k in keys if r in k]
        print(f"[OK] {r}: {matched[0] if matched else 'N/A'}")
    if "dr-ewti" in variant.lower():
        n_dr = sum(1 for k in keys if "dr_wti_blocks" in k)
        print(f"[OK] dr_wti_blocks keys in merged model: {n_dr}")
PY
}

echo "========================================"
echo "[0/8] Preflight — VARIANT=${VARIANT}"
echo "========================================"
echo "[INFO] REPO_DIR=${REPO_DIR}"
echo "[INFO] VARIANT=${VARIANT}  RUN_TAG=${RUN_TAG}"
echo "[INFO] MASK_CONFIG=${MASK_CONFIG}"
echo "[INFO] INIT checkpoint=${MODEL_PATH}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] GPU_ID=${GPU_ID}"
echo "[INFO] MAX_STEPS=${MAX_STEPS}  batch=${PER_DEVICE_TRAIN_BATCH_SIZE}  grad_accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "[INFO] SAVE_STEPS=${SAVE_STEPS}  SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
echo "[INFO] WANDB_NAME=${WANDB_NAME}"
echo "[INFO] RUN_TRAIN=${RUN_TRAIN} RUN_MERGE=${RUN_MERGE} RUN_EVAL=${RUN_EVAL} RUN_CROSS_DATASET_EVAL=${RUN_CROSS_DATASET_EVAL}"

assert_output_dir_safe "${OUTPUT_DIR}"
cleanup_failed_checkpoints "${OUTPUT_DIR}"

MASK_CONFIG="${MASK_CONFIG}" "${PYTHON}" - <<'PY'
import os
from segearth_r2.datasets.dataset import get_mask_config
cfg = get_mask_config(os.environ["MASK_CONFIG"])
tg = getattr(cfg, "TG_SWIN", None)
if not tg:
    print("[ERROR] TG_SWIN block missing"); raise SystemExit(1)
print(f"[INFO] TG_SWIN VERSION={getattr(tg, 'VERSION', 'v1')}")
print(f"[INFO] TG_SWIN NUM_STAGES={getattr(tg, 'NUM_STAGES', 4)}")
print(f"[INFO] TG_SWIN WTI_STAGES={list(getattr(tg, 'WTI_STAGES', []))}")
print(f"[INFO] TG_SWIN USE_COARSE_EVIDENCE={getattr(tg, 'USE_COARSE_EVIDENCE', False)}")
print(f"[INFO] TG_SWIN HEAD_AWARE={getattr(tg, 'HEAD_AWARE', False)}")
PY

echo "[INFO] Swin repeat multiplier estimate: ~mean(mask_num) per batch (per-target encoding)"
echo "[INFO] Running trainable param summary requires model init (see train.py logs)"

[[ -d "${MODEL_PATH}" ]] || { echo "[ERROR] model not found: ${MODEL_PATH}"; exit 1; }
[[ -d "${VISION_TOWER}" ]] || { echo "[ERROR] vision tower not found: ${VISION_TOWER}"; exit 1; }
[[ -f "${VISION_TOWER_MASK}" ]] || { echo "[ERROR] vision tower mask not found: ${VISION_TOWER_MASK}"; exit 1; }
[[ -f "${MASK_CONFIG}" ]] || { echo "[ERROR] mask config not found: ${MASK_CONFIG}"; exit 1; }

if [[ ! -f "${BASE_DATA_PATH}/train/annotations/train_data.json" ]]; then
  echo "[ERROR] train annotation not found: ${BASE_DATA_PATH}/train/annotations/train_data.json"
  echo "[HINT] 解压: tar -xzf ${BASE_DATA_PATH}/train.tar.gz -C ${BASE_DATA_PATH}"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/train/images" ]]; then
  echo "[ERROR] train images not found: ${BASE_DATA_PATH}/train/images"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/test/annotations" ]]; then
  echo "[ERROR] test annotations not found: ${BASE_DATA_PATH}/test/annotations"
  echo "[HINT] 解压: tar -xzf ${BASE_DATA_PATH}/test.tar.gz -C ${BASE_DATA_PATH}"
  exit 1
fi

if [[ ! -d "${BASE_DATA_PATH}/test/images" ]]; then
  echo "[ERROR] test images not found: ${BASE_DATA_PATH}/test/images"
  exit 1
fi

if [[ ! -f "${BASE_DATA_PATH}/test/annotations/test_multi_cate.json" ]]; then
  echo "[ERROR] taxonomy ann not found: ${BASE_DATA_PATH}/test/annotations/test_multi_cate.json"
  exit 1
fi

if [[ ! -f "scripts/zero1.json" ]]; then
  echo "[ERROR] DeepSpeed config not found: ${REPO_DIR}/scripts/zero1.json"
  exit 1
fi

echo "[OK] preflight passed (train + test present, no val required)"

if [[ "${VARIANT}" == "dr-ewti" || "${VARIANT}" == "DR-EWTI" || "${VARIANT}" == "dr_ewti" ]]; then
  if [[ "${RUN_PREFLIGHT}" == "1" && "${RUN_TRAIN}" == "1" ]]; then
    run_dr_ewti_preflight
    echo "[OK] DR-EWTI preflight probes passed"
  fi
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
assert_gpu_available
echo "========================================"
echo "[1/8] Training TG-Swin-WTI on LaSeRS train (holdout eval)"
echo "========================================"
# train_data.json；eval 用同文件 holdout 5%（非 test）
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
"${DEEPSPEED}" --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --bf16 "${BF16}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --logging_steps "${LOGGING_STEPS}" \
  --tf32 "${TF32}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio "${DATA_RATIO}" \
  --switch_bs "${SWITCH_BS}" \
  --seed "${SEED}" \
  --data_seed "${DATA_SEED}" \
  --lasers_holdout_ratio "${LASERS_HOLDOUT_RATIO}" \
  --lasers_holdout_seed "${DATA_SEED}" \
  --report_to wandb
else
echo "[1/8] SKIP training"
fi

BEST_CHECKPOINT="${MERGE_CHECKPOINT:-$(read_hf_trainer_checkpoint "${OUTPUT_DIR}" "${MAX_STEPS}" "${R2_USE_BEST_CHECKPOINT:-0}")}"
[[ -n "${BEST_CHECKPOINT}" && -d "${BEST_CHECKPOINT}" ]] || { echo "[ERROR] no checkpoint (expected checkpoint-${MAX_STEPS}) under ${OUTPUT_DIR}"; exit 1; }
echo "[2/8] SELECTED_CHECKPOINT=${BEST_CHECKPOINT}"

if [[ "${RUN_MERGE}" == "1" ]]; then
echo "[3/8] Merge"
merge_ckpt "${BEST_CHECKPOINT}" "${MERGED_DIR}"
else
echo "[3/8] SKIP merge"
fi

echo "[4/8] Check merged model"
check_merged_tgswin_model

if [[ "${RUN_EVAL}" == "1" ]]; then
echo "[5/8] LaSeRS test eval"
eval_model "${MERGED_DIR}" "${EVAL_OUTPUT_DIR}"
run_eval_metrics "${EVAL_OUTPUT_DIR}" "${EVAL_WANDB_RUN_NAME}"
if [[ "${RUN_TAXONOMY}" == "1" ]]; then
  echo "[6/8] Taxonomy (test_multi_cate)"
  run_taxonomy "${EVAL_OUTPUT_DIR}"
else
  echo "[6/8] SKIP taxonomy"
fi
else
echo "[5/8] SKIP LaSeRS eval"
echo "[6/8] SKIP taxonomy"
fi

if [[ "${RUN_CROSS_DATASET_EVAL}" == "1" ]]; then
  run_cross_dataset_eval
else
  echo "[7/8] SKIP cross-dataset eval"
fi

echo "[8/8] DONE — output=${OUTPUT_DIR}"
