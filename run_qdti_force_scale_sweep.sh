#!/usr/bin/env bash
set -euo pipefail

########################################
# QDTI eval-only force_scale sweep
# - No training.
# - Reuses the existing QDTI merged_model.
# - Tests whether QDTI is under-injected by scaling only eval-time decoder bias.
########################################

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

REPO_DIR="/root/rivermind-data/huangziyi/reseg/resegearth+tgi"
cd "${REPO_DIR}"

PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
GPU_ID="${GPU_ID:-0}"

BASE_ROOT="/root/rivermind-data/huangziyi/reseg"
BASE_DATA_PATH="${BASE_DATA_PATH:-/root/rivermind-data/huangziyi/data/RRSISD}"
DATASET_NAME="${DATASET_NAME:-rrsisd}"
TEST_SPLIT="${TEST_SPLIT:-test}"

VISION_TOWER="${VISION_TOWER:-${BASE_ROOT}/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-${BASE_ROOT}/pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"

QDTI_OUTPUT_DIR="${QDTI_OUTPUT_DIR:-${BASE_ROOT}/output/tgi/qdti-8w}"
MERGED_DIR="${MERGED_DIR:-${QDTI_OUTPUT_DIR}/merged_model}"
SWEEP_DIR="${SWEEP_DIR:-${QDTI_OUTPUT_DIR}/force_scale_sweep}"
BASELINE_METRICS="${BASELINE_METRICS:-${BASE_ROOT}/output/base/standard-base-siglip1-28w-gd4/rrsisd_test_metrics.json}"
EVAL_METRICS_SCRIPT="${EVAL_METRICS_SCRIPT:-${BASE_ROOT}/eval_val_metrics.py}"

# Space-separated list. Override example: SCALES="1 2 5 10 20" bash run_qdti_force_scale_sweep.sh
SCALES="${SCALES:-1 5 10 20}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"

# Default off to avoid creating many W&B runs during diagnostic sweeps.
EVAL_USE_WANDB="${EVAL_USE_WANDB:-False}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-tgi-force-scale}"
EVAL_WANDB_RUN_PREFIX="${EVAL_WANDB_RUN_PREFIX:-qdti-force-scale}"
PRECHECK_ONLY="${PRECHECK_ONLY:-False}"

safe_scale_name() {
  local scale="$1"
  scale="${scale//./p}"
  scale="${scale//-/_neg_}"
  echo "${scale}"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] required file not found: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] required directory not found: ${path}" >&2
    exit 1
  fi
}

preflight() {
  require_dir "${MERGED_DIR}"
  require_file "${MERGED_DIR}/config.json"
  require_file "${EVAL_METRICS_SCRIPT}"
  require_file "${MASK_CONFIG}"
  require_dir "${VISION_TOWER}"
  require_file "${VISION_TOWER_MASK}"

  "${PYTHON}" - <<PY
import json, os, sys
merged = "${MERGED_DIR}"
cfg_path = os.path.join(merged, "config.json")
cfg = json.load(open(cfg_path, encoding="utf-8"))
errors = []
if cfg.get("use_query_aware_decoder_bias") is not True:
    errors.append("use_query_aware_decoder_bias is not True")
if cfg.get("allow_random_qdti_init") is not False:
    errors.append("allow_random_qdti_init is not False")
if cfg.get("decoder_attn_bias_apply_layers") != "last3":
    errors.append("decoder_attn_bias_apply_layers is not last3")
idx_path = os.path.join(merged, "model.safetensors.index.json")
keys = []
if os.path.isfile(idx_path):
    idx = json.load(open(idx_path, encoding="utf-8"))
    keys = [k for k in idx.get("weight_map", {}) if "qdti_core" in k]
else:
    for name in ("model.safetensors", "pytorch_model.bin"):
        if os.path.isfile(os.path.join(merged, name)):
            keys = [name]
            break
if not keys:
    errors.append("merged weights contain no qdti_core keys")
if errors:
    print("[ERROR] QDTI merged model preflight failed:", file=sys.stderr)
    for e in errors:
        print("  - " + e, file=sys.stderr)
    sys.exit(1)
print(f"[INFO] QDTI preflight OK: qdti_core_keys={len(keys)}")
print("[INFO] eval will override decoder_attn_bias_eval_mode=force_scale per scale")
PY
}

run_eval_for_scale() {
  local scale="$1"
  local scale_name
  scale_name="$(safe_scale_name "${scale}")"
  local scale_dir="${SWEEP_DIR}/scale_${scale_name}"
  local pred_dir="${scale_dir}/test_results"
  local eval_log="${scale_dir}/eval.log"
  local metric_file="${scale_dir}/${DATASET_NAME}_${TEST_SPLIT}_metrics_force_scale_${scale_name}.json"

  if [[ -f "${metric_file}" ]]; then
    echo "[SKIP] scale=${scale}: metrics already exist: ${metric_file}"
    return 0
  fi

  if [[ -d "${pred_dir}" && ! -f "${metric_file}" ]]; then
    echo "[ERROR] partial output exists without metrics for scale=${scale}: ${pred_dir}" >&2
    echo "        move/remove that scale directory manually before rerunning." >&2
    exit 1
  fi

  mkdir -p "${pred_dir}"

  echo "========================================"
  echo "[Eval] QDTI force_scale=${scale}"
  echo "model : ${MERGED_DIR}"
  echo "pred  : ${pred_dir}"
  echo "log   : ${eval_log}"
  echo "========================================"

  local max_eval_args=()
  if [[ "${MAX_EVAL_SAMPLES}" != "0" ]]; then
    max_eval_args+=(--max_eval_samples "${MAX_EVAL_SAMPLES}")
  fi

  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON}" segearth_r2/eval/eval.py \
    --base_data_path "${BASE_DATA_PATH}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --model_path "${MERGED_DIR}" \
    --output_dir "${pred_dir}" \
    --dataset_name "${DATASET_NAME}" \
    --split "${TEST_SPLIT}" \
    --eval_batch_size 1 \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --zip_results False \
    --allow_random_qdti_init False \
    --decoder_attn_bias_eval_mode force_scale \
    --decoder_attn_bias_force_scale "${scale}" \
    "${max_eval_args[@]}" \
    2>&1 | tee "${eval_log}"

  echo "[Metrics] scale=${scale}"
  USE_WANDB="${EVAL_USE_WANDB}" \
  WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
  WANDB_RUN_NAME="${EVAL_WANDB_RUN_PREFIX}-${scale_name}" \
  DATASET_TYPE="${DATASET_NAME}" \
  BASE_DATA_PATH="${BASE_DATA_PATH}" \
  SPLIT="${TEST_SPLIT}" \
  PRED_DIR="${pred_dir}" \
  METRICS_TAG="force_scale_${scale_name}" \
  "${PYTHON}" "${EVAL_METRICS_SCRIPT}" | tee "${scale_dir}/metrics.log"
}

write_summary() {
  "${PYTHON}" - <<PY
import csv, json, os
sweep_dir = "${SWEEP_DIR}"
baseline_path = "${BASELINE_METRICS}"
scales = "${SCALES}".split()
dataset = "${DATASET_NAME}"
split = "${TEST_SPLIT}"
keys = ["mIoU", "gIoU", "cIoU", "mDice", "mRecall", "mPrecision", "Pr@0.5", "Pr@0.6", "Pr@0.7", "Pr@0.8", "Pr@0.9", "box_level_iou", "box_level_giou", "box_level_ciou"]

def safe_scale(scale):
    return scale.replace('.', 'p').replace('-', '_neg_')

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

baseline = load(baseline_path) if os.path.isfile(baseline_path) else None
rows = []
scale1 = None
for scale in scales:
    name = safe_scale(scale)
    path = os.path.join(sweep_dir, f"scale_{name}", f"{dataset}_{split}_metrics_force_scale_{name}.json")
    if not os.path.isfile(path):
        print(f"[WARN] missing metrics for scale={scale}: {path}")
        continue
    m = load(path)
    if scale == "1":
        scale1 = m
    row = {"scale": scale, "metrics_file": path}
    for k in keys:
        row[k] = m.get(k)
    rows.append(row)

summary_csv = os.path.join(sweep_dir, "summary_force_scale_sweep.csv")
os.makedirs(sweep_dir, exist_ok=True)
fieldnames = ["scale"] + keys + [f"delta_vs_base_gd4_{k}" for k in keys] + [f"delta_vs_scale1_{k}" for k in keys] + ["metrics_file"]
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in rows:
        out = dict(row)
        for k in keys:
            v = row.get(k)
            out[f"delta_vs_base_gd4_{k}"] = None if baseline is None or v is None else v - baseline.get(k, 0.0)
            out[f"delta_vs_scale1_{k}"] = None if scale1 is None or v is None else v - scale1.get(k, 0.0)
        w.writerow(out)

print("\n=== QDTI force_scale sweep summary ===")
print(f"summary_csv: {summary_csv}")
if baseline:
    print(f"baseline: {baseline_path}")
print("scale\tmIoU\tcIoU\tPr@0.8\tPr@0.9\td_mIoU_base\td_cIoU_base")
for row in rows:
    def fmt(x):
        return "NA" if x is None else f"{x:.6f}"
    miou = row.get("mIoU")
    ciou = row.get("cIoU")
    pr8 = row.get("Pr@0.8")
    pr9 = row.get("Pr@0.9")
    dmiou = None if baseline is None or miou is None else miou - baseline.get("mIoU", 0.0)
    dciou = None if baseline is None or ciou is None else ciou - baseline.get("cIoU", 0.0)
    print(f"{row['scale']}\t{fmt(miou)}\t{fmt(ciou)}\t{fmt(pr8)}\t{fmt(pr9)}\t{fmt(dmiou)}\t{fmt(dciou)}")
PY
}

main() {
  echo "[INFO] REPO_DIR=${REPO_DIR}"
  echo "[INFO] MERGED_DIR=${MERGED_DIR}"
  echo "[INFO] SWEEP_DIR=${SWEEP_DIR}"
  echo "[INFO] SCALES=${SCALES}"
  echo "[INFO] GPU_ID=${GPU_ID}"
  echo "[INFO] EVAL_USE_WANDB=${EVAL_USE_WANDB}"
  echo "[INFO] PRECHECK_ONLY=${PRECHECK_ONLY}"
  preflight
  if [[ "${PRECHECK_ONLY}" == "True" ]]; then
    echo "[INFO] PRECHECK_ONLY=True; skip eval runs."
    exit 0
  fi
  mkdir -p "${SWEEP_DIR}"
  for scale in ${SCALES}; do
    run_eval_for_scale "${scale}"
  done
  write_summary
}

main "$@"
