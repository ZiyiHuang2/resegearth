#!/usr/bin/env bash
# Q_seg-only sanity: baseline merged model + fresh DGP modules must match ~67% gIoU.
set -euo pipefail

REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+DGP"
cd "${REPO_DIR}"

PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
BASE_ROOT="${BASE_ROOT:-/root/rivermind-data/huangziyi/reseg/output/base/standard-base-siglip1-28w-gd4}"
BASE_MERGED="${BASE_MERGED:-${BASE_ROOT}/merged_model}"
BASE_DATA_PATH="${BASE_DATA_PATH:-/root/rivermind-data/huangziyi/data/RRSISD}"
VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
OUT_ROOT="${OUT_ROOT:-/root/rivermind-data/huangziyi/reseg/output/dgp/v6.1-stage-a-from-base/sanity_qseg}"
PRED_DIR="${OUT_ROOT}/test_results_qseg"
METRICS_JSON="${OUT_ROOT}/rrsisd_test_metrics_qseg_sanity.json"
BASE_METRICS="${BASE_ROOT}/rrsisd_test_metrics.json"

mkdir -p "${OUT_ROOT}"

echo "========== [1] Baseline merged model reference metrics =========="
if [[ -f "${BASE_METRICS}" ]]; then
  "${PYTHON}" - <<PY
import json
m = json.load(open("${BASE_METRICS}"))
print(f"  path: ${BASE_METRICS}")
print(f"  gIoU={m['gIoU']*100:.2f}% cIoU={m['cIoU']*100:.2f}% Pr@0.5={m['Pr@0.5']*100:.2f}%")
PY
else
  echo "  [WARN] missing ${BASE_METRICS}; run baseline test eval first"
fi

echo ""
echo "========== [2] Q_seg-only inference (DGP attached, dgp_use_refined_query=False) =========="
rm -rf "${PRED_DIR}"
mkdir -p "${PRED_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON}" segearth_r2/eval/eval.py \
  --model_path "${BASE_MERGED}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name rrsisd \
  --split test \
  --output_dir "${PRED_DIR}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --use_dgp_qdti True \
  --use_qdti_bias False \
  --dgp_training_stage a \
  --gate_g_init 0.01 \
  --gate_l_init 0.02 \
  --dgp_use_refined_query False \
  --eval_batch_size 1 \
  --dataloader_num_workers 4 \
  --max_eval_samples 0 \
  --zip_results False

echo ""
echo "========== [3] Metrics =========="
USE_WANDB=false DATASET_TYPE=rrsisd SPLIT=test \
  BASE_DATA_PATH="${BASE_DATA_PATH}" \
  PRED_DIR="${PRED_DIR}" \
  METRICS_TAG="qseg_sanity" \
  "${PYTHON}" /root/rivermind-data/huangziyi/reseg/eval_val_metrics.py \
  > "${OUT_ROOT}/eval_val_metrics.log" 2>&1

LATEST=$(ls -t "${OUT_ROOT}"/rrsisd_test_metrics_*.json 2>/dev/null | head -1 || true)
if [[ -n "${LATEST}" ]]; then
  cp -f "${LATEST}" "${METRICS_JSON}"
fi

echo ""
echo "========== [4] Verdict =========="
"${PYTHON}" - <<PY
import json, sys
path = "${METRICS_JSON}"
try:
    m = json.load(open(path))
except FileNotFoundError:
    print("INIT_BROKEN: metrics json missing")
    sys.exit(2)
giou = float(m["gIoU"])
ciou = float(m["cIoU"])
pr05 = float(m["Pr@0.5"])
print(f"Q_seg-only: gIoU={giou*100:.2f}% cIoU={ciou*100:.2f}% Pr@0.5={pr05*100:.2f}%")
if giou < 0.50:
    print("VERDICT: INIT_BROKEN (gIoU < 50%, loading chain wrong)")
    sys.exit(2)
if giou < 0.60:
    print("VERDICT: INIT_WARN (gIoU 50-60%, below baseline ~67%)")
    sys.exit(1)
print("VERDICT: INIT_OK (Q_seg-only near baseline, safe to train Stage A)")
sys.exit(0)
PY
