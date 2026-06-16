#!/usr/bin/env bash
# Acceptance checks for Stage 1 Seg Spatial Refiner (no 10k training).
set -euo pipefail
RESEG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${RESEG_ROOT}"
export PYTHONPATH="detectron2:${RESEG_ROOT}/segearth_r2/train:${RESEG_ROOT}:${PYTHONPATH:-}"
PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"

SMOKE_DIR="${SMOKE_DIR:-${RESEG_ROOT}/output/tgi/smoke_seg_spatial_refiner_v1}"
CKPT_DIR="${CKPT_DIR:-}"
MERGED_DIR="${MERGED_DIR:-${SMOKE_DIR}/merged_model}"
EVAL_OUT="${EVAL_OUT:-${SMOKE_DIR}/eval_smoke}"
BASE_MODEL="${BASE_MODEL:-/root/rivermind-data/huangziyi/reseg/output/base/standard-base-siglip1-28w-gd4/merged_model}"
VISION_TOWER="${VISION_TOWER:-/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"
BASE_DATA_PATH="${BASE_DATA_PATH:-/root/rivermind-data/huangziyi/data/RRSISD}"

echo "========================================"
echo "[A] Smoke train (2 steps) + sidecar save"
echo "========================================"
if [[ ! -f "${SMOKE_DIR}/seg_spatial_refiner_trainable.pt" ]]; then
  OUTPUT_DIR="${SMOKE_DIR}" SAVE_STEPS=1000000 bash scripts/smoke_train_seg_spatial_refiner.sh
else
  echo "[SKIP] sidecar already exists: ${SMOKE_DIR}/seg_spatial_refiner_trainable.pt"
fi
if [[ ! -f "${SMOKE_DIR}/seg_spatial_refiner_trainable.pt" ]]; then
  echo "[ERROR] missing sidecar: ${SMOKE_DIR}/seg_spatial_refiner_trainable.pt"
  exit 1
fi
echo "[OK] sidecar saved"

echo "========================================"
echo "[B] Sidecar refiner keys"
echo "========================================"
SMOKE_DIR="${SMOKE_DIR}" "${PYTHON}" - <<'PY'
import os, torch
sidecar=os.path.join(os.environ["SMOKE_DIR"], "seg_spatial_refiner_trainable.pt")
sd=torch.load(sidecar, map_location="cpu")
keys=[k for k in sd if "seg_spatial_refiner" in k]
print(f"[INFO] sidecar seg_spatial_refiner keys={len(keys)}")
print("[OK] sample:", keys[:4])
if len(keys) != 8:
    raise SystemExit(f"[ERROR] expected 8 refiner keys, got {len(keys)}")
PY

echo "========================================"
echo "[C] Merge from sidecar"
echo "========================================"
rm -rf "${MERGED_DIR}"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
  --model_path "${SMOKE_DIR}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --save_path "${MERGED_DIR}" \
  --use_seg_spatial_refiner True \
  --seg_spatial_refiner_alpha 0.1 \
  --seg_spatial_refiner_loss_weight 0.1 \
  --seg_spatial_refiner_dice_weight 1.0 \
  --seg_spatial_refiner_bce_weight 1.0 \
  --use_query_aware_decoder_bias False \
  --use_decoder_attn_bias False \
  --use_mstva False \
  --use_text_film False

echo "========================================"
echo "[D] Verify merged config/weights"
echo "========================================"
MERGED_DIR="${MERGED_DIR}" "${PYTHON}" - <<'PY'
import json, os, sys
from pathlib import Path
from safetensors import safe_open

merged = Path(os.environ["MERGED_DIR"])
cfg = json.load(open(merged / "config.json", encoding="utf-8"))
checks = [
    ("use_seg_spatial_refiner", True),
    ("seg_spatial_refiner_alpha", 0.1),
    ("seg_spatial_refiner_loss_weight", 0.1),
]
for k, exp in checks:
    val = cfg.get(k)
    print(f"[INFO] config {k}={val}")
    if k == "use_seg_spatial_refiner" and not bool(val):
        sys.exit(f"[ERROR] config {k} != True")
keys = []
idx = merged / "model.safetensors.index.json"
if idx.is_file():
    wm = json.load(open(idx, encoding="utf-8")).get("weight_map", {})
    keys = [k for k in wm if "seg_spatial_refiner" in k]
single = merged / "model.safetensors"
if not keys and single.is_file():
    with safe_open(str(single), framework="pt") as f:
        keys = [k for k in f.keys() if "seg_spatial_refiner" in k]
print(f"[INFO] merged seg_spatial_refiner keys={len(keys)}")
if not keys:
    sys.exit("[ERROR] merged safetensors missing seg_spatial_refiner weights")
print("[OK] sample:", keys[:4])
PY

echo "========================================"
echo "[E] Eval smoke (max_eval_samples=2)"
echo "========================================"
rm -rf "${EVAL_OUT}"
mkdir -p "${EVAL_OUT}"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" segearth_r2/eval/eval.py \
  --base_data_path "${BASE_DATA_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --model_path "${MERGED_DIR}" \
  --output_dir "${EVAL_OUT}" \
  --dataset_name rrsisd \
  --split test \
  --eval_batch_size 1 \
  --max_eval_samples 2 \
  --zip_results False

TIF_COUNT=$(find "${EVAL_OUT}" -name '*.tif' | wc -l)
echo "[INFO] tif count under ${EVAL_OUT}: ${TIF_COUNT}"
if [[ "${TIF_COUNT}" -lt 1 ]]; then
  echo "[ERROR] eval did not produce tif"
  exit 1
fi
echo "[OK] eval produced tif"

echo "========================================"
echo "[F] Frozen-only trainable param audit"
echo "========================================"
CKPT_DIR="${CKPT_DIR:-}" BASE_MODEL="${BASE_MODEL}" VISION_TOWER="${VISION_TOWER}" \
VISION_TOWER_MASK="${VISION_TOWER_MASK}" MASK_CONFIG="${MASK_CONFIG}" BASE_DATA_PATH="${BASE_DATA_PATH}" \
PYTHONPATH="${RESEG_ROOT}/detectron2:${RESEG_ROOT}/segearth_r2/train:${RESEG_ROOT}:${PYTHONPATH:-}" \
"${PYTHON}" - <<'PY'
import os, sys
sys.path.insert(0, os.environ.get("RESEG_ROOT", "."))
sys.path.insert(0, os.path.join(os.environ.get("RESEG_ROOT", "."), "detectron2"))
sys.path.insert(0, os.path.join(os.environ.get("RESEG_ROOT", "."), "segearth_r2", "train"))
import torch
from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from peft import LoraConfig, get_peft_model

base = os.environ["BASE_MODEL"]
mask_cfg = get_mask_config(os.environ["MASK_CONFIG"])
model = SegEarthR2.from_pretrained(base, mask_decoder_cfg=mask_cfg, add_cross_attn=True)
model.config.use_seg_spatial_refiner = True
model.config.seg_spatial_refiner_alpha = 0.1
model.ensure_seg_spatial_refiner_branch(allow_init=True)
train_module_list = ["lm_head", "pixel_decoder", "predictor", "SEG_token_projector", "mid_stage_text_recalibration", "mstva", "text_film_branch"]
from segearth_r2.train.train import find_linear_layers, _enable_seg_spatial_refiner_trainable, log_seg_spatial_refiner_trainable_params
lora_target_modules = find_linear_layers(model, train_module_list=train_module_list)
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=lora_target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
for p in model.parameters():
    p.requires_grad = False
_enable_seg_spatial_refiner_trainable(model)
log_seg_spatial_refiner_trainable_params(model, local_rank_value=0)
lora_trainable = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
if lora_trainable:
    raise RuntimeError(f"[FAIL-FAST] LoRA params trainable: {lora_trainable[:8]}")
print("[OK] LoRA trainable count=0")
PY

echo "========================================"
echo "[G] Default-off forward check"
echo "========================================"
BASE_MODEL="${BASE_MODEL}" MASK_CONFIG="${MASK_CONFIG}" RESEG_ROOT="${RESEG_ROOT}" \
PYTHONPATH="${RESEG_ROOT}/detectron2:${RESEG_ROOT}:${PYTHONPATH:-}" \
"${PYTHON}" tools/verify_seg_refiner_default_off.py

echo "========================================"
echo "[PASS] acceptance_seg_spatial_refiner"
echo "========================================"
