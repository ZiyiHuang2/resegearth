#!/usr/bin/env bash
set -euo pipefail

PY=/home/wangchengjun/miniconda3/envs/reseg/bin/python
REPO=/home/wangchengjun/huangziyi/reseg/segearth+att
OUT=/home/wangchengjun/huangziyi/reseg/output/segearth+att-spim-alpha005
SRC="$OUT/merged_checkpoint-116000"
A5="$OUT/merged_model_e_spim_alpha5"
A10="$OUT/merged_model_e_spim_alpha10"
E0="$OUT/test_results_e0_no_spim"
FULL5="$OUT/test_results_spim_alpha5"
FULL10="$OUT/test_results_spim_alpha10"

BASE=/home/wangchengjun/huangziyi/data/RRSISD
SIG=/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384
M2F=/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl
YAML=segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml
GPU="${CUDA_VISIBLE_DEVICES:-2}"

patch_cfg () {
  local dir="$1" alpha="$2" debug="$3"
  "$PY" - <<PY
import json
from pathlib import Path
p = Path("$dir") / "config.json"
d = json.loads(p.read_text())
d["use_spim"] = True
d["spim_alpha"] = float("$alpha")
d["spim_layer_idx"] = -1
d["spim_detach"] = True
d["spim_norm"] = True
d["spim_seg_agg"] = "mean"
d["spim_near_zero_eps"] = 1e-8
d["spim_debug"] = $debug == 1
p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
PY
}

echo "== copy merged dirs =="
rm -rf "$A5" "$A10"
cp -a "$SRC" "$A5"
cp -a "$SRC" "$A10"

patch_cfg "$A5" 5.0 1
patch_cfg "$A10" 10.0 1

run_smoke () {
  local mp od log
  mp="$1"; od="$2"; log="$3"
  rm -rf "$od"
  mkdir -p "$od"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$REPO/segearth_r2/eval/eval.py" \
    --base_data_path "$BASE" \
    --vision_tower "$SIG" \
    --vision_tower_mask "$M2F" \
    --mask_config "$YAML" \
    --model_path "$mp" \
    --output_dir "$od" \
    --dataset_name rrsisd --split test \
    --eval_batch_size 1 --zip_results False --dataloader_num_workers 2 \
    --max_eval_samples 40 \
    2>&1 | tee "$log"
}

echo "== smoke alpha=5 =="
run_smoke "$A5" "$A5/_smoke_out" "$OUT/e_alpha5_smoke.log"
echo "== smoke alpha=10 =="
run_smoke "$A10" "$A10/_smoke_out" "$OUT/e_alpha10_smoke.log"

echo "== grep smoke =="
G='\[SPIM eval debug\]|prior_shape|near_zero_count|NaN|OOM|Traceback|spatial_bias\.shape'
grep -E "$G" "$OUT/e_alpha5_smoke.log" | head -30 || true
grep -E "$G" "$OUT/e_alpha10_smoke.log" | head -30 || true

if grep -E 'Traceback|CUDA out of memory|nan|NaN' "$OUT/e_alpha5_smoke.log" "$OUT/e_alpha10_smoke.log"; then
  echo "SMOKE had error keywords — 请人工决定是否继续全量"; exit 1
fi

echo "== turn spim_debug off for full eval =="
patch_cfg "$A5" 5.0 0
patch_cfg "$A10" 10.0 0

echo "== full eval =="
rm -rf "$FULL5" "$FULL10"
mkdir -p "$FULL5" "$FULL10"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$REPO/segearth_r2/eval/eval.py" \
  --base_data_path "$BASE" --vision_tower "$SIG" --vision_tower_mask "$M2F" \
  --mask_config "$YAML" --model_path "$A5" --output_dir "$FULL5" \
  --dataset_name rrsisd --split test --eval_batch_size 1 --zip_results False --dataloader_num_workers 4 \
  2>&1 | tee "$OUT/e_full_alpha5.log"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$REPO/segearth_r2/eval/eval.py" \
  --base_data_path "$BASE" --vision_tower "$SIG" --vision_tower_mask "$M2F" \
  --mask_config "$YAML" --model_path "$A10" --output_dir "$FULL10" \
  --dataset_name rrsisd --split test --eval_batch_size 1 --zip_results False --dataloader_num_workers 4 \
  2>&1 | tee "$OUT/e_full_alpha10.log"

echo "== metrics (copy json immediately after each) =="
cd /home/wangchengjun/huangziyi/reseg
export USE_WANDB=0 DATASET_TYPE=rrsisd SPLIT=test BASE_DATA_PATH="$BASE"

export PRED_DIR="$FULL5"
"$PY" eval_val_metrics.py
cp -f "$OUT/rrsisd_test_metrics.json" "$OUT/rrsisd_test_metrics_alpha5.json"

export PRED_DIR="$FULL10"
"$PY" eval_val_metrics.py
cp -f "$OUT/rrsisd_test_metrics.json" "$OUT/rrsisd_test_metrics_alpha10.json"

echo "== prediction diff (tif, seed=42, n=200) =="
"$PY" - <<'PY'
import os, random, numpy as np
import tifffile as tiff

def stats(d0, d1, n=200, seed=42):
    f0 = sorted(os.listdir(d0))
    f1 = sorted(os.listdir(d1))
    common = sorted(set(f0) & set(f1))
    rng = random.Random(seed)
    if len(common) > n:
        common = rng.sample(common, n)
    changed = 0
    abs_sum = 0.0
    pix = 0
    ratios = []
    for fn in common:
        a = tiff.imread(os.path.join(d0, fn))
        b = tiff.imread(os.path.join(d1, fn))
        if a.shape != b.shape:
            continue
        a = (a > 0).astype(np.uint8)
        b = (b > 0).astype(np.uint8)
        if not (a == b).all():
            changed += 1
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        abs_sum += float(d.sum())
        pix += d.size
        ratios.append(float(np.mean(a != b)))
    ratios.sort()
    m = len(ratios)
    return {
        "sampled_count": m,
        "changed_count": changed,
        "changed_ratio": changed / m if m else 0.0,
        "mean_abs_diff": abs_sum / max(pix, 1),
        "mean_pixel_diff_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "median_pixel_diff_ratio": ratios[m // 2] if m else 0.0,
        "max_pixel_diff_ratio": max(ratios) if ratios else 0.0,
    }

E0 = r"/home/wangchengjun/huangziyi/reseg/output/segearth+att-spim-alpha005/test_results_e0_no_spim"
A5 = r"/home/wangchengjun/huangziyi/reseg/output/segearth+att-spim-alpha005/test_results_spim_alpha5"
A10 = r"/home/wangchengjun/huangziyi/reseg/output/segearth+att-spim-alpha005/test_results_spim_alpha10"
for name, da, db in [
    ("E0 vs alpha5", E0, A5),
    ("E0 vs alpha10", E0, A10),
    ("alpha5 vs alpha10", A5, A10),
]:
    print(name, stats(da, db))
PY

echo "DONE"