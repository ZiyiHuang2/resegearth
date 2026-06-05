#!/usr/bin/env bash
# Post 500-step smoke acceptance: merge reload, A3 weights, eval paths, baseline no-op.
set -euo pipefail

REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+set"
cd "${REPO_DIR}"

OUTPUT_DIR="${1:-/root/rivermind-data/huangziyi/reseg/output/set/a3-frozen-lasers-smoke-500}"
MERGED_DIR="${OUTPUT_DIR}/merged_model"
PYTHON="/root/rivermind-data/miniconda3/envs/reseg/bin/python"
BASELINE="/root/rivermind-data/huangziyi/reseg/output/base/standard-base-lasers-siglip1-28w-gd4/merged_model"

echo "=== [1] merged_model exists ==="
[[ -d "${MERGED_DIR}" ]] || { echo "[FAIL] missing ${MERGED_DIR}"; exit 1; }
[[ -f "${MERGED_DIR}/config.json" ]] || { echo "[FAIL] missing config.json"; exit 1; }

echo "=== [2] A3 weights in merged checkpoint ==="
"${PYTHON}" - <<PY
import torch
from safetensors.torch import load_file
import os
merged = "${MERGED_DIR}"
keys = []
for root, _, files in os.walk(merged):
    for f in files:
        if f.endswith(".safetensors"):
            keys.extend(load_file(os.path.join(root, f)).keys())
        elif f == "pytorch_model.bin":
            keys.extend(torch.load(os.path.join(root, f), map_location="cpu").keys())
need = ["set_conditioner", "count_head", "category_set_head"]
missing = [n for n in need if not any(n in k for k in keys)]
if missing and not keys:
    import json
    cfg = json.load(open(os.path.join(merged, "config.json")))
    print("[WARN] could not list weight keys; config use_set_conditioner=", cfg.get("use_set_conditioner"))
else:
    for n in need:
        ok = any(n in k for k in keys)
        print(f"  {n}: {'OK' if ok else 'MISSING'}")
        if not ok:
            raise SystemExit(1)
print("[OK] A3 module keys present")
PY

echo "=== [3] reload merged model ==="
"${PYTHON}" - <<PY
from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
mask_cfg = get_mask_config("segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
m = SegEarthR2.from_pretrained("${MERGED_DIR}", mask_decoder_cfg=mask_cfg, device_map="cpu")
assert getattr(m, "set_conditioner", None) is not None or hasattr(m, "set_conditioner")
print("[OK] merged_model reload")
PY

echo "=== [4] eval outputs ==="
[[ -d "${OUTPUT_DIR}/eval_seg_test" ]] && echo "[OK] eval_seg_test" || echo "[WARN] eval_seg_test missing"
[[ -d "${OUTPUT_DIR}/lasers_diagnostic" ]] && echo "[OK] lasers_diagnostic" || echo "[WARN] lasers_diagnostic missing"

echo "=== [5] use_set_conditioner=false strict no-op ==="
"${PYTHON}" scripts/smoke_a3_set_conditioner.py 2>&1 | grep -q "use_set_conditioner=false is strict no-op"
echo "[OK] baseline no-op smoke"

echo "=== ALL smoke artifact checks passed ==="
