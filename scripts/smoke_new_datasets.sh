#!/usr/bin/env bash
# Smoke tests for RefSegRS / RISBench integration (GPU 0).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

REPO_DIR="/home/wangchengjun/huangziyi/reseg/segearth+base"
cd "${REPO_DIR}"

PYTHON="/home/wangchengjun/miniconda3/envs/reseg/bin/python3"
DEEPSPEED="/home/wangchengjun/miniconda3/envs/reseg/bin/deepspeed"
export PATH="/home/wangchengjun/miniconda3/envs/reseg/bin:${PATH}"
MODEL_PATH="/home/wangchengjun/huangziyi/reseg/output/itaa/coarse_refine_layer2_w005_unfreezePD-5w/merged_model"
VISION_TOWER="/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
MODEL_BASE="/home/wangchengjun/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
SMOKE_ROOT="/tmp/segearth_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SMOKE_ROOT}"
LOG="${SMOKE_ROOT}/smoke.log"
exec > >(tee -a "${LOG}") 2>&1

echo "========== smoke log: ${LOG} =========="

echo "[1/7] py_compile"
"${PYTHON}" -m py_compile \
  segearth_r2/datasets/dataset.py \
  segearth_r2/train/train.py \
  segearth_r2/eval/eval.py

echo "[2/7] eval import check"
"${PYTHON}" -c "
import ast
s=open('segearth_r2/eval/eval.py').read()
imp={n.name for n in ast.walk(ast.parse(s))
     if isinstance(n, ast.ImportFrom) and n.module=='segearth_r2.datasets.dataset'
     for n in n.names}
for c in ('RefSegRSDataset','RISBenchDataset','RRSISDDataset','LaSeRSDataset'):
    assert c in imp, (c, imp)
print('eval imports OK:', sorted(imp))
"

echo "[3/7] dataset + collator smoke"
"${PYTHON}" - <<'PY'
import sys
sys.path.insert(0, ".")
from transformers import AutoTokenizer, SiglipImageProcessor
from segearth_r2.datasets.dataset import (
    RefSegRSDataset, RISBenchDataset, RRSISDDataset, LaSeRSDataset,
    DataCollatorForCOCODatasetV2,
)

tok = AutoTokenizer.from_pretrained(
    "/home/wangchengjun/huangziyi/reseg/pretrained_model/mllm/Mipha-3B",
    trust_remote_code=True,
)
clip = SiglipImageProcessor.from_pretrained(
    "/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384",
)
collator = DataCollatorForCOCODatasetV2(tokenizer=tok, clip_image_processor=clip)
da = object()

cases = [
    ("refsegrs-train", RefSegRSDataset, "/home/wangchengjun/huangziyi/data/RefSegRS", "train"),
    ("refsegrs-val", RefSegRSDataset, "/home/wangchengjun/huangziyi/data/RefSegRS", "val"),
    ("risbench-train", RISBenchDataset, "/home/wangchengjun/huangziyi/data/RISBench", "train"),
    ("risbench-val", RISBenchDataset, "/home/wangchengjun/huangziyi/data/RISBench", "val"),
    ("risbench-test", RISBenchDataset, "/home/wangchengjun/huangziyi/data/RISBench", "test"),
    ("rrsisd-val", RRSISDDataset, "/home/wangchengjun/huangziyi/data/RRSISD", "val"),
    ("lasers-val", LaSeRSDataset, "/home/wangchengjun/huangziyi/data/LaSeRS", "val_data.json"),
]
for name, cls, root, split in cases:
    ds = cls(root, tok, da, split=split)
    batch = collator([ds[0], ds[1]])
    assert batch["images"].shape[0] == 2
    assert len(batch["seg_info"]) == 2
    print(name, "OK len=", len(ds))

# RISBench duplicate key output names
ds = RISBenchDataset("/home/wangchengjun/huangziyi/data/RISBench", tok, da, split="test")
names = []
for i, rec in enumerate(ds.reason_file):
    if rec["stem"] == "test_979_0":
        ann = ds[i]["annotations"][0]
        names.append(f"{ann['image_id']}_{ann['data_id']}_test_0.tif")
assert len(names) == 2 and names[0] != names[1], names
print("risbench dup-key naming OK:", names)
PY

run_eval () {
  local ds_name="$1"
  local split="$2"
  local data_path="$3"
  local out_dir="${SMOKE_ROOT}/eval_${ds_name}_${split}"
  echo "[eval] ${ds_name} split=${split} -> ${out_dir}"
  "${PYTHON}" segearth_r2/eval/eval.py \
    --base_data_path "${data_path}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${out_dir}" \
    --dataset_name "${ds_name}" \
    --split "${split}" \
    --eval_batch_size 1 \
    --max_eval_samples 1 \
    --dataloader_num_workers 0 \
    --zip_results False
  local n
  n=$(find "${out_dir}" -maxdepth 1 -name '*.tif' | wc -l)
  echo "[eval] ${ds_name}/${split} wrote ${n} mask(s)"
  test "${n}" -ge 1
}

echo "[4/7] eval refsegrs val"
run_eval refsegrs val /home/wangchengjun/huangziyi/data/RefSegRS

echo "[5/7] eval risbench test (dup-key split)"
run_eval risbench test /home/wangchengjun/huangziyi/data/RISBench

echo "[6/7] eval regression rrsisd val + lasers val"
run_eval rrsisd val /home/wangchengjun/huangziyi/data/RRSISD
# LaSeRS uses all val json files; pick val_data.json via split loop — use dataset_name lasers (auto)
OUT_LASERS="${SMOKE_ROOT}/eval_lasers"
mkdir -p "${OUT_LASERS}"
"${PYTHON}" segearth_r2/eval/eval.py \
  --base_data_path /home/wangchengjun/huangziyi/data/LaSeRS \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUT_LASERS}" \
  --dataset_name lasers \
  --split val \
  --eval_batch_size 1 \
  --max_eval_samples 1 \
  --dataloader_num_workers 0 \
  --zip_results False
n=$(find "${OUT_LASERS}" -maxdepth 1 -name '*.tif' | wc -l)
echo "[eval] lasers wrote ${n} mask(s)"
test "${n}" -ge 1

run_train_smoke () {
  local ds_name="$1"
  local data_path="$2"
  local out_dir="${SMOKE_ROOT}/train_${ds_name}"
  echo "[train] ${ds_name} 1 step -> ${out_dir}"
  "${DEEPSPEED}" --include localhost:0 --master_port=29610 \
    segearth_r2/train/train.py \
    --model_name_or_path "${MODEL_BASE}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --base_data_path "${data_path}" \
    --dataset_name "${ds_name}" \
    --output_dir "${out_dir}" \
    --max_steps 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy no \
    --logging_steps 1 \
    --bf16 True \
    --dataloader_num_workers 2 \
    --deepspeed scripts/zero1.json
}

echo "[7/7] train smoke (refsegrs, risbench, rrsisd)"
run_train_smoke refsegrs /home/wangchengjun/huangziyi/data/RefSegRS
run_train_smoke risbench /home/wangchengjun/huangziyi/data/RISBench
run_train_smoke rrsisd /home/wangchengjun/huangziyi/data/RRSISD

echo "========== ALL SMOKE PASSED =========="
echo "Artifacts: ${SMOKE_ROOT}"
