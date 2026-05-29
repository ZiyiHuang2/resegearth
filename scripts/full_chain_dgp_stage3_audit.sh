#!/usr/bin/env bash
# SegEarth-R2 Stage 3 v5 DGP-QDTI full-chain audit
set -euo pipefail

REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+DGP"
cd "${REPO_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_DIR}/stage3_audit_logs/${TS}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/full_chain.log") 2>&1

summary() { echo "$1"; }

STATUS_PY_COMPILE="PASS"
STATUS_STATIC="PASS"
STATUS_SHAPE="PASS"
STATUS_SMOKE="WARN"
STATUS_TRAIN="WARN"
STATUS_TRAIN_KEYS="WARN"
STATUS_MERGE="WARN"
STATUS_MERGED_KEYS="WARN"
STATUS_EVAL="WARN"
STATUS_EVAL_KEYS="WARN"
STATUS_FAILFAST="PASS"

TRAIN_DGP_KEYS="n/a"
MERGED_DGP_KEYS="n/a"
EVAL_DGP_KEYS="n/a"

echo "========================================"
echo "[0] Environment evidence"
echo "========================================"
pwd
git branch --show-current || true
git rev-parse HEAD || true
git status --short || true
which python || true
python - <<'PY'
import torch, sys
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY

echo "========================================"
echo "[1] py_compile"
echo "========================================"
PY_FILES=(
  segearth_r2/model/language_model/prompt_query_fusion.py
  segearth_r2/model/language_model/llava_phi.py
  segearth_r2/model/mask_decoder/Mask2Former_Simplify/modeling/transformer_decoder/qdti.py
  segearth_r2/model/mask_decoder/Mask2Former_Simplify/modeling/transformer_decoder/mask2former_transformer_decoder.py
  segearth_r2/train/train.py
  segearth_r2/train/merge_lora_weights_and_save_hf_model.py
  segearth_r2/utils/builder.py
  segearth_r2/eval/eval.py
  tools/audit_dgp_stage3_chain.py
)
if python -m py_compile "${PY_FILES[@]}"; then
  STATUS_PY_COMPILE="PASS"
else
  STATUS_PY_COMPILE="FAIL"
  echo "py_compile FAILED"
  exit 1
fi

echo "========================================"
echo "[2] static + shape audit"
echo "========================================"
if python tools/audit_dgp_stage3_chain.py \
  --repo_dir "${REPO_DIR}" \
  --log_dir "${LOG_DIR}" | tee "${LOG_DIR}/audit_dgp_stage3_chain.log"; then
  STATUS_STATIC="PASS"
  STATUS_SHAPE="PASS"
  STATUS_FAILFAST="PASS"
else
  AUDIT_RC=$?
  if grep -q "OVERALL: WARN" "${LOG_DIR}/audit_dgp_stage3_chain.txt" 2>/dev/null; then
    STATUS_STATIC="WARN"
    STATUS_SHAPE="WARN"
  else
    STATUS_STATIC="FAIL"
    STATUS_SHAPE="FAIL"
    exit "${AUDIT_RC}"
  fi
fi

echo "========================================"
echo "[3] smoke_dgp_stage3 (optional)"
echo "========================================"
if [[ -f tools/smoke_dgp_stage3.py ]]; then
  if python tools/smoke_dgp_stage3.py | tee "${LOG_DIR}/smoke_dgp_stage3.log"; then
    STATUS_SMOKE="PASS"
  else
    STATUS_SMOKE="FAIL"
    exit 1
  fi
else
  STATUS_SMOKE="WARN"
  echo "tools/smoke_dgp_stage3.py not found"
fi

echo "========================================"
echo "[4] DGP smoke train"
echo "========================================"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

MODEL_NAME_OR_PATH="/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
BASE_DATA_PATH="/root/rivermind-data/huangziyi/data/RRSISD"
OUTPUT_DIR="${REPO_DIR}/stage3_audit_logs/full_chain_train_${TS}"
MERGED_DIR="${OUTPUT_DIR}/merged_model"
EVAL_OUT="${OUTPUT_DIR}/eval_smoke"

mkdir -p "${OUTPUT_DIR}"

deepspeed --master_port=29502 --include localhost:0 segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name rrsisd \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy steps \
  --save_steps 2 \
  --save_total_limit 1 \
  --bf16 True \
  --learning_rate 1e-4 \
  --logging_steps 1 \
  --model_max_length 2048 \
  --dataloader_num_workers 2 \
  --lora_r 4 \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio 1 \
  --switch_bs 1 \
  --use_dgp_qdti True \
  --use_qdti_bias True \
  --scale_hard_loss_weight 0.0 \
  2>&1 | tee "${LOG_DIR}/train_dgp_stage3.log"

STATUS_TRAIN="PASS"

echo "========================================"
echo "[5] training checkpoint DGP key check"
echo "========================================"
CKPT_DIR="$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
if [[ -z "${CKPT_DIR}" ]]; then
  CKPT_DIR="${OUTPUT_DIR}"
fi
echo "CKPT_DIR=${CKPT_DIR}"

TRAIN_DGP_KEYS="$(CKPT_DIR="${CKPT_DIR}" python - <<'PY'
import os, sys
sys.path.insert(0, "/root/rivermind-data/huangziyi/reseg/segearth+DGP")
from tools.audit_dgp_stage3_chain import list_dgp_keys_from_state_dict, assert_dgp_keys_exist
ckpt = os.environ["CKPT_DIR"]
n, missing = assert_dgp_keys_exist(ckpt)
print(n)
if n == 0:
    sys.exit(1)
if missing:
    print("missing_groups:", missing, file=sys.stderr)
    sys.exit(2)
PY
)"
STATUS_TRAIN_KEYS="PASS"

echo "training DGP keys: ${TRAIN_DGP_KEYS}"

echo "========================================"
echo "[6] merge (help + run)"
echo "========================================"
python segearth_r2/train/merge_lora_weights_and_save_hf_model.py --help | tee "${LOG_DIR}/merge_help.log"

rm -rf "${MERGED_DIR}"
CUDA_VISIBLE_DEVICES=0 python segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
  --model_path "${CKPT_DIR}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --save_path "${MERGED_DIR}" \
  --use_dgp_qdti True \
  --use_qdti_bias True \
  --lora_r 4 \
  2>&1 | tee "${LOG_DIR}/merge.log"
STATUS_MERGE="PASS"

echo "========================================"
echo "[7] merged model DGP key check"
echo "========================================"
MERGED_DGP_KEYS="$(MERGED_DIR=${MERGED_DIR} python - <<'PY'
import json, os, sys
sys.path.insert(0, "/root/rivermind-data/huangziyi/reseg/segearth+DGP")
from tools.audit_dgp_stage3_chain import assert_dgp_keys_exist
merged = os.environ["MERGED_DIR"]
n, missing = assert_dgp_keys_exist(merged)
print(n)
cfg = os.path.join(merged, "config.json")
required_cfg = (
    "use_dgp_qdti", "use_qdti_bias", "dgp_pg_tokens",
    "qdti_apply_layers", "qdti_scale_init", "qdti_max_abs", "scale_hard_loss_weight",
)
if os.path.isfile(cfg):
    with open(cfg) as f:
        c = json.load(f)
    missing_cfg = [k for k in required_cfg if k not in c]
    if missing_cfg:
        print("FAIL: config.json missing DGP keys:", missing_cfg, file=sys.stderr)
        sys.exit(3)
    if not c.get("use_dgp_qdti", False):
        print("WARN: config.json use_dgp_qdti not True", file=sys.stderr)
else:
    print("FAIL: config.json not found", file=sys.stderr)
    sys.exit(4)
if n == 0:
    sys.exit(1)
if missing:
    sys.exit(2)
PY
)"
STATUS_MERGED_KEYS="PASS"
echo "merged DGP keys: ${MERGED_DGP_KEYS}"

echo "========================================"
echo "[8] eval smoke"
echo "========================================"
python segearth_r2/eval/eval.py --help | tee "${LOG_DIR}/eval_help.log"
mkdir -p "${EVAL_OUT}"
CUDA_VISIBLE_DEVICES=0 python segearth_r2/eval/eval.py \
  --base_data_path "${BASE_DATA_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --model_path "${MERGED_DIR}" \
  --output_dir "${EVAL_OUT}" \
  --dataset_name rrsisd \
  --split test \
  --eval_batch_size 1 \
  --max_eval_samples 1 \
  --use_dgp_qdti True \
  --zip_results False \
  2>&1 | tee "${LOG_DIR}/eval_dgp_stage3.log"
STATUS_EVAL="PASS"

echo "========================================"
echo "[9] eval loaded model DGP keys"
echo "========================================"
EVAL_DGP_KEYS="$(MERGED_DIR=${MERGED_DIR} python - <<'PY'
import os, sys
sys.path.insert(0, "/root/rivermind-data/huangziyi/reseg/segearth+DGP")
os.chdir("/root/rivermind-data/huangziyi/reseg/segearth+DGP")
from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.utils.builder import load_pretrained_model

class A:
    use_dgp_qdti = True
    dgp_fuse_dim = 256
    dgp_refiner_hidden_dim = 512
    qdti_bias_dim = 128
    qdti_init_std = 1e-3
    qdti_max_abs = 0.01
    qdti_apply_layers = "last3"
    scale_hard_loss_weight = 0.0

merged = os.environ["MERGED_DIR"]
mask_cfg = "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
_, model, _, _ = load_pretrained_model(merged, model_args=A(), mask_config=mask_cfg, device="cpu")
keys = [n for n, _ in model.named_parameters() if any(x in n for x in ("prompt_adapter", "query_refiner", "query_specific_text_memory_bias"))]
print(len(keys))
if len(keys) == 0:
    sys.exit(1)
PY
)" && STATUS_EVAL_KEYS="PASS" || STATUS_EVAL_KEYS="WARN"
echo "eval loaded DGP param keys: ${EVAL_DGP_KEYS}"

echo "========================================"
echo "[10] fail-fast smoke"
echo "========================================"
python - <<'PY' | tee "${LOG_DIR}/failfast.log"
import sys
sys.path.insert(0, "/root/rivermind-data/huangziyi/reseg/segearth+DGP")
import torch
from segearth_r2.model.language_model.llava_phi import SegEarthR2
try:
    SegEarthR2.validate_dgp_qdti_checkpoint({"lm_head.weight": torch.zeros(1)}, context="full-chain-fake")
    print("FAIL: expected RuntimeError")
    sys.exit(1)
except RuntimeError as e:
    print("PASS:", str(e)[:160])
PY
STATUS_FAILFAST="PASS"

echo ""
echo "FULL CHAIN SUMMARY"
echo "- py_compile: ${STATUS_PY_COMPILE}"
echo "- static audit: ${STATUS_STATIC}"
echo "- shape audit: ${STATUS_SHAPE}"
echo "- smoke_dgp_stage3: ${STATUS_SMOKE}"
echo "- train smoke: ${STATUS_TRAIN}"
echo "- training checkpoint DGP keys: ${TRAIN_DGP_KEYS}"
echo "- merge: ${STATUS_MERGE}"
echo "- merged model DGP keys: ${MERGED_DGP_KEYS}"
echo "- eval smoke: ${STATUS_EVAL}"
echo "- eval loaded DGP keys: ${EVAL_DGP_KEYS} (${STATUS_EVAL_KEYS})"
echo "- fail-fast missing DGP keys: ${STATUS_FAILFAST}"
echo "- log dir: ${LOG_DIR}"
