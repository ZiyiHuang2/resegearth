#!/usr/bin/env bash
# 100 step smoke：decoder_attn_bias + ranking loss（max_abs=0.03 诊断档）
# 训练结束自动断言 checkpoint config / trainer_state / 日志。
#
# 用法：
#   conda activate reseg
#   GPU_SLOT=localhost:1 GPU_ID=1 bash run_smoke_decoder_attn_bias_rank_100.sh
#
# 换输出目录：OUTPUT_DIR=/path/to/smoke bash run_smoke_decoder_attn_bias_rank_100.sh
set -euo pipefail

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="${GPU_SLOT:-localhost:1}"
GPU_ID="${GPU_ID:-1}"
MASTER_PORT="${MASTER_PORT:-29543}"

REPO_DIR="${REPO_DIR:-/home/wangchengjun/huangziyi/reseg/resegearth+tgi}"
RESEG_ROOT="${RESEG_ROOT:-/home/wangchengjun/huangziyi/reseg}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/tgi/smoke_decoder_attn_bias_rank_100}"
REPORT_TO="${REPORT_TO:-none}"
SMOKE_LOG="${SMOKE_LOG:-${OUTPUT_DIR}/smoke_rank_train.log}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${RESEG_ROOT}/output/standard-base-siglip11/merged_model}"
VISION_TOWER="${VISION_TOWER:-${RESEG_ROOT}/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-${RESEG_ROOT}/pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-scripts/zero1.json}"
BASE_DATA_PATH="${BASE_DATA_PATH:-/home/wangchengjun/huangziyi/data/RRSISD}"
DATASET_NAME="${DATASET_NAME:-rrsisd}"

mkdir -p "${OUTPUT_DIR}"

echo "[smoke] OUTPUT_DIR=${OUTPUT_DIR} MAX_STEPS=${MAX_STEPS} SAVE_STEPS=${SAVE_STEPS} REPORT_TO=${REPORT_TO}"
echo "[smoke] log -> ${SMOKE_LOG}"

deepspeed --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 1 \
  --logging_steps 10 \
  --bf16 False \
  --fp16 True \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --model_max_length 2048 \
  --dataloader_num_workers 2 \
  --max_grad_norm 1.0 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --deepspeed "${DEEPSPEED_CFG}" \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio 1 \
  --switch_bs 4 \
  --seed 42 \
  --data_seed 42 \
  --report_to "${REPORT_TO}" \
  --use_attention_loss False \
  --use_midstage_gate_loss False \
  --use_mstva False \
  --use_mstva_loss False \
  --use_text_film False \
  --use_decoder_attn_bias True \
  --decoder_attn_bias_dim 128 \
  --decoder_attn_bias_init_std 1e-3 \
  --decoder_attn_bias_max_abs 0.03 \
  --decoder_attn_bias_apply_layers last3 \
  --use_decoder_attn_bias_rank_loss True \
  --decoder_attn_bias_rank_margin 0.1 \
  --decoder_attn_bias_rank_loss_weight 0.001 \
  2>&1 | tee "${SMOKE_LOG}"

echo "[smoke] running post checks..."

# 禁止训练期 all-accessible fallback / shape mismatch / FORBIDDEN
if grep -qF "Use all-accessible fallback only in eval." "${SMOKE_LOG}"; then
  echo "[smoke][FAIL] found eval-only all-accessible fallback in training log."
  exit 1
fi
if grep -qiE "FORBIDDEN during training|shape mismatch|\[DecoderAttnBiasRank\].*mismatch" "${SMOKE_LOG}"; then
  echo "[smoke][FAIL] found FORBIDDEN/shape mismatch in training log."
  exit 1
fi
if grep -qiE "\bnan\b|\binf\b" "${SMOKE_LOG}"; then
  echo "[smoke][FAIL] found nan/inf in training log."
  exit 1
fi

# stdout 元信息（last3 → 6,7,8）
if ! grep -qF "[DecoderAttnBiasRank] rank_num_layers=3" "${SMOKE_LOG}"; then
  echo "[smoke][FAIL] missing rank_num_layers=3 log line."
  exit 1
fi
if ! grep -qE "layer_indices=6,7,8" "${SMOKE_LOG}"; then
  echo "[smoke][FAIL] missing layer_indices=6,7,8 (if decoder depth changed, update smoke assert)."
  exit 1
fi

export OUTPUT_DIR MAX_STEPS SAVE_STEPS
python3 - <<'PYCHK'
import glob
import json
import math
import os
import re

out = os.environ["OUTPUT_DIR"]
max_steps = int(os.environ.get("MAX_STEPS", "100"))
save_steps = int(os.environ.get("SAVE_STEPS", "100"))

def find_latest_checkpoint_config(root: str) -> str:
    preferred = os.path.join(root, f"checkpoint-{max_steps}", "config.json")
    if os.path.isfile(preferred):
        return preferred
    preferred2 = os.path.join(root, f"checkpoint-{save_steps}", "config.json")
    if os.path.isfile(preferred2):
        return preferred2
    cands = sorted(glob.glob(os.path.join(root, "checkpoint-*", "config.json")))
    if not cands:
        return ""
    def step_of(p: str) -> int:
        m = re.search(r"checkpoint-(\d+)", p)
        return int(m.group(1)) if m else -1
    cands.sort(key=step_of)
    return cands[-1]

cfg_path = find_latest_checkpoint_config(out)
assert cfg_path and os.path.isfile(cfg_path), f"no checkpoint-*/config.json under {out}"
print(f"[smoke] config_check_path={cfg_path}")

c = json.load(open(cfg_path, "r", encoding="utf-8"))
w = float(c.get("decoder_attn_bias_rank_loss_weight", -1.0))
assert abs(w - 0.001) < 1e-12, c.get("decoder_attn_bias_rank_loss_weight")
ma = float(c.get("decoder_attn_bias_max_abs", -1.0))
assert abs(ma - 0.03) < 1e-12, c.get("decoder_attn_bias_max_abs")

ts_path = os.path.join(out, "trainer_state.json")
assert os.path.isfile(ts_path), f"missing trainer_state.json: {ts_path}"
ts = json.load(open(ts_path, "r", encoding="utf-8"))
rows = [r for r in ts.get("log_history", []) if isinstance(r, dict)]

required_numeric = [
    "loss_decoder_attn_bias_rank",
    "decoder_attn_bias_rank_loss_raw",
    "decoder_attn_bias_inside_mean",
    "decoder_attn_bias_outside_mean",
    "decoder_attn_bias_inside_outside_gap",
    "decoder_attn_bias_rank_valid_count",
    "decoder_attn_bias_rank_num_layers",
    "decoder_attn_bias_rank_fg_access_ratio",
    "decoder_attn_bias_rank_bg_access_ratio",
    "decoder_attn_bias_abs_mean",
    "loss_mask",
    "loss_dice",
]

def last_row_with_key(key: str):
    hits = [r for r in rows if key in r and r[key] is not None]
    return hits[-1] if hits else None

for key in required_numeric:
    row = last_row_with_key(key)
    assert row is not None, f"missing {key} in trainer_state log_history"
    val = row[key]
    assert isinstance(val, (int, float)) and math.isfinite(float(val)), f"non-finite {key}: {val!r}"

last_rank = last_row_with_key("loss_decoder_attn_bias_rank")
assert last_rank is not None
assert abs(float(last_rank["loss_decoder_attn_bias_rank"])) < 50.0

raw_row = last_row_with_key("decoder_attn_bias_rank_loss_raw")
assert raw_row is not None and float(raw_row["decoder_attn_bias_rank_loss_raw"]) >= 0.0

vc_row = last_row_with_key("decoder_attn_bias_rank_valid_count")
assert float(vc_row["decoder_attn_bias_rank_valid_count"]) > 0.0

nl_row = last_row_with_key("decoder_attn_bias_rank_num_layers")
assert abs(float(nl_row["decoder_attn_bias_rank_num_layers"]) - 3.0) < 1e-6

idx_row = last_row_with_key("decoder_attn_bias_rank_layer_indices")
assert idx_row is not None
idx_val = idx_row["decoder_attn_bias_rank_layer_indices"]
assert str(idx_val).replace(" ", "") == "6,7,8", f"layer_indices expected 6,7,8 got {idx_val!r}"

print("[smoke][OK] checkpoint config weight=0.001 max_abs=0.03; trainer_state rank metrics complete.")
PYCHK

echo "[smoke] all checks passed."
