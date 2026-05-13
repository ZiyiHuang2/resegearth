#!/usr/bin/env bash
# Decoder Cross-Attention Token-Level Bias — 100 step smoke (train → merge → eval normal → eval bypass).
# 前置：在完整依赖环境执行；按需修改路径与 GPU_SLOT。
set -euo pipefail

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

REPO_DIR="/home/wangchengjun/huangziyi/reseg/resegearth+tgi"
cd "${REPO_DIR}"

# -----------------------------------------------------------------------------
# 0) 路径（与需求一致；可按机器修改）
# -----------------------------------------------------------------------------
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/wangchengjun/huangziyi/reseg/output/bseg/baseline_standard-base_5w/merged_model}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/wangchengjun/huangziyi/reseg/output/tgi/decoder_attn_bias_last3_smoke_100}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}/merged_smoke}"
EVAL_OUT_NORMAL="${EVAL_OUT_NORMAL:-${OUTPUT_DIR}/eval_normal}"
EVAL_OUT_BYPASS="${EVAL_OUT_BYPASS:-${OUTPUT_DIR}/eval_bypass}"
SMOKE_LOG="${SMOKE_LOG:-${OUTPUT_DIR}/smoke_train.log}"

VISION_TOWER="${VISION_TOWER:-/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"
BASE_DATA_PATH="${BASE_DATA_PATH:-/home/wangchengjun/huangziyi/data/RRSISD}"
DATASET_NAME="${DATASET_NAME:-rrsisd}"

GPU_SLOT="${GPU_SLOT:-localhost:0}"
MASTER_PORT="${MASTER_PORT:-29540}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-scripts/zero1.json}"

mkdir -p "${OUTPUT_DIR}"

echo "== [1/9] Preflight: train.py --help (decoder_attn_bias) =="
python segearth_r2/train/train.py --help | grep -E "use_decoder_attn_bias|decoder_attn_bias_dim|decoder_attn_bias_init_std|decoder_attn_bias_max_abs|decoder_attn_bias_apply_layers" || {
  echo "FAIL: train help missing decoder_attn_bias args"
  exit 1
}

echo "== [2/9] Preflight: eval.py --help (decoder_attn_bias) =="
python segearth_r2/eval/eval.py --help | grep -E "decoder_attn_bias_eval_mode|decoder_attn_bias_force_scale" || {
  echo "FAIL: eval help missing decoder_attn_bias eval args"
  exit 1
}

echo "== [3/9] Train 100 steps (deepspeed) → ${SMOKE_LOG} =="
# 注意：argparse type=bool 对字符串不友好；此处用 True/False 与现有 shell 脚本一致。
deepspeed --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps 100 \
  --save_steps 100 \
  --save_total_limit 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --bf16 False \
  --fp16 True \
  --learning_rate 3e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 5 \
  --tf32 False \
  --model_max_length 2048 \
  --gradient_checkpointing False \
  --dataloader_num_workers 4 \
  --lora_r 8 \
  --deepspeed "${DEEPSPEED_CFG}" \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio 1 \
  --switch_bs 4 \
  --seed 42 \
  --data_seed 42 \
  --max_grad_norm 1.0 \
  --use_attention_loss False \
  --use_midstage_gate_loss False \
  --midstage_gate_loss_weight 0.0 \
  --use_mstva False \
  --use_mstva_loss False \
  --use_text_film False \
  --use_decoder_attn_bias True \
  --decoder_attn_bias_dim 128 \
  --decoder_attn_bias_init_std 1e-3 \
  --decoder_attn_bias_max_abs 0.01 \
  --decoder_attn_bias_apply_layers last3 \
  2>&1 | tee "${SMOKE_LOG}"

echo "== [4/9] Grep smoke log (first 200 hits) =="
grep -E "decoder_attn_bias|DecoderAttnBias|attn_bias|text_tokens|text_mask|WARNING|warning|nan|inf|shape|mismatch|loss_mask|loss_dice" "${SMOKE_LOG}" | head -n 200 || true

echo "== [5/9] Select latest checkpoint under ${OUTPUT_DIR} =="
CKPT_DIR="$(ls -td "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | head -1 || true)"
if [[ -z "${CKPT_DIR}" || ! -d "${CKPT_DIR}" ]]; then
  echo "FAIL: no checkpoint-* under ${OUTPUT_DIR}"
  exit 1
fi
echo "Using checkpoint: ${CKPT_DIR}"

echo "== [6/9] Merge LoRA → ${MERGED_DIR} =="
mkdir -p "${MERGED_DIR}"
python segearth_r2/train/merge_lora_weights_and_save_hf_model.py \
  --model_path "${CKPT_DIR}" \
  --save_path "${MERGED_DIR}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --use_mstva False \
  --use_mstva_loss False \
  --use_text_film False \
  --use_decoder_attn_bias True \
  --decoder_attn_bias_dim 128 \
  --decoder_attn_bias_init_std 1e-3 \
  --decoder_attn_bias_max_abs 0.01 \
  --decoder_attn_bias_apply_layers last3 \
  --decoder_attn_bias_eval_mode normal \
  --decoder_attn_bias_force_scale 1.0 \
  --lora_enable True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05

echo "== [7/9] Grep merged config.json =="
CFG="${MERGED_DIR}/config.json"
if [[ ! -f "${CFG}" ]]; then
  echo "FAIL: missing ${CFG}"
  exit 1
fi
grep -n '"use_decoder_attn_bias"\|"decoder_attn_bias_dim"\|"decoder_attn_bias_init_std"\|"decoder_attn_bias_max_abs"\|"decoder_attn_bias_apply_layers"' "${CFG}" || {
  echo "WARN: expected decoder_attn_bias keys not found in config.json (检查 HF 是否序列化自定义字段)"
}

echo "== [8/9] Eval NORMAL (decoder_attn_bias_eval_mode=normal) =="
mkdir -p "${EVAL_OUT_NORMAL}"
python segearth_r2/eval/eval.py \
  --model_path "${MERGED_DIR}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --split val \
  --mask_config "${MASK_CONFIG}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --output_dir "${EVAL_OUT_NORMAL}" \
  --decoder_attn_bias_eval_mode normal \
  --decoder_attn_bias_force_scale 1.0

echo "== [9/9] Eval BYPASS (decoder_attn_bias_eval_mode=bypass) =="
mkdir -p "${EVAL_OUT_BYPASS}"
python segearth_r2/eval/eval.py \
  --model_path "${MERGED_DIR}" \
  --base_data_path "${BASE_DATA_PATH}" \
  --dataset_name "${DATASET_NAME}" \
  --split val \
  --mask_config "${MASK_CONFIG}" \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --output_dir "${EVAL_OUT_BYPASS}" \
  --decoder_attn_bias_eval_mode bypass \
  --decoder_attn_bias_force_scale 1.0

echo "== Smoke 流水线结束。请人工对照 §十四 通过标准检查 ${SMOKE_LOG} 与两次 eval 日志。 =="
