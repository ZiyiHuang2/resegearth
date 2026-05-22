#!/usr/bin/env bash
# 7w = 70000 steps：Decoder Cross-Attention Token-Level Bias（last3）全链路
# train → best checkpoint → merge（含 decoder_attn_bias / rank config）→ config 检查
# → eval normal → eval bypass → metrics(normal) → metrics(bypass)
#
# 支持 A/B/C 通过环境变量调用：
# A:
#   DECODER_ATTN_BIAS_MAX_ABS=0.03 USE_DECODER_ATTN_BIAS_RANK_LOSS=False bash run_train_decoder_attn_bias_7w.sh
# B:
#   DECODER_ATTN_BIAS_MAX_ABS=0.05 USE_DECODER_ATTN_BIAS_RANK_LOSS=False bash run_train_decoder_attn_bias_7w.sh
# C:
#   DECODER_ATTN_BIAS_MAX_ABS=0.03 USE_DECODER_ATTN_BIAS_RANK_LOSS=True DECODER_ATTN_BIAS_RANK_LOSS_WEIGHT=0.001 bash run_train_decoder_attn_bias_7w.sh
#
# 用法示例：
#   conda activate reseg
#   GPU_SLOT=localhost:1 GPU_ID=1 bash run_train_decoder_attn_bias_7w.sh
#
# 仅续训已存在 OUTPUT_DIR 时：
#   RESUME_OK=1 bash run_train_decoder_attn_bias_7w.sh
#
# 覆盖已有 merged：
#   OVERWRITE_MERGE=1 bash run_train_decoder_attn_bias_7w.sh
#
# 仅对已 merge 的模型做 eval + metrics（跳过 train / pick / merge）：
#   EVAL_ONLY_FROM_MERGED=1 bash run_train_decoder_attn_bias_C_rank_w0001_m01_7w.sh
#
# 当前文件已改成 C 实验默认配置：
#   max_abs=0.03
#   apply_layers=last3
#   use_decoder_attn_bias_rank_loss=True
#   rank_loss_weight=0.001
#   rank_margin=0.1
#   max_steps=70000
#

set -euo pipefail

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="${GPU_SLOT:-localhost:2}"
GPU_ID="${GPU_ID:-2}"
MASTER_PORT="${MASTER_PORT:-29543}"

REPO_DIR="${REPO_DIR:-/home/wangchengjun/huangziyi/reseg/resegearth+tgi}"
RESEG_ROOT="${RESEG_ROOT:-/home/wangchengjun/huangziyi/reseg}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MAX_STEPS="${MAX_STEPS:-70000}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"

OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/tgi/decoder_attn_bias_last3_max003_7w_C_rank_w0001_m01}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}/merged_best}"
TEST_NORMAL_DIR="${TEST_NORMAL_DIR:-${OUTPUT_DIR}/eval_best/test_results_normal}"
TEST_BYPASS_DIR="${TEST_BYPASS_DIR:-${OUTPUT_DIR}/eval_best/test_results_bypass}"
TRAIN_LOG="${TRAIN_LOG:-${OUTPUT_DIR}/train_decoder_attn_bias_C_rank_w0001_m01_7w.log}"
BEST_INFO_JSON="${BEST_INFO_JSON:-${OUTPUT_DIR}/best_checkpoint_info.json}"

export WANDB_PROJECT="${WANDB_PROJECT:-segearth-tgi}"
export WANDB_NAME="${WANDB_NAME:-decoder_attn_bias_last3_max003_7w_C_rank_w0001_m01}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${RESEG_ROOT}/output/standard-base-siglip11/merged_model}"
VISION_TOWER="${VISION_TOWER:-${RESEG_ROOT}/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-${RESEG_ROOT}/pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-scripts/zero1.json}"

BASE_DATA_PATH="${BASE_DATA_PATH:-/home/wangchengjun/huangziyi/data/RRSISD}"
DATASET_NAME="${DATASET_NAME:-rrsisd}"
TEST_SPLIT="${TEST_SPLIT:-test}"

REPORT_TO="${REPORT_TO:-wandb}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

BF16="${BF16:-False}"
FP16="${FP16:-True}"
TF32="${TF32:-False}"

MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

DATA_RATIO="${DATA_RATIO:-1}"
SWITCH_BS="${SWITCH_BS:-4}"

# 当前 A/B/C 诊断建议：
# A/C 默认 0.03，B 外部覆盖 0.05。
DECODER_ATTN_BIAS_MAX_ABS="${DECODER_ATTN_BIAS_MAX_ABS:-0.03}"
USE_DECODER_ATTN_BIAS_RANK_LOSS="${USE_DECODER_ATTN_BIAS_RANK_LOSS:-True}"
DECODER_ATTN_BIAS_RANK_LOSS_WEIGHT="${DECODER_ATTN_BIAS_RANK_LOSS_WEIGHT:-0.001}"
DECODER_ATTN_BIAS_RANK_MARGIN="${DECODER_ATTN_BIAS_RANK_MARGIN:-0.1}"

EVAL_METRICS_SCRIPT="${EVAL_METRICS_SCRIPT:-${RESEG_ROOT}/eval_val_metrics.py}"
EVAL_USE_WANDB="${EVAL_USE_WANDB:-True}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-tgi-val}"
EVAL_METRICS_RUN_NAME="${EVAL_METRICS_RUN_NAME:-decoder_attn_bias_last3_max003_7w_C_rank_w0001_m01}"

MERGE_PY="${REPO_DIR}/segearth_r2/train/merge_lora_weights_and_save_hf_model.py"
EVAL_PY="${REPO_DIR}/segearth_r2/eval/eval.py"

########################################
# Pick best checkpoint
# 输出格式：
# OK:metric:/path/checkpoint-xxx|metric_name=value
# OK:root_state:/path/checkpoint-xxx|best_model_checkpoint
# WARN:fallback_max_step:/path/checkpoint-xxx|step=xxx
########################################
pick_best_checkpoint () {
  export _CK_ROOT="${OUTPUT_DIR}"
  export _CK_REPO="${REPO_DIR}"
  export _CK_SEL="${SELECTED_CHECKPOINT:-}"
  export _CK_METRIC_NAME="${BEST_METRIC_NAME:-}"
  export _CK_METRIC_MODE="${BEST_METRIC_MODE:-max}"

  python - <<'PY'
import glob, json, os, re, sys

def step_of(path: str) -> int:
    m = re.search(r"checkpoint-(\d+)$", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1

def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def resolve_path(raw: str, out_root: str, repo_root: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if os.path.isdir(raw):
        return os.path.abspath(raw)
    c1 = os.path.normpath(os.path.join(out_root, raw))
    if os.path.isdir(c1):
        return os.path.abspath(c1)
    c2 = os.path.normpath(os.path.join(repo_root, raw))
    if os.path.isdir(c2):
        return os.path.abspath(c2)
    return ""

def latest_log_metric_from_trainer_state(ts_path, user_metric=""):
    data = read_json(ts_path)
    if not isinstance(data, dict):
        return None
    hist = data.get("log_history")
    if not isinstance(hist, list):
        return None

    chain = [
        "eval_score",
        "eval_giou", "eval_gIoU", "eval_mIoU",
        "gIoU", "giou", "mIoU",
        "eval_ciou", "eval_cIoU", "eval_oIoU",
        "cIoU", "oIoU",
        "eval_mDice", "mDice",
    ]
    if user_metric:
        chain = [user_metric]

    for row in reversed(hist):
        if not isinstance(row, dict):
            continue
        for k in chain:
            if k in row and isinstance(row[k], (int, float)):
                return k, float(row[k])
    return None

def metrics_from_dir(ckpt_dir, user_metric=""):
    chain = [
        "eval_score",
        "eval_giou", "eval_gIoU", "eval_mIoU",
        "gIoU", "giou", "mIoU",
        "eval_ciou", "eval_cIoU", "eval_oIoU",
        "cIoU", "oIoU",
        "eval_mDice", "mDice",
    ]
    if user_metric:
        chain = [user_metric]

    for fname in ("eval_results.json", "metrics.json", "all_results.json", "eval_metrics.json"):
        p = os.path.join(ckpt_dir, fname)
        data = read_json(p)
        if isinstance(data, dict):
            for k in chain:
                if k in data and isinstance(data[k], (int, float)):
                    return k, float(data[k])

    ts_path = os.path.join(ckpt_dir, "trainer_state.json")
    got = latest_log_metric_from_trainer_state(ts_path, user_metric)
    if got:
        return got
    return None

out_root = os.path.abspath(os.environ["_CK_ROOT"])
repo_root = os.path.abspath(os.environ["_CK_REPO"])
sel = os.environ.get("_CK_SEL", "").strip()
mode = os.environ.get("_CK_METRIC_MODE", "max").strip().lower()
user_metric = os.environ.get("_CK_METRIC_NAME", "").strip()

if sel:
    r = resolve_path(sel, out_root, repo_root)
    if not r:
        print(f"ERROR:user_not_dir:{sel}", file=sys.stderr)
        sys.exit(2)
    print(f"OK:user:{r}|selected_checkpoint")
    sys.exit(0)

# 1. root trainer_state best_model_checkpoint
root_ts = os.path.join(out_root, "trainer_state.json")
data = read_json(root_ts)
if isinstance(data, dict):
    b = data.get("best_model_checkpoint")
    if isinstance(b, str) and b.strip():
        r = resolve_path(b, out_root, repo_root)
        if r and os.path.isdir(r):
            metric = latest_log_metric_from_trainer_state(root_ts, user_metric)
            if metric:
                print(f"OK:root_state:{r}|{metric[0]}={metric[1]}")
            else:
                print(f"OK:root_state:{r}|best_model_checkpoint")
            sys.exit(0)

# 2. sub checkpoint trainer_state best_model_checkpoint
ts_paths = sorted(
    glob.glob(os.path.join(out_root, "checkpoint-*", "trainer_state.json")),
    key=lambda p: step_of(os.path.dirname(p)),
    reverse=True,
)
for ts_path in ts_paths:
    data = read_json(ts_path)
    if not isinstance(data, dict):
        continue
    b = data.get("best_model_checkpoint")
    if not isinstance(b, str) or not b.strip():
        continue
    r = resolve_path(b, out_root, repo_root)
    if r and os.path.isdir(r):
        metric = latest_log_metric_from_trainer_state(ts_path, user_metric)
        if metric:
            print(f"OK:subdir_state:{r}|{metric[0]}={metric[1]}")
        else:
            print(f"OK:subdir_state:{r}|best_model_checkpoint")
        sys.exit(0)

# 3. checkpoint metrics
candidates = sorted(
    [p for p in glob.glob(os.path.join(out_root, "checkpoint-*")) if os.path.isdir(p)],
    key=step_of,
    reverse=True,
)
scores = []
for ck in candidates:
    metric = metrics_from_dir(ck, user_metric)
    if metric:
        k, v = metric
        scores.append((ck, k, v))

if scores:
    print("[metric candidates]", file=sys.stderr)
    for ck, k, v in sorted(scores, key=lambda x: step_of(x[0]), reverse=True):
        print(f"  {ck}  {k}={v}", file=sys.stderr)
    scores.sort(key=lambda x: x[2], reverse=(mode != "min"))
    ck, k, v = scores[0]
    print(f"OK:metric:{ck}|{k}={v}")
    sys.exit(0)

# 4. fallback max step
if not candidates:
    print("ERROR:no_checkpoint_dirs", file=sys.stderr)
    sys.exit(3)

latest = max(candidates, key=step_of)
print(f"WARN:fallback_max_step:{latest}|step={step_of(latest)}")
sys.exit(0)
PY
}

merge_lora_dac () {
  local ckpt="$1"
  local save_dir="$2"

  rm -rf "${save_dir}"
  mkdir -p "${save_dir}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${MERGE_PY}" \
    --model_path "${ckpt}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --save_path "${save_dir}" \
    --use_mstva False \
    --use_mstva_loss False \
    --use_text_film False \
    --use_decoder_attn_bias True \
    --decoder_attn_bias_dim 128 \
    --decoder_attn_bias_init_std 1e-3 \
    --decoder_attn_bias_max_abs "${DECODER_ATTN_BIAS_MAX_ABS}" \
    --decoder_attn_bias_apply_layers last3 \
    --decoder_attn_bias_eval_mode normal \
    --decoder_attn_bias_force_scale 1.0 \
    --use_decoder_attn_bias_rank_loss "${USE_DECODER_ATTN_BIAS_RANK_LOSS}" \
    --decoder_attn_bias_rank_loss_weight "${DECODER_ATTN_BIAS_RANK_LOSS_WEIGHT}" \
    --decoder_attn_bias_rank_margin "${DECODER_ATTN_BIAS_RANK_MARGIN}" \
    --lora_enable True \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}"
}

run_eval_infer () {
  local model_dir="$1"
  local out_dir="$2"
  local mode="$3"

  mkdir -p "${out_dir}"

  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python "${EVAL_PY}" \
      --base_data_path "${BASE_DATA_PATH}" \
      --vision_tower "${VISION_TOWER}" \
      --vision_tower_mask "${VISION_TOWER_MASK}" \
      --mask_config "${MASK_CONFIG}" \
      --model_path "${model_dir}" \
      --output_dir "${out_dir}" \
      --dataset_name "${DATASET_NAME}" \
      --split "${TEST_SPLIT}" \
      --eval_batch_size 1 \
      --zip_results False \
      --decoder_attn_bias_eval_mode "${mode}" \
      --decoder_attn_bias_force_scale 1.0
}

check_merged_config () {
  local merged_dir="$1"
  if [[ ! -f "${merged_dir}/config.json" ]]; then
    echo "[ERROR] merged config 不存在: ${merged_dir}/config.json"
    exit 1
  fi
  grep -n '"use_decoder_attn_bias"\|"decoder_attn_bias_dim"\|"decoder_attn_bias_init_std"\|"decoder_attn_bias_max_abs"\|"decoder_attn_bias_apply_layers"\|"use_decoder_attn_bias_rank_loss"\|"decoder_attn_bias_rank_loss_weight"\|"decoder_attn_bias_rank_margin"' \
    "${merged_dir}/config.json" || {
      echo "[ERROR] merged config.json missing decoder_attn_bias / rank fields"
      exit 1
    }
  python - <<PY
import json
p = "${merged_dir}/config.json"
c = json.load(open(p, "r", encoding="utf-8"))

assert c.get("use_decoder_attn_bias") is True, c.get("use_decoder_attn_bias")
assert str(c.get("decoder_attn_bias_apply_layers")) == "last3", c.get("decoder_attn_bias_apply_layers")
assert abs(float(c.get("decoder_attn_bias_max_abs")) - float("${DECODER_ATTN_BIAS_MAX_ABS}")) < 1e-12, c.get("decoder_attn_bias_max_abs")

expected_rank = str("${USE_DECODER_ATTN_BIAS_RANK_LOSS}").lower() == "true"
assert bool(c.get("use_decoder_attn_bias_rank_loss")) == expected_rank, c.get("use_decoder_attn_bias_rank_loss")
assert abs(float(c.get("decoder_attn_bias_rank_loss_weight", 0.001)) - float("${DECODER_ATTN_BIAS_RANK_LOSS_WEIGHT}")) < 1e-12, c.get("decoder_attn_bias_rank_loss_weight")
assert abs(float(c.get("decoder_attn_bias_rank_margin", 0.1)) - float("${DECODER_ATTN_BIAS_RANK_MARGIN}")) < 1e-12, c.get("decoder_attn_bias_rank_margin")

print("[PASS] merged decoder_attn_bias config checked:", p)
PY
}

run_eval_metrics () {
  local pred_dir="$1"
  local run_name="$2"
  local metrics_tag="$3"

  if [[ ! -f "${EVAL_METRICS_SCRIPT}" ]]; then
    echo "[WARN] eval metrics script not found: ${EVAL_METRICS_SCRIPT} — skip metrics."
    return 0
  fi

  USE_WANDB="${EVAL_USE_WANDB}" \
    WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
    WANDB_RUN_NAME="${run_name}" \
    DATASET_TYPE="${DATASET_NAME}" \
    BASE_DATA_PATH="${BASE_DATA_PATH}" \
    SPLIT="${TEST_SPLIT}" \
    PRED_DIR="${pred_dir}" \
    METRICS_TAG="${metrics_tag}" \
    python "${EVAL_METRICS_SCRIPT}"
}

########################################
# Preflight
########################################
echo "========================================"
echo "[CONFIG] decoder_attn_bias 7w (${MAX_STEPS} steps)"
echo "  REPO_DIR=${REPO_DIR}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  MERGED_DIR=${MERGED_DIR}"
echo "  TEST_NORMAL_DIR=${TEST_NORMAL_DIR}"
echo "  TEST_BYPASS_DIR=${TEST_BYPASS_DIR}"
echo "  GPU_SLOT=${GPU_SLOT}  GPU_ID=${GPU_ID}  MASTER_PORT=${MASTER_PORT}"
echo "  MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "  DECODER_ATTN_BIAS_MAX_ABS=${DECODER_ATTN_BIAS_MAX_ABS}"
echo "  USE_DECODER_ATTN_BIAS_RANK_LOSS=${USE_DECODER_ATTN_BIAS_RANK_LOSS}"
echo "  DECODER_ATTN_BIAS_RANK_LOSS_WEIGHT=${DECODER_ATTN_BIAS_RANK_LOSS_WEIGHT}"
echo "  DECODER_ATTN_BIAS_RANK_MARGIN=${DECODER_ATTN_BIAS_RANK_MARGIN}"
echo "  EVAL_ONLY_FROM_MERGED=${EVAL_ONLY_FROM_MERGED:-0}"
echo "========================================"

if ! python -c "import transformers" 2>/dev/null; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda activate reseg 2>/dev/null || true
  fi
fi

if ! python -c "import transformers" 2>/dev/null; then
  echo "[ERROR] Python cannot import transformers. 请先 conda activate reseg."
  exit 1
fi

if ! python segearth_r2/train/train.py --help 2>&1 | grep -q "use_decoder_attn_bias"; then
  echo "[ERROR] train.py --help 未发现 use_decoder_attn_bias。"
  exit 1
fi

if ! python segearth_r2/eval/eval.py --help 2>&1 | grep -q "decoder_attn_bias_eval_mode"; then
  echo "[ERROR] eval.py --help 未发现 decoder_attn_bias_eval_mode。"
  exit 1
fi

if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "[ERROR] MODEL_NAME_OR_PATH 不存在: ${MODEL_NAME_OR_PATH}"
  exit 1
fi

if [[ "${EVAL_ONLY_FROM_MERGED:-0}" != "1" ]]; then
  if [[ -d "${OUTPUT_DIR}" ]]; then
    shopt -s nullglob
    _existing=( "${OUTPUT_DIR}"/checkpoint-* )
    shopt -u nullglob
    if [[ ${#_existing[@]} -gt 0 ]]; then
      if [[ "${RESUME_OK:-0}" != "1" ]]; then
        echo "[ERROR] OUTPUT_DIR 下已有 checkpoint-*，可能误续训旧实验。"
        echo "        请换 OUTPUT_DIR 或显式 RESUME_OK=1 续训。"
        echo "        若只需 eval/metrics：EVAL_ONLY_FROM_MERGED=1 bash $0"
        exit 1
      fi
      echo "[WARN] RESUME_OK=1 — 将在已有 checkpoint 上续训。"
    fi
  fi
fi

mkdir -p "${OUTPUT_DIR}"

if [[ "${EVAL_ONLY_FROM_MERGED:-0}" == "1" ]]; then
  echo "========================================"
  echo "[EVAL_ONLY] 跳过 train / pick / merge，使用已有 MERGED_DIR"
  echo "  MERGED_DIR=${MERGED_DIR}"
  echo "========================================"
  if [[ ! -d "${MERGED_DIR}" ]]; then
    echo "[ERROR] MERGED_DIR 不存在: ${MERGED_DIR}"
    exit 1
  fi
  check_merged_config "${MERGED_DIR}"
else

########################################
# [1/5] Train
########################################
echo "========================================"
echo "[1/5] Training decoder_attn_bias last3, max_steps=${MAX_STEPS}"
echo "  TRAIN_LOG=${TRAIN_LOG}"
echo "========================================"

DAC_RANK_ARGS=()
if [[ "${USE_DECODER_ATTN_BIAS_RANK_LOSS}" == "True" || "${USE_DECODER_ATTN_BIAS_RANK_LOSS}" == "true" ]]; then
  DAC_RANK_ARGS=(
    --use_decoder_attn_bias_rank_loss True
    --decoder_attn_bias_rank_loss_weight "${DECODER_ATTN_BIAS_RANK_LOSS_WEIGHT}"
    --decoder_attn_bias_rank_margin "${DECODER_ATTN_BIAS_RANK_MARGIN}"
  )
fi

deepspeed --master_port="${MASTER_PORT}" --include="${GPU_SLOT}" segearth_r2/train/train.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
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
  --fp16 "${FP16}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --logging_steps "${LOGGING_STEPS}" \
  --tf32 "${TF32}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --deepspeed "${DEEPSPEED_CFG}" \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio "${DATA_RATIO}" \
  --switch_bs "${SWITCH_BS}" \
  --seed "${SEED}" \
  --data_seed "${DATA_SEED}" \
  --report_to "${REPORT_TO}" \
  --use_attention_loss False \
  --use_midstage_gate_loss False \
  --midstage_gate_loss_weight 0.0 \
  --use_mstva False \
  --use_mstva_loss False \
  --use_text_film False \
  --use_decoder_attn_bias True \
  --decoder_attn_bias_dim 128 \
  --decoder_attn_bias_init_std 1e-3 \
  --decoder_attn_bias_max_abs "${DECODER_ATTN_BIAS_MAX_ABS}" \
  --decoder_attn_bias_apply_layers last3 \
  "${DAC_RANK_ARGS[@]}" 2>&1 | tee "${TRAIN_LOG}"

########################################
# [2/5] Select best checkpoint
########################################
echo "========================================"
echo "[2/5] Select best checkpoint"
echo "========================================"

_CK_PICK_OUT="$(mktemp)"
set +e
pick_best_checkpoint > "${_CK_PICK_OUT}"
_CK_RC=$?
set -e

PICK="$(head -n 1 "${_CK_PICK_OUT}" | tr -d '\r')"
rm -f "${_CK_PICK_OUT}"

if [[ "${_CK_RC}" -eq 2 ]]; then
  echo "[ERROR] SELECTED_CHECKPOINT 已设置但不是目录。"
  exit 1
fi

if [[ "${_CK_RC}" -eq 3 ]]; then
  echo "[ERROR] OUTPUT_DIR 下无 checkpoint-*。"
  exit 1
fi

BEST_CHECKPOINT=""
BEST_SELECTION_STRATEGY=""
BEST_METRIC_NAME=""
BEST_METRIC_VALUE=""
BEST_CHECKPOINT_VALUE=""

if echo "${PICK}" | grep -q '^WARN:fallback_max_step:'; then
  _LINE="${PICK#WARN:fallback_max_step:}"
  BEST_CHECKPOINT="${_LINE%%|*}"
  BEST_SELECTION_STRATEGY="fallback_max_step"
  BEST_METRIC_NAME="step"
  BEST_METRIC_VALUE="${_LINE#*|step=}"
  BEST_CHECKPOINT_VALUE="${BEST_METRIC_VALUE}"

elif echo "${PICK}" | grep -q '^OK:user:'; then
  _LINE="${PICK#OK:user:}"
  BEST_CHECKPOINT="${_LINE%%|*}"
  BEST_SELECTION_STRATEGY="user_SELECTED_CHECKPOINT"
  BEST_METRIC_NAME="selected_checkpoint"
  BEST_METRIC_VALUE=""
  BEST_CHECKPOINT_VALUE=""

elif echo "${PICK}" | grep -q '^OK:root_state:'; then
  _LINE="${PICK#OK:root_state:}"
  BEST_CHECKPOINT="${_LINE%%|*}"
  BEST_SELECTION_STRATEGY="root_trainer_state"
  _META="${_LINE#*|}"
  if echo "${_META}" | grep -q '='; then
    BEST_METRIC_NAME="${_META%%=*}"
    BEST_METRIC_VALUE="${_META#*=}"
    BEST_CHECKPOINT_VALUE="${BEST_METRIC_VALUE}"
  else
    BEST_METRIC_NAME="${_META}"
    BEST_METRIC_VALUE=""
    BEST_CHECKPOINT_VALUE=""
  fi

elif echo "${PICK}" | grep -q '^OK:subdir_state:'; then
  _LINE="${PICK#OK:subdir_state:}"
  BEST_CHECKPOINT="${_LINE%%|*}"
  BEST_SELECTION_STRATEGY="subdir_trainer_state"
  _META="${_LINE#*|}"
  if echo "${_META}" | grep -q '='; then
    BEST_METRIC_NAME="${_META%%=*}"
    BEST_METRIC_VALUE="${_META#*=}"
    BEST_CHECKPOINT_VALUE="${BEST_METRIC_VALUE}"
  else
    BEST_METRIC_NAME="${_META}"
    BEST_METRIC_VALUE=""
    BEST_CHECKPOINT_VALUE=""
  fi

elif echo "${PICK}" | grep -q '^OK:metric:'; then
  _LINE="${PICK#OK:metric:}"
  BEST_CHECKPOINT="${_LINE%%|*}"
  BEST_SELECTION_STRATEGY="metric"
  _META="${_LINE#*|}"
  BEST_METRIC_NAME="${_META%%=*}"
  BEST_METRIC_VALUE="${_META#*=}"
  BEST_CHECKPOINT_VALUE="${BEST_METRIC_VALUE}"

else
  echo "[ERROR] checkpoint 选择器异常输出: ${PICK}"
  exit 1
fi

if [[ ! -d "${BEST_CHECKPOINT}" ]]; then
  echo "[ERROR] BEST_CHECKPOINT 不是目录: ${BEST_CHECKPOINT}"
  exit 1
fi

cat > "${BEST_INFO_JSON}" <<EOF
{
  "best_checkpoint": "${BEST_CHECKPOINT}",
  "best_selection_strategy": "${BEST_SELECTION_STRATEGY}",
  "best_metric_name": "${BEST_METRIC_NAME}",
  "best_metric_value": "${BEST_METRIC_VALUE}",
  "best_checkpoint_value": "${BEST_CHECKPOINT_VALUE}",
  "raw_pick": "${PICK}"
}
EOF

echo "[OK] BEST_CHECKPOINT=${BEST_CHECKPOINT}"
echo "[OK] BEST_SELECTION_STRATEGY=${BEST_SELECTION_STRATEGY}"
echo "[OK] BEST_METRIC_NAME=${BEST_METRIC_NAME}"
echo "[OK] BEST_METRIC_VALUE=${BEST_METRIC_VALUE}"
echo "[OK] BEST_CHECKPOINT_VALUE=${BEST_CHECKPOINT_VALUE}"
echo "[OK] BEST_INFO_JSON=${BEST_INFO_JSON}"

########################################
# [3/5] Merge LoRA
########################################
echo "========================================"
echo "[3/5] Merge LoRA → ${MERGED_DIR}"
echo "========================================"

if [[ ! -f "${MERGE_PY}" ]]; then
  echo "[ERROR] 未找到 merge 脚本: ${MERGE_PY}"
  exit 1
fi

if [[ -e "${MERGED_DIR}" ]]; then
  if [[ "${OVERWRITE_MERGE:-0}" != "1" ]]; then
    echo "[ERROR] MERGED_DIR 已存在: ${MERGED_DIR} — 设置 OVERWRITE_MERGE=1 可覆盖。"
    exit 1
  fi
  rm -rf "${MERGED_DIR}"
fi

merge_lora_dac "${BEST_CHECKPOINT}" "${MERGED_DIR}"

########################################
# [CHECK] merged config
########################################
echo "========================================"
echo "[CHECK] merged config decoder_attn_bias fields"
echo "========================================"

check_merged_config "${MERGED_DIR}"

fi  # end EVAL_ONLY_FROM_MERGED else (train → merge)

########################################
# [4/5] Eval normal + bypass
########################################
echo "========================================"
echo "[4/5] Eval inference split=${TEST_SPLIT}: normal + bypass"
echo "========================================"

echo "[INFO] Eval normal"
run_eval_infer "${MERGED_DIR}" "${TEST_NORMAL_DIR}" "normal"

echo "[INFO] Eval bypass"
run_eval_infer "${MERGED_DIR}" "${TEST_BYPASS_DIR}" "bypass"

########################################
# [5/5] Metrics normal + bypass
########################################
echo "========================================"
echo "[5/5] Metrics normal + bypass"
echo "========================================"

echo "[INFO] Metrics normal"
run_eval_metrics "${TEST_NORMAL_DIR}" "${EVAL_METRICS_RUN_NAME}_normal" "normal"

echo "[INFO] Metrics bypass"
run_eval_metrics "${TEST_BYPASS_DIR}" "${EVAL_METRICS_RUN_NAME}_bypass" "bypass"

echo "========================================"
if [[ "${EVAL_ONLY_FROM_MERGED:-0}" == "1" ]]; then
  echo "[DONE] eval-only pipeline finished."
else
  echo "[DONE] 7w pipeline finished."
  echo "  BEST_CHECKPOINT=${BEST_CHECKPOINT:-}"
  echo "  BEST_SELECTION_STRATEGY=${BEST_SELECTION_STRATEGY:-}"
  echo "  BEST_METRIC_NAME=${BEST_METRIC_NAME:-}"
  echo "  BEST_METRIC_VALUE=${BEST_METRIC_VALUE:-}"
  echo "  BEST_CHECKPOINT_VALUE=${BEST_CHECKPOINT_VALUE:-}"
  echo "  BEST_INFO_JSON=${BEST_INFO_JSON:-}"
fi
echo "  MERGED_DIR=${MERGED_DIR}"
echo "  TEST_NORMAL_DIR=${TEST_NORMAL_DIR}"
echo "  TEST_BYPASS_DIR=${TEST_BYPASS_DIR}"
echo "========================================"