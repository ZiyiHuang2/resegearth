#!/usr/bin/env bash
# 28w = 280000 steps：Decoder Cross-Attention Token-Level Bias（last3）全链路
# train → best checkpoint → merge（含 decoder_attn_bias）→ config 检查
# → eval normal → eval bypass → metrics(normal) → metrics(bypass)
#
# 与 smoke 对齐的核心开关（勿与 MSTVA / TextFiLM / attention loss 混开）：
#   --use_decoder_attn_bias True --decoder_attn_bias_apply_layers last3 ...
#   --use_mstva False --use_mstva_loss False --use_text_film False --use_attention_loss False
#
# 用法示例：
#   conda activate reseg
#   GPU_SLOT=localhost:1 GPU_ID=1 bash run_train_decoder_attn_bias_28w.sh
#
# 仅续训已存在 OUTPUT_DIR 时：RESUME_OK=1 bash run_train_decoder_attn_bias_28w.sh
# 覆盖已有 merged：OVERWRITE_MERGE=1 bash run_train_decoder_attn_bias_28w.sh
set -euo pipefail

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="${GPU_SLOT:-localhost:1}"
GPU_ID="${GPU_ID:-1}"
MASTER_PORT="${MASTER_PORT:-29542}"

REPO_DIR="${REPO_DIR:-/home/wangchengjun/huangziyi/reseg/resegearth+tgi}"
RESEG_ROOT="${RESEG_ROOT:-/home/wangchengjun/huangziyi/reseg}"
cd "${REPO_DIR}"
# 确保优先加载本仓库的 segearth_r2（避免 PYTHONPATH 指向其它旧副本）
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MAX_STEPS="${MAX_STEPS:-280000}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"

OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/tgi/decoder_attn_bias_last3_28w}"
MERGED_DIR="${MERGED_DIR:-${RESEG_ROOT}/output/tgi/decoder_attn_bias_last3_28w/merged_best}"
TEST_NORMAL_DIR="${TEST_NORMAL_DIR:-${RESEG_ROOT}/output/tgi/decoder_attn_bias_last3_28w/eval_best/test_results_normal}"
TEST_BYPASS_DIR="${TEST_BYPASS_DIR:-${RESEG_ROOT}/output/tgi/decoder_attn_bias_last3_28w/eval_best/test_results_bypass}"

export WANDB_PROJECT="${WANDB_PROJECT:-segearth-tgi}"
export WANDB_NAME="${WANDB_NAME:-decoder_attn_bias_last3_28w}"

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
# 与已通过 smoke 一致（fp16）；若改 bf16，请同时自行做短跑验证
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

EVAL_METRICS_SCRIPT="${EVAL_METRICS_SCRIPT:-${RESEG_ROOT}/eval_val_metrics.py}"
EVAL_USE_WANDB="${EVAL_USE_WANDB:-True}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-tgi}"
EVAL_METRICS_RUN_NAME="${EVAL_METRICS_RUN_NAME:-decoder_attn_bias_last3_28w}"

MERGE_PY="${REPO_DIR}/segearth_r2/train/merge_lora_weights_and_save_hf_model.py"
EVAL_PY="${REPO_DIR}/segearth_r2/eval/eval.py"

########################################
# Pick best checkpoint（stdout 首行为控制行）
########################################
pick_best_checkpoint () {
  export _CK_ROOT="${OUTPUT_DIR}"
  export _CK_REPO="${REPO_DIR}"
  export _CK_SEL="${SELECTED_CHECKPOINT:-}"
  export _CK_METRIC_NAME="${BEST_METRIC_NAME:-}"
  export _CK_METRIC_MODE="${BEST_METRIC_MODE:-max}"
  python - <<'PY'
import glob, json, os, re, sys

def step_of(name: str) -> int:
    m = re.search(r"checkpoint-(\d+)$", name)
    return int(m.group(1)) if m else -1

def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def resolve_best_path(raw: str, out_root: str, repo_root: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if os.path.isdir(raw):
        return os.path.abspath(raw)
    cand1 = os.path.normpath(os.path.join(out_root, raw))
    if os.path.isdir(cand1):
        return os.path.abspath(cand1)
    cand2 = os.path.normpath(os.path.join(repo_root, raw))
    if os.path.isdir(cand2):
        return os.path.abspath(cand2)
    return ""

def metrics_from_dir(ckpt_dir):
    for fname in ("eval_results.json", "metrics.json", "all_results.json", "eval_metrics.json"):
        p = os.path.join(ckpt_dir, fname)
        data = read_json(p)
        if isinstance(data, dict):
            return data
    ts = read_json(os.path.join(ckpt_dir, "trainer_state.json"))
    if isinstance(ts, dict) and isinstance(ts.get("log_history"), list):
        for row in reversed(ts["log_history"]):
            if isinstance(row, dict):
                return row
    return None

out_root = os.path.abspath(os.environ["_CK_ROOT"])
repo_root = os.path.abspath(os.environ["_CK_REPO"])
sel = os.environ.get("_CK_SEL", "").strip()
mode = os.environ.get("_CK_METRIC_MODE", "max").strip().lower()
user_metric = os.environ.get("_CK_METRIC_NAME", "").strip()

if sel:
    if not os.path.isdir(sel):
        print(f"ERROR:user_not_dir:{sel}", file=sys.stderr)
        sys.exit(2)
    print(f"OK:user:{os.path.abspath(sel)}")
    sys.exit(0)

ts0 = read_json(os.path.join(out_root, "trainer_state.json"))
if isinstance(ts0, dict):
    b = ts0.get("best_model_checkpoint")
    if isinstance(b, str) and b.strip():
        r = resolve_best_path(b, out_root, repo_root)
        if r and os.path.isdir(r) and (r == out_root or r.startswith(out_root + os.sep)):
            print(f"OK:root_state:{r}")
            sys.exit(0)

for ts_path in sorted(
    glob.glob(os.path.join(out_root, "checkpoint-*", "trainer_state.json")),
    key=lambda p: step_of(os.path.dirname(p)),
    reverse=True,
):
    data = read_json(ts_path)
    if not isinstance(data, dict):
        continue
    b = data.get("best_model_checkpoint")
    if not isinstance(b, str) or not b.strip():
        continue
    r = resolve_best_path(b, out_root, repo_root)
    if r and os.path.isdir(r) and (r == out_root or r.startswith(out_root + os.sep)):
        print(f"OK:subdir_state:{r}|{ts_path}")
        sys.exit(0)

chain = [
    "eval_score", "eval_giou", "eval_gIoU", "gIoU", "giou",
    "eval_mDice", "mDice", "eval_ciou", "eval_cIoU", "cIoU",
]
if user_metric:
    chain = [user_metric]

candidates = sorted(
    [p for p in glob.glob(os.path.join(out_root, "checkpoint-*")) if os.path.isdir(p)],
    key=step_of,
    reverse=True,
)
scores = []
for ck in candidates:
    m = metrics_from_dir(ck)
    if not isinstance(m, dict):
        continue
    val, kn = None, None
    for k in chain:
        if k in m and isinstance(m[k], (int, float)):
            val, kn = float(m[k]), k
            break
    if val is not None:
        scores.append((ck, val, kn))

if scores:
    print("[metric candidates]", file=sys.stderr)
    for ck, v, kn in sorted(scores, key=lambda x: step_of(x[0]), reverse=True):
        print(f"  {ck}  {kn}={v}", file=sys.stderr)
    scores.sort(key=lambda x: x[1], reverse=(mode != "min"))
    best_ck, val, kn = scores[0]
    print(f"OK:metric:{best_ck}|{kn}={val}")
    sys.exit(0)

if not candidates:
    print("ERROR:no_checkpoint_dirs", file=sys.stderr)
    sys.exit(3)

latest = max(candidates, key=step_of)
print(f"WARN:fallback_max_step:{latest}")
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
    --decoder_attn_bias_max_abs 0.01 \
    --decoder_attn_bias_apply_layers last3 \
    --decoder_attn_bias_eval_mode normal \
    --decoder_attn_bias_force_scale 1.0 \
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

run_eval_metrics () {
  local pred_dir="$1"
  local run_name="$2"

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
    python "${EVAL_METRICS_SCRIPT}"
}

########################################
# Preflight
########################################
echo "========================================"
echo "[CONFIG] decoder_attn_bias 28w (${MAX_STEPS} steps)"
echo "  REPO_DIR=${REPO_DIR}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  MERGED_DIR=${MERGED_DIR}"
echo "  TEST_NORMAL_DIR=${TEST_NORMAL_DIR}"
echo "  TEST_BYPASS_DIR=${TEST_BYPASS_DIR}"
echo "  GPU_SLOT=${GPU_SLOT}  GPU_ID=${GPU_ID}  MASTER_PORT=${MASTER_PORT}"
echo "========================================"

if ! python -c "import transformers" 2>/dev/null; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda activate reseg 2>/dev/null || true
  fi
fi
if ! python -c "import transformers" 2>/dev/null; then
  echo "[ERROR] Python cannot import transformers (请先 conda activate reseg)."
  exit 1
fi

HELP_HAS_DAC=0
if python segearth_r2/train/train.py --help 2>&1 | grep -q "use_decoder_attn_bias"; then
  HELP_HAS_DAC=1
fi
if [[ "${HELP_HAS_DAC}" -ne 1 ]]; then
  if grep -q "use_decoder_attn_bias" "${REPO_DIR}/segearth_r2/train/train.py" 2>/dev/null; then
    echo "[WARN] --help 输出中未匹配到 use_decoder_attn_bias（可能被截断或 argparse 格式差异），"
    echo "       但 ${REPO_DIR}/segearth_r2/train/train.py 源码中含该字段，继续执行。"
    HELP_HAS_DAC=1
  fi
fi
if [[ "${HELP_HAS_DAC}" -ne 1 ]]; then
  echo "[ERROR] 无法确认 train 支持 decoder_attn_bias：--help 与源码均未发现 use_decoder_attn_bias。"
  echo "        请确认已在含 dac 代码的 resegearth+tgi 目录下运行，并执行: python segearth_r2/train/train.py --help | head"
  exit 1
fi

if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "[ERROR] MODEL_NAME_OR_PATH 不存在: ${MODEL_NAME_OR_PATH}"
  exit 1
fi

if [[ -d "${OUTPUT_DIR}" ]]; then
  shopt -s nullglob
  _existing=( "${OUTPUT_DIR}"/checkpoint-* )
  shopt -u nullglob
  if [[ ${#_existing[@]} -gt 0 ]]; then
    if [[ "${RESUME_OK:-0}" != "1" ]]; then
      echo "[ERROR] OUTPUT_DIR 下已有 checkpoint-*，可能误续训旧实验。"
      echo "        请换 OUTPUT_DIR 或显式 RESUME_OK=1 续训。"
      exit 1
    fi
    echo "[WARN] RESUME_OK=1 — 将在已有 checkpoint 上续训。"
  fi
fi

mkdir -p "${OUTPUT_DIR}"

########################################
# [1/5] Train
########################################
echo "========================================"
echo "[1/5] Training decoder_attn_bias last3, max_steps=${MAX_STEPS}"
echo "========================================"

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
  --decoder_attn_bias_max_abs 0.01 \
  --decoder_attn_bias_apply_layers last3

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

if echo "${PICK}" | grep -q '^WARN:fallback_max_step:'; then
  echo "[WARN] 未找到 best_model_checkpoint 或可解析的 eval 指标，回退为步数最大 checkpoint。"
  BEST_CHECKPOINT="${PICK#WARN:fallback_max_step:}"
  _STRATEGY="fallback_max_step"
elif echo "${PICK}" | grep -q '^OK:user:'; then
  BEST_CHECKPOINT="${PICK#OK:user:}"
  _STRATEGY="user SELECTED_CHECKPOINT"
elif echo "${PICK}" | grep -q '^OK:root_state:'; then
  BEST_CHECKPOINT="${PICK#OK:root_state:}"
  _STRATEGY="OUTPUT_DIR/trainer_state.json best_model_checkpoint"
elif echo "${PICK}" | grep -q '^OK:subdir_state:'; then
  _LINE="${PICK#OK:subdir_state:}"
  BEST_CHECKPOINT="${_LINE%%|*}"
  _STRATEGY="checkpoint-*/trainer_state.json"
elif echo "${PICK}" | grep -q '^OK:metric:'; then
  _LINE="${PICK#OK:metric:}"
  BEST_CHECKPOINT="${_LINE%%|*}"
  _STRATEGY="metric ${_LINE#*|}"
else
  echo "[ERROR] checkpoint 选择器异常输出: ${PICK}"
  exit 1
fi

if [[ ! -d "${BEST_CHECKPOINT}" ]]; then
  echo "[ERROR] BEST_CHECKPOINT 不是目录: ${BEST_CHECKPOINT}"
  exit 1
fi
echo "[OK] strategy=${_STRATEGY}"
echo "[OK] BEST_CHECKPOINT=${BEST_CHECKPOINT}"

########################################
# [3/5] Merge LoRA（含 decoder_attn_bias）
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
# [CHECK] merged config：decoder_attn_bias 字段必须存在
########################################
echo "========================================"
echo "[CHECK] merged config decoder_attn_bias fields"
echo "========================================"

grep -n '"use_decoder_attn_bias"\|"decoder_attn_bias_dim"\|"decoder_attn_bias_init_std"\|"decoder_attn_bias_max_abs"\|"decoder_attn_bias_apply_layers"' \
  "${MERGED_DIR}/config.json" || {
    echo "[ERROR] merged config.json missing decoder_attn_bias fields"
    exit 1
  }

########################################
# [4/5] Eval：normal + bypass（各一次）
########################################
echo "========================================"
echo "[4/5] Eval inference split=${TEST_SPLIT}: normal + bypass"
echo "========================================"

echo "[INFO] Eval normal"
run_eval_infer "${MERGED_DIR}" "${TEST_NORMAL_DIR}" "normal"

echo "[INFO] Eval bypass"
run_eval_infer "${MERGED_DIR}" "${TEST_BYPASS_DIR}" "bypass"

########################################
# [5/5] Metrics：normal + bypass 分别计算
########################################
echo "========================================"
echo "[5/5] Metrics normal + bypass"
echo "========================================"

echo "[INFO] Metrics normal"
run_eval_metrics "${TEST_NORMAL_DIR}" "${EVAL_METRICS_RUN_NAME}_normal"

echo "[INFO] Metrics bypass"
run_eval_metrics "${TEST_BYPASS_DIR}" "${EVAL_METRICS_RUN_NAME}_bypass"

echo "========================================"
echo "[DONE] 28w pipeline finished."
echo "  BEST_CHECKPOINT=${BEST_CHECKPOINT}"
echo "  MERGED_DIR=${MERGED_DIR}"
echo "  TEST_NORMAL_DIR=${TEST_NORMAL_DIR}"
echo "  TEST_BYPASS_DIR=${TEST_BYPASS_DIR}"
echo "========================================"
