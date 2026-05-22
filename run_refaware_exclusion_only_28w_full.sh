#!/usr/bin/env bash
# =============================================================================
# rrsisd_public_semantic_v2_refaware_exclusion_only_28w（默认 max_steps=280000）
# 全链路：train → 选点 checkpoint → merge LoRA → eval 推理 → eval_val_metrics
#
# Phase-1 基建封口（本脚本内）：
#   - OUTPUT_DIR 下若已有 checkpoint-* 且未设置 RESUME_OK=1，则直接退出（防无意续训污染）。
#   - MERGED_DIR / TEST_OUTPUT_DIR 若已存在，须分别设置 OVERWRITE_MERGE=1 /
#     OVERWRITE_EVAL=1 才允许覆盖；否则在训练前 preflight 即退出。
#   - checkpoint 选取：仅允许 SELECTED_CHECKPOINT、trainer_state.json 的
#     best_model_checkpoint，或各 checkpoint-* 下可解析的 eval 指标链；禁止在
#     无可用选点依据时静默回退到「最大 step」checkpoint。
#   - 最终 BEST_CHECKPOINT（无论来自用户 / trainer_state / metric scan）必须通过
#     统一指标校验：在 checkpoint 目录下须能从 eval_results.json 等或
#     trainer_state.json 的 log_history 解析到 eval_gIoU/gIoU/giou/… 之一；
#     否则 FATAL 退出。仅当 ALLOW_NO_METRIC_CHECKPOINT=1 时可跳过该校验。
# =============================================================================
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-source}"
export WANDB_INIT_TIMEOUT=300
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="${GPU_SLOT:-localhost:2}"
GPU_ID="${GPU_ID:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"

REPO_DIR="${REPO_DIR:-/home/wangchengjun/huangziyi/reseg/resegearth+source}"
RESEG_ROOT="${RESEG_ROOT:-/home/wangchengjun/huangziyi/reseg}"
cd "${REPO_DIR}"

MAX_STEPS="${MAX_STEPS:-280000}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"

OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/source/rrsisd_public_semantic_v2_refaware_exclusion_only_28w}"
MERGED_DIR="${MERGED_DIR:-${RESEG_ROOT}/output/source/rrsisd_public_semantic_v2_refaware_exclusion_only_28w_merged_best}"
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-${RESEG_ROOT}/output/source/rrsisd_public_semantic_v2_refaware_exclusion_only_28w_eval_best/test_results}"

export WANDB_NAME="${WANDB_NAME:-rrsisd_public_semantic_v2_refaware_exclusion_only_28w}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/wangchengjun/huangziyi/reseg/output/bseg/baseline_standard-base_5w/merged_model}"
VISION_TOWER="${VISION_TOWER:-/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"

BASE_DATA_PATH="${BASE_DATA_PATH:-/home/wangchengjun/huangziyi/data/RRSISD}"
DATASET_NAME="rrsisd"
TEST_SPLIT="${TEST_SPLIT:-test}"
CONCEPT_PUBLIC_SEMANTIC_LIBRARY="${CONCEPT_PUBLIC_SEMANTIC_LIBRARY:-configs/concept_public_semantic_library_v2.json}"

EVAL_METRICS_SCRIPT="${EVAL_METRICS_SCRIPT:-${RESEG_ROOT}/eval_val_metrics.py}"
EVAL_USE_WANDB="${EVAL_USE_WANDB:-True}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-source-val}"
EVAL_METRICS_RUN_NAME="${EVAL_METRICS_RUN_NAME:-rrsisd_public_semantic_v2_refaware_exclusion_only_28w}"

REPORT_TO="${REPORT_TO:-wandb}"

PER_DEVICE_TRAIN_BATCH_SIZE="1"
GRADIENT_ACCUMULATION_STEPS="1"
SAVE_STEPS="2000"
SAVE_TOTAL_LIMIT="2"
LEARNING_RATE="1e-4"
WEIGHT_DECAY="0.0"
WARMUP_RATIO="0.03"
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS="10"
BF16="True"
TF32="False"
MODEL_MAX_LENGTH="2048"
GRADIENT_CHECKPOINTING="False"
DATALOADER_NUM_WORKERS="4"
LORA_R="8"
LORA_ALPHA="16"
LORA_DROPOUT="0.05"
DATA_RATIO="1"
SWITCH_BS="4"

MERGE_PY="${REPO_DIR}/segearth_r2/train/merge_lora_weights_and_save_hf_model.py"
EVAL_PY="${REPO_DIR}/segearth_r2/eval/eval.py"

########################################
# Pick checkpoint for merge（stdout: 单行控制串；失败时 stderr + 非 0 退出）
# 优先级：SELECTED_CHECKPOINT → 根 trainer_state best_model_checkpoint →
# 各 checkpoint-*/trainer_state.json 的 best → 在 checkpoint 目录中解析
# eval_gIoU / gIoU / eval_mDice / mDice 等（见 chain）。禁止无依据时取 latest。
# 最终路径仍须通过 validate_final_checkpoint_metrics（见下）。
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

# 1) User
if sel:
    if not os.path.isdir(sel):
        print(f"ERROR:user_not_dir:{sel}", file=sys.stderr)
        sys.exit(2)
    print(f"OK:user:{os.path.abspath(sel)}")
    sys.exit(0)

# 2) OUTPUT_DIR/trainer_state.json
ts0 = read_json(os.path.join(out_root, "trainer_state.json"))
if isinstance(ts0, dict):
    b = ts0.get("best_model_checkpoint")
    if isinstance(b, str) and b.strip():
        r = resolve_best_path(b, out_root, repo_root)
        if r and os.path.isdir(r) and (r == out_root or r.startswith(out_root + os.sep)):
            print(f"OK:root_state:{r}")
            sys.exit(0)

# 3) checkpoint-*/trainer_state.json
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

chain = ["eval_gIoU", "gIoU", "giou", "eval_mDice", "mDice", "eval_cIoU", "cIoU"]
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

print(
    "ERROR:no_checkpoint_selection: no valid SELECTED_CHECKPOINT, no usable "
    "best_model_checkpoint in trainer_state.json, and no eval_gIoU/gIoU/eval_mDice/mDice/… "
    "in checkpoint eval files or log_history under any checkpoint-*.",
    file=sys.stderr,
)
sys.exit(4)
PY
}

########################################
# 对最终 BEST_CHECKPOINT 做专家级硬校验：目录内须能解析到规定 eval 指标之一
#（eval_results.json / metrics.json / … 或 该目录下 trainer_state.json 的 log_history）。
# ALLOW_NO_METRIC_CHECKPOINT=1 时跳过（默认不允许）。
########################################
validate_final_checkpoint_metrics () {
  export _VALIDATE_CKPT_DIR="${BEST_CHECKPOINT}"
  export ALLOW_NO_METRIC_CHECKPOINT="${ALLOW_NO_METRIC_CHECKPOINT:-0}"
  python - <<'PY'
import json
import os
import sys

KEYS = ("eval_gIoU", "gIoU", "giou", "eval_mDice", "mDice", "eval_cIoU", "cIoU")


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def dict_has_metric(d):
    if not isinstance(d, dict):
        return None
    for k in KEYS:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return k, float(v)
    return None


def scan_checkpoint_dir(ckpt_dir):
    # 与需求一致：先 log_history，再各 eval 落盘 json
    ts = read_json(os.path.join(ckpt_dir, "trainer_state.json"))
    if isinstance(ts, dict) and isinstance(ts.get("log_history"), list):
        for row in reversed(ts["log_history"]):
            hit = dict_has_metric(row)
            if hit:
                return hit[0], hit[1], "trainer_state.json:log_history"
    for fname in ("eval_results.json", "metrics.json", "all_results.json", "eval_metrics.json"):
        p = os.path.join(ckpt_dir, fname)
        data = read_json(p)
        hit = dict_has_metric(data)
        if hit:
            return hit[0], hit[1], fname
    return None


allow = os.environ.get("ALLOW_NO_METRIC_CHECKPOINT", "0").strip() == "1"
ckpt = (os.environ.get("_VALIDATE_CKPT_DIR") or "").strip()

if allow:
    print(
        "[INFO] ALLOW_NO_METRIC_CHECKPOINT=1 — skipping mandatory metric validation.",
        file=sys.stderr,
    )
    sys.exit(0)

if not ckpt or not os.path.isdir(ckpt):
    print("FATAL: No valid best checkpoint metrics found", file=sys.stderr)
    sys.exit(1)

found = scan_checkpoint_dir(os.path.abspath(ckpt))
if not found:
    print("FATAL: No valid best checkpoint metrics found", file=sys.stderr)
    sys.exit(1)

k, v, src = found
print(f"[OK] final checkpoint metric validation: {k}={v} (source={src})")
sys.exit(0)
PY
}

merge_lora () {
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
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}"
}

run_eval_infer () {
  local model_dir="$1"
  local out_dir="$2"
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
      --zip_results False
}

run_eval_metrics () {
  if [[ ! -f "${EVAL_METRICS_SCRIPT}" ]]; then
    echo "[ERROR] eval metrics script not found: ${EVAL_METRICS_SCRIPT}"
    exit 1
  fi
  USE_WANDB="${EVAL_USE_WANDB}" \
    WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
    WANDB_RUN_NAME="${EVAL_METRICS_RUN_NAME}" \
    DATASET_TYPE="${DATASET_NAME}" \
    BASE_DATA_PATH="${BASE_DATA_PATH}" \
    SPLIT="${TEST_SPLIT}" \
    PRED_DIR="$1" \
    python "${EVAL_METRICS_SCRIPT}"
}

########################################
# Preflight
########################################
echo "========================================"
echo "[CONFIG] rrsisd_public_semantic_v2_refaware_exclusion_only_28w（full pipeline）"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  MERGED_DIR=${MERGED_DIR}"
echo "  TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}"
echo "  MAX_STEPS=${MAX_STEPS}"
echo "  SEED=${SEED}"
echo "  DATA_SEED=${DATA_SEED}"
echo "  GPU_SLOT=${GPU_SLOT}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "  RESUME_OK=${RESUME_OK:-0}  (1=allow OUTPUT_DIR with checkpoint-* / resume train)"
echo "  OVERWRITE_MERGE=${OVERWRITE_MERGE:-0}  (1=allow replacing existing MERGED_DIR)"
echo "  OVERWRITE_EVAL=${OVERWRITE_EVAL:-0}  (1=allow replacing existing TEST_OUTPUT_DIR)"
echo "  ALLOW_NO_METRIC_CHECKPOINT=${ALLOW_NO_METRIC_CHECKPOINT:-0}  (1=skip mandatory metric validation on BEST_CHECKPOINT)"
echo "  concept_refaware_prior=True"
echo "  concept_match_strict=False"
echo "========================================"

if ! python -c "import transformers" 2>/dev/null; then
  if command -v conda >/dev/null 2>&1; then
    _PY="$(conda run -n reseg which python 2>/dev/null || true)"
    if [[ -n "${_PY}" && -x "${_PY}" ]]; then export PATH="$(dirname "${_PY}"):${PATH}"; fi
  fi
fi
if ! python -c "import transformers" 2>/dev/null; then
  for _c in "${HOME}/miniconda3/envs/reseg/bin" "${HOME}/anaconda3/envs/reseg/bin"; do
    if [[ -x "${_c}/python" ]]; then export PATH="${_c}:${PATH}"; break; fi
  done
fi
if ! python -c "import transformers" 2>/dev/null; then
  echo "[ERROR] Python cannot import transformers."
  exit 1
fi

HELP_OUT="$(python segearth_r2/train/train.py --help 2>&1 || true)"
if ! echo "${HELP_OUT}" | grep -q "concept_refaware_prior"; then
  echo "[ERROR] train.py --help must contain concept_refaware_prior."
  exit 1
fi
if ! echo "${HELP_OUT}" | grep -q "concept_match_strict"; then
  echo "[ERROR] train.py --help must contain concept_match_strict."
  exit 1
fi

LIB_PATH="${CONCEPT_PUBLIC_SEMANTIC_LIBRARY}"
if [[ "${LIB_PATH}" != /* ]]; then LIB_PATH="${REPO_DIR}/${CONCEPT_PUBLIC_SEMANTIC_LIBRARY}"; fi
if [[ ! -f "${LIB_PATH}" ]]; then
  echo "[ERROR] Semantic library not found: ${LIB_PATH}"
  exit 1
fi
python - <<PY || { echo "[ERROR] Cannot load semantic library JSON."; exit 1; }
import json
with open("${LIB_PATH}", "r", encoding="utf-8") as f:
    json.load(f)
print("[OK] semantic library JSON loads.")
PY

if [[ ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "[ERROR] MODEL_NAME_OR_PATH not found: ${MODEL_NAME_OR_PATH}"
  exit 1
fi
if [[ ! -f "${EVAL_METRICS_SCRIPT}" ]]; then
  echo "[ERROR] EVAL_METRICS_SCRIPT not found: ${EVAL_METRICS_SCRIPT}"
  exit 1
fi

# MERGED_DIR / TEST_OUTPUT_DIR：训练前即检查，避免跑满训练后才发现无法 merge/eval
if [[ -e "${MERGED_DIR}" ]]; then
  if [[ "${OVERWRITE_MERGE:-0}" != "1" ]]; then
    echo "[ERROR] MERGED_DIR already exists: ${MERGED_DIR}"
    echo "Set OVERWRITE_MERGE=1 to allow remove and re-merge, or choose a new MERGED_DIR."
    exit 1
  fi
fi
if [[ -e "${TEST_OUTPUT_DIR}" ]]; then
  if [[ "${OVERWRITE_EVAL:-0}" != "1" ]]; then
    echo "[ERROR] TEST_OUTPUT_DIR already exists: ${TEST_OUTPUT_DIR}"
    echo "Set OVERWRITE_EVAL=1 to allow overwrite, or choose a new TEST_OUTPUT_DIR."
    exit 1
  fi
fi

if [[ -d "${OUTPUT_DIR}" ]]; then
  shopt -s nullglob
  _existing=( "${OUTPUT_DIR}"/checkpoint-* )
  shopt -u nullglob
  if [[ ${#_existing[@]} -gt 0 ]]; then
    if [[ "${RESUME_OK:-0}" != "1" ]]; then
      echo "ERROR: OUTPUT_DIR already contains checkpoint-*."
      echo "This may resume from an old run and pollute the experiment."
      echo "Please set RESUME_OK=1 to resume intentionally, or use a new OUTPUT_DIR."
      exit 1
    fi
    echo "WARNING: RESUME_OK=1 — OUTPUT_DIR contains checkpoint-*; training may resume from existing run."
  fi
fi

mkdir -p "${OUTPUT_DIR}"

########################################
# [1/5] Train
########################################
echo "========================================"
echo "[1/5] Training rrsisd_public_semantic_v2_refaware_exclusion_only_28w (${MAX_STEPS} steps)"
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
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --logging_steps "${LOGGING_STEPS}" \
  --tf32 "${TF32}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --deepspeed scripts/zero1.json \
  --mask_config "${MASK_CONFIG}" \
  --data_ratio "${DATA_RATIO}" \
  --switch_bs "${SWITCH_BS}" \
  --seed "${SEED}" \
  --data_seed "${DATA_SEED}" \
  --report_to "${REPORT_TO}" \
  --concept_public_semantic_library "${CONCEPT_PUBLIC_SEMANTIC_LIBRARY}" \
  --concept_refaware_prior True

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
  echo "[ERROR] SELECTED_CHECKPOINT is set but not a directory."
  exit 1
fi
if [[ "${_CK_RC}" -eq 3 ]]; then
  echo "[ERROR] No checkpoint-* under OUTPUT_DIR."
  exit 1
fi
if [[ "${_CK_RC}" -eq 4 ]]; then
  echo "[ERROR] Checkpoint selection failed: no SELECTED_CHECKPOINT, no valid best_model_checkpoint,"
  echo "        and no eval_gIoU/gIoU/eval_mDice/mDice (or configured BEST_METRIC_NAME) under checkpoint-*."
  echo "        Silent fallback to latest step is disabled. Fix training eval saving or set SELECTED_CHECKPOINT=..."
  exit 1
fi

if echo "${PICK}" | grep -q '^OK:user:'; then
  BEST_CHECKPOINT="${PICK#OK:user:}"
  _STRATEGY="user SELECTED_CHECKPOINT"
  echo "Using user-specified SELECTED_CHECKPOINT=${BEST_CHECKPOINT}"
elif echo "${PICK}" | grep -q '^OK:root_state:'; then
  BEST_CHECKPOINT="${PICK#OK:root_state:}"
  _STRATEGY="OUTPUT_DIR/trainer_state.json best_model_checkpoint"
  echo "Using best_model_checkpoint from OUTPUT_DIR/trainer_state.json: ${BEST_CHECKPOINT}"
elif echo "${PICK}" | grep -q '^OK:subdir_state:'; then
  _LINE="${PICK#OK:subdir_state:}"
  BEST_CHECKPOINT="${_LINE%%|*}"
  _STRATEGY="checkpoint-*/trainer_state.json (${_LINE#*|})"
  echo "Using best_model_checkpoint from nested trainer_state: ${BEST_CHECKPOINT}"
elif echo "${PICK}" | grep -q '^OK:metric:'; then
  _LINE="${PICK#OK:metric:}"
  BEST_CHECKPOINT="${_LINE%%|*}"
  _STRATEGY="metric ${_LINE#*|}"
  echo "Using metric-selected checkpoint: ${BEST_CHECKPOINT} (${_LINE#*|})"
else
  echo "[ERROR] Unexpected picker output: ${PICK}"
  exit 1
fi

if [[ ! -d "${BEST_CHECKPOINT}" ]]; then
  echo "[ERROR] BEST_CHECKPOINT is not a directory: ${BEST_CHECKPOINT}"
  exit 1
fi

echo "[OK] checkpoint selection strategy: ${_STRATEGY}"
echo "[OK] BEST_CHECKPOINT=${BEST_CHECKPOINT}"

echo "========================================"
echo "[2b/5] Mandatory metric validation on BEST_CHECKPOINT"
echo "========================================"
validate_final_checkpoint_metrics

########################################
# [3/5] Merge LoRA
########################################
echo "========================================"
echo "[3/5] Merge LoRA"
echo "========================================"
echo "  CHECKPOINT selection strategy: ${_STRATEGY}"
echo "  SELECTED_CHECKPOINT (resolved): ${BEST_CHECKPOINT}"
echo "  MERGED_DIR=${MERGED_DIR}"
echo "  MODEL_NAME_OR_PATH (train base): ${MODEL_NAME_OR_PATH}"
echo "  VISION_TOWER=${VISION_TOWER}"
echo "  VISION_TOWER_MASK=${VISION_TOWER_MASK}"
echo "  MASK_CONFIG=${MASK_CONFIG}"

if [[ ! -f "${MERGE_PY}" ]]; then
  echo "[ERROR] merge_lora_weights_and_save_hf_model.py not found: ${MERGE_PY}"
  exit 1
fi

if [[ -e "${MERGED_DIR}" ]]; then
  if [[ "${OVERWRITE_MERGE:-0}" != "1" ]]; then
    echo "[ERROR] MERGED_DIR already exists: ${MERGED_DIR}"
    echo "Set OVERWRITE_MERGE=1 to remove and re-merge."
    exit 1
  fi
  echo "[INFO] OVERWRITE_MERGE=1 — removing existing MERGED_DIR."
  rm -rf "${MERGED_DIR}"
fi

merge_lora "${BEST_CHECKPOINT}" "${MERGED_DIR}"

if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  echo "[ERROR] Missing merged config: ${MERGED_DIR}/config.json"
  exit 1
fi

########################################
# [4/5] Eval (inference → test_results)
########################################
echo "========================================"
echo "[4/5] Eval (inference)"
echo "========================================"
echo "  MODEL_PATH (MERGED_DIR)=${MERGED_DIR}"
echo "  TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}"
echo "  BASE_DATA_PATH=${BASE_DATA_PATH}"
echo "  TEST_SPLIT=${TEST_SPLIT}"
echo "  Semantic prior in inference/eval: No, follows original v2 setting."

if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  echo "[ERROR] MERGED_DIR/config.json missing."
  exit 1
fi
if [[ ! -f "${EVAL_PY}" ]]; then
  echo "[ERROR] eval.py not found: ${EVAL_PY}"
  exit 1
fi
if [[ ! -d "${BASE_DATA_PATH}" ]]; then
  echo "[ERROR] BASE_DATA_PATH not found: ${BASE_DATA_PATH}"
  exit 1
fi
if [[ ! -f "${BASE_DATA_PATH}/rrsisd/refs(unc).p" ]]; then
  echo "[ERROR] Missing ${BASE_DATA_PATH}/rrsisd/refs(unc).p"
  exit 1
fi
if [[ ! -f "${BASE_DATA_PATH}/rrsisd/instances.json" ]]; then
  echo "[ERROR] Missing ${BASE_DATA_PATH}/rrsisd/instances.json"
  exit 1
fi

if [[ -e "${TEST_OUTPUT_DIR}" ]]; then
  if [[ "${OVERWRITE_EVAL:-0}" != "1" ]]; then
    echo "[ERROR] TEST_OUTPUT_DIR already exists: ${TEST_OUTPUT_DIR}"
    echo "Set OVERWRITE_EVAL=1 to overwrite."
    exit 1
  fi
  echo "[INFO] OVERWRITE_EVAL=1 — removing existing TEST_OUTPUT_DIR."
  rm -rf "${TEST_OUTPUT_DIR}"
fi

run_eval_infer "${MERGED_DIR}" "${TEST_OUTPUT_DIR}"

########################################
# [5/5] Metrics
########################################
echo "========================================"
echo "[5/5] eval_val_metrics.py"
echo "========================================"

run_eval_metrics "${TEST_OUTPUT_DIR}"

echo "========================================"
echo "DONE rrsisd_public_semantic_v2_refaware_exclusion_only_28w"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  BEST_CHECKPOINT=${BEST_CHECKPOINT}"
echo "  MERGED_DIR=${MERGED_DIR}"
echo "  TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}"
echo "  eval metrics run name: ${EVAL_METRICS_RUN_NAME}"
echo "========================================"
