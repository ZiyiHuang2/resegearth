#!/usr/bin/env bash
# Mask Quality v1（OHEM-BCE, ratio=0.5）：train → best checkpoint → merge/export → verify → eval → metrics
#
# 本脚本固定：OHEM_RATIO=0.5，仅 7w（MAX_STEPS=70000）。
#   bash run_mq_v1_0.5_full_pipeline.sh
#
# 与 QMC 脚本相同风格的可选环境变量：
#   MERGE_KEEP_EXISTING=1     — MERGED_DIR 已存在则退出（默认 merge 内会 rm -rf 覆盖）
#   REQUIRE_DEEPSPEED_STATE=0 — 无 mp_rank_*_model_states.pt 时仅警告
#   ALLOW_BEST_CKPT_OUTSIDE_OUTPUT=1
#   REQUIRE_EVAL_METRICS=0    — 跳过 [5/6] 指标
#   REQUIRE_RRSISD_FILES=0
#   KEEP_TEST_OUTPUT=1        — TEST_OUTPUT_DIR 已存在则退出（默认先 rm -rf）
#   RESUME_OK=1               — OUTPUT_DIR 已有 checkpoint-* 时允许续训
#   PYTHON=…/reseg/bin/python — merge/eval/内嵌脚本解释器（训练仍由 PATH 上 deepspeed 决定）
#   VERIFY_OHEM_YAML=0        — 不检查 MASK_CONFIG 内 USE_OHEM_BCE: True（调 baseline yaml 时设）
#   EXPECT_QMC=0              — 默认已是 0；merge/checkpoint 不要求 qmc_* 键
set -euo pipefail

########################################
# Basic env
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-mq}"
export WANDB_INIT_TIMEOUT=300

# MQ-v1 不使用 QMC：merge / checkpoint / eval 校验不要求 qmc_* 投影权重
export EXPECT_QMC="${EXPECT_QMC:-0}"

unset CUDA_VISIBLE_DEVICES

GPU_SLOT="${GPU_SLOT:-localhost:0}"
GPU_ID="${GPU_ID:-0}"
MASTER_PORT="${MASTER_PORT:-29510}"

########################################
# Paths
########################################
RESEG_ROOT="${RESEG_ROOT:-/home/wangchengjun/huangziyi/reseg}"
REPO_DIR="${REPO_DIR:-${RESEG_ROOT}/segearth+maskquality}"
PYTHON="${PYTHON:-python}"
cd "${REPO_DIR}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mllm/Mipha-3B}"
VISION_TOWER="${VISION_TOWER:-/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep_mq_v1_ohem_ratio0.5.yaml}"

BASE_DATA_PATH="${BASE_DATA_PATH:-/home/wangchengjun/huangziyi/data/RRSISD}"
DATASET_NAME="${DATASET_NAME:-rrsisd}"
TEST_SPLIT="${TEST_SPLIT:-test}"

SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"
OHEM_RATIO_TAG="${OHEM_RATIO_TAG:-0.5}"

# 用户若在启动前已 export，仅单档位模式生效（both 时会报错）
_USER_MAX_STEPS="${MAX_STEPS:-}"
_USER_SAVE_STEPS="${SAVE_STEPS:-}"
_USER_RUN_TAG="${RUN_TAG:-}"
_USER_OUTPUT_DIR="${OUTPUT_DIR:-}"
_USER_MERGED_DIR="${MERGED_DIR:-}"
_USER_TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-}"
_USER_WANDB_NAME="${WANDB_NAME:-}"
_USER_EVAL_METRICS_RUN_NAME="${EVAL_METRICS_RUN_NAME:-}"

########################################
# Training hyperparams（与 baseline-7w / smoke 对齐；可按需 env 覆盖）
########################################
REPORT_TO="${REPORT_TO:-wandb}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

BF16="${BF16:-True}"
TF32="${TF32:-False}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

DATA_RATIO="${DATA_RATIO:-1}"
SWITCH_BS="${SWITCH_BS:-4}"

MERGE_PY="${MERGE_PY:-${REPO_DIR}/segearth_r2/train/merge_lora_weights_and_save_hf_model.py}"
EVAL_PY="${EVAL_PY:-${REPO_DIR}/segearth_r2/eval/eval.py}"
EVAL_METRICS_SCRIPT="${EVAL_METRICS_SCRIPT:-${RESEG_ROOT}/eval_val_metrics.py}"

EVAL_USE_WANDB="${EVAL_USE_WANDB:-True}"
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-mq}"

export REQUIRE_DEEPSPEED_STATE="${REQUIRE_DEEPSPEED_STATE:-1}"
export REQUIRE_EVAL_METRICS="${REQUIRE_EVAL_METRICS:-1}"
export ALLOW_BEST_CKPT_OUTSIDE_OUTPUT="${ALLOW_BEST_CKPT_OUTSIDE_OUTPUT:-0}"
VERIFY_OHEM_YAML="${VERIFY_OHEM_YAML:-1}"

########################################
# Helpers
########################################
die() {
  echo "[ERROR] $*" >&2
  exit 1
}

info() {
  echo "[INFO] $*"
}

configure_mq_7w_profile () {
  MQ_PROFILE=7w
  MAX_STEPS="${_USER_MAX_STEPS:-70000}"
  SAVE_STEPS="${_USER_SAVE_STEPS:-2000}"
  if [[ -n "${_USER_RUN_TAG}" ]]; then
    RUN_TAG="${_USER_RUN_TAG}"
  else
    RUN_TAG="mq_v1_ohem_ratio${OHEM_RATIO_TAG}_siglip2_7w"
  fi
  OUTPUT_DIR="${_USER_OUTPUT_DIR:-${RESEG_ROOT}/output/mq/${RUN_TAG}}"
  MERGED_DIR="${_USER_MERGED_DIR:-${RESEG_ROOT}/output/mq/${RUN_TAG}/merged_best}"
  TEST_OUTPUT_DIR="${_USER_TEST_OUTPUT_DIR:-${RESEG_ROOT}/output/mq/${RUN_TAG}/test_results}"
  export WANDB_NAME="${_USER_WANDB_NAME:-rrsisd_${RUN_TAG}}"
  EVAL_METRICS_RUN_NAME="${_USER_EVAL_METRICS_RUN_NAME:-rrsisd_${RUN_TAG}}"
}

########################################
# Pick best checkpoint（与 QMC 脚本相同）
########################################
pick_best_checkpoint () {
  export _CK_ROOT="${OUTPUT_DIR}"
  export _CK_REPO="${REPO_DIR}"
  export _CK_SEL="${SELECTED_CHECKPOINT:-}"
  export _CK_METRIC_NAME="${BEST_METRIC_NAME:-}"
  export _CK_METRIC_MODE="${BEST_METRIC_MODE:-max}"
  export _CK_ALLOW_EXTERNAL="${ALLOW_BEST_CKPT_OUTSIDE_OUTPUT:-0}"

  "${PYTHON}" - <<'PY'
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
allow_ext = os.environ.get("_CK_ALLOW_EXTERNAL", "0").strip() in ("1", "true", "True", "yes", "YES")

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
        if r and os.path.isdir(r) and (allow_ext or r == out_root or r.startswith(out_root + os.sep)):
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
    if r and os.path.isdir(r) and (allow_ext or r == out_root or r.startswith(out_root + os.sep)):
        print(f"OK:subdir_state:{r}|{ts_path}")
        sys.exit(0)

chain = ["eval_score", "eval_giou", "eval_ciou", "eval_gIoU", "gIoU", "giou", "eval_cIoU", "cIoU"]
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

########################################
# Merge / eval / metrics
########################################
merge_lora () {
  local ckpt="$1"
  local save_dir="$2"

  rm -rf "${save_dir}"
  mkdir -p "${save_dir}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" "${MERGE_PY}" \
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
    "${PYTHON}" "${EVAL_PY}" \
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
  if [[ "${REQUIRE_EVAL_METRICS:-1}" == "0" ]]; then
    info "REQUIRE_EVAL_METRICS=0 — skip metrics aggregation."
    return 0
  fi
  if [[ ! -f "${EVAL_METRICS_SCRIPT}" ]]; then
    die "eval metrics script not found: ${EVAL_METRICS_SCRIPT}"
  fi

  USE_WANDB="${EVAL_USE_WANDB}" \
    WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
    WANDB_RUN_NAME="${EVAL_METRICS_RUN_NAME}" \
    DATASET_TYPE="${DATASET_NAME}" \
    BASE_DATA_PATH="${BASE_DATA_PATH}" \
    SPLIT="${TEST_SPLIT}" \
    PRED_DIR="$1" \
    "${PYTHON}" "${EVAL_METRICS_SCRIPT}"
}

########################################
# Verification functions
########################################
verify_checkpoint_full_state () {
  local ckpt="$1"

  echo "========================================"
  echo "[VERIFY] checkpoint full state"
  echo "  CHECKPOINT=${ckpt}"
  echo "========================================"

  REQUIRE_DS="${REQUIRE_DEEPSPEED_STATE:-1}"
  EXPECT_QMC_FLAG="${EXPECT_QMC:-0}"

  "${PYTHON}" - <<PY
import os, glob, sys, torch

ckpt = "${ckpt}"
require_ds = "${REQUIRE_DS}".strip() not in ("0", "false", "False", "no", "NO")
expect_qmc = "${EXPECT_QMC_FLAG}".strip() in ("1", "true", "True", "yes", "YES")

cands = glob.glob(os.path.join(ckpt, "global_step*", "mp_rank_*_model_states.pt"))
if not cands:
    msg = "No DeepSpeed full model state under checkpoint: " + ckpt
    if require_ds:
        print("[ERROR] " + msg, file=sys.stderr)
        sys.exit(1)
    print("[WARN] " + msg)
    print("[OK] checkpoint verify skipped (REQUIRE_DEEPSPEED_STATE=0).")
    sys.exit(0)

p = cands[0]
print("[OK] Found DeepSpeed full state:", p)

d = torch.load(p, map_location="cpu")
sd = d.get("module", d) if isinstance(d, dict) else d
keys = list(sd.keys())

need_patterns = [
    "SEG_token_projector",
    "pixel_decoder",
    "predictor",
]

for pat in need_patterns:
    hit = [k for k in keys if pat in k]
    if not hit:
        print(f"[ERROR] Missing trained module in checkpoint: {pat}")
        sys.exit(2)
    t = sd[hit[0]].float()
    print(f"[OK] checkpoint contains {pat}: key={hit[0]}, shape={tuple(t.shape)}, norm={float(t.norm()):.6f}")

if expect_qmc:
    for pat in ["qmc_q_projector", "qmc_v_projector"]:
        hit = [k for k in keys if pat in k]
        if not hit:
            print(f"[ERROR] EXPECT_QMC=1 but checkpoint missing: {pat}")
            sys.exit(3)
        t = sd[hit[0]].float()
        print(f"[OK] checkpoint contains {pat}: key={hit[0]}, shape={tuple(t.shape)}, norm={float(t.norm()):.6f}")
else:
    print("[INFO] EXPECT_QMC=0 — skip qmc_* key requirement in checkpoint.")

print("[OK] checkpoint full-state check passed.")
PY
}

verify_merged_dir_not_adapter_only () {
  local merged="$1"

  echo "========================================"
  echo "[VERIFY] merged dir completeness"
  echo "  MERGED_DIR=${merged}"
  echo "========================================"

  [[ -f "${merged}/config.json" ]] || die "Missing config.json in MERGED_DIR."

  shopt -s nullglob
  local model_files=( "${merged}"/*.safetensors "${merged}"/pytorch_model*.bin "${merged}"/model*.bin )
  shopt -u nullglob

  if [[ ${#model_files[@]} -eq 0 ]]; then
    die "No full model weight file found in MERGED_DIR. It may be adapter-only or merge failed."
  fi

  if [[ -f "${merged}/adapter_model.safetensors" && ${#model_files[@]} -eq 1 ]]; then
    echo "[WARN] MERGED_DIR appears to contain only adapter_model.safetensors."
    die "Adapter-only MERGED_DIR is unsafe for eval.py."
  fi

  echo "[OK] MERGED_DIR has model weight files:"
  printf '  %s\n' "${model_files[@]}"
}

verify_eval_loader_modules () {
  local merged="$1"

  echo "========================================"
  echo "[VERIFY] eval loader can load merged model"
  echo "  MERGED_DIR=${merged}"
  echo "========================================"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" - <<PY
import os, sys, torch

repo = "${REPO_DIR}"
sys.path.insert(0, repo)

from segearth_r2.utils.builder import load_pretrained_model

model_path = "${merged}"
vision_tower = "${VISION_TOWER}"
vision_tower_mask = "${VISION_TOWER_MASK}"
mask_config = "${MASK_CONFIG}"

class Args:
    pass

args = Args()
args.vision_tower = vision_tower
args.vision_tower_mask = vision_tower_mask
args.mask_config = mask_config

args.swin_type = "base"
args.load_mask2former = True
args.with_norm = True
args.with_layernorm = False
args.skip_init_vision = False
args.projector_outdim = 2048
args.mm_projector_type = "swin_conv"
args.model_version = "v1"
args.model_name_or_path = model_path

try:
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_args=args,
        mask_config=mask_config,
        device="cuda",
    )
except Exception as e:
    print("[ERROR] eval loader failed to load MERGED_DIR:", repr(e))
    sys.exit(1)

sd = model.state_dict()
keys = list(sd.keys())

patterns = [
    "SEG_token_projector.weight",
    "pixel_decoder",
    "predictor",
]

expect_qmc = os.environ.get("EXPECT_QMC", "0").strip() in ("1", "true", "True", "yes", "YES")
if expect_qmc:
    patterns.append("qmc_q_projector.0.weight")

for pat in patterns:
    hits = [k for k in keys if pat in k]
    if not hits:
        print(f"[ERROR] Loaded eval model missing param pattern: {pat}")
        sys.exit(2)
    k = hits[0]
    t = sd[k].detach().float()
    print(f"[OK] loaded {pat}: key={k}, shape={tuple(t.shape)}, norm={float(t.norm()):.6f}, mean={float(t.mean()):.8f}")

print("[OK] eval loader loaded merged model with expected modules.")
PY
}

extract_train_summary () {
  local out_dir="$1"
  local out_json="${out_dir}/trainer_state.json"

  if [[ ! -f "${out_json}" ]]; then
    echo "[WARN] trainer_state.json not found: ${out_json}"
    return 0
  fi

  echo "========================================"
  echo "[SUMMARY] trainer_state"
  echo "========================================"

  "${PYTHON}" - <<PY
import json
p = "${out_json}"
h = json.load(open(p, "r", encoding="utf-8")).get("log_history", [])
last_eval = None
last_loss = None
last_mask = None
for row in h:
    if "eval_score" in row:
        last_eval = row
    if "loss" in row:
        last_loss = row
    if "loss_mask" in row:
        last_mask = row

print("[last_loss]", last_loss)
print("[last_loss_mask]", last_mask)
print("[last_eval]", last_eval)

if last_eval is None:
    print("[WARN] No eval_* record found.")
else:
    print("[OK] eval_giou=", last_eval.get("eval_giou"))
    print("[OK] eval_ciou=", last_eval.get("eval_ciou"))
    print("[OK] eval_score=", last_eval.get("eval_score"))
PY
}

run_mq_v1_pipeline_once () {
########################################
# Preflight（单档位）
########################################
echo "========================================"
echo "[CONFIG] MQ-v1 OHEM-BCE full pipeline"
echo "  MQ_PROFILE=${MQ_PROFILE}"
echo "  REPO_DIR=${REPO_DIR}"
echo "  PYTHON=${PYTHON}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  MERGED_DIR=${MERGED_DIR}"
echo "  TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}"
echo "  MAX_STEPS=${MAX_STEPS}"
echo "  SAVE_STEPS=${SAVE_STEPS}"
echo "  SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"
echo "  SEED=${SEED}"
echo "  DATA_SEED=${DATA_SEED}"
echo "  GPU_SLOT=${GPU_SLOT}"
echo "  GPU_ID=${GPU_ID}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "  MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "  VISION_TOWER=${VISION_TOWER}"
echo "  VISION_TOWER_MASK=${VISION_TOWER_MASK}"
echo "  MASK_CONFIG=${MASK_CONFIG}"
echo "  USE_QMC=False (MQ-v1)"
echo "  PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "  EXPECT_QMC=${EXPECT_QMC}"
echo "========================================"

if [[ -d "${OUTPUT_DIR}" ]]; then
  shopt -s nullglob
  existing=( "${OUTPUT_DIR}"/checkpoint-* )
  shopt -u nullglob

  if [[ ${#existing[@]} -gt 0 ]]; then
    if [[ "${RESUME_OK:-0}" != "1" ]]; then
      echo "[ERROR] OUTPUT_DIR already contains checkpoint-*:"
      printf '  %s\n' "${existing[@]}"
      echo "This may resume from an old run and pollute the experiment."
      echo "Set RESUME_OK=1 to resume intentionally, or use a new OUTPUT_DIR."
      exit 1
    fi
    echo "[WARN] RESUME_OK=1 — OUTPUT_DIR contains checkpoint-*; training may resume."
  fi
fi

mkdir -p "${OUTPUT_DIR}"

########################################
# [1/6] Train
########################################
echo "========================================"
echo "[1/6] Training MQ-v1 OHEM-BCE (${MAX_STEPS} steps)"
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
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
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
  --use_qmc False

extract_train_summary "${OUTPUT_DIR}"

########################################
# [2/6] Select best checkpoint
########################################
echo "========================================"
echo "[2/6] Select best checkpoint"
echo "========================================"

CK_PICK_OUT="$(mktemp)"
set +e
pick_best_checkpoint > "${CK_PICK_OUT}"
CK_RC=$?
set -e
PICK="$(head -n 1 "${CK_PICK_OUT}" | tr -d '\r')"
rm -f "${CK_PICK_OUT}"

if [[ "${CK_RC}" -eq 2 ]]; then
  die "SELECTED_CHECKPOINT is set but not a directory."
fi
if [[ "${CK_RC}" -eq 3 ]]; then
  die "No checkpoint-* under OUTPUT_DIR."
fi

if echo "${PICK}" | grep -q '^WARN:fallback_max_step:'; then
  echo "[WARN] No best_model_checkpoint or eval metric found. Falling back to latest checkpoint."
  BEST_CHECKPOINT="${PICK#WARN:fallback_max_step:}"
  CK_STRATEGY="fallback_max_step"
elif echo "${PICK}" | grep -q '^OK:user:'; then
  BEST_CHECKPOINT="${PICK#OK:user:}"
  CK_STRATEGY="user SELECTED_CHECKPOINT"
elif echo "${PICK}" | grep -q '^OK:root_state:'; then
  BEST_CHECKPOINT="${PICK#OK:root_state:}"
  CK_STRATEGY="OUTPUT_DIR/trainer_state.json best_model_checkpoint"
elif echo "${PICK}" | grep -q '^OK:subdir_state:'; then
  LINE="${PICK#OK:subdir_state:}"
  BEST_CHECKPOINT="${LINE%%|*}"
  CK_STRATEGY="checkpoint-*/trainer_state.json (${LINE#*|})"
elif echo "${PICK}" | grep -q '^OK:metric:'; then
  LINE="${PICK#OK:metric:}"
  BEST_CHECKPOINT="${LINE%%|*}"
  CK_STRATEGY="metric ${LINE#*|}"
else
  die "Unexpected picker output: ${PICK}"
fi

[[ -d "${BEST_CHECKPOINT}" ]] || die "BEST_CHECKPOINT is not a directory: ${BEST_CHECKPOINT}"

echo "[OK] checkpoint selection strategy: ${CK_STRATEGY}"
echo "[OK] BEST_CHECKPOINT=${BEST_CHECKPOINT}"

verify_checkpoint_full_state "${BEST_CHECKPOINT}"

########################################
# [3/6] Merge / export
########################################
echo "========================================"
echo "[3/6] Merge / export"
echo "========================================"
echo "  BEST_CHECKPOINT=${BEST_CHECKPOINT}"
echo "  MERGED_DIR=${MERGED_DIR}"

if [[ -e "${MERGED_DIR}" && "${MERGE_KEEP_EXISTING:-0}" == "1" ]]; then
  die "MERGED_DIR already exists and MERGE_KEEP_EXISTING=1 refuses overwrite. Unset MERGE_KEEP_EXISTING or remove ${MERGED_DIR}."
fi

merge_lora "${BEST_CHECKPOINT}" "${MERGED_DIR}"

[[ -f "${MERGED_DIR}/config.json" ]] || die "Missing merged config: ${MERGED_DIR}/config.json"

verify_merged_dir_not_adapter_only "${MERGED_DIR}"
verify_eval_loader_modules "${MERGED_DIR}"

########################################
# [4/6] Eval inference
########################################
echo "========================================"
echo "[4/6] Eval inference"
echo "========================================"
echo "  MODEL_PATH=${MERGED_DIR}"
echo "  TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}"
echo "  TEST_SPLIT=${TEST_SPLIT}"

[[ -f "${MERGED_DIR}/config.json" ]] || die "MERGED_DIR/config.json missing."
[[ -d "${BASE_DATA_PATH}" ]] || die "BASE_DATA_PATH not found: ${BASE_DATA_PATH}"
if [[ "${REQUIRE_RRSISD_FILES:-1}" != "0" ]]; then
  [[ -f "${BASE_DATA_PATH}/rrsisd/refs(unc).p" ]] || die "Missing ${BASE_DATA_PATH}/rrsisd/refs(unc).p (set REQUIRE_RRSISD_FILES=0 to skip)"
  [[ -f "${BASE_DATA_PATH}/rrsisd/instances.json" ]] || die "Missing ${BASE_DATA_PATH}/rrsisd/instances.json (set REQUIRE_RRSISD_FILES=0 to skip)"
else
  info "REQUIRE_RRSISD_FILES=0 — skip refs/instances.json existence check."
fi

if [[ -e "${TEST_OUTPUT_DIR}" && "${KEEP_TEST_OUTPUT:-0}" == "1" ]]; then
  die "TEST_OUTPUT_DIR already exists: ${TEST_OUTPUT_DIR}. Unset KEEP_TEST_OUTPUT or pick a new TEST_OUTPUT_DIR."
fi
rm -rf "${TEST_OUTPUT_DIR}"

run_eval_infer "${MERGED_DIR}" "${TEST_OUTPUT_DIR}"

########################################
# [5/6] Metrics
########################################
echo "========================================"
echo "[5/6] Metrics"
echo "========================================"

run_eval_metrics "${TEST_OUTPUT_DIR}"

########################################
# [6/6] Done
########################################
echo "========================================"
echo "DONE MQ-v1 OHEM-BCE full pipeline"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  BEST_CHECKPOINT=${BEST_CHECKPOINT}"
echo "  MERGED_DIR=${MERGED_DIR}"
echo "  TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}"
echo "  MASK_CONFIG=${MASK_CONFIG}"
echo "  MAX_STEPS=${MAX_STEPS}"
echo "========================================"
}

########################################
# Global preflight + 7w only
########################################
if [[ $# -ge 1 && "${1}" != "7w" ]]; then
  die "run_mq_v1_0.5_full_pipeline.sh only supports 7w (got: ${1}). Use run_mq_v1_0.3_full_pipeline.sh for 28w/both."
fi

configure_mq_7w_profile

"${PYTHON}" -c "import transformers" 2>/dev/null || die "Python cannot import transformers (set PYTHON to your training env's interpreter)."

[[ -d "${MODEL_NAME_OR_PATH}" ]] || die "MODEL_NAME_OR_PATH not found: ${MODEL_NAME_OR_PATH}"
[[ -e "${VISION_TOWER}" ]] || die "VISION_TOWER not found: ${VISION_TOWER}"
[[ -f "${VISION_TOWER_MASK}" ]] || die "VISION_TOWER_MASK not found: ${VISION_TOWER_MASK}"
[[ -f "${MASK_CONFIG}" ]] || die "MASK_CONFIG not found: ${MASK_CONFIG}"
[[ -d "${BASE_DATA_PATH}" ]] || die "BASE_DATA_PATH not found: ${BASE_DATA_PATH}"
[[ -f "${MERGE_PY}" ]] || die "MERGE_PY not found: ${MERGE_PY}"
[[ -f "${EVAL_PY}" ]] || die "EVAL_PY not found: ${EVAL_PY}"
if [[ "${REQUIRE_EVAL_METRICS:-1}" != "0" ]]; then
  [[ -f "${EVAL_METRICS_SCRIPT}" ]] || die "EVAL_METRICS_SCRIPT not found: ${EVAL_METRICS_SCRIPT} (set REQUIRE_EVAL_METRICS=0 to skip metrics step)"
else
  [[ -f "${EVAL_METRICS_SCRIPT}" ]] || info "REQUIRE_EVAL_METRICS=0 — metrics script missing, [5/6] will be skipped."
fi

TRAIN_PY="${REPO_DIR}/segearth_r2/train/train.py"
grep -q "use_qmc" "${TRAIN_PY}" || die "train.py must define use_qmc (${TRAIN_PY})."

if [[ "${VERIFY_OHEM_YAML:-1}" != "0" ]]; then
  if echo "${MASK_CONFIG}" | grep -q "mq_v1_ohem"; then
    grep -qE "USE_OHEM_BCE:[[:space:]]*True" "${MASK_CONFIG}" \
      || die "MQ-v1 expects USE_OHEM_BCE: True in ${MASK_CONFIG} (set VERIFY_OHEM_YAML=0 to skip)."
    grep -qE "OHEM_RATIO:[[:space:]]*${OHEM_RATIO_TAG}" "${MASK_CONFIG}" \
      || die "MASK_CONFIG OHEM_RATIO must match OHEM_RATIO_TAG=${OHEM_RATIO_TAG} in ${MASK_CONFIG}"
  else
    info "MASK_CONFIG does not contain 'mq_v1_ohem' — VERIFY_OHEM_YAML skipped (e.g. baseline yaml)."
  fi
fi

echo "========================================"
echo "[PLAN] MQ-v1 OHEM ratio=${OHEM_RATIO_TAG}, 7w (MAX_STEPS=${MAX_STEPS})"
echo "  RUN_TAG=${RUN_TAG}"
echo "========================================"

run_mq_v1_pipeline_once

echo "========================================"
echo "ALL DONE MQ-v1 OHEM-BCE ratio=${OHEM_RATIO_TAG} (7w)"
echo "========================================"
