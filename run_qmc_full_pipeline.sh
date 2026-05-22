#!/usr/bin/env bash
# QMC-GT v1：一次完成 train → best checkpoint → merge/export → verify → eval → metrics
# 默认跑 7w；如需 28w：MAX_STEPS=280000 bash run_qmc_full_pipeline.sh
#
# Merge / 校验相关环境变量（按需放宽，默认仍偏「正式实验」）：
#   MERGE_KEEP_EXISTING=1     — 若 MERGED_DIR 已存在则报错退出（默认会由 merge_lora 内 rm -rf 覆盖）
#   REQUIRE_DEEPSPEED_STATE=0 — checkpoint 无 mp_rank_*_model_states.pt 时仅警告，不中断
#   EXPECT_QMC=0            — checkpoint / merged 加载校验不要求 qmc_* 键（baseline merge 时用）
#   ALLOW_BEST_CKPT_OUTSIDE_OUTPUT=1 — best_model_checkpoint 可指向 OUTPUT_DIR 外的绝对路径
#   REQUIRE_EVAL_METRICS=0  — 无 eval_val_metrics.py 时跳过 [5/6]
#   REQUIRE_RRSISD_FILES=0  — 不强制检查 refs(unc).p / instances.json
#   KEEP_TEST_OUTPUT=1      — 若 TEST_OUTPUT_DIR 已存在则报错（默认会先 rm -rf 再推理）
#   PYTHON=…/envs/reseg/bin/python — merge/eval/内嵌脚本与 preflight 的解释器（默认 python；训练仍由 deepspeed 在 PATH 上解析）

set -euo pipefail

########################################
# Basic env
########################################
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT="${WANDB_PROJECT:-segearth-qmc}"
export WANDB_INIT_TIMEOUT=300

# 如果你想固定单卡，可以设置：
# GPU_SLOT=localhost:0 GPU_ID=0 bash run_qmc_full_pipeline.sh
unset CUDA_VISIBLE_DEVICES

GPU_SLOT="${GPU_SLOT:-localhost:1}"
GPU_ID="${GPU_ID:-1}"
MASTER_PORT="${MASTER_PORT:-29530}"

########################################
# Paths
########################################
RESEG_ROOT="${RESEG_ROOT:-/home/wangchengjun/huangziyi/reseg}"
REPO_DIR="${REPO_DIR:-${RESEG_ROOT}/segearth+maskquality}"
# 全流程（merge / eval / 内嵌脚本）与 preflight 使用同一解释器；若默认 python 非训练环境，请显式设置，例如:
#   PYTHON=/home/wangchengjun/miniconda3/envs/reseg/bin/python bash run_qmc_full_pipeline.sh
PYTHON="${PYTHON:-python}"
cd "${REPO_DIR}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mllm/Mipha-3B}"
VISION_TOWER="${VISION_TOWER:-/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384}"
VISION_TOWER_MASK="${VISION_TOWER_MASK:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl}"
MASK_CONFIG="${MASK_CONFIG:-segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml}"

BASE_DATA_PATH="${BASE_DATA_PATH:-/home/wangchengjun/huangziyi/data/RRSISD}"
DATASET_NAME="${DATASET_NAME:-rrsisd}"
TEST_SPLIT="${TEST_SPLIT:-test}"

########################################
# Experiment config
########################################
MAX_STEPS="${MAX_STEPS:-70000}"
SEED="${SEED:-42}"
DATA_SEED="${DATA_SEED:-42}"

QMC_LOSS_WEIGHT="${QMC_LOSS_WEIGHT:-0.05}"
QMC_MIN_MASK_SUM="${QMC_MIN_MASK_SUM:-1e-6}"
QMC_DETACH_VISUAL="${QMC_DETACH_VISUAL:-False}"

RUN_TAG="${RUN_TAG:-qmc_gt_v1_w${QMC_LOSS_WEIGHT}_${MAX_STEPS}}"

OUTPUT_DIR="${OUTPUT_DIR:-${RESEG_ROOT}/output/qmc/${RUN_TAG}}"
MERGED_DIR="${MERGED_DIR:-${RESEG_ROOT}/output/qmc/${RUN_TAG}/merged_best}"
TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-${RESEG_ROOT}/output/qmc/${RUN_TAG}/test_results}"

export WANDB_NAME="${WANDB_NAME:-rrsisd_${RUN_TAG}}"

########################################
# Training hyperparams
########################################
REPORT_TO="${REPORT_TO:-wandb}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

# train.py 会把 eval_steps 覆盖成 save_steps，所以二者必须一起规划。
# 7w 建议 2000；28w 可设 5000。
SAVE_STEPS="${SAVE_STEPS:-2000}"
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
EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-segearth-eval-qmc}"
EVAL_METRICS_RUN_NAME="${EVAL_METRICS_RUN_NAME:-rrsisd_${RUN_TAG}}"

# QMC 权重校验：1=merge 前要求 checkpoint DeepSpeed state 中含 qmc_*（本脚本固定训练 QMC，一般保持 1）。
# 若你只做 merge 调试或 checkpoint 来自无 QMC 的旧实验，可设 EXPECT_QMC=0。
export EXPECT_QMC="${EXPECT_QMC:-1}"

# 1=merge 前必须存在 DeepSpeed mp_rank_*_model_states.pt；0=仅警告不中断（不推荐正式实验）。
export REQUIRE_DEEPSPEED_STATE="${REQUIRE_DEEPSPEED_STATE:-1}"

# 1=强制存在 eval_val_metrics.py；0=跳过 [5/6] 指标步骤（仅跑通 train→merge→infer）。
export REQUIRE_EVAL_METRICS="${REQUIRE_EVAL_METRICS:-1}"

# 1=允许 trainer_state 里的 best_model_checkpoint 落在 OUTPUT_DIR 外（绝对路径仍须为目录）。
export ALLOW_BEST_CKPT_OUTSIDE_OUTPUT="${ALLOW_BEST_CKPT_OUTSIDE_OUTPUT:-0}"

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

########################################
# Pick best checkpoint
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

# 1. user selected checkpoint
if sel:
    if not os.path.isdir(sel):
        print(f"ERROR:user_not_dir:{sel}", file=sys.stderr)
        sys.exit(2)
    print(f"OK:user:{os.path.abspath(sel)}")
    sys.exit(0)

# 2. root trainer_state best_model_checkpoint
ts0 = read_json(os.path.join(out_root, "trainer_state.json"))
if isinstance(ts0, dict):
    b = ts0.get("best_model_checkpoint")
    if isinstance(b, str) and b.strip():
        r = resolve_best_path(b, out_root, repo_root)
        if r and os.path.isdir(r) and (allow_ext or r == out_root or r.startswith(out_root + os.sep)):
            print(f"OK:root_state:{r}")
            sys.exit(0)

# 3. nested checkpoint trainer_state best_model_checkpoint
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

# 4. metric candidates
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
  EXPECT_QMC_FLAG="${EXPECT_QMC:-1}"

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

# 兼容 builder.py 可能读取的字段；没有用到也没关系。
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

expect_qmc = os.environ.get("EXPECT_QMC", "1").strip() in ("1", "true", "True", "yes", "YES")
if expect_qmc:
    # eval_seg 不直接调用 qmc projector，但完整模型目录最好保留。
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
import json, os, math
p = "${out_json}"
h = json.load(open(p, "r", encoding="utf-8")).get("log_history", [])
last_eval = None
last_loss = None
last_qmc = None
for row in h:
    if "eval_score" in row:
        last_eval = row
    if "loss" in row:
        last_loss = row
    if "loss_qmc" in row:
        last_qmc = row

print("[last_loss]", last_loss)
print("[last_qmc]", last_qmc)
print("[last_eval]", last_eval)

if last_eval is None:
    print("[WARN] No eval_* record found.")
else:
    print("[OK] eval_giou=", last_eval.get("eval_giou"))
    print("[OK] eval_ciou=", last_eval.get("eval_ciou"))
    print("[OK] eval_score=", last_eval.get("eval_score"))
PY
}

########################################
# Preflight
########################################
echo "========================================"
echo "[CONFIG] QMC-GT v1 full pipeline"
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
echo "  USE_QMC=True"
echo "  QMC_LOSS_WEIGHT=${QMC_LOSS_WEIGHT}"
echo "  QMC_DETACH_VISUAL=${QMC_DETACH_VISUAL}"
echo "========================================"

"${PYTHON}" -c "import transformers" 2>/dev/null || die "Python cannot import transformers (set PYTHON to your training env's interpreter)."

[[ -d "${MODEL_NAME_OR_PATH}" ]] || die "MODEL_NAME_OR_PATH not found: ${MODEL_NAME_OR_PATH}"
# vision tower 可能是 HF 缓存目录或单文件快照，用 -e 比仅 -d 更宽松。
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

# train.py 在打印 --help 前会 import deepspeed/peft 等；用「默认 python」时导入失败会导致 help 为空而误报。
# 此处改为检查源码中的 TrainingArguments 字段；若仍希望校验 CLI，可设 TRAIN_HELP_CHECK=1 且保证 PYTHON 为训练环境。
TRAIN_PY="${REPO_DIR}/segearth_r2/train/train.py"
grep -q "use_qmc" "${TRAIN_PY}" || die "train.py must define use_qmc (${TRAIN_PY})."
grep -q "qmc_loss_weight" "${TRAIN_PY}" || die "train.py must define qmc_loss_weight (${TRAIN_PY})."
if [[ "${TRAIN_HELP_CHECK:-0}" == "1" ]]; then
  HELP_OUT="$("${PYTHON}" "${TRAIN_PY}" --help 2>&1 || true)"
  echo "${HELP_OUT}" | grep -q "use_qmc" || die "train.py --help must contain use_qmc (TRAIN_HELP_CHECK=1; check PYTHON / imports)."
  echo "${HELP_OUT}" | grep -q "qmc_loss_weight" || die "train.py --help must contain qmc_loss_weight (TRAIN_HELP_CHECK=1)."
fi

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
echo "[1/6] Training QMC-GT v1 (${MAX_STEPS} steps)"
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
  --use_qmc True \
  --qmc_loss_weight "${QMC_LOSS_WEIGHT}" \
  --qmc_min_mask_sum "${QMC_MIN_MASK_SUM}" \
  --qmc_detach_visual "${QMC_DETACH_VISUAL}"

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

# merge_lora 内部会对 save_dir 执行 rm -rf；此处不再因目录已存在而强制要求 OVERWRITE_MERGE。
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

# 默认可覆盖已有推理输出；若需保留历史结果设 KEEP_TEST_OUTPUT=1。
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
echo "DONE QMC-GT v1 full pipeline"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  BEST_CHECKPOINT=${BEST_CHECKPOINT}"
echo "  MERGED_DIR=${MERGED_DIR}"
echo "  TEST_OUTPUT_DIR=${TEST_OUTPUT_DIR}"
echo "  QMC_LOSS_WEIGHT=${QMC_LOSS_WEIGHT}"
echo "  MAX_STEPS=${MAX_STEPS}"
echo "========================================"