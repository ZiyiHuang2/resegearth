#!/usr/bin/env bash
# 当 GPU 2 显存占用（nvidia-smi memory.used）低于阈值时，在「启动本脚本时」终端所在目录下，
# 激活 conda 环境 reseg，并执行 bash run_abc_decoder_attn_bias_7w.sh
#
# 用法（请先 cd 到含 run_abc_decoder_attn_bias_7w.sh 的目录，一般为本目录）：
#   bash wait_gpu2_lt_1g_then_run_abc_decoder_attn_bias_7w.sh
#
# 可选环境变量：
#   GPU_INDEX=2                   # 监视的 GPU 编号（默认 2）
#   MEM_THRESHOLD_MB=1024         # 占用低于该值（MiB）才启动，默认约 1GiB
#   POLL_INTERVAL_SEC=30          # 轮询间隔（秒）
#   ABC_SCRIPT=run_abc_decoder_attn_bias_7w.sh
#   CONDA_ENV=reseg
#
set -uo pipefail

GPU_INDEX="${GPU_INDEX:-2}"
MEM_THRESHOLD_MB="${MEM_THRESHOLD_MB:-1024}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-30}"
ABC_SCRIPT="${ABC_SCRIPT:-run_abc_decoder_attn_bias_7w.sh}"
CONDA_ENV="${CONDA_ENV:-reseg}"

# 记录「当前终端目录」：以执行本脚本时的 cwd 为准
START_DIR="$(pwd -P)"

echo "========================================"
echo "[wait] START_DIR=${START_DIR}"
echo "[wait] GPU_INDEX=${GPU_INDEX}  MEM_THRESHOLD_MB=${MEM_THRESHOLD_MB} (< 则启动)"
echo "[wait] ABC_SCRIPT=${ABC_SCRIPT}  CONDA_ENV=${CONDA_ENV}"
echo "========================================"

if [[ ! -f "${START_DIR}/${ABC_SCRIPT}" ]]; then
  echo "[ERROR] 在 START_DIR 下找不到脚本: ${START_DIR}/${ABC_SCRIPT}"
  echo "        请先 cd 到 resegearth+tgi（或放置该 sh 的目录）再运行本 wait 脚本。"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] 未找到 nvidia-smi"
  exit 1
fi

query_used_mb () {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_INDEX}" 2>/dev/null | head -n 1 | tr -d '[:space:]'
}

while true; do
  used="$(query_used_mb)"
  if [[ -z "${used}" ]] || ! [[ "${used}" =~ ^[0-9]+$ ]]; then
    echo "[WARN] 无法解析 GPU ${GPU_INDEX} memory.used（got '${used}'），${POLL_INTERVAL_SEC}s 后重试"
    sleep "${POLL_INTERVAL_SEC}"
    continue
  fi
  if (( used < MEM_THRESHOLD_MB )); then
    echo "[OK] GPU ${GPU_INDEX} memory.used=${used} MiB < ${MEM_THRESHOLD_MB} MiB — 开始执行训练脚本"
    break
  fi
  echo "[WAIT] GPU ${GPU_INDEX} memory.used=${used} MiB（需 < ${MEM_THRESHOLD_MB}），${POLL_INTERVAL_SEC}s 后再查"
  sleep "${POLL_INTERVAL_SEC}"
done

cd "${START_DIR}" || {
  echo "[ERROR] cd 失败: ${START_DIR}"
  exit 1
}

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
fi
if ! conda activate "${CONDA_ENV}" 2>/dev/null; then
  echo "[ERROR] conda activate ${CONDA_ENV} 失败（请先确保该环境存在）"
  exit 1
fi

exec bash "${ABC_SCRIPT}"
