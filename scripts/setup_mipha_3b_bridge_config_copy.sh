#!/usr/bin/env bash
# 在本地生成一份「与 Mipha-3B 权重硬链接相同、仅 config.json 独立且打开 Bridge」的模型目录，
# 供 bridge_short.sh / 训练脚本的 MODEL_NAME_OR_PATH 使用，不改动原始 Mipha-3B。
#
# 用法：
#   ./scripts/setup_mipha_3b_bridge_config_copy.sh
#   SRC=/path/to/Mipha-3B DST=/path/to/Mipha-3B_bridge_config ./scripts/setup_mipha_3b_bridge_config_copy.sh
#
set -euo pipefail

SRC="${SRC:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mllm/Mipha-3B}"
DST="${DST:-/home/wangchengjun/huangziyi/reseg/pretrained_model/mllm/Mipha-3B_bridge_config}"

if [[ ! -d "${SRC}" ]]; then
  echo "[ERROR] 源目录不存在: ${SRC}"
  exit 1
fi

if [[ ! -f "${SRC}/config.json" ]]; then
  echo "[ERROR] 源目录缺少 config.json: ${SRC}/config.json"
  exit 1
fi

if [[ -e "${DST}" ]]; then
  if python3 - <<PY
import json
import sys
p = "${DST}/config.json"
try:
    with open(p, "r", encoding="utf-8") as f:
        c = json.load(f)
    if c.get("use_seg_query_feature_bridge") is True:
        sys.exit(0)
except Exception:
    sys.exit(1)
sys.exit(1)
PY
  then
    echo "[OK] 已存在且 use_seg_query_feature_bridge=true，跳过: ${DST}"
    echo "MODEL_NAME_OR_PATH=${DST}"
    exit 0
  fi
  echo "[ERROR] 目标已存在但不是可用的 Bridge config 副本: ${DST}"
  echo "        请删除后重试，或换一个 DST 环境变量。"
  exit 1
fi

echo "[INFO] 硬链接副本（不占双倍权重空间）: ${SRC} -> ${DST}"
cp -al "${SRC}" "${DST}"

# 断开 config.json 硬链接，避免改 config 时动到原始 Mipha
rm -f "${DST}/config.json"
cp -a "${SRC}/config.json" "${DST}/config.json"

python3 - <<PY
import json
from pathlib import Path
p = Path("${DST}") / "config.json"
with p.open("r", encoding="utf-8") as f:
    cfg = json.load(f)
cfg["use_seg_query_feature_bridge"] = True
with p.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
    f.write("\n")
print("[OK] 已写入", p, "use_seg_query_feature_bridge=true")
PY

echo "[OK] 模型目录: ${DST}"
echo "请在训练脚本中设置: MODEL_NAME_OR_PATH=\"${DST}\""
