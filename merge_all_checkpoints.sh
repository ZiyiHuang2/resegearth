#!/bin/bash

# ==========================================
# 配置区域：根据你的实际路径修改
# ==========================================

# 1. 包含所有 checkpoint 的父目录
BASE_DIR="/home/wangchengjun/huangziyi/reseg/output/RRSISD/phrase_5w_r8_2+all"

# 2. 合并脚本的路径
MERGE_SCRIPT="/home/wangchengjun/huangziyi/reseg/segearth+ov/segearth_r2/train/merge_lora_weights_and_save_hf_model.py"

# 3. 公共参数（Vision Tower, Mask Config 等）
VISION_TOWER="/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384"
VISION_TOWER_MASK="/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
LORA_R=8

# ==========================================
# 脚本逻辑：查找并合并
# ==========================================

echo "🔍 正在搜索目录: ${BASE_DIR}"
echo "🔍 搜索前缀: checkpoint-*"

# 查找所有以 checkpoint- 开头的目录
CHECKPOINT_DIRS=$(find "${BASE_DIR}" -maxdepth 1 -type d -name "checkpoint-*")

if [ -z "$CHECKPOINT_DIRS" ]; then
    echo "❌ 错误：在 ${BASE_DIR} 下未找到任何 checkpoint-* 目录"
    exit 1
fi

# 遍历每个 checkpoint 目录
for CKPT_PATH in $CHECKPOINT_DIRS; do
    # 提取 checkpoint 名称 (例如: checkpoint-33000)
    CKPT_NAME=$(basename "$CKPT_PATH")
    
    # 定义保存路径 (例如: merged_model-33000)
    SAVE_PATH="${BASE_DIR}/merged_${CKPT_NAME}"

    echo ""
    echo "=========================================="
    echo "🚀 开始合并: ${CKPT_NAME}"
    echo "📂 来源: ${CKPT_PATH}"
    echo "💾 保存至: ${SAVE_PATH}"
    echo "=========================================="

    # 执行合并命令
    python ${MERGE_SCRIPT} \
        --model_path "${CKPT_PATH}" \
        --vision_tower "${VISION_TOWER}" \
        --vision_tower_mask "${VISION_TOWER_MASK}" \
        --mask_config "${MASK_CONFIG}" \
        --save_path "${SAVE_PATH}" \
        --lora_r ${LORA_R}

    # 检查是否成功
    if [ $? -eq 0 ]; then
        echo "✅ ${CKPT_NAME} 合并成功！"
    else
        echo "❌ ${CKPT_NAME} 合并失败！请检查日志。"
    fi
done

echo ""
echo "🎉 所有 Checkpoint 处理完毕！"0