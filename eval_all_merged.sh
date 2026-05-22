#!/bin/bash

# ==========================================
# 配置区域：根据你的实际路径修改
# ==========================================

# 1. 包含 merged 模型的父目录
BASE_MERGED_DIR="/home/wangchengjun/huangziyi/reseg/output/RRSISD/phrase_5w_r8_2+all"

# 2. 评估结果（test）的父目录
BASE_TEST_DIR="/home/wangchengjun/huangziyi/reseg/output/RRSISD/phrase_5w_r8_2+all"

# 3. Eval 脚本路径
EVAL_SCRIPT="/home/wangchengjun/huangziyi/reseg/segearth+ov/segearth_r2/eval/eval.py"

# 4. 公共参数
BASE_DATA_PATH="/home/wangchengjun/huangziyi/data/RRSISD"
VISION_TOWER="/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384"
VISION_TOWER_MASK="/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"

# ==========================================
# 脚本逻辑：扫描并评估
# ==========================================

echo "🔍 正在扫描 Merged 模型目录: ${BASE_MERGED_DIR}"
echo ""

# 查找所有 merged_checkpoint-* 目录
MERGED_DIRS=$(find "${BASE_MERGED_DIR}" -maxdepth 1 -type d -name "merged_checkpoint-*")

if [ -z "$MERGED_DIRS" ]; then
    echo "❌ 错误：未找到任何 merged_checkpoint-* 目录"
    exit 1
fi

# 遍历每个 merged 目录
for MERGED_PATH in $MERGED_DIRS; do
    # 提取 checkpoint 名称 (例如: merged_checkpoint-37000)
    MERGED_NAME=$(basename "$MERGED_PATH")
    
    # 提取 step 数字 (例如: 37000)
    STEP_NUM=$(echo "$MERGED_NAME" | sed 's/merged_checkpoint-//')
    
    # 构造对应的 test 目录路径
    TEST_PATH="${BASE_TEST_DIR}/test_results-${STEP_NUM}"

    echo "=========================================="
    echo "🚀 开始评估: ${MERGED_NAME}"
    echo "📂 模型路径: ${MERGED_PATH}"
    echo "📊 结果保存至: ${TEST_PATH}"
    echo "=========================================="

    # 如果 test 目录不存在，则创建
    mkdir -p "${TEST_PATH}"

    # 执行评估命令
    NCCL_P2P_DISABLE=1 \
    NCCL_IB_DISABLE=1 \
    CUDA_VISIBLE_DEVICES=2 \
    python ${EVAL_SCRIPT} \
        --base_data_path "${BASE_DATA_PATH}" \
        --vision_tower "${VISION_TOWER}" \
        --vision_tower_mask "${VISION_TOWER_MASK}" \
        --mask_config "${MASK_CONFIG}" \
        --model_path "${MERGED_PATH}" \
        --output_dir "${TEST_PATH}" \
        --dataset_name rrsisd \
        --split test \
        --eval_batch_size 1 \
        --zip_results False

    # 检查是否成功
    if [ $? -eq 0 ]; then
        echo "✅ ${MERGED_NAME} 评估完成！"
    else
        echo "❌ ${MERGED_NAME} 评估失败！请检查日志。"
    fi

    echo ""
done

echo "🎉 所有 Merged 模型评估完毕！"