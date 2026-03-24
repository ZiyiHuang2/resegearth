export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
# ------ main-training ------
# 显存不足时可选择 zero2.json / zero3.json
# 用法示例：
#   BASE_DATA_PATH=/path/to/LaSeRS OUTPUT_DIR=output/train_2gpu INCLUDE_GPUS=localhost:0,1 bash scripts/train.sh
BASE_DATA_PATH=${BASE_DATA_PATH:-/data1/xzp/data}
OUTPUT_DIR=${OUTPUT_DIR:-output_folder}
INCLUDE_GPUS=${INCLUDE_GPUS:-localhost:0}
MASTER_PORT=${MASTER_PORT:-29500}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-scripts/zero3.json}

# data_ratio 在代码中会被解析为 int，必须是整数（如 1）
DATA_RATIO=${DATA_RATIO:-1}
SWITCH_BS=${SWITCH_BS:-4}
MAX_STEPS=${MAX_STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-1}
SAVE_STEPS=${SAVE_STEPS:-1000}
LOGGING_STEPS=${LOGGING_STEPS:-10}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
NUM_WORKERS=${NUM_WORKERS:-8}

set -euo pipefail

# 打印关键参数，便于排查环境变量是否生效
echo "[train.sh] BASE_DATA_PATH=${BASE_DATA_PATH}"
echo "[train.sh] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[train.sh] INCLUDE_GPUS=${INCLUDE_GPUS}"
echo "[train.sh] DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"

# 训练前检查 GPU slot 是否有效，避免 DeepSpeed 报 "No slot 'x'"
GPU_COUNT=$(python - <<'PY2'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print("unknown")
PY2
)
echo "[train.sh] GPU_COUNT=${GPU_COUNT}"

if [[ "${INCLUDE_GPUS}" == localhost:* ]] && [[ "${GPU_COUNT}" != "unknown" ]]; then
    SLOT_LIST=${INCLUDE_GPUS#localhost:}
    IFS=',' read -r -a SLOT_ARRAY <<< "${SLOT_LIST}"
    for SLOT in "${SLOT_ARRAY[@]}"; do
        if ! [[ "${SLOT}" =~ ^[0-9]+$ ]]; then
            echo "[train.sh][ERROR] INCLUDE_GPUS contains non-numeric slot: ${SLOT}" >&2
            exit 1
        fi
        if (( SLOT >= GPU_COUNT )); then
            echo "[train.sh][ERROR] INCLUDE_GPUS=${INCLUDE_GPUS} is invalid for GPU_COUNT=${GPU_COUNT}." >&2
            echo "[train.sh][HINT] Set INCLUDE_GPUS to valid slots, e.g. localhost:0 or localhost:0,1" >&2
            exit 1
        fi
    done
fi

deepspeed --master_port="${MASTER_PORT}" --include "${INCLUDE_GPUS}" segearth_r2/train/train.py \
    --model_name_or_path "pretrained_model/mllm/Mipha-3B" \
    #--vision_tower "pretrained_model/CLIP/siglip-so400m-patch14-384" \
    --vision_tower "pretrained_model/CLIP/siglip2-so400m-patch14-384" \
    --vision_tower_mask "pretrained_model/mask2former/model_final_54b88a.pkl" \
    --base_data_path "${BASE_DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS}" \
    --bf16 True \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps "${LOGGING_STEPS}" \
    --tf32 False \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing False \
    --dataloader_num_workers "${NUM_WORKERS}" \
    --lora_r 4 \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --mask_config "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml" \
    --data_ratio "${DATA_RATIO}" \
    --switch_bs "${SWITCH_BS}"