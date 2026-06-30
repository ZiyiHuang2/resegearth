#!/usr/bin/env bash
# Full = Enhanced TG-Swin encoder grounding + SET++ decoder set consistency
# (coarse evidence / DR-EWTI disabled via ours_full_enhanced_tgswin_setpp.yaml)
set -euo pipefail

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

deepspeed --master_port="${MASTER_PORT:-29500}" --include "${GPU_SLOT:-localhost:0}" segearth_r2/train/train.py \
    --model_name_or_path "${MODEL_PATH:-pretrained_model/mllm/Mipha-3B}" \
    --vision_tower "${VISION_TOWER:-pretrained_model/CLIP/siglip-so400m-patch14-384}" \
    --vision_tower_mask "${VISION_TOWER_MASK:-pretrained_model/mask2former/model_final_54b88a.pkl}" \
    --base_data_path "${BASE_DATA_PATH:-/data1/xzp/data}" \
    --output_dir "${OUTPUT_DIR:-output_full_enhanced_tgswin_setpp}" \
    --max_steps "${MAX_STEPS:-5000}" \
    --per_device_train_batch_size "${BATCH_SIZE:-1}" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS:-1000}" \
    --bf16 True \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
    --learning_rate "${LR:-5e-5}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-False}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}" \
    --lora_r "${LORA_R:-4}" \
    --deepspeed "${DEEPSPEED:-scripts/zero3.json}" \
    --mask_config 'segearth_r2/model/mask_decoder/mask_config/ours_full_enhanced_tgswin_setpp.yaml' \
    --data_ratio '1' \
    --switch_bs "${SWITCH_BS:-4}" \
    --dataset_name "${DATASET_NAME:-rrsisd}" \
    --setpp_enable True \
    --setpp_csqr_enable True \
    --setpp_closed_loop True \
    --setpp_regroup_set_loss True \
    "$@"
