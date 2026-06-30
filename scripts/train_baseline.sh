#!/usr/bin/env bash
# Baseline ablation: no TG-Swin, no SET++ auxiliary losses / CSQR.
set -euo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

deepspeed --master_port="${MASTER_PORT:-29501}" --include "${GPU_SLOT:-localhost:0}" segearth_r2/train/train.py \
    --model_name_or_path "${MODEL_PATH:-pretrained_model/mllm/Mipha-3B}" \
    --vision_tower "${VISION_TOWER:-pretrained_model/CLIP/siglip-so400m-patch14-384}" \
    --vision_tower_mask "${VISION_TOWER_MASK:-pretrained_model/mask2former/model_final_54b88a.pkl}" \
    --base_data_path "${BASE_DATA_PATH:-/data1/xzp/data}" \
    --output_dir "${OUTPUT_DIR:-output/ablation/baseline}" \
    --max_steps "${MAX_STEPS:-5000}" \
    --per_device_train_batch_size "${BATCH_SIZE:-2}" \
    --gradient_accumulation_steps "${GRAD_ACC:-1}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS:-1000}" \
    --bf16 True \
    --save_total_limit 2 \
    --learning_rate "${LR:-5e-5}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lora_r "${LORA_R:-8}" \
    --deepspeed scripts/zero1.json \
    --mask_config 'segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml' \
    --data_ratio '1' \
    --switch_bs "${SWITCH_BS:-4}" \
    --dataset_name "${DATASET_NAME:-rrsisd}" \
    --setpp_enable False \
    --setpp_csqr_enable False \
    --setpp_closed_loop False \
    --setpp_regroup_set_loss False \
    "$@"
