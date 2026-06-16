#!/usr/bin/env bash
set -euo pipefail

# Minimal BQER training entry (override paths as needed)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

deepspeed --master_port=29500 --include localhost:0 segearth_r2/train/train.py \
  --model_name_or_path pretrained_model/mllm/Mipha-3B \
  --vision_tower pretrained_model/CLIP/siglip-so400m-patch14-384 \
  --vision_tower_mask pretrained_model/mask2former/model_final_54b88a.pkl \
  --mask_config segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
  --base_data_path /path/to/data \
  --dataset_name rrsisd \
  --output_dir output/bqer/run1 \
  --max_steps 70000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --save_steps 2000 \
  --bf16 True \
  --learning_rate 1e-4 \
  --lora_r 8 \
  --deepspeed scripts/zero1.json \
  --data_ratio 1 \
  --switch_bs 4 \
  --bqer_enable True \
  --bqer_k_layers 2 \
  --bqer_boundary_weight 0.4 \
  --bqer_query_consistency_weight 0.2 \
  --bqer_small_object_weight 1.8 \
  --bqer_small_object_percentile 30.0 \
  --bqer_mod_alpha 0.1 \
  --bqer_token_drift_weight 0.02 \
  --bqer_q2b_detach_query False
