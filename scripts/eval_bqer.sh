#!/usr/bin/env bash
set -euo pipefail

# Minimal BQER eval entry (override paths as needed)
python segearth_r2/eval/eval.py \
  --base_data_path /path/to/data \
  --vision_tower pretrained_model/CLIP/siglip-so400m-patch14-384 \
  --vision_tower_mask pretrained_model/mask2former/model_final_54b88a.pkl \
  --mask_config segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml \
  --model_path /path/to/merged_model \
  --output_dir /path/to/eval_output \
  --dataset_name rrsisd \
  --split test \
  --eval_batch_size 1 \
  --zip_results False \
  --bqer_enable True \
  --bqer_k_layers 2 \
  --bqer_mod_alpha 0.1 \
  --bqer_token_drift_weight 0.02 \
  --bqer_q2b_detach_query False
