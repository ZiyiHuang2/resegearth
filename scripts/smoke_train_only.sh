#!/usr/bin/env bash
# Legacy: points to segearth+base; for Test 4 use scripts/test4_smoke_train.sh
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_DIR="/home/wangchengjun/huangziyi/reseg/segearth+base"
cd "${REPO_DIR}"
DEEPSPEED="/home/wangchengjun/miniconda3/envs/reseg/bin/deepspeed"
export PATH="/home/wangchengjun/miniconda3/envs/reseg/bin:${PATH}"
MODEL_BASE="/home/wangchengjun/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
VISION_TOWER="/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
SMOKE_ROOT="/tmp/segearth_smoke_train_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SMOKE_ROOT}"
exec > >(tee "${SMOKE_ROOT}/smoke.log") 2>&1

run_train () {
  local ds="$1" path="$2"
  local out="${SMOKE_ROOT}/train_${ds}"
  echo "[train] ${ds} 1 step"
  "${DEEPSPEED}" --include localhost:0 --master_port=29611 \
    segearth_r2/train/train.py \
    --model_name_or_path "${MODEL_BASE}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --base_data_path "${path}" \
    --dataset_name "${ds}" \
    --output_dir "${out}" \
    --max_steps 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy no \
    --logging_steps 1 \
    --bf16 True \
    --dataloader_num_workers 2 \
    --deepspeed scripts/zero1.json
}

run_train refsegrs /home/wangchengjun/huangziyi/data/RefSegRS
run_train risbench /home/wangchengjun/huangziyi/data/RISBench
run_train rrsisd /home/wangchengjun/huangziyi/data/RRSISD
echo "TRAIN SMOKE OK: ${SMOKE_ROOT}"
