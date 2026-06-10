#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_DIR="/home/wangchengjun/huangziyi/reseg/segearth+base"
cd "${REPO_DIR}"
PYTHON="/home/wangchengjun/miniconda3/envs/reseg/bin/python3"
MODEL_PATH="/home/wangchengjun/huangziyi/reseg/output/itaa/coarse_refine_layer2_w005_unfreezePD-5w/merged_model"
VISION_TOWER="/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
VISION_TOWER_MASK="/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
SMOKE_ROOT="/tmp/segearth_smoke_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SMOKE_ROOT}"
exec > >(tee "${SMOKE_ROOT}/smoke.log") 2>&1

run_eval () {
  local ds_name="$1" split="$2" data_path="$3"
  local out_dir="${SMOKE_ROOT}/eval_${ds_name}_${split}"
  echo "[eval] ${ds_name}/${split}"
  "${PYTHON}" -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
  "${PYTHON}" segearth_r2/eval/eval.py \
    --base_data_path "${data_path}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_mask "${VISION_TOWER_MASK}" \
    --mask_config "${MASK_CONFIG}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${out_dir}" \
    --dataset_name "${ds_name}" \
    --split "${split}" \
    --eval_batch_size 1 \
    --max_eval_samples 1 \
    --dataloader_num_workers 0 \
    --zip_results False
  test "$(find "${out_dir}" -maxdepth 1 -name '*.tif' | wc -l)" -ge 1
}

run_eval refsegrs val /home/wangchengjun/huangziyi/data/RefSegRS
run_eval risbench test /home/wangchengjun/huangziyi/data/RISBench
run_eval rrsisd val /home/wangchengjun/huangziyi/data/RRSISD
OUT_LASERS="${SMOKE_ROOT}/eval_lasers"
mkdir -p "${OUT_LASERS}"
"${PYTHON}" segearth_r2/eval/eval.py \
  --base_data_path /home/wangchengjun/huangziyi/data/LaSeRS \
  --vision_tower "${VISION_TOWER}" \
  --vision_tower_mask "${VISION_TOWER_MASK}" \
  --mask_config "${MASK_CONFIG}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUT_LASERS}" \
  --dataset_name lasers --split val \
  --eval_batch_size 1 --max_eval_samples 1 \
  --dataloader_num_workers 0 --zip_results False
test "$(find "${OUT_LASERS}" -maxdepth 1 -name '*.tif' | wc -l)" -ge 1
echo "EVAL SMOKE OK: ${SMOKE_ROOT}"
