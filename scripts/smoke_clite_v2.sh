#!/usr/bin/env bash
# C-lite-v2 smoke: verify train entry, merge args, dataset/collator, forward/backward, eval_seg.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_DIR="/root/rivermind-data/huangziyi/reseg/segearth+set"
PYTHON="${PYTHON:-/root/rivermind-data/miniconda3/envs/reseg/bin/python}"
DEEPSPEED="${DEEPSPEED:-/root/rivermind-data/miniconda3/envs/reseg/bin/deepspeed}"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

MODEL_BASE="/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
VISION_TOWER="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384"
VISION_TOWER_MASK="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
MASK_CONFIG="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
LASERS_DATA_PATH="/root/rivermind-data/huangziyi/data/LaSeRS"
CATEGORY_VOCAB="segearth_r2/model/lasers_category_vocab.json"

SMOKE_ROOT="/tmp/segearth_smoke_clite_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SMOKE_ROOT}"
exec > >(tee "${SMOKE_ROOT}/smoke.log") 2>&1

echo "========== C-lite-v2 smoke test =========="
echo "Log: ${SMOKE_ROOT}/smoke.log"

[[ -x "${PYTHON}" ]] || { echo "[ERROR] python not found: ${PYTHON}"; exit 1; }
[[ -x "${DEEPSPEED}" ]] || { echo "[ERROR] deepspeed not found: ${DEEPSPEED}"; exit 1; }
[[ -f "${CATEGORY_VOCAB}" ]] || { echo "[ERROR] vocab not found: ${CATEGORY_VOCAB}"; exit 1; }

echo "[1/5] train.py --help (C-lite-v2 flags present)"
"${PYTHON}" segearth_r2/train/train.py --help | grep -q use_explicit_set_token
"${PYTHON}" segearth_r2/train/train.py --help | grep -q q_set_fusion_hidden
echo "[OK] train entry parses"

echo "[2/5] merge.py accepts --use_explicit_set_token"
"${PYTHON}" segearth_r2/train/merge_lora_weights_and_save_hf_model.py --help | grep -q use_explicit_set_token
echo "[OK] merge argparse"

echo "[3/5] dataset/collator: [SET] auto-insert + SET_token_embedding_indices"
"${PYTHON}" - <<PY
import sys
sys.path.insert(0, ".")
from transformers import AutoTokenizer, SiglipImageProcessor
from segearth_r2.datasets.dataset import LaSeRSDataset, DataCollatorForCOCODatasetV2

class DummyArgs:
    base_data_path = "${LASERS_DATA_PATH}"
    dataset_name = "lasers"
    is_multimodal = True
    version = "llava_phi"
    use_explicit_set_token = True
    vision_tower = "${VISION_TOWER}"
    mm_use_im_start_end = False
    mm_use_im_patch_token = False
    lasers_category_vocab_path = "${CATEGORY_VOCAB}"

tokenizer = AutoTokenizer.from_pretrained("${MODEL_BASE}", use_fast=False)
tokenizer.add_tokens("[SEG]")
tokenizer.add_tokens("[SET]")

data_args = DummyArgs()
dataset = LaSeRSDataset(
    base_data_path=data_args.base_data_path,
    tokenizer=tokenizer,
    data_args=data_args,
    split="train_data.json",
)
sample = dataset[0]
set_id = tokenizer.convert_tokens_to_ids("[SET]")
assert (sample["input_ids"] == set_id).any(), "missing [SET] in input_ids"
assert "SET_token_embedding_indices" in sample, "missing SET_token_embedding_indices"

clip_image_processor = SiglipImageProcessor.from_pretrained(data_args.vision_tower)
collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_image_processor)
batch = collator([dataset[0], dataset[1]])
assert "SET_token_embedding_indices" in batch
assert batch["SET_token_embedding_indices"].shape == batch["SEG_token_embedding_indices"].shape
print("[OK] dataset/collator")
PY

echo "[4/5] q_set_fusion: zero residual at init + learnable explicit path"
"${PYTHON}" - <<PY
import sys
sys.path.insert(0, ".")
import torch
from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2

class Args:
    model_name_or_path = "${MODEL_BASE}"
    vision_tower = "${VISION_TOWER}"
    vision_tower_mask = "${VISION_TOWER_MASK}"
    mask_config = "${MASK_CONFIG}"
    use_set_conditioner = True
    use_set_count_loss = False
    use_set_category_loss = False
    use_explicit_set_token = True
    q_set_fusion_hidden = 512
    set_conditioner_layers = 1
    set_conditioner_heads = 4
    set_conditioner_gate_init = 1e-3
    set_max_count = 10
    lasers_category_vocab_path = "${CATEGORY_VOCAB}"
    mm_use_im_start_end = False
    mm_use_im_patch_token = False

mask_cfg = get_mask_config(Args.mask_config)
model = SegEarthR2.from_pretrained(
    Args.model_name_or_path,
    mask_decoder_cfg=mask_cfg,
    torch_dtype=torch.float32,
    device_map="cpu",
)
model.initial_mask_module(Args.vision_tower_mask, Args)
model.init_set_conditioning_modules(Args)

hidden_dim = mask_cfg.MODEL.MASK_FORMER.HIDDEN_DIM
assert model.q_set_fusion[0].weight.shape == (512, hidden_dim * 2)
assert model.q_set_fusion[2].weight.shape == (hidden_dim, 512)

B = 2
q_explicit = torch.randn(B, hidden_dim, requires_grad=True)
q_implicit = torch.randn(B, hidden_dim, requires_grad=True)
q_concat = torch.cat([q_explicit, q_implicit], dim=-1)

with torch.no_grad():
    delta0 = model.q_set_fusion(q_concat)
    fused0 = q_implicit + delta0
assert torch.allclose(fused0, q_implicit, atol=1e-6), "initial delta must be zero"

fused = q_implicit + model.q_set_fusion(q_concat)
fused.sum().backward()

step0 = {
    "fusion_0_weight": model.q_set_fusion[0].weight.grad.norm().item(),
    "fusion_2_weight": model.q_set_fusion[2].weight.grad.norm().item(),
    "q_explicit": q_explicit.grad.norm().item(),
}
print("  step0 grad norms:", step0)
assert step0["fusion_2_weight"] > 0, "fusion layer2 weight must receive grad at step 0"
assert step0["fusion_0_weight"] == 0, "fusion layer0 blocked until layer2 weight is non-zero"
assert step0["q_explicit"] == 0, "explicit q_set blocked until layer2 weight is non-zero"

# Simulate one optimizer step on fusion layer2 weight.
with torch.no_grad():
    model.q_set_fusion[2].weight.add_(model.q_set_fusion[2].weight.grad, alpha=-1e-3)
model.zero_grad(set_to_none=True)
q_explicit.grad = None
q_implicit.grad = None

fused = q_implicit + model.q_set_fusion(q_concat)
fused.sum().backward()
step1 = {
    "fusion_0_weight": model.q_set_fusion[0].weight.grad.norm().item(),
    "fusion_2_weight": model.q_set_fusion[2].weight.grad.norm().item(),
    "q_explicit": q_explicit.grad.norm().item(),
}
print("  step1 grad norms:", step1)
for name, norm in step1.items():
    assert norm > 0, f"{name} grad norm must be > 0 after layer2 update, got {norm}"
print("[OK] q_set_fusion residual init + explicit path unlocks after layer2 update")
PY

echo "[5/5] eval_seg path accepts SET_token_embedding_indices"
"${PYTHON}" - <<PY
import sys
sys.path.insert(0, ".")
import inspect
from segearth_r2.model.language_model.llava_phi import SegEarthR2
sig = inspect.signature(SegEarthR2.eval_seg)
assert "SET_token_embedding_indices" in sig.parameters
print("[OK] eval_seg signature includes SET_token_embedding_indices")
PY

echo "========== ALL C-lite-v2 smoke tests passed =========="
echo "Artifacts: ${SMOKE_ROOT}"
