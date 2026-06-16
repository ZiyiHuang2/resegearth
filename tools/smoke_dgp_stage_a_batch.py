#!/usr/bin/env python3
"""Full single-batch DGP v6.1 Stage A smoke (MLLM -> DGP -> Mask Decoder)."""
from __future__ import annotations

import argparse
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F
from transformers import SiglipImageProcessor

from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2, RRSISDDataset, get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.utils import conversation as conversation_lib


def _finite(x) -> bool:
    if torch.is_tensor(x):
        t = x.detach()
        return bool(torch.isfinite(t).all().item()) and not bool(torch.isnan(t).any().item())
    return math.isfinite(float(x))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B")
    parser.add_argument("--vision-tower", default="/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384")
    parser.add_argument("--vision-tower-mask", default="/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl")
    parser.add_argument("--mask-config", default="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    parser.add_argument("--data", default="/root/rivermind-data/huangziyi/data/RRSISD")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        print(f"[FAIL] model path missing: {args.model}")
        return 2
    if not os.path.isdir(args.data):
        print(f"[FAIL] data path missing: {args.data}")
        return 2

    print("=" * 60)
    print("[SMOKE] Environment")
    print("=" * 60)
    import cv2
    import detectron2
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"cv2={cv2.__version__} detectron2={detectron2.__version__}")

    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    mask_cfg = get_mask_config(args.mask_config)

    class MArgs:
        use_dgp_qdti = True
        use_qdti_bias = False
        dgp_training_stage = "a"
        dgp_version = "v6.1"
        gate_g_init = 0.01
        gate_l_init = 0.02
        dgp_fuse_dim = 256
        dgp_refiner_hidden_dim = 512
        dgp_pg_tokens = 1
        qdti_apply_layers = "disabled"
        qdti_scale_init = 0.0

    model_args = MArgs()
    print("=" * 60)
    print("[SMOKE] Load model + mask decoder")
    print("=" * 60)
    model = SegEarthR2.from_pretrained(args.model, mask_decoder_cfg=mask_cfg, torch_dtype=dtype)
    model.initial_mask_module(pretrained_path=args.vision_tower_mask, model_args=model_args)
    SegEarthR2.sync_dgp_config_from_args(model.config, model_args)
    model.ensure_dgp_qdti_modules()
    model.to(device=device, dtype=dtype)
    model.eval()

    class ModelArgs:
        vision_tower = args.vision_tower
        vision_tower_mask = args.vision_tower_mask
        train_clip_backbone = False
        train_swin_backbone = False
        version = "phi-2"

    model.get_model().initialize_vision_modules(model_args=ModelArgs(), fsdp=None)
    vt = model.get_vision_tower()
    vtm = model.model.get_vision_tower_mask()
    if vt is not None:
        vt.to(device=device, dtype=dtype)
    if vtm is not None:
        vtm.to(device=device, dtype=dtype)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("[SEG]")
    model.resize_token_embeddings(len(tokenizer))
    model.get_special_token(SEG=tokenizer("[SEG]", return_tensors="pt", add_special_tokens=False)["input_ids"], EOS=tokenizer.eos_token_id)
    model.tokenizer = tokenizer
    conversation_lib.default_conversation = conversation_lib.conv_templates.get("phi-2", conversation_lib.conv_templates["vicuna_v1"])

    clip_proc = SiglipImageProcessor.from_pretrained(args.vision_tower)
    class DArgs:
        base_data_path = args.data
        data_ratio = "1"
        switch_bs = 1
        segmentation = True
        dataset_name = "rrsisd"

    ds = RRSISDDataset(args.data, tokenizer, DArgs(), split="train")
    collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_proc)
    batch = collator([ds[0]])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch["token_refer_id"] = [x.to(device) for x in batch["token_refer_id"]]

    print("=" * 60)
    print("[SMOKE] Forward (train path)")
    print("=" * 60)
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    for n, p in model.named_parameters():
        if "prompt_adapter" in n or "query_refiner" in n:
            p.requires_grad = True
    SegEarthR2.validate_stage_a_trainable_params(model, rank0_log=True)

    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        images=batch["images"].to(dtype=dtype),
        images_clip=batch["images_clip"].to(dtype=dtype),
        seg_info=batch["seg_info"],
        token_refer_id=batch["token_refer_id"],
        SEG_token_embedding_indices=batch["SEG_token_embedding_indices"],
        mask_num=batch["mask_num"],
    )

    loss = out.loss
    print(f"loss={float(loss.detach().cpu()):.6f} finite={_finite(loss)}")
    assert _finite(loss), "loss not finite"

    health = getattr(model, "_dgp_last_health", {})
    print("gate_g", health.get("query_refiner_gate_g"), "gate_l", health.get("query_refiner_gate_l"))
    print("delta_g_norm", health.get("delta_g_norm"), "delta_l_norm", health.get("delta_l_norm"))
    print("cos_qref_qseg", health.get("cos_qref_qseg"), "cos_pg_pl", health.get("cos_pg_pl"))
    print("entropy_pg_norm", health.get("entropy_pg_attention_norm"), "entropy_pl_norm", health.get("entropy_pl_attention_norm"))

    qdti_mod = getattr(model.predictor, "query_specific_text_memory_bias", None)
    assert qdti_mod is None, "QDTI module must not exist when use_qdti_bias=False"

    loss.backward()
    grad_ok = {}
    for key in ("prompt_adapter.proj.weight", "prompt_adapter.detail_proj.weight", "query_refiner.gate_g", "query_refiner.gate_l"):
        p = dict(model.named_parameters()).get(key)
        grad_ok[key] = p is not None and p.grad is not None and float(p.grad.norm()) > 0
    frozen_ok = all(
        p.grad is None
        for n, p in model.named_parameters()
        if p.requires_grad is False and "prompt_adapter" not in n and "query_refiner" not in n
    )
    print("grad_ok", grad_ok)
    assert all(grad_ok.values()), f"missing DGP grads: {grad_ok}"
    assert not getattr(model.predictor, "query_specific_text_memory_bias", None)

    print("=" * 60)
    print("[SMOKE] Eval path pred_masks")
    print("=" * 60)
    with torch.no_grad():
        eval_out = model.eval_seg(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            images=batch["images"].to(dtype=dtype),
            images_clip=batch["images_clip"].to(dtype=dtype),
            seg_info=batch["seg_info"],
            token_refer_id=batch["token_refer_id"],
            SEG_token_embedding_indices=batch["SEG_token_embedding_indices"],
            mask_num=batch["mask_num"],
            dgp_use_refined_query=True,
        )
    pred = eval_out[0]["pred"]
    print(f"pred_masks shape={getattr(pred, 'shape', type(pred))}")
    assert pred is not None

    print("=" * 60)
    print("[SMOKE] PASS")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
