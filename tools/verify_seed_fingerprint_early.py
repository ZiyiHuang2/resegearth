#!/usr/bin/env python3
"""
Minimal fingerprint: load SegEarthR2, vision, tokenizer, [SEG] resize, LoRA (same as train.py),
then print mean/std/md5 of selected tensors. No training, no dataset.

Run twice and diff stdout:
  python tools/verify_seed_fingerprint_early.py [args] > /tmp/fp1.txt
  python tools/verify_seed_fingerprint_early.py [args] > /tmp/fp2.txt
  diff -u /tmp/fp1.txt /tmp/fp2.txt
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import transformers

# repo root = parent of tools/
_CURRENT = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_TRAIN_DIR = os.path.join(_PROJECT_ROOT, "segearth_r2", "train")
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)


def _load_train_module():
    path = os.path.join(_PROJECT_ROOT, "segearth_r2", "train", "train.py")
    spec = importlib.util.spec_from_file_location("segearth_train_fp", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _fp_line(name: str, t: torch.Tensor) -> str:
    x = t.detach().float().cpu().contiguous().reshape(-1).numpy()
    h = hashlib.md5(x.tobytes()).hexdigest()
    return f"{name}\tshape={tuple(t.shape)}\tmean={float(x.mean()):.10g}\tstd={float(x.std()):.10g}\tmd5={h}"


def _collect_lora_lines(model: torch.nn.Module) -> List[str]:
    rows: List[Tuple[str, str]] = []
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            if p.requires_grad or True:
                rows.append((n, _fp_line(f"lora::{n}", p)))
    rows.sort(key=lambda x: x[0])
    return [r[1] for r in rows]


def _find_seg_projector_tensors(model: torch.nn.Module) -> List[str]:
    out: List[str] = []
    for n, m in model.named_modules():
        if n.endswith("SEG_token_projector") and isinstance(m, torch.nn.Linear):
            out.append(_fp_line(f"SEG_token_projector::{n}.weight", m.weight))
            if m.bias is not None:
                out.append(_fp_line(f"SEG_token_projector::{n}.bias", m.bias))
    out.sort()
    return out


def _other_named_tensors(model: torch.nn.Module) -> List[str]:
    """Train-relevant modules outside LoRA / vision backbones (subset to keep output small)."""
    keys = ("mm_projector", "pixel_decoder", "predictor")
    lines: List[str] = []
    for n, p in model.named_parameters():
        if not p.ndim:
            continue
        if "vision_tower" in n:
            continue
        if any(k in n for k in keys):
            lines.append(_fp_line(f"other::{n}", p))
    lines.sort()
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_name_or_path",
        default=os.environ.get(
            "MODEL_NAME_OR_PATH",
            "/home/wangchengjun/huangziyi/reseg/output/bseg/baseline_standard-base_5w/merged_model",
        ),
    )
    p.add_argument(
        "--vision_tower",
        default=os.environ.get(
            "VISION_TOWER",
            "/home/wangchengjun/huangziyi/reseg/pretrained_model/CLIP/siglip2-so400m-patch14-384",
        ),
    )
    p.add_argument(
        "--vision_tower_mask",
        default=os.environ.get(
            "VISION_TOWER_MASK",
            "/home/wangchengjun/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl",
        ),
    )
    p.add_argument(
        "--mask_config",
        default=os.environ.get(
            "MASK_CONFIG",
            "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml",
        ),
    )
    p.add_argument("--base_data_path", default=os.environ.get("BASE_DATA_PATH", "/home/wangchengjun/huangziyi/data/RRSISD"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="/tmp/seed_fingerprint_verify")
    args = p.parse_args()

    train_mod = _load_train_module()
    ModelArguments = train_mod.ModelArguments
    DataArguments = train_mod.DataArguments
    TrainingArguments = train_mod.TrainingArguments
    find_linear_layers = train_mod.find_linear_layers
    smart_tokenizer_and_embedding_resize = train_mod.smart_tokenizer_and_embedding_resize
    get_mask_config = train_mod.get_mask_config
    SegEarthR2 = train_mod.SegEarthR2

    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        vision_tower=args.vision_tower,
        vision_tower_mask=args.vision_tower_mask,
        mask_config=args.mask_config,
    )
    data_args = DataArguments(base_data_path=args.base_data_path, dataset_name="rrsisd")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        use_cpu=True,
        bf16=False,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=1,
        logging_steps=1,
        save_steps=500,
        eval_steps=500,
        gradient_checkpointing=False,
        dataloader_num_workers=2,
        dataloader_prefetch_factor=2,
        deepspeed=None,
        lora_enable=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )

    if training_args.seed is None:
        training_args.seed = 42
    if training_args.data_seed is None:
        training_args.data_seed = 42

    print(f"[verify] PROJECT_ROOT={_PROJECT_ROOT}")
    print(f"[verify] seed={training_args.seed} (P0: set before from_pretrained)")
    transformers.set_seed(training_args.seed)

    mask_cfg = get_mask_config(config=model_args.mask_config)
    model = SegEarthR2.from_pretrained(
        model_args.model_name_or_path,
        mask_decoder_cfg=mask_cfg,
        add_cross_attn=True,
        cache_dir=training_args.cache_dir,
    )
    if not model.is_train_mask_decode:
        mask2former_ckpt = model_args.vision_tower_mask if model_args.load_mask2former else None
        model.initial_mask_module(mask2former_ckpt, model_args)

    model.config.use_cache = False
    model = model.to(device=training_args.device, dtype=torch.float32)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )

    from segearth_r2.model.mipha import conversation as conversation_lib

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
        vision_tower = model.get_vision_tower()
        vision_tower_mask = model.model.get_vision_tower_mask()
        dtype = torch.float32
        vision_tower.to(dtype=dtype, device=training_args.device)
        vision_tower_mask.to(dtype=dtype, device=training_args.device)
        data_args.is_multimodal = True
        if not model_args.train_clip_backbone:
            model.model.vision_tower.requires_grad_(False)
        if not model_args.train_swin_backbone:
            model.model.vision_tower_mask.requires_grad_(False)
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

    tokenizer.add_tokens("[SEG]")
    model.resize_token_embeddings(len(tokenizer))

    emb_in = model.get_input_embeddings().weight
    out_emb = model.get_output_embeddings()
    if out_emb is not None:
        emb_out_w = out_emb.weight
    else:
        emb_out_w = model.lm_head.weight
    n_new = 1
    lines: List[str] = []
    lines.append(_fp_line("new_token_embed::input_embeddings_last_row", emb_in[-n_new:].reshape(-1)))
    lines.append(_fp_line("new_token_embed::lm_head_last_row", emb_out_w[-n_new:].reshape(-1)))

    train_module_list = ["lm_head", "pixel_decoder", "predictor", "SEG_token_projector"]
    if model_args.train_swin_backbone:
        train_module_list.append("vision_tower_mask")

    lora_target_modules = find_linear_layers(model, train_module_list=train_module_list)
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=training_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    for n, p in model.named_parameters():
        if any(x in n for x in train_module_list):
            p.requires_grad = True

    seg_id = tokenizer("[SEG]", return_tensors="pt", add_special_tokens=False)["input_ids"]
    model.get_special_token(SEG=seg_id, EOS=tokenizer.eos_token_id)

    lines.extend(_find_seg_projector_tensors(model))
    lines.extend(_collect_lora_lines(model))
    lines.extend(_other_named_tensors(model))

    print("--- FINGERPRINT BEGIN ---")
    for row in sorted(lines):
        print(row)
    print("--- FINGERPRINT END ---")


if __name__ == "__main__":
    main()
