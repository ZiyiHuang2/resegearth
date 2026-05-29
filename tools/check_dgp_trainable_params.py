#!/usr/bin/env python3
"""Verify DGP-QDTI parameters exist, are trainable, in optimizer, and receive gradients."""
from __future__ import annotations

import argparse
import copy
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import SiglipImageProcessor

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2


def find_linear_layers(model, lora_target_modules=("q_proj", "v_proj"), train_module_list=()):
    cur_train_module_list = copy.deepcopy(list(train_module_list))
    cur_train_module_list.extend(["vision_tower", "vision_tower_mask"])
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            and all(x not in name for x in cur_train_module_list)
            and any(x in name for x in lora_target_modules)
        ):
            lora_module_names.add(name)
    return sorted(lora_module_names)

DGP_KEYWORDS = (
    "prompt_adapter",
    "query_refiner",
    "query_specific_text_memory_bias",
    "qdti",
    "dgp",
    "gate",
    "qdti_scale",
)


def real_trainable_report(model, tag: str = "") -> dict:
    total = 0
    trainable = 0
    dgp_total = 0
    dgp_trainable = 0
    dgp_items = []

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        if any(k in name for k in DGP_KEYWORDS):
            dgp_total += n
            if p.requires_grad:
                dgp_trainable += n
            dgp_items.append((name, n, p.requires_grad, tuple(p.shape)))

    prefix = f"[{tag}] " if tag else ""
    print(f"{prefix}[REAL_PARAM] total={total:,}")
    print(f"{prefix}[REAL_PARAM] trainable={trainable:,}")
    pct = 100.0 * trainable / total if total else 0.0
    print(f"{prefix}[REAL_PARAM] trainable_percent={pct:.6f}%")
    print(f"{prefix}[DGP_PARAM] total={dgp_total:,}")
    print(f"{prefix}[DGP_PARAM] trainable={dgp_trainable:,}")
    print(f"{prefix}[DGP_PARAM] matched parameters:")
    for name, n, req, shape in dgp_items:
        print(f"  requires_grad={req} numel={n:,} shape={shape} name={name}")

    cfg = getattr(model, "config", None) or getattr(getattr(model, "base_model", None), "config", None)
    use_dgp = bool(getattr(cfg, "use_dgp_qdti", False)) if cfg is not None else False
    if use_dgp and dgp_total == 0:
        raise RuntimeError("use_dgp_qdti=True but no DGP-QDTI parameters found in model.named_parameters().")
    if use_dgp and dgp_trainable == 0:
        raise RuntimeError("use_dgp_qdti=True but no DGP-QDTI trainable parameters found.")

    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": pct,
        "dgp_total": dgp_total,
        "dgp_trainable": dgp_trainable,
        "dgp_items": dgp_items,
    }


def optimizer_dgp_report(model, optimizer) -> dict:
    id_to_name = {id(p): name for name, p in model.named_parameters()}
    opt_names = []
    opt_total = 0
    opt_dgp_total = 0

    for group_idx, group in enumerate(optimizer.param_groups):
        for p in group["params"]:
            name = id_to_name.get(id(p), None)
            if name is None:
                continue
            n = p.numel()
            opt_total += n
            if any(k in name for k in DGP_KEYWORDS):
                opt_dgp_total += n
                opt_names.append(
                    (group_idx, name, n, tuple(p.shape), group.get("lr", None), group.get("weight_decay", None))
                )

    print(f"[OPT_PARAM] total_params_in_optimizer={opt_total:,}")
    print(f"[OPT_DGP_PARAM] total_dgp_params_in_optimizer={opt_dgp_total:,}")
    print("[OPT_DGP_PARAM] matched optimizer parameters:")
    for group_idx, name, n, shape, lr, wd in opt_names:
        print(f"  group={group_idx} lr={lr} wd={wd} numel={n:,} shape={shape} name={name}")

    cfg = getattr(model, "config", None) or getattr(getattr(model, "base_model", None), "config", None)
    use_dgp = bool(getattr(cfg, "use_dgp_qdti", False)) if cfg is not None else False
    if use_dgp and opt_dgp_total == 0:
        raise RuntimeError("use_dgp_qdti=True but optimizer contains no DGP-QDTI parameters.")

    return {"opt_total": opt_total, "opt_dgp_total": opt_dgp_total, "opt_names": opt_names}


def _grad_norm(param) -> str:
    if param is None:
        return "param_missing"
    if param.grad is None:
        return "grad_is_None"
    return f"{param.grad.detach().float().norm().item():.6e}"


from segearth_r2.model.language_model.prompt_query_fusion import pack_seg_hidden_states_bq


def backward_grad_report(model, run_backward: bool = True) -> dict:
    """Run a lightweight forward/backward through DGP + QDTI paths."""
    base = model.base_model if hasattr(model, "base_model") else model
    m = base

    if not hasattr(m, "prompt_adapter") or not hasattr(m, "query_refiner"):
        raise RuntimeError("prompt_adapter/query_refiner not found on model")

    device = next(model.parameters()).device
    dtype = m.SEG_token_projector.weight.dtype
    B, L, H_lm, Q, P, S = 1, 32, int(m.config.hidden_size), 1, 2, 64
    hidden = torch.randn(B, L, H_lm, device=device, dtype=dtype, requires_grad=False)
    attn = torch.ones(B, L, dtype=torch.bool, device=device)
    seg_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    seg_mask[:, -1] = True

    seg_hidden_bq, seg_valid = pack_seg_hidden_states_bq(hidden, seg_mask)
    seg_emb = m.SEG_token_projector(seg_hidden_bq)
    _, _, prompt_tokens, prompt_mask = m.prompt_adapter(
        hidden_states=hidden,
        attention_mask=attn,
        seg_mask=seg_mask,
    )
    q_ref = m.query_refiner(seg_emb, prompt_tokens, seg_query_mask=seg_valid, prompt_mask=prompt_mask)
    loss_dgp = q_ref.sum()
    loss_dgp.backward(retain_graph=True)

    gate_mod = m.query_refiner
    gate = gate_mod.gate
    refiner_linear = gate_mod.ffn[0].weight

    report = {
        "query_refiner.gate": _grad_norm(gate),
        "query_refiner.ffn0.weight": _grad_norm(refiner_linear),
        "prompt_adapter.proj.weight": _grad_norm(m.prompt_adapter.proj.weight),
    }

    qdti_mod = getattr(m.predictor, "query_specific_text_memory_bias", None)
    if qdti_mod is not None:
        for p in qdti_mod.parameters():
            p.grad = None

        memory = torch.randn(S, B, 256, device=device, dtype=dtype)
        query = q_ref.permute(1, 0, 2).detach().requires_grad_(True)
        text_memory = prompt_tokens.detach()
        text_mask = prompt_mask
        bias, _ = qdti_mod(memory, query, text_memory, text_mask, num_heads=8)
        if bias is None:
            loss_qdti = (query * 0).sum()
            note = "extra_attn_bias=None (QDTI disabled or invalid inputs)"
        else:
            loss_qdti = bias.sum()
            note = f"extra_attn_bias shape={tuple(bias.shape)} absmax={bias.detach().abs().max().item():.4e}"
        loss_qdti.backward()
        report["qdti_scale"] = _grad_norm(qdti_mod.qdti_scale)
        report["qdti.bias_mlp.0.weight"] = _grad_norm(qdti_mod.bias_mlp[0].weight)
        report["qdti.text_proj.weight"] = _grad_norm(qdti_mod.text_proj.weight)
        report["qdti.visual_proj.weight"] = _grad_norm(qdti_mod.visual_proj.weight)
        report["qdti.query_proj.weight"] = _grad_norm(qdti_mod.query_proj.weight)
        report["qdti_note"] = note
    else:
        report["qdti_note"] = "query_specific_text_memory_bias not present (use_qdti_bias=False?)"

    print("[GRAD_REPORT]")
    for k, v in report.items():
        print(f"  {k}: {v}")

    return report


class ModelArgs:
    model_name_or_path = "/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
    vision_tower = "/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
    vision_tower_mask = "/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
    mask_config = "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
    load_mask2former = True
    use_dgp_qdti = True
    use_qdti_bias = True
    dgp_fuse_dim = 256
    dgp_refiner_hidden_dim = 512
    dgp_pg_tokens = 1
    qdti_bias_dim = 128
    qdti_init_std = 1e-3
    qdti_max_abs = 0.01
    qdti_apply_layers = "last3"
    qdti_scale_init = 0.0
    scale_hard_loss_weight = 0.0
    freeze_backbone = False
    train_clip_backbone = False
    train_swin_backbone = False
    version = "phi-2"


class TrainArgs:
    lora_enable = True
    lora_r = 4
    lora_alpha = 16
    lora_dropout = 0.05
    fp16 = False
    bf16 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_mm_mlp_adapter = True
    gradient_checkpointing = False
    per_device_train_batch_size = 1
    learning_rate = 1e-4
    weight_decay = 0.0


def build_model_like_train(model_args, training_args):
    mask_cfg = get_mask_config(config=model_args.mask_config)
    model = SegEarthR2.from_pretrained(
        model_args.model_name_or_path,
        mask_decoder_cfg=mask_cfg,
        add_cross_attn=True,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
    )

    if not model.is_train_mask_decode:
        mask2former_ckpt = model_args.vision_tower_mask if model_args.load_mask2former else None
        model.initial_mask_module(mask2former_ckpt, model_args)
    else:
        SegEarthR2.sync_dgp_config_from_args(model.config, model_args)
        model.ensure_dgp_qdti_modules()

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    return model


def apply_lora_and_trainable_flags(model, model_args, training_args):
    train_module_list = [
        "lm_head", "pixel_decoder", "predictor", "SEG_token_projector",
    ]
    if getattr(model_args, "use_dgp_qdti", False):
        train_module_list.extend(["prompt_adapter", "query_refiner"])

    peft_trainable_before_unfreeze = None
    if training_args.lora_enable:
        lora_target_modules = find_linear_layers(model, train_module_list=train_module_list)
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        peft_stats = model.get_nb_trainable_parameters()
        peft_trainable_before_unfreeze = peft_stats[0]
        print(
            f"[PEFT_BEFORE_UNFREEZE] trainable={peft_stats[0]:,} "
            f"all={peft_stats[1]:,} percent={100 * peft_stats[0] / peft_stats[1]:.6f}%"
        )
        model.print_trainable_parameters()

        for n, p in model.named_parameters():
            if any(x in n for x in train_module_list):
                p.requires_grad = True

    return model, train_module_list, peft_trainable_before_unfreeze


def checkpoint_key_report(ckpt_dir: str) -> None:
    from tools.audit_dgp_stage3_chain import list_dgp_keys_from_state_dict

    keys = list_dgp_keys_from_state_dict(ckpt_dir)
    if not keys:
        try:
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            sd = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)
            keys = [k for k in sd.keys() if any(x in k for x in DGP_KEYWORDS)]
        except Exception as e:
            print(f"[CKPT] failed to read checkpoint: {e}")
            return

    print(f"[CKPT] dgp_key_count={len(keys)}")
    for k in sorted(keys)[:30]:
        print(f"  {k}")
    if len(keys) > 30:
        print(f"  ... and {len(keys) - 30} more")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("--checkpoint", default="", help="optional ZeRO checkpoint dir")
    parser.add_argument("--lora-r", type=int, default=4)
    args = parser.parse_args()

    model_args = ModelArgs()
    training_args = TrainArgs()
    training_args.lora_r = args.lora_r

    if not os.path.isdir(model_args.model_name_or_path):
        print(f"[ERROR] model path missing: {model_args.model_name_or_path}")
        return 2

    print("=" * 60)
    print("[1] Build model (train.py equivalent init)")
    print("=" * 60)
    model = build_model_like_train(model_args, training_args)
    device = "cpu"
    training_args.device = device
    model = model.to(device)

    print("=" * 60)
    print("[2] Apply LoRA + train_module_list requires_grad (train.py order)")
    print("=" * 60)
    model, train_module_list, peft_before = apply_lora_and_trainable_flags(model, model_args, training_args)
    print(f"[INFO] train_module_list={train_module_list}")

    print("=" * 60)
    print("[3] real_trainable_report AFTER requires_grad unfreeze")
    print("=" * 60)
    stats_after = real_trainable_report(model, tag="AFTER_UNFREEZE")

    print("=" * 60)
    print("[4] optimizer_dgp_report")
    print("=" * 60)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )
    opt_stats = optimizer_dgp_report(model, optimizer)

    if not args.skip_backward:
        print("=" * 60)
        print("[5] backward_grad_report (micro forward)")
        print("=" * 60)
        model.zero_grad(set_to_none=True)
        grad_report = backward_grad_report(model)

    if args.checkpoint:
        print("=" * 60)
        print(f"[6] checkpoint_key_report: {args.checkpoint}")
        print("=" * 60)
        checkpoint_key_report(args.checkpoint)

    print("=" * 60)
    print("[SUMMARY]")
    print("=" * 60)
    print(f"PEFT print_trainable_parameters (before unfreeze): {peft_before:,}" if peft_before else "n/a")
    print(f"real trainable after unfreeze: {stats_after['trainable']:,}")
    print(f"DGP trainable after unfreeze: {stats_after['dgp_trainable']:,}")
    print(f"optimizer DGP params: {opt_stats['opt_dgp_total']:,}")

    if stats_after["dgp_trainable"] == 0 or opt_stats["opt_dgp_total"] == 0:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
