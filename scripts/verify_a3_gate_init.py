#!/usr/bin/env python3
"""Print A3 SetConditioner gate-init safety stats (constructed + optional real batch)."""

from __future__ import annotations

import argparse
import os
import sys

import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from segearth_r2.model.set_conditioner import SetConditioner, regroup_seg_embeddings


def _print_stats(tag: str, original: torch.Tensor, refined: torch.Tensor, gate: torch.Tensor,
                 valid_mask: torch.Tensor, residual_gate: torch.Tensor):
    orig_flat = original.squeeze(1) if original.dim() == 3 else original
    ref_flat = refined.squeeze(1) if refined.dim() == 3 else refined
    diff = (ref_flat - orig_flat).abs()
    gate_valid = gate.masked_select(valid_mask.unsqueeze(-1)) if gate.numel() else gate

    print(f"\n=== {tag} ===")
    print(f"mean_abs(refined - original): {diff.mean().item():.6e}")
    print(f"max_abs(refined - original):  {diff.max().item():.6e}")
    if gate_valid.numel():
        print(f"gate mean: {gate_valid.mean().item():.6f}")
        print(f"gate min:  {gate_valid.min().item():.6f}")
        print(f"gate max:  {gate_valid.max().item():.6f}")
    else:
        print("gate: (empty)")
    print(f"residual_gate param: {residual_gate.item():.6f}")


def _print_grad_stats(module: SetConditioner):
    output_weight_grad = module.output_proj.weight.grad
    output_bias_grad = module.output_proj.bias.grad
    residual_gate_grad = module.residual_gate.grad
    print(
        f"output_proj.weight grad sum: {0.0 if output_weight_grad is None else output_weight_grad.abs().sum().item():.6e}"
    )
    print(
        f"output_proj.bias grad sum:   {0.0 if output_bias_grad is None else output_bias_grad.abs().sum().item():.6e}"
    )
    print(
        f"residual_gate grad:         {0.0 if residual_gate_grad is None else residual_gate_grad.item():.6e}"
    )


def constructed_batch(hidden_dim: int = 256):
    mask_num = [2, 1, 3, 1]
    total = sum(mask_num)
    seg = torch.randn(total, 1, hidden_dim)
    return seg, mask_num


def real_batch_from_model(model_path: str, hidden_dim: int):
    from types import SimpleNamespace

    from transformers import SiglipImageProcessor

    from segearth_r2.datasets.dataset import get_mask_config
    sys.path.insert(0, os.path.join(REPO_DIR, "segearth_r2", "train"))
    from train import make_unify_datamodule
    from segearth_r2.model.language_model.llava_phi import SegEarthR2

    mask_cfg = get_mask_config(
        config="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
    )
    model = SegEarthR2.from_pretrained(
        model_path, mask_decoder_cfg=mask_cfg, add_cross_attn=True, device_map="cpu"
    )
    model.init_set_conditioning_modules(
        SimpleNamespace(
            use_set_conditioner=True,
            use_set_count_loss=True,
            use_set_category_loss=True,
            set_conditioner_layers=1,
            set_conditioner_heads=4,
            set_conditioner_gate_init=1e-3,
            lasers_category_vocab_path="segearth_r2/model/lasers_category_vocab.json",
        )
    )
    model.eval()
    hidden_dim = model.set_conditioner.hidden_dim

    clip_proc = SiglipImageProcessor.from_pretrained(
        "/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
    )
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = "[PAD]"

    data = make_unify_datamodule(
        clip_image_processor=clip_proc,
        tokenizer=tok,
        data_args=SimpleNamespace(
            base_data_path="/root/rivermind-data/huangziyi/data/LaSeRS",
            dataset_name="lasers",
            data_ratio="1",
            switch_bs=4,
            lasers_holdout_ratio=0.05,
            lasers_holdout_seed=42,
            is_multimodal=True,
            image_aspect_ratio="square",
            image_grid_pinpoints=None,
            lazy_preprocess=True,
            segmentation=True,
            fix_dataset_len=0,
        ),
        training_args=SimpleNamespace(
            per_device_train_batch_size=2,
            dataloader_num_workers=0,
            seed=42,
        ),
    )["train_dataset"]

    batch = data[0]
    for i in range(1, min(4, len(data))):
        # simple collate for smoke: only need one sample with SEG tokens
        pass

    from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2

    collator = data.collator if hasattr(data, "collator") else None
    samples = [data[i] for i in range(min(2, len(data)))]
    if collator is None:
        from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2

        collator = DataCollatorForCOCODatasetV2(tokenizer=tok, clip_image_processor=clip_proc)
    batch = collator(samples)

    with torch.no_grad():
        if (batch.get("SEG_token_embedding_indices") == 1).sum() == 0:
            raise RuntimeError("real batch has no [SEG] tokens")
        image_features = model.get_vision_tower_feature(batch["images"])
        _, _, _, _, _, seg_idx, _ = model.prepare_inputs_labels_for_multimodal(
            batch["input_ids"],
            batch["attention_mask"],
            None,
            batch["labels"],
            batch["images_clip"],
            token_refer_id=batch["token_refer_id"],
            SEG_token_embedding_indices=batch["SEG_token_embedding_indices"],
        )
        hidden = model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
        seg_emb = model.SEG_token_projector(
            model.get_SEG_embedding(hidden, seg_idx)
        ).unsqueeze(1)
        if seg_emb.dim() == 2:
            seg_emb = seg_emb.unsqueeze(1)
    return seg_emb, batch["mask_num"], model.set_conditioner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--gate_init", type=float, default=1e-3)
    p.add_argument("--real_batch", action="store_true")
    p.add_argument(
        "--model_name_or_path",
        default="/root/rivermind-data/huangziyi/reseg/output/base/standard-base-lasers-siglip1-28w-gd4/merged_model",
    )
    args = p.parse_args()

    mod = SetConditioner(args.hidden_dim, gate_init=args.gate_init)
    mod.eval()
    seg, mask_num = constructed_batch(args.hidden_dim)
    refined, _, valid_mask, gate_mean, gate = mod(seg, mask_num)
    seg_group, _, _ = regroup_seg_embeddings(seg, mask_num)
    _print_stats(
        "constructed batch (fresh SetConditioner)",
        seg_group,
        regroup_seg_embeddings(refined, mask_num)[0],
        gate,
        valid_mask,
        mod.residual_gate,
    )

    mod.zero_grad(set_to_none=True)
    seg_for_grad = seg.clone().detach().requires_grad_(True)
    refined_for_grad, _, _, _, _ = mod(seg_for_grad, mask_num)
    refined_for_grad.sum().backward()
    _print_grad_stats(mod)

    if args.real_batch:
        try:
            seg_r, mask_num_r, mod_r = real_batch_from_model(args.model_name_or_path, args.hidden_dim)
            refined_r, _, vm_r, _, gate_r = mod_r(seg_r, mask_num_r)
            sg, vm, _ = regroup_seg_embeddings(seg_r, mask_num_r)
            rg, _, _ = regroup_seg_embeddings(refined_r, mask_num_r)
            _print_stats("real LaSeRS batch (baseline+init modules)", sg, rg, gate_r, vm_r, mod_r.residual_gate)
            mod_r.zero_grad(set_to_none=True)
            seg_r_for_grad = seg_r.clone().detach().requires_grad_(True)
            refined_r_for_grad, _, _, _, _ = mod_r(seg_r_for_grad, mask_num_r)
            refined_r_for_grad.sum().backward()
            _print_grad_stats(mod_r)
        except Exception as e:
            print(f"\n[WARN] real batch skipped: {e}")

    print("\n[OK] gate init verification complete")


if __name__ == "__main__":
    main()
