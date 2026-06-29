#!/usr/bin/env python3
"""Single LaSeRS batch preflight: real warmstart + segmentation loss backward."""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import torch
import transformers

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESEG_ROOT = os.path.abspath(os.path.join(REPO, ".."))
DEFAULT_BASE_8W = os.path.join(
    RESEG_ROOT,
    "output/base/standard-base-lasers-siglip1-8w-gd4/merged_model",
)
DEFAULT_LASERS = "/root/rivermind-data/huangziyi/data/LaSeRS"
sys.path.insert(0, REPO)

from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2, LaSeRSDataset, get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2

DR_EWTI_CFG = "segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin_dr_ewti.yaml"


def _grad_norm(p) -> float:
    if p.grad is None:
        return 0.0
    return float(p.grad.detach().norm().item())


def _pick_batch_indices(dataset, want_multi: bool = True):
    for i in range(min(len(dataset), 512)):
        item = dataset[i]
        n = int(item.get("mask_num", 0))
        if want_multi and n >= 2:
            return [i]
        if not want_multi and n >= 1:
            return [i]
    return [0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_BASE_8W)
    parser.add_argument("--lasers-path", default=DEFAULT_LASERS)
    parser.add_argument("--batch-index", type=int, default=None)
    args = parser.parse_args()
    os.chdir(REPO)

    dr_cfg = get_mask_config(DR_EWTI_CFG)
    dr_cfg.TG_SWIN.LOG_STATS = True

    print(f"[info] loading warmstart: {args.model_path}")
    model = SegEarthR2.from_pretrained(
        args.model_path,
        mask_decoder_cfg=dr_cfg,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    model.train()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vision_path = getattr(model.config, "mm_vision_tower", None) or os.path.join(
        RESEG_ROOT, "pretrained_model/CLIP/siglip-so400m-patch14-384"
    )
    clip_processor = transformers.SiglipImageProcessor.from_pretrained(vision_path)

    data_args = SimpleNamespace(lasers_holdout_ratio=0.05, lasers_holdout_seed=42)
    dataset = LaSeRSDataset(
        base_data_path=args.lasers_path,
        tokenizer=tokenizer,
        data_args=data_args,
        split="train_data.json",
        holdout_mode="train",
        holdout_seed=42,
    )
    indices = [args.batch_index] if args.batch_index is not None else _pick_batch_indices(dataset)
    batch = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_processor)(
        [dataset[i] for i in indices]
    )
    print(f"[info] LaSeRS batch indices={indices}, mask_num={batch['mask_num']}")

    for p in model.parameters():
        p.requires_grad = False
    for p in model.tg_swin_controller.parameters():
        p.requires_grad = True

    model.tg_swin_controller.log_stats = True
    model.tg_swin_controller.zero_grad(set_to_none=True)

    with torch.no_grad():
        prep = model.prepare_inputs_labels_for_multimodal(
            batch["input_ids"],
            batch["attention_mask"],
            None,
            batch["labels"],
            batch["images_clip"],
            token_refer_id=batch["token_refer_id"],
            SEG_token_embedding_indices=batch["SEG_token_embedding_indices"],
            SET_token_embedding_indices=batch.get("SET_token_embedding_indices"),
        )
        _, attention_mask, _, inputs_embeds, _, seg_idx, set_idx, _, refer_span = prep
        backbone = model.model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = backbone.last_hidden_state
        seg_hidden, _, _ = model.build_text_cond(hidden, seg_idx, refer_span_mask=refer_span)
        seg_emb = model.SEG_token_projector(seg_hidden.unsqueeze(1))
        set_emb_b = model.SET_token_projector(model.get_SET_embedding(hidden, set_idx))
        coarse = model.get_shared_coarse_evidence(
            batch["images"], seg_emb, batch["mask_num"], set_embedding_b=set_emb_b
        )
        print(
            f"[info] coarse_prob mean={float(coarse.mean()):.6f} "
            f"std={float(coarse.std()):.6f} shape={tuple(coarse.shape)}"
        )

    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        images=batch["images"],
        images_clip=batch["images_clip"],
        labels=batch["labels"],
        seg_info=batch["seg_info"],
        mask_num=batch["mask_num"],
        token_refer_id=batch["token_refer_id"],
        SEG_token_embedding_indices=batch["SEG_token_embedding_indices"],
        SET_token_embedding_indices=batch.get("SET_token_embedding_indices"),
        dataset_type=batch.get("dataset_type"),
    )
    loss = out.loss
    assert loss is not None and torch.isfinite(loss)
    print(f"[info] total loss={float(loss.detach()):.6e}")
    if out.loss_mask is not None:
        print(f"[info] loss_mask={float(out.loss_mask):.6e}")
    if out.loss_dice is not None:
        print(f"[info] loss_dice={float(out.loss_dice):.6e}")

    loss.backward()

    ctrl = model.tg_swin_controller
    print("[PASS] warmstart LaSeRS preflight — per-stage values/gradients:")
    for stage in ("1", "2", "3"):
        wti = ctrl.wti_blocks[stage]
        dr = ctrl.dr_wti_blocks[stage]
        alpha_val = float(torch.tanh(wti.alpha).detach())
        gate_val = float(torch.tanh(dr.evidence_relation_gate).detach())
        print(
            f"  stage {stage}: alpha={alpha_val:.6e} grad={_grad_norm(wti.alpha):.6e} | "
            f"gate={gate_val:.6e} grad={_grad_norm(dr.evidence_relation_gate):.6e}"
        )

    last = getattr(ctrl, "_last_stats", {})
    if last:
        dyn = last.get("dynamic_bias_abs_mean", float("nan"))
        bias_max = last.get("raw_bias_abs_mean", float("nan"))
        print(
            f"[info] LOG_STATS: dynamic_bias_abs_mean={dyn} "
            f"raw_bias_abs_mean={bias_max} "
            f"evidence_prob_mean={last.get('evidence_prob_mean', 'n/a')}"
        )

    gate_grads = [_grad_norm(ctrl.dr_wti_blocks[s].evidence_relation_gate) for s in ("1", "2", "3")]
    if all(g > 0 for g in gate_grads):
        print(f"[PASS] evidence_relation_gate grads: {gate_grads}")
    else:
        print(f"[WARN] evidence_relation_gate grads (alpha may still be ~0): {gate_grads}")

    print("[PASS] probe_tgswin_dr_ewti_lasers_preflight")


if __name__ == "__main__":
    main()
