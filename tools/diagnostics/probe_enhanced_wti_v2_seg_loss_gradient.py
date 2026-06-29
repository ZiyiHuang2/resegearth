#!/usr/bin/env python3
"""Enhanced WTI v2 / v1.5: real seg-loss backward; report WTI grads (no coarse)."""

from __future__ import annotations

import argparse
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESEG_ROOT = os.path.abspath(os.path.join(REPO, ".."))
DEFAULT_MODEL = os.path.join(
    RESEG_ROOT, "output/base/standard-base-lasers-siglip1-8w-gd4/merged_model"
)
DEFAULT_CFG = "segearth_r2/model/mask_decoder/mask_config/maskformer2_enhanced_wti_v2_setpp.yaml"

sys.path.insert(0, REPO)

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.model.mask_encoder.swin_trans import build_swin_b
from segearth_r2.model.mask_encoder.tg_swin import StageEnhancedWTIHeadAware, StageWTIHeadAware


def _grad_norm(param) -> float:
    if param is None or param.grad is None:
        return 0.0
    return float(param.grad.detach().norm().item())


def _report_enhanced(wti: StageEnhancedWTIHeadAware, stage: str):
    print(f"  stage {stage} (Enhanced WTI v2):")
    print(f"    alpha.grad norm           = {_grad_norm(wti.alpha):.6e}")
    print(f"    head_mixer_gamma.grad norm= {_grad_norm(wti.head_mixer_gamma):.6e}")
    print(f"    visual_proj[-1].grad norm = {_grad_norm(wti.visual_proj[-1].weight):.6e}")
    print(f"    text_proj[-1].grad norm   = {_grad_norm(wti.text_proj[-1].weight):.6e}")


def _report_v15(wti: StageWTIHeadAware, stage: str):
    print(f"  stage {stage} (v1.5 WTI):")
    print(f"    alpha.grad norm           = {_grad_norm(wti.alpha):.6e}")
    print(f"    visual_q.grad norm        = {_grad_norm(wti.visual_q.weight):.6e}")
    print(f"    text_q.grad norm          = {_grad_norm(wti.text_q.weight):.6e}")


def _run_seg_loss_backward(model, images, mask_num, text_cond, reliability, seg_emb, set_emb_b):
    n_target = sum(mask_num)
    assert not model._use_coarse_evidence(), "probe expects no-coarse config"

    set_emb_t = model._repeat_set_embedding_per_target(set_emb_b, mask_num)
    set_control = model.build_set_control(set_emb_t)
    images_exp = model._repeat_images_per_target(images, mask_num)
    image_features = model.get_vision_tower_feature(
        images_exp,
        text_cond=text_cond,
        reliability=reliability,
        set_control=set_control,
        coarse_evidence=None,
        enable_tg_swin=True,
    )
    mask_features, _, multi_scale = model.pixel_decoder.forward_features(image_features)
    mask_outputs = model.predictor(
        multi_scale,
        mask_features,
        None,
        None,
        SEG_embedding=seg_emb,
        SET_embedding=set_emb_t,
        per_target_mode=True,
    )
    mask_outputs["per_target_mode"] = True
    mask_outputs["target_to_image"] = model._build_target_to_image(mask_num, images.device)
    mask_outputs["mask_num"] = list(mask_num)

    gt_h, gt_w = mask_outputs["pred_masks"].shape[-2:]
    targets = []
    for i in range(n_target):
        m = torch.zeros(1, gt_h, gt_w)
        m[:, i * 4 : (i + 1) * 4, i * 4 : (i + 1) * 4] = 1.0
        targets.append({"labels": torch.tensor([0]), "masks": m, "valid": None, "inst_id": None})

    model.tg_swin_controller.zero_grad(set_to_none=True)
    losses = model.criterion(mask_outputs, targets)
    mask_loss = sum(
        v * model.weight_dict[k]
        for k, v in losses.items()
        if v is not None and k in model.weight_dict
    )
    mask_loss.backward()
    return float(mask_loss.detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--mask-config", default=DEFAULT_CFG)
    parser.add_argument("--phase", choices=("init", "alpha005"), default="alpha005")
    args = parser.parse_args()
    os.chdir(REPO)

    if not os.path.isdir(args.model_path):
        print(f"[ERROR] model path not found: {args.model_path}")
        sys.exit(1)

    cfg = get_mask_config(args.mask_config)
    tg = cfg.TG_SWIN
    enhanced = bool(getattr(tg, "ENHANCED_WTI", False) or str(tg.VERSION).lower() == "enhanced-wti-v2")
    num_stages = int(getattr(tg, "NUM_STAGES", 3))
    cond_dim = int(tg.COND_DIM)
    print(f"[info] mask_config={args.mask_config}")
    print(f"[info] VERSION={tg.VERSION} enhanced={enhanced} coarse={getattr(tg, 'USE_COARSE_EVIDENCE', False)}")

    model = SegEarthR2.from_pretrained(args.model_path, mask_decoder_cfg=cfg, torch_dtype=torch.float32)
    model.initial_mask_module(None, None)
    model.get_model().vision_tower_mask = build_swin_b(None)
    model.train()

    for p in model.parameters():
        p.requires_grad = False
    for p in model.tg_swin_controller.parameters():
        p.requires_grad = True
    if model.tg_swin_set_control is not None:
        for p in model.tg_swin_set_control.parameters():
            p.requires_grad = True
    for p in model.pixel_decoder.parameters():
        p.requires_grad = True
    for p in model.predictor.parameters():
        p.requires_grad = True

    images = torch.randn(2, 3, 384, 384)
    mask_num = [2, 1]
    n_target = sum(mask_num)
    hidden = int(cfg.MODEL.MASK_FORMER.HIDDEN_DIM)

    text_cond = torch.randn(n_target, num_stages, cond_dim)
    reliability = torch.full((n_target, num_stages, 1), 0.5)
    seg_emb = torch.randn(n_target, 1, hidden)
    set_emb_b = torch.randn(2, 1, hidden)

    if args.phase == "alpha005":
        with torch.no_grad():
            for wti in model.tg_swin_controller.wti_blocks.values():
                wti.alpha.fill_(0.05)

    loss = _run_seg_loss_backward(
        model, images, mask_num, text_cond, reliability, seg_emb, set_emb_b
    )
    print(f"[info] segmentation loss = {loss:.6e} phase={args.phase}")

    ctrl = model.tg_swin_controller
    report_fn = _report_enhanced if enhanced else _report_v15
    wti_cls = StageEnhancedWTIHeadAware if enhanced else StageWTIHeadAware
    ok = True
    for stage in ("1", "2", "3"):
        wti = ctrl.wti_blocks[stage]
        assert isinstance(wti, wti_cls)
        report_fn(wti, stage)
        if _grad_norm(wti.alpha) <= 0:
            ok = False
            print(f"    [WARN] stage {stage} alpha grad is zero")
        if enhanced and args.phase == "alpha005":
            checks = (
                ("head_mixer_gamma", wti.head_mixer_gamma),
                ("visual_proj", wti.visual_proj[-1].weight),
                ("text_proj", wti.text_proj[-1].weight),
            )
            for name, p in checks:
                if _grad_norm(p) <= 0:
                    ok = False
                    print(f"    [WARN] stage {stage} {name} grad is zero")

    if model.tg_swin_set_control is not None:
        sc_grads = []
        for i, head in enumerate(model.tg_swin_set_control.stage_heads):
            wg = _grad_norm(head.weight)
            bg = _grad_norm(head.bias)
            sc_grads.append(max(wg, bg))
            print(f"  tg_swin_set_control stage_heads[{i}] grad norm (w/b) = {wg:.6e} / {bg:.6e}")
        if max(sc_grads) <= 0:
            ok = False
            print("    [WARN] tg_swin_set_control grad is zero on all stages")

    if ok:
        print("[PASS] probe_enhanced_wti_v2_seg_loss_gradient")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
