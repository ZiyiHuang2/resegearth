#!/usr/bin/env python3
"""DR-EWTI: single-batch segmentation loss backward; report TG-Swin gradient norms."""

from __future__ import annotations

import argparse
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESEG_ROOT = os.path.abspath(os.path.join(REPO, ".."))
DEFAULT_MODEL = os.path.join(RESEG_ROOT, "pretrained_model/mllm/Mipha-3B")
sys.path.insert(0, REPO)

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.model.mask_encoder.swin_trans import build_swin_b

DR_EWTI_CFG = "segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin_dr_ewti.yaml"


def _grad_norm(param) -> float:
    if param.grad is None:
        return 0.0
    return float(param.grad.detach().norm().item())


def _report_stage(controller, stage: str):
    wti = controller.wti_blocks[stage]
    dr = controller.dr_wti_blocks[stage]
    fusion = dr.evidence_fusion
    print(f"  stage {stage}:")
    print(f"    alpha.grad norm          = {_grad_norm(wti.alpha):.6e}")
    print(f"    evidence_relation_gate   = {_grad_norm(dr.evidence_relation_gate):.6e}")
    print(f"    evidence_feature_proj    = {_grad_norm(fusion.evidence_feature_proj.weight):.6e}")
    print(f"    evidence_q               = {_grad_norm(dr.evidence_q.weight):.6e}")
    print(f"    evidence_k               = {_grad_norm(dr.evidence_k.weight):.6e}")


def _run_seg_loss_backward(model, images, mask_num, text_cond, reliability, seg_emb):
    n_target = sum(mask_num)
    coarse = model.get_shared_coarse_evidence(images, seg_emb, mask_num)
    images_exp = model._repeat_images_per_target(images, mask_num)
    image_features = model.get_vision_tower_feature(
        images_exp,
        text_cond=text_cond,
        reliability=reliability,
        coarse_evidence=coarse,
        enable_tg_swin=True,
    )
    mask_features, _, multi_scale = model.pixel_decoder.forward_features(image_features)
    mask_outputs = model.predictor(multi_scale, mask_features, None, None, seg_emb)

    gt_h, gt_w = mask_outputs["pred_masks"].shape[-2:]
    targets = []
    for i in range(n_target):
        m = torch.zeros(1, gt_h, gt_w)
        m[:, i * 4:(i + 1) * 4, i * 4:(i + 1) * 4] = 1.0
        targets.append({"labels": torch.tensor([0]), "masks": m, "valid": None, "inst_id": None})

    model.tg_swin_controller.zero_grad(set_to_none=True)
    losses = model.criterion(mask_outputs, targets)
    mask_loss = sum(
        v * model.weight_dict[k]
        for k, v in losses.items()
        if v is not None and k in model.weight_dict
    )
    mask_loss.backward()
    return float(mask_loss.detach()), mask_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    args = parser.parse_args()
    os.chdir(REPO)

    if not os.path.isdir(args.model_path):
        print(f"[ERROR] model path not found: {args.model_path}")
        sys.exit(1)

    cfg = get_mask_config(DR_EWTI_CFG)
    model = SegEarthR2.from_pretrained(args.model_path, mask_decoder_cfg=cfg, torch_dtype=torch.float32)
    model.initial_mask_module(None, None)
    model.get_model().vision_tower_mask = build_swin_b(None)
    model.train()

    for p in model.parameters():
        p.requires_grad = False
    for p in model.tg_swin_controller.parameters():
        p.requires_grad = True
    for p in model.pixel_decoder.parameters():
        p.requires_grad = True
    for p in model.predictor.parameters():
        p.requires_grad = True

    images = torch.randn(2, 3, 384, 384)
    mask_num = [2, 1]
    n_target = sum(mask_num)
    hidden = int(cfg.MODEL.MASK_FORMER.HIDDEN_DIM)

    text_cond = torch.randn(n_target, 4, int(cfg.TG_SWIN.COND_DIM))
    reliability = torch.full((n_target, 4, 1), 0.5)
    seg_emb = torch.randn(n_target, 1, hidden)

    print("[phase A] init alpha=0, evidence_relation_gate=0")
    loss_a, _ = _run_seg_loss_backward(model, images, mask_num, text_cond, reliability, seg_emb)
    print(f"[info] segmentation loss = {loss_a:.6e}")
    print("[PASS] real segmentation loss backward — gradient norms (init):")
    ctrl = model.tg_swin_controller
    for stage in ("1", "2", "3"):
        _report_stage(ctrl, stage)

    with torch.no_grad():
        for wti in ctrl.wti_blocks.values():
            wti.alpha.fill_(0.05)

    print("\n[phase B] alpha=0.05 (post-init train step), gate still 0")
    loss_b, _ = _run_seg_loss_backward(model, images, mask_num, text_cond, reliability, seg_emb)
    print(f"[info] segmentation loss = {loss_b:.6e}")
    print("[PASS] real segmentation loss backward — gradient norms (alpha=0.05):")
    finite_ok = True
    nonzero_ok = True
    for stage in ("1", "2", "3"):
        _report_stage(ctrl, stage)
        wti = ctrl.wti_blocks[stage]
        dr = ctrl.dr_wti_blocks[stage]
        checks = (
            ("alpha", wti.alpha),
            ("gate", dr.evidence_relation_gate),
            ("feat_proj", dr.evidence_fusion.evidence_feature_proj.weight),
            ("evidence_q", dr.evidence_q.weight),
            ("evidence_k", dr.evidence_k.weight),
        )
        for name, p in checks:
            g = p.grad
            if g is None or not torch.isfinite(g).all():
                finite_ok = False
                print(f"    [WARN] stage {stage} {name}: grad missing or non-finite")
            elif float(g.abs().sum()) == 0.0 and name != "feat_proj":
                # gate@0: evidence_q/k may be 0; gate itself must be non-zero
                if name == "gate":
                    nonzero_ok = False
                    print(f"    [WARN] stage {stage} {name}: grad is exactly zero")
            elif name in ("evidence_q", "evidence_k") and float(g.abs().sum()) == 0.0:
                print(f"    [info] stage {stage} {name}: grad zero at gate=0 (expected)")

    gate_norms = [
        float(ctrl.dr_wti_blocks[s].evidence_relation_gate.grad.norm())
        for s in ("1", "2", "3")
        if ctrl.dr_wti_blocks[s].evidence_relation_gate.grad is not None
    ]
    if gate_norms and min(gate_norms) > 0:
        print(f"[PASS] evidence_relation_gate grad norms (alpha=0.05): {gate_norms}")
    else:
        nonzero_ok = False

    if finite_ok and nonzero_ok:
        print("[PASS] probe_tgswin_dr_ewti_seg_loss_gradient")
    else:
        print("[FAIL] probe_tgswin_dr_ewti_seg_loss_gradient — see warnings above")
        sys.exit(1)


if __name__ == "__main__":
    main()
