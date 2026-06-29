#!/usr/bin/env python3
"""Integration smoke: per-target DR-EWTI + SET++ (B=2, mask_num=[1,2], T=3)."""
import sys

import torch

REPO = "/root/rivermind-data/huangziyi/reseg/segearth+dr-ewti-setpp"
sys.path.insert(0, REPO)

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MultiScaleMaskedTransformerDecoderForOPTPreTrain,
)
from segearth_r2.model.mask_decoder.mask_criterion.Mask_Criterion import Criterion, hungarian_matcher_InstructSeg
from segearth_r2.model.language_model.llava_phi import SegEarthR2


def build_predictor(cfg, use_csqr=True):
    return MultiScaleMaskedTransformerDecoderForOPTPreTrain(
        cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
        cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
        cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
        cfg.MODEL.MASK_FORMER.NHEADS,
        cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
        cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1,
        cfg.MODEL.MASK_FORMER.PRE_NORM,
        cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
        False,
        cfg.MODEL.MASK_FORMER.SEG_NORM,
        cfg.MODEL.MASK_FORMER.SEG_PROJ,
        cfg.MODEL.MASK_FORMER.FUSE_SCORE,
        use_csqr=use_csqr,
    )


def test_predictor_per_target():
    cfg = get_mask_config(f"{REPO}/segearth_r2/model/mask_decoder/mask_config/maskformer2_dr_ewti_setpp.yaml")
    pred = build_predictor(cfg)
    B, T = 2, 3
    mask_num = [1, 2]
    mask_features = torch.randn(T, 256, 32, 32)
    multi_scale = [torch.randn(T, 256, s, s) for s in (128, 64, 32)]
    SEG = torch.randn(T, 1, 256)
    SET_b = torch.randn(B, 1, 256)
    repeats = torch.tensor(mask_num, dtype=torch.long)
    SET = torch.repeat_interleave(SET_b, repeats, dim=0)
    assert SET.shape[0] == T
    out = pred(
        multi_scale, mask_features, None, None,
        SEG_embedding=SEG, SET_embedding=SET, per_target_mode=True,
    )
    assert out["pred_masks"].shape == (T, 2, 32, 32), out["pred_masks"].shape
    print(f"[PASS] predictor per_target T={T} -> {tuple(out['pred_masks'].shape)}")


def test_criterion_per_target():
    cfg = get_mask_config(f"{REPO}/segearth_r2/model/mask_decoder/mask_config/maskformer2_dr_ewti_setpp.yaml")
    pred = build_predictor(cfg)
    matcher = hungarian_matcher_InstructSeg(2.0, 5.0, 5.0, num_points=12544)
    crit = Criterion(
        matcher=matcher,
        losses=["SEG_labels", "masks", "union"],
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        device=torch.device("cpu"),
        setpp_closed_loop=True,
    )
    B, T = 2, 3
    mask_num = [1, 2]
    mask_features = torch.randn(T, 256, 32, 32)
    multi_scale = [torch.randn(T, 256, s, s) for s in (128, 64, 32)]
    SEG = torch.randn(T, 1, 256, requires_grad=True)
    SET_b = torch.randn(B, 1, 256)
    SET = torch.repeat_interleave(SET_b, torch.tensor(mask_num), dim=0)
    out = pred(
        multi_scale, mask_features, None, None,
        SEG_embedding=SEG, SET_embedding=SET, per_target_mode=True,
    )
    out["per_target_mode"] = True
    out["target_to_image"] = torch.tensor([0, 1, 1])
    out["mask_num"] = mask_num
    targets = []
    for _ in range(T):
        targets.append({
            "labels": torch.zeros(1, dtype=torch.long),
            "masks": torch.randint(0, 2, (1, 32, 32)).float(),
        })
    losses = crit(out, targets)
    total = sum(v for v in losses.values() if v is not None)
    total.backward()
    print(f"[PASS] criterion per_target backward ok, loss={float(total.detach()):.4f}")


def test_helpers():
    mask_num = [1, 2]
    set_b = torch.randn(2, 1, 64)
    set_t = SegEarthR2._repeat_set_embedding_per_target(set_b, mask_num)
    assert set_t.shape[0] == 3
    t2i = SegEarthR2._build_target_to_image(mask_num, torch.device("cpu"))
    assert t2i.tolist() == [0, 1, 1]
    print("[PASS] target_to_image + SET repeat")


def main():
    test_helpers()
    test_predictor_per_target()
    test_criterion_per_target()
    print("[PASS] smoke_dr_ewti_setpp_integration")


if __name__ == "__main__":
    main()
