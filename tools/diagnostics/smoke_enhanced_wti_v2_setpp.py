#!/usr/bin/env python3
"""Smoke: no-coarse Enhanced WTI v2 + SET++ per-target alignment."""

from __future__ import annotations

import sys
from unittest import mock

import torch

REPO = "/root/rivermind-data/huangziyi/reseg/segearth+dr-ewti-setpp"
sys.path.insert(0, REPO)

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.model.mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MultiScaleMaskedTransformerDecoderForOPTPreTrain,
)
from segearth_r2.model.mask_encoder.tg_swin import (
    SETControlHead,
    StageDynamicRelationalWTI,
    StageEnhancedWTIHeadAware,
    StageWTIHeadAware,
    TGSwimController,
)

ENHANCED_CFG = f"{REPO}/segearth_r2/model/mask_decoder/mask_config/maskformer2_enhanced_wti_v2_setpp.yaml"
DR_CFG = f"{REPO}/segearth_r2/model/mask_decoder/mask_config/maskformer2_dr_ewti_setpp.yaml"
V15_CFG = f"{REPO}/segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin_v15_setpp.yaml"


def test_config_no_coarse():
    cfg = get_mask_config(ENHANCED_CFG)
    tg = cfg.TG_SWIN
    assert tg.VERSION == "enhanced-wti-v2", tg.VERSION
    assert not tg.USE_COARSE_EVIDENCE
    assert tg.ENHANCED_WTI
    assert not getattr(tg, "USE_DR_EWTI", True)
    assert getattr(tg, "USE_SET_TGSWIN_CONTROL", False)
    print("[PASS] config: enhanced-wti-v2, no coarse, no DR, set_control=true")


def test_config_flag():
    enhanced = get_mask_config(ENHANCED_CFG)
    assert getattr(enhanced.TG_SWIN, "USE_SET_TGSWIN_CONTROL", False)
    v15 = get_mask_config(V15_CFG)
    assert not getattr(v15.TG_SWIN, "USE_SET_TGSWIN_CONTROL", False)
    print("[PASS] config flag: enhanced=true, v15=false")


def test_set_control_head_shape():
    T, S, D = 3, 3, 256
    head = SETControlHead(text_dim=D, num_stages=S, init_bias=4.0)
    set_hidden = torch.randn(T, D)
    out = head(set_hidden)
    assert out.shape == (T, S, 1), out.shape
    mean_val = out.mean().item()
    assert 0.95 <= mean_val <= 1.0, f"init mean={mean_val}"
    print(f"[PASS] SETControlHead shape={tuple(out.shape)} init_mean={mean_val:.4f}")


def test_compute_bias_with_set_control():
    torch.manual_seed(42)
    ctrl = TGSwimController(
        cond_dim=64,
        wti_rank=16,
        wti_stages=[1],
        window_size=12,
        enhanced_wti=True,
        log_stats=True,
    )
    T, BW, Nw = 2, 4, 144
    x_windows = torch.randn(BW, Nw, 256)
    text_cond = torch.randn(T, 4, 64)
    reliability = torch.rand(T, 4, 1)

    bias_none = ctrl.compute_bias(1, 0, x_windows, text_cond, reliability, set_control=None)
    set_control = torch.ones(T, 4, 1)
    bias_one = ctrl.compute_bias(1, 0, x_windows, text_cond, reliability, set_control=set_control)
    assert bias_none is not None and bias_one is not None
    assert bias_none.shape == (BW, 8, Nw, Nw), bias_none.shape
    assert bias_one.shape == bias_none.shape
    torch.testing.assert_close(bias_none, bias_one, rtol=1e-5, atol=1e-6)

    set_control_mod = torch.full((T, 4, 1), 0.5)
    bias_half = ctrl.compute_bias(1, 0, x_windows, text_cond, reliability, set_control=set_control_mod)
    torch.testing.assert_close(bias_half, bias_none * 0.5, rtol=1e-5, atol=1e-6)
    print(f"[PASS] compute_bias with set_control shape={tuple(bias_none.shape)}, regression ok")


def test_identity_init():
    torch.manual_seed(0)
    H, Nw, r = 8, 144, 16
    T, BW = 2, 4
    block = StageEnhancedWTIHeadAware(
        dim=256,
        cond_dim=64,
        num_heads=H,
        window_size=12,
        rank=r,
        alpha_init=0.0,
    )
    x_windows = torch.randn(BW, Nw, 256, requires_grad=True)
    text_cond = torch.randn(T, 64, requires_grad=True)
    reliability = torch.ones(T, 1)

    set_control = torch.ones(BW, 1, 1, 1)
    attn_bias, _, _ = block(x_windows, text_cond, reliability, set_control=set_control)
    assert attn_bias.abs().max().item() == 0.0, "alpha=0 should yield zero attn_bias"

    block.alpha.data.fill_(0.05)
    set_control_var = torch.ones(BW, 1, 1, 1, requires_grad=True)
    attn_bias2, _, _ = block(x_windows, text_cond, reliability, set_control=set_control_var)
    attn_bias2.mean().backward()
    assert set_control_var.grad is not None
    assert set_control_var.grad.abs().sum().item() > 0
    print("[PASS] identity init: alpha=0 -> zero bias; set_control receives grad")


def test_enhanced_wti_module():
    torch.manual_seed(0)
    H, Nw, r = 8, 144, 16
    T, BW = 3, 6
    block = StageEnhancedWTIHeadAware(
        dim=256,
        cond_dim=64,
        num_heads=H,
        window_size=12,
        rank=r,
        use_head_mixer=True,
        gamma_init=0.0,
        alpha_init=0.05,
    )
    x_windows = torch.randn(BW, Nw, 256, requires_grad=True)
    text_cond = torch.randn(T, 64, requires_grad=True)
    reliability = torch.rand(T, 1)

    attn_bias, raw_bias, gate = block(x_windows, text_cond, reliability)
    assert attn_bias.shape == (BW, H, Nw, Nw), attn_bias.shape
    assert raw_bias.shape == (BW, H, Nw, Nw)

    # gamma_init=0: mixer delta is active (default Linear init), but gated by tanh(gamma)=0.
    # Verify gamma receives grad via raw path (no alpha gate).
    loss_raw = raw_bias.mean()
    loss_raw.backward(retain_graph=True)
    gamma_grad_raw = block.head_mixer_gamma.grad.abs().sum().item()
    assert gamma_grad_raw > 0, f"head_mixer_gamma grad vanishes on raw path: {gamma_grad_raw}"

    block.zero_grad(set_to_none=True)
    x_windows.grad = None
    text_cond.grad = None

    loss = attn_bias.mean()
    loss.backward()
    assert block.visual_proj[-1].weight.grad is not None
    assert block.text_proj[-1].weight.grad is not None
    assert block.alpha.grad is not None
    assert block.alpha.grad.abs().sum().item() > 0
    assert block.head_mixer_gamma.grad is not None
    gamma_grad = block.head_mixer_gamma.grad.abs().sum().item()
    assert gamma_grad > 0, f"head_mixer_gamma grad vanishes on attn path: {gamma_grad}"
    print(
        f"[PASS] StageEnhancedWTIHeadAware shape={tuple(attn_bias.shape)}, "
        f"gamma.grad raw={gamma_grad_raw:.6e} attn={gamma_grad:.6e}"
    )


def test_controller_enhanced_no_dr():
    ctrl = TGSwimController(
        cond_dim=64,
        wti_rank=16,
        wti_stages=[1, 2, 3],
        window_size=12,
        enhanced_wti=True,
        use_dr_ewti=False,
    )
    assert ctrl.enhanced_wti
    assert not ctrl.use_dr_ewti
    assert ctrl.dr_wti_blocks is None
    for key, block in ctrl.wti_blocks.items():
        assert isinstance(block, StageEnhancedWTIHeadAware), type(block)
    print("[PASS] TGSwimController instantiates StageEnhancedWTIHeadAware, no DR blocks")


def test_controller_dr_unchanged():
    ctrl = TGSwimController(
        cond_dim=64,
        wti_rank=16,
        wti_stages=[1, 2, 3],
        window_size=12,
        enhanced_wti=False,
        use_dr_ewti=True,
    )
    assert ctrl.dr_wti_blocks is not None
    assert len(ctrl.dr_wti_blocks) == 3
    assert isinstance(ctrl.wti_blocks["1"], StageWTIHeadAware)
    assert isinstance(ctrl.dr_wti_blocks["1"], StageDynamicRelationalWTI)
    print("[PASS] DR config still builds StageWTIHeadAware + StageDynamicRelationalWTI")


def test_compute_bias_shape():
    ctrl = TGSwimController(
        cond_dim=64,
        wti_rank=16,
        wti_stages=[1],
        window_size=12,
        enhanced_wti=True,
    )
    T, BW, Nw = 2, 4, 144
    x_windows = torch.randn(BW, Nw, 256)
    text_cond = torch.randn(T, 4, 64)
    reliability = torch.rand(T, 4, 1)
    bias = ctrl.compute_bias(1, 0, x_windows, text_cond, reliability)
    assert bias is not None
    assert bias.shape == (BW, 8, Nw, Nw), bias.shape
    print(f"[PASS] compute_bias shape={tuple(bias.shape)}")


def test_no_pass_a_on_enhanced_config():
    """get_shared_coarse_evidence must not run when USE_COARSE_EVIDENCE=false."""
    cfg = get_mask_config(ENHANCED_CFG)
    model = mock.MagicMock()
    model._get_tg_swin_cfg.return_value = cfg.TG_SWIN
    model._use_coarse_evidence = SegEarthR2._use_coarse_evidence.__get__(model, SegEarthR2)
    assert not model._use_coarse_evidence()
    print("[PASS] _use_coarse_evidence() false for enhanced config")


def test_setpp_per_target():
    cfg = get_mask_config(ENHANCED_CFG)
    pred = MultiScaleMaskedTransformerDecoderForOPTPreTrain(
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
        use_csqr=True,
    )
    T = 3
    mask_num = [1, 2]
    SET = torch.repeat_interleave(torch.randn(2, 1, 256), torch.tensor(mask_num), dim=0)
    SEG = torch.randn(T, 1, 256)
    out = pred(
        [torch.randn(T, 256, 128, 128), torch.randn(T, 256, 64, 64), torch.randn(T, 256, 32, 32)],
        torch.randn(T, 256, 32, 32),
        None,
        None,
        SEG_embedding=SEG,
        SET_embedding=SET,
        per_target_mode=True,
    )
    assert out["pred_masks"].shape[0] == T
    print(f"[PASS] SET++ per_target T={T}")


def main():
    test_config_no_coarse()
    test_config_flag()
    test_set_control_head_shape()
    test_compute_bias_with_set_control()
    test_identity_init()
    test_enhanced_wti_module()
    test_controller_enhanced_no_dr()
    test_controller_dr_unchanged()
    test_compute_bias_shape()
    test_no_pass_a_on_enhanced_config()
    test_setpp_per_target()
    print("[PASS] smoke_enhanced_wti_v2_setpp")


if __name__ == "__main__":
    main()
