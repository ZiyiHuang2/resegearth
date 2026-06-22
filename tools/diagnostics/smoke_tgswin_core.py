#!/usr/bin/env python3
"""Unified TG-Swin-WTI v1.5 smoke checks. Exit 0 only if all PASS."""

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
from segearth_r2.model.mask_encoder.tg_swin import StageWTIHeadAware, TextConditionFactory

IDENTITY_THRESH = 1e-5


def check_1_tcf_v15_init():
    """v1.5 TCF: stage-wise router → [N,S,C] + per-stage reliability."""
    tcf = TextConditionFactory(
        text_dim=64, cond_dim=32, reliability_init=0.0, version="v1.5", num_stages=4, stage_router=True
    )
    seg_hidden = torch.randn(3, 64)
    phrase_hidden = torch.randn(3, 5, 64)
    phrase_mask = torch.ones(3, 5, dtype=torch.bool)
    text_cond, reliability = tcf(
        seg_hidden, phrase_hidden=phrase_hidden, phrase_mask=phrase_mask
    )
    assert text_cond.shape == (3, 4, 32), text_cond.shape
    assert reliability.shape == (3, 4, 1), reliability.shape
    assert reliability.mean().item() > 0.4
    assert text_cond.abs().max().item() > 0
    print("[PASS] 1/7 v1.5 TCF stage_text_cond [N,S,C] + reliability [N,S,1]")


def check_2_wti_identity():
    """WTI v1.5: alpha=0 gate → attn_bias ≈ 0 despite non-zero raw pairwise bias."""
    torch.manual_seed(42)
    wti = StageWTIHeadAware(dim=256, cond_dim=64, num_heads=8, window_size=12, alpha_init=0.0)
    x = torch.randn(4, 144, 256)
    tc = torch.randn(2, 64)
    rel = torch.full((2, 1), 0.5)
    bias, raw, _ = wti(x, tc, rel)
    assert bias.shape == (4, 8, 144, 144), bias.shape
    assert bias.abs().max().item() < IDENTITY_THRESH, f"attn_bias max={bias.abs().max()}"
    assert raw.abs().max().item() > 1e-8, "raw bias should be non-zero at init (small random weights)"
    print("[PASS] 2/7 WTI v1.5 identity attn_bias≈0, raw bias non-zero")


def check_3_disabled_no_modules(model_path: str):
    cfg = get_mask_config(
        "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
    )
    cfg.TG_SWIN.ENABLED = False
    model = SegEarthR2.from_pretrained(model_path, mask_decoder_cfg=cfg, torch_dtype=torch.float32)
    model.initial_mask_module(None, None)
    assert model.tg_swin_tcf is None
    assert model.tg_swin_controller is None
    print("[PASS] 3/7 TG_SWIN disabled: no tg_swin_tcf / tg_swin_controller")


def check_4_forward_reorder(model_path: str):
    cfg = get_mask_config(
        "segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin.yaml"
    )
    model = SegEarthR2.from_pretrained(model_path, mask_decoder_cfg=cfg, torch_dtype=torch.float32)
    model.initial_mask_module(None, None)

    mask_num = [2, 1, 3]
    n_expanded = sum(mask_num)
    images = torch.randn(3, 3, 384, 384)
    expanded = model._repeat_images_per_target(images, mask_num)
    assert expanded.shape[0] == n_expanded

    hidden = torch.randn(1, 20, model.config.hidden_size)
    seg_idx = torch.zeros(1, 20, dtype=torch.bool)
    refer_span = torch.zeros(1, 20, dtype=torch.bool)
    refer_span[0, 2:6] = True
    seg_idx[0, 10] = True
    seg_idx[0, 15] = True
    n_seg = int(seg_idx.sum().item())

    phrase_hidden, phrase_mask = model._gather_refer_phrase_hidden(hidden, seg_idx, refer_span)
    assert phrase_hidden is not None and phrase_mask.any()
    seg_hidden, text_cond, reliability = model.build_text_cond(hidden, seg_idx, refer_span_mask=refer_span)
    assert seg_hidden.shape[0] == n_seg

    assert text_cond.dim() == 3, f"expected v1.5 stage_text_cond, got {text_cond.shape}"
    assert text_cond.shape[0] == n_seg
    assert reliability.shape[0] == n_seg

    swin = build_swin_b(None)
    model.get_model().vision_tower_mask = swin
    imgs = torch.randn(n_seg, 3, 384, 384)
    outs = swin(
        imgs,
        text_cond=text_cond,
        reliability=reliability,
        tg_swin_controller=model.tg_swin_controller,
    )
    for feat in outs:
        assert feat.shape[0] == n_seg
    print(f"[PASS] 4/7 forward reorder: images_expanded={n_expanded}, n_target={n_seg}, stage_text_cond={tuple(text_cond.shape)}")


def check_5_refer_span_not_fallback():
    tcf = TextConditionFactory(text_dim=32, cond_dim=16, reliability_init=0.0, version="v1.5", stage_router=True)

    def _boom(*args, **kwargs):
        raise AssertionError("_pool_local_context should not be called when refer span is present")

    tcf.router._pool_local_context = _boom  # type: ignore[method-assign]
    seg_hidden = torch.randn(2, 32)
    phrase_hidden = torch.randn(2, 4, 32)
    phrase_mask = torch.ones(2, 4, dtype=torch.bool)
    text_cond, _ = tcf(seg_hidden, phrase_hidden=phrase_hidden, phrase_mask=phrase_mask)
    assert text_cond.shape == (2, 4, 16)
    print("[PASS] 5/7 refer span → v1.5 router skips local pool fallback")


def check_6_train_init_gradient_v15():
    """Delegate to v1.5 dedicated probe (no manual param perturbation)."""
    import subprocess

    probe = os.path.join(os.path.dirname(__file__), "probe_tgswin_v15_train_init_gradient.py")
    python = sys.executable
    subprocess.run([python, probe], check=True, cwd=REPO)
    print("[PASS] 6/7 v1.5 train-init gradient probe (subprocess)")


def check_7_v1_legacy_path():
    tcf = TextConditionFactory(text_dim=32, cond_dim=16, version="v1", stage_router=False)
    seg_hidden = torch.randn(2, 32)
    text_cond, reliability = tcf(seg_hidden)
    assert text_cond.shape == (2, 16)
    assert reliability.shape == (2, 1)
    print("[PASS] 7/7 v1 legacy TCF path still works")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--version", default="v1.5", choices=["v1.5", "v1.6"])
    args = parser.parse_args()
    os.chdir(REPO)

    if not os.path.isdir(args.model_path):
        print(f"[ERROR] model path not found: {args.model_path}")
        sys.exit(1)

    if args.version == "v1.6":
        import subprocess
        v16_smoke = os.path.join(os.path.dirname(__file__), "smoke_tgswin_v16.py")
        subprocess.run([sys.executable, v16_smoke], check=True, cwd=REPO)
        print("[PASS] smoke_tgswin_core — delegated to smoke_tgswin_v16")
        return

    check_1_tcf_v15_init()
    check_2_wti_identity()
    check_3_disabled_no_modules(args.model_path)
    check_4_forward_reorder(args.model_path)
    check_5_refer_span_not_fallback()
    check_6_train_init_gradient_v15()
    check_7_v1_legacy_path()

    print("[PASS] smoke_tgswin_core — all checks passed")


if __name__ == "__main__":
    main()
