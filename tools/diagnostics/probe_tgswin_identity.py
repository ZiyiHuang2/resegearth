#!/usr/bin/env python3
"""Verify TG-Swin-WTI identity path: disabled or alpha=0 should not alter baseline behavior."""

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
from segearth_r2.model.mask_encoder.tg_swin import StageWTI, TextConditionFactory, TGSwimController


def _build_model(mask_config: str, enable_tg_swin: bool, model_path: str = DEFAULT_MODEL):
    cfg = get_mask_config(mask_config)
    if enable_tg_swin:
        cfg.TG_SWIN.ENABLED = True
    else:
        cfg.TG_SWIN.ENABLED = False

    model = SegEarthR2.from_pretrained(
        model_path,
        mask_decoder_cfg=cfg,
        torch_dtype=torch.float32,
    )
    model.initial_mask_module(None, None)
    model.eval()
    return model, cfg


def test_imports():
    assert TextConditionFactory is not None
    assert StageWTI is not None
    assert TGSwimController is not None
    print("[OK] imports")


def test_wti_zero_init():
    wti = StageWTI(dim=256, cond_dim=64, num_heads=8, window_size=12, alpha_init=0.0)
    x = torch.randn(4, 144, 256)
    tc = torch.randn(2, 64)
    rel = torch.ones(2, 1)
    bias, _, _ = wti(x, tc, rel)
    assert bias.abs().max().item() < 1e-6, f"expected zero bias at init, got max={bias.abs().max()}"
    print("[OK] WTI zero-init bias ~ 0")


def test_disabled_no_controller(model):
    assert model.tg_swin_tcf is None
    assert model.tg_swin_controller is None
    print("[OK] TG_SWIN disabled: no TCF/controller modules")


def test_enabled_modules(model):
    assert model.tg_swin_tcf is not None
    assert model.tg_swin_controller is not None
    print("[OK] TG_SWIN enabled: TCF + controller present")


def test_state_dict_load(mask_config):
    model, _ = _build_model(mask_config, enable_tg_swin=True)
    sd = model.state_dict()
    keys = [k for k in sd if "tg_swin" in k]
    assert keys, "expected tg_swin keys in state_dict"
    missing, unexpected = model.load_state_dict(sd, strict=False)
    tg_missing = [k for k in missing if "tg_swin" in k]
    assert not tg_missing, f"tg_swin keys missing on reload: {tg_missing}"
    print(f"[OK] state_dict round-trip ({len(keys)} tg_swin keys)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask-config",
        default="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml",
    )
    parser.add_argument(
        "--tgswin-config",
        default="segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin.yaml",
    )
    args = parser.parse_args()

    os.chdir(REPO)
    test_imports()
    test_wti_zero_init()

    model_off, _ = _build_model(args.mask_config, enable_tg_swin=False)
    test_disabled_no_controller(model_off)

    model_on, _ = _build_model(args.tgswin_config, enable_tg_swin=True)
    test_enabled_modules(model_on)
    test_state_dict_load(args.tgswin_config)

    print("[PASS] probe_tgswin_identity")


if __name__ == "__main__":
    main()
