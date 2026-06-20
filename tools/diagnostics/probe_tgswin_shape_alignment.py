#!/usr/bin/env python3
"""Verify per-target Swin repeat alignment for TG-Swin-WTI."""

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


def test_repeat_alignment(model):
    mask_num = [2, 1, 3]
    images = torch.randn(3, 3, 384, 384)
    expanded = model._repeat_images_per_target(images, mask_num)
    n_target = sum(mask_num)
    assert expanded.shape[0] == n_target
    print(f"[OK] image repeat: batch={expanded.shape[0]} == sum(mask_num)={n_target}")


def test_swin_output_shapes(model):
    mask_num = [2, 1]
    n_target = sum(mask_num)
    images = torch.randn(n_target, 3, 384, 384)
    num_stages = int(getattr(model.mask_decoder_cfg.TG_SWIN, "NUM_STAGES", 4))
    cond_dim = model.tg_swin_controller.cond_dim
    text_cond = torch.randn(n_target, num_stages, cond_dim)
    reliability = torch.full((n_target, num_stages, 1), 0.5)

    swin = build_swin_b(None)
    model.get_model().vision_tower_mask = swin
    outs = swin(
        images,
        text_cond=text_cond,
        reliability=reliability,
        tg_swin_controller=model.tg_swin_controller,
    )
    for i, feat in enumerate(outs):
        assert feat.shape[0] == n_target
    print(f"[OK] Swin outputs batch={n_target} for all stages")


def test_tcf_forward(model):
    hidden = torch.randn(2, 32, model.config.hidden_size)
    seg_idx = torch.zeros(2, 32, dtype=torch.bool)
    seg_idx[0, 10] = True
    seg_idx[1, 15] = True
    refer_span = torch.zeros(2, 32, dtype=torch.bool)
    refer_span[0, 5:8] = True
    refer_span[1, 12:14] = True
    seg_hidden, text_cond, reliability = model.build_text_cond(
        hidden, seg_idx, refer_span_mask=refer_span
    )
    num_stages = int(getattr(model.mask_decoder_cfg.TG_SWIN, "NUM_STAGES", 4))
    cond_dim = model.tg_swin_controller.cond_dim
    assert text_cond.shape == (2, num_stages, cond_dim), text_cond.shape
    assert reliability.shape == (2, num_stages, 1), reliability.shape
    print(f"[OK] TCF v1.5 output shapes stage_text_cond={tuple(text_cond.shape)}")


def test_frozen_swin_no_grad(model):
    swin = build_swin_b(None)
    model.get_model().vision_tower_mask = swin
    for p in swin.parameters():
        p.requires_grad = False

    x = torch.randn(2, 3, 384, 384)
    hidden = torch.randn(1, 16, model.config.hidden_size)
    seg_idx = torch.zeros(1, 16, dtype=torch.bool)
    seg_idx[0, 8] = True
    refer_span = torch.zeros(1, 16, dtype=torch.bool)
    refer_span[0, 2:6] = True
    _, text_cond, reliability = model.build_text_cond(hidden, seg_idx, refer_span_mask=refer_span)

    outs = swin(
        x,
        text_cond=text_cond,
        reliability=reliability,
        tg_swin_controller=model.tg_swin_controller,
    )
    loss = sum(o.float().sum() for o in outs)
    loss.backward()

    swin_grad = sum(
        float(p.grad.abs().sum().item()) for p in swin.parameters() if p.grad is not None
    )
    alpha_grad = float(model.tg_swin_controller.wti_blocks["1"].alpha.grad.abs().sum().item())
    assert swin_grad == 0.0
    assert alpha_grad > 0
    print(f"[OK] frozen Swin grad=0, alpha.grad={alpha_grad:.6e} (real init)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask-config",
        default="segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin_v15.yaml",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    args = parser.parse_args()
    os.chdir(REPO)

    cfg = get_mask_config(args.mask_config)
    cfg.TG_SWIN.ENABLED = True

    model = SegEarthR2.from_pretrained(
        args.model_path,
        mask_decoder_cfg=cfg,
        torch_dtype=torch.float32,
    )
    model.initial_mask_module(None, None)
    model.train()

    test_repeat_alignment(model)
    test_tcf_forward(model)
    test_swin_output_shapes(model)
    test_frozen_swin_no_grad(model)

    print("[PASS] probe_tgswin_shape_alignment")


if __name__ == "__main__":
    main()
