#!/usr/bin/env python3
"""Pass A integration: feature repeat, mock coarse evidence, real SegEarthR2 forward."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESEG_ROOT = os.path.abspath(os.path.join(REPO, ".."))
DEFAULT_MODEL = os.path.join(RESEG_ROOT, "pretrained_model/mllm/Mipha-3B")
sys.path.insert(0, REPO)

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.model.mask_encoder.swin_trans import build_swin_b

DR_EWTI_CFG = "segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin_dr_ewti.yaml"


def _load_dr_model(model_path: str) -> SegEarthR2:
    cfg = get_mask_config(DR_EWTI_CFG)
    model = SegEarthR2.from_pretrained(model_path, mask_decoder_cfg=cfg, torch_dtype=torch.float32)
    model.initial_mask_module(None, None)
    swin = build_swin_b(None)
    model.get_model().vision_tower_mask = swin
    return model


def test_repeat_features_per_target(model: SegEarthR2):
    mask_features = torch.zeros(2, 8, 32, 32)
    mask_features[0] = 1.0
    mask_features[1] = 2.0
    ms0 = torch.zeros(2, 16, 64, 64)
    ms0[0] = 10.0
    ms0[1] = 20.0
    ms1 = torch.zeros(2, 32, 32, 32)
    ms1[0] = 100.0
    ms1[1] = 200.0

    out_mf = model._repeat_features_per_target({"mask_features": mask_features}, [2, 1])
    mf = out_mf["mask_features"]
    assert mf.shape[0] == 3
    assert torch.allclose(mf[0], mf[1]) and torch.allclose(mf[0], torch.ones_like(mf[0]))
    assert torch.allclose(mf[2], torch.full_like(mf[2], 2.0))

    out_ms = model._repeat_features_per_target({"ms0": ms0, "ms1": ms1}, [2, 1])
    assert out_ms["ms0"].shape[0] == 3
    assert torch.allclose(out_ms["ms0"][0], out_ms["ms0"][1])
    assert torch.allclose(out_ms["ms0"][0, 0], torch.tensor(10.0))
    assert torch.allclose(out_ms["ms0"][2, 0], torch.tensor(20.0))
    assert torch.allclose(out_ms["ms1"][2, 0], torch.tensor(200.0))
    print("[PASS] _repeat_features_per_target mask_features/ms* order sample0,sample0,sample1")


class _MockPixelDecoder(nn.Module):
    def __init__(self, shared_batch: int, hidden: int = 8):
        super().__init__()
        self.shared_batch = shared_batch
        self.hidden = hidden
        self.last_mask_batch = None
        self.last_ms_batches: List[int] = []

    def forward_features(self, features_dict):
        b = int(next(iter(features_dict.values())).shape[0])
        assert b == self.shared_batch, f"Pass A shared batch={self.shared_batch}, got {b}"
        self.last_mask_batch = b
        mask_features = torch.zeros(b, self.hidden, 32, 32, device=next(iter(features_dict.values())).device)
        for i in range(b):
            mask_features[i] = float(i + 1)
        ms = [
            torch.full((b, self.hidden, 64, 64), float(i + 1), device=mask_features.device)
            for i in range(3)
        ]
        return mask_features, None, ms


class _MockPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_seg_batch = None
        self.last_mask_batch = None

    def forward(self, multi_scale_list, mask_features, _a, _b, seg_embeddings):
        self.last_seg_batch = int(seg_embeddings.shape[0])
        self.last_mask_batch = int(mask_features.shape[0])
        t = self.last_seg_batch
        return {"pred_masks": torch.randn(t, 1, 32, 32)}


def test_pass_a_mock(model: SegEarthR2):
    images = torch.randn(2, 3, 384, 384)
    mask_num = [2, 1]
    n_target = sum(mask_num)
    hidden = int(model.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
    seg_emb = torch.randn(n_target, 1, hidden)

    real_get_feat = model.get_vision_tower_feature
    mock_pd = _MockPixelDecoder(shared_batch=2)
    mock_pred = _MockPredictor()
    model.pixel_decoder = mock_pd
    model.predictor = mock_pred

    def _fake_shared(images_in, enable_tg_swin=True, **kwargs):
        assert enable_tg_swin is False
        assert images_in.shape[0] == 2
        return {
            "res2": torch.randn(2, 96, 96, 96),
            "res3": torch.randn(2, 192, 48, 48),
            "res4": torch.randn(2, 384, 24, 24),
            "res5": torch.randn(2, 768, 12, 12),
        }

    model.get_vision_tower_feature = _fake_shared  # type: ignore[method-assign]
    coarse = model.get_shared_coarse_evidence(images, seg_emb, mask_num)

    assert coarse.shape == (n_target, 1, 32, 32)
    assert not coarse.requires_grad
    assert mock_pd.last_mask_batch == 2
    assert mock_pred.last_mask_batch == n_target
    assert mock_pred.last_seg_batch == n_target
    repeated = model._repeat_features_per_target(
        {"mask_features": mock_pd.forward_features({"x": torch.zeros(2, 1)})[0]}, mask_num
    )["mask_features"]
    assert repeated.shape[0] == n_target
    assert torch.allclose(repeated[0], repeated[1])
    assert not torch.allclose(repeated[0], repeated[2])
    print("[PASS] Pass A mock: shared B=2 → repeated T=3 coarse_prob detached")

    model.get_vision_tower_feature = real_get_feat  # type: ignore[method-assign]


def test_pass_a_real_segearthr2(model: SegEarthR2):
    images = torch.randn(2, 3, 384, 384)
    mask_num = [2, 1]
    n_target = sum(mask_num)
    hidden = int(model.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
    seg_emb = torch.randn(n_target, 1, hidden)
    try:
        coarse = model.get_shared_coarse_evidence(images, seg_emb, mask_num)
    except Exception as exc:
        print(f"[FAIL] Pass A real SegEarthR2: {type(exc).__name__}: {exc}")
        raise
    assert coarse.shape[0] == n_target
    assert coarse.shape[1] == 1
    assert not coarse.requires_grad
    print(f"[PASS] Pass A real SegEarthR2 coarse_prob {tuple(coarse.shape)} requires_grad=False")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    args = parser.parse_args()
    os.chdir(REPO)

    if not os.path.isdir(args.model_path):
        print(f"[ERROR] model path not found: {args.model_path}")
        sys.exit(1)

    model = _load_dr_model(args.model_path)
    assert model._use_coarse_evidence()
    assert model._is_dr_ewti()

    test_repeat_features_per_target(model)
    test_pass_a_mock(model)
    test_pass_a_real_segearthr2(model)
    print("[PASS] smoke_tgswin_dr_ewti_pass_a")


if __name__ == "__main__":
    main()
