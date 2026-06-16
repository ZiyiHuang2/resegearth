#!/usr/bin/env python3
"""Lightweight smoke test for per-image joint [SEG] decoder ablation."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.model.mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MultiScaleMaskedTransformerDecoderForOPTPreTrain,
)


def test_decoder_query_embed_shape():
    hidden_dim = 256
    predictor = MultiScaleMaskedTransformerDecoderForOPTPreTrain(
        in_channels=256,
        hidden_dim=hidden_dim,
        num_queries=100,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=3,
        pre_norm=False,
        mask_dim=256,
        enforce_input_project=False,
        seg_norm=False,
        seg_proj=True,
        seg_fuse_score=False,
        use_seg_query=False,
    ).eval()

    h, w = 32, 32
    mask_features = torch.randn(1, hidden_dim, h, w)
    multi_scale = [
        torch.randn(1, 256, h, w),
        torch.randn(1, 256, h // 2, w // 2),
        torch.randn(1, 256, h // 4, w // 4),
    ]

    for k_i in [1, 2, 3]:
        seg_i = torch.randn(1, k_i, hidden_dim)
        out = predictor(multi_scale, mask_features, None, None, seg_i)
        assert out["pred_masks"].shape == (1, k_i, h, w), out["pred_masks"].shape
        for aux in out["aux_outputs"]:
            assert aux["pred_masks"].shape == (1, k_i, h, w)
    print("[PASS] decoder per-image shapes: K=1,2,3 -> pred_masks [1,K,H,W]")


def test_flatten_and_joint_predictor():
    mask_cfg = get_mask_config(
        str(ROOT / "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    )
    hidden_dim = mask_cfg.MODEL.MASK_FORMER.HIDDEN_DIM
    h, w = 64, 64
    bs = 4
    mask_num = [2, 1, 3, 1]
    sum_k = sum(mask_num)

    predictor = MultiScaleMaskedTransformerDecoderForOPTPreTrain(
        in_channels=mask_cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
        hidden_dim=hidden_dim,
        num_queries=mask_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
        nheads=mask_cfg.MODEL.MASK_FORMER.NHEADS,
        dim_feedforward=mask_cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
        dec_layers=mask_cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1,
        pre_norm=mask_cfg.MODEL.MASK_FORMER.PRE_NORM,
        mask_dim=mask_cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
        enforce_input_project=False,
        seg_norm=mask_cfg.MODEL.MASK_FORMER.SEG_NORM,
        seg_proj=mask_cfg.MODEL.MASK_FORMER.SEG_PROJ,
        seg_fuse_score=mask_cfg.MODEL.MASK_FORMER.FUSE_SCORE,
    ).eval()

    predictor_module = predictor

    class _StubModel:
        predictor = predictor_module

        _run_joint_predictor_by_image = SegEarthR2._run_joint_predictor_by_image
        _flatten_joint_decoder_outputs = SegEarthR2._flatten_joint_decoder_outputs

    stub = _StubModel()
    mask_features = torch.randn(bs, hidden_dim, h, w)
    multi_scale = [
        torch.randn(bs, 256, h, w),
        torch.randn(bs, 256, h // 2, w // 2),
        torch.randn(bs, 256, h // 4, w // 4),
    ]
    seg_embedding = torch.randn(sum_k, 1, hidden_dim)

    per_image = []
    seg_groups = torch.split(seg_embedding.squeeze(1), mask_num, dim=0)
    for i, seg_group in enumerate(seg_groups):
        seg_i = seg_group.unsqueeze(0)
        out_i = predictor(
            [feat[i : i + 1] for feat in multi_scale],
            mask_features[i : i + 1],
            None,
            None,
            seg_i,
        )
        per_image.append(out_i)
        print(f"  image {i}: seg_i {tuple(seg_i.shape)} -> pred_masks {tuple(out_i['pred_masks'].shape)}")

    flat = SegEarthR2._flatten_joint_decoder_outputs(per_image)
    assert flat["pred_masks"].shape == (sum_k, 1, h, w), flat["pred_masks"].shape
    for layer_idx, aux in enumerate(flat["aux_outputs"]):
        assert aux["pred_masks"].shape == (sum_k, 1, h, w), (
            layer_idx,
            aux["pred_masks"].shape,
        )

    joint_out = stub._run_joint_predictor_by_image(
        multi_scale, mask_features, seg_embedding, mask_num
    )
    assert joint_out["pred_masks"].shape == (sum_k, 1, h, w)
    for aux in joint_out["aux_outputs"]:
        assert aux["pred_masks"].shape == (sum_k, 1, h, w)

    # flatten order: img0 (2), img1 (1), img2 (3), img3 (1)
    ref = torch.cat([out["pred_masks"].squeeze(0).unsqueeze(1) for out in per_image], dim=0)
    assert torch.equal(joint_out["pred_masks"], ref)
    print(f"[PASS] mask_num={mask_num}, sumK={sum_k}, pred_masks {tuple(joint_out['pred_masks'].shape)}")
    print(f"[PASS] aux_outputs layers={len(joint_out['aux_outputs'])}, each pred_masks [7,1,{h},{w}]")


def test_real_checkpoint_smoke():
    if os.environ.get("SKIP_REAL_CKPT"):
        print("[SKIP] real checkpoint smoke (pipeline mode)")
        return
    ckpt = Path("/root/rivermind-data/huangziyi/reseg/output/base/standard-base-lasers-siglip1-28w-gd4/merged_model")
    if not ckpt.exists():
        print("[SKIP] baseline checkpoint not found")
        return

    mask_cfg = get_mask_config(
        str(ROOT / "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    )
    print("[INFO] loading merged model (may take a minute)...")
    use_cuda = torch.cuda.is_available()
    model = SegEarthR2.from_pretrained(
        str(ckpt),
        mask_decoder_cfg=mask_cfg,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
    )
    if not model.is_train_mask_decode:
        model.initial_mask_module()
    if use_cuda:
        model = model.cuda()
    model.eval()
    pred_device = next(model.predictor.parameters()).device
    dtype = next(model.predictor.parameters()).dtype

    hidden_dim = mask_cfg.MODEL.MASK_FORMER.HIDDEN_DIM
    h, w = 64, 64

    # single SEG
    mask_num_single = [1]
    seg_single = torch.randn(1, 1, hidden_dim, dtype=dtype, device=pred_device)
    mf = torch.randn(1, hidden_dim, h, w, dtype=dtype, device=pred_device)
    ms = [
        torch.randn(1, 256, h, w, dtype=dtype, device=pred_device),
        torch.randn(1, 256, h // 2, w // 2, dtype=dtype, device=pred_device),
        torch.randn(1, 256, h // 4, w // 4, dtype=dtype, device=pred_device),
    ]
    out_single = model._run_joint_predictor_by_image(ms, mf, seg_single, mask_num_single)
    assert out_single["pred_masks"].shape[0] == 1
    print(f"[PASS] real ckpt single-SEG: pred_masks {tuple(out_single['pred_masks'].shape)}")

    # multi SEG
    mask_num_multi = [3]
    seg_multi = torch.randn(3, 1, hidden_dim, dtype=dtype, device=pred_device)
    out_multi = model._run_joint_predictor_by_image(ms, mf, seg_multi, mask_num_multi)
    assert out_multi["pred_masks"].shape[0] == 3
    print(f"[PASS] real ckpt multi-SEG K=3: pred_masks {tuple(out_multi['pred_masks'].shape)}")


if __name__ == "__main__":
    test_decoder_query_embed_shape()
    test_flatten_and_joint_predictor()
    test_real_checkpoint_smoke()
    print("\nAll smoke checks passed.")
