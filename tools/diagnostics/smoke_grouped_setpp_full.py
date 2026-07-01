#!/usr/bin/env python3
"""Smoke: grouped SET++ Full path (B=2, mask_num=[2,3], T=5, Kmax=3)."""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import torch

from segearth_r2.model.mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MultiScaleMaskedTransformerDecoderForOPTPreTrain,
    deterministic_slot_pe,
    flatten_grouped_bank,
    regroup_target_maps_to_bank,
    regroup_mask_features_bank,
)
from segearth_r2.model.mask_encoder.tg_swin.set_control_head import SETControlHead
from segearth_r2.datasets.dataset import get_mask_config

ENHANCED_CFG = f"{REPO}/segearth_r2/model/mask_decoder/mask_config/ours_full_enhanced_tgswin_setpp.yaml"


def test_set_gate_broadcast():
    B, T, S, C = 2, 5, 3, 256
    mask_num = [2, 3]
    head = SETControlHead(text_dim=C, num_stages=S)
    set_emb_b = torch.randn(B, 1, C)
    set_gate_b = head(set_emb_b.squeeze(1))
    assert set_gate_b.shape == (B, S, 1), set_gate_b.shape
    target_to_image = torch.repeat_interleave(
        torch.arange(B, dtype=torch.long), torch.tensor(mask_num, dtype=torch.long)
    )
    set_gate_t = set_gate_b[target_to_image]
    assert set_gate_t.shape == (T, S, 1), set_gate_t.shape
    print(f"[PASS] set_gate_b={tuple(set_gate_b.shape)} set_gate_t={tuple(set_gate_t.shape)}")


def test_regroup_banks():
    B, T, Kmax, C, H, W = 2, 5, 3, 256, 16, 16
    mask_num = [2, 3]
    feat_t = torch.randn(T, C, H, W)
    slot_pe = deterministic_slot_pe(Kmax, C, feat_t.device, feat_t.dtype)
    mem_bank, valid = regroup_target_maps_to_bank(feat_t, mask_num, slot_pe=slot_pe)
    pos_bank, _ = regroup_target_maps_to_bank(feat_t, mask_num, slot_pe=None)
    assert mem_bank.shape == (B, Kmax, H * W, C), mem_bank.shape
    assert pos_bank.shape == (B, Kmax, H * W, C), pos_bank.shape
    assert valid.tolist() == [[True, True, False], [True, True, True]]
    mem_flat, key_pad = flatten_grouped_bank(mem_bank, valid)
    pos_flat, _ = flatten_grouped_bank(pos_bank, valid)
    assert mem_flat.shape == (Kmax * H * W, B, C), mem_flat.shape
    assert pos_flat.shape == mem_flat.shape
    assert key_pad.shape == (B, Kmax * H * W)
    print(
        f"[PASS] memory_bank={tuple(mem_bank.shape)} "
        f"memory_flat={tuple(mem_flat.shape)} grouped_pos_flat={tuple(pos_flat.shape)}"
    )


def test_grouped_decoder_forward():
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
    B, T, Kmax = 2, 5, 3
    mask_num = [2, 3]
    H, W = 32, 32
    multi_scale = [
        torch.randn(T, 256, 128, 128),
        torch.randn(T, 256, 64, 64),
        torch.randn(T, 256, 32, 32),
    ]
    mask_features = torch.randn(T, 256, H, W)
    mem_banks, pos_banks, mf_bank, slot_valid = pred.prepare_grouped_banks(
        multi_scale, mask_features, mask_num
    )
    SET_b = torch.randn(B, 1, 256)
    SEG_t = torch.randn(T, 1, 256)
    target_to_image = torch.repeat_interleave(
        torch.arange(B, dtype=torch.long), torch.tensor(mask_num, dtype=torch.long)
    )
    out = pred(
        multi_scale,
        mask_features,
        None,
        None,
        SEG_embedding=SEG_t,
        SET_embedding=SET_b,
        mask_num=mask_num,
        per_target_mode=False,
        grouped_setpp_mode=True,
        grouped_memory_banks=mem_banks,
        grouped_pos_banks=pos_banks,
        mask_features_bank=mf_bank,
        slot_valid_mask=slot_valid,
        target_to_image=target_to_image,
    )
    assert out["grouped_setpp_mode"] is True
    assert out["per_target_mode"] is False
    assert out["pred_set_union_mask"].shape == (B, 1, H, W), out["pred_set_union_mask"].shape
    assert out["pred_seg_masks_grouped"].shape == (B, Kmax, H, W), out["pred_seg_masks_grouped"].shape
    assert out["pred_seg_masks"].shape == (T, 1, H, W), out["pred_seg_masks"].shape
    assert out["pred_masks"].shape == (B, 1 + Kmax, H, W), out["pred_masks"].shape
    q_shape = (B, 1 + Kmax, 256)
    print(f"[PASS] decoder queries implied={q_shape}")
    print(f"[PASS] pred_set_union={tuple(out['pred_set_union_mask'].shape)}")
    print(f"[PASS] pred_seg_grouped={tuple(out['pred_seg_masks_grouped'].shape)}")
    print(f"[PASS] pred_seg_flat={tuple(out['pred_seg_masks'].shape)}")


def test_legacy_per_target_still_works():
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
    assert out["per_target_mode"] is True
    print(f"[PASS] legacy per_target T={T}")


def main():
    test_set_gate_broadcast()
    test_regroup_banks()
    test_grouped_decoder_forward()
    test_legacy_per_target_still_works()
    print("[PASS] smoke_grouped_setpp_full")


if __name__ == "__main__":
    main()
