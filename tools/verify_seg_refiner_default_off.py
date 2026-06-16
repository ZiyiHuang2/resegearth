#!/usr/bin/env python3
"""Verify use_seg_spatial_refiner=False keeps legacy decoder output path."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detectron2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2


def main():
    base = os.environ.get(
        "BASE_MODEL",
        "/root/rivermind-data/huangziyi/reseg/output/base/standard-base-siglip1-28w-gd4/merged_model",
    )
    mask_cfg_path = os.environ.get(
        "MASK_CONFIG",
        "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml",
    )
    mask_cfg = get_mask_config(mask_cfg_path)
    model = SegEarthR2.from_pretrained(base, mask_decoder_cfg=mask_cfg, add_cross_attn=True)
    model.config.use_seg_spatial_refiner = False
    model.eval()
    predictor = model.predictor
    has_refiner_module = getattr(predictor, "seg_spatial_refiner", None) is not None
    use_flag = bool(getattr(predictor, "use_seg_spatial_refiner", False))
    print(f"[INFO] seg_spatial_refiner module present={has_refiner_module} use_flag={use_flag}")

    b, c, h, w = 1, 256, 64, 64
    seg = torch.randn(b, 1, 256)
    mask_features = torch.randn(b, c, h, w)
    multi_scale = [torch.randn(b, c, h, w), torch.randn(b, c, h // 2, w // 2), torch.randn(b, c, h // 4, w // 4)]
    with torch.no_grad():
        out = predictor(
            multi_scale,
            mask_features,
            None,
            None,
            seg,
            text_tokens=None,
            text_mask=None,
        )
    extra_keys = [k for k in ("P_refine", "pred_masks_base") if k in out]
    if extra_keys:
        raise RuntimeError(f"[FAIL] default-off output contains refiner keys: {extra_keys}")
    if getattr(predictor, "_last_seg_refiner_log", None) is not None:
        raise RuntimeError("[FAIL] default-off set _last_seg_refiner_log")
    print(f"[OK] pred_masks shape={tuple(out['pred_masks'].shape)}")
    print("[OK] default-off path: no P_refine/pred_masks_base, refiner not invoked")


if __name__ == "__main__":
    main()
