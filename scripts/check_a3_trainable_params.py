#!/usr/bin/env python3
"""Verify A3-frozen trainable parameter scope without launching training."""

import argparse
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.train.a3_training_utils import (
    A3_FORBIDDEN_KEYWORDS,
    A3_TRAINABLE_KEYWORDS,
    apply_a3_frozen_training,
    assert_no_forbidden_trainables,
    collect_trainable_names,
    print_trainable_parameter_report,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_name_or_path",
        default="/root/rivermind-data/huangziyi/reseg/output/base/standard-base-lasers-siglip1-28w-gd4/merged_model",
    )
    p.add_argument("--mask_config", default="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    p.add_argument("--use_set_conditioner", action="store_true", default=True)
    p.add_argument("--a3_train_only_set_modules", action="store_true", default=True)
    p.add_argument("--lasers_category_vocab_path", default="segearth_r2/model/lasers_category_vocab.json")
    return p.parse_args()


def main():
    args = parse_args()
    mask_cfg = get_mask_config(config=args.mask_config)
    print(f"[load] {args.model_name_or_path}")
    model = SegEarthR2.from_pretrained(
        args.model_name_or_path,
        mask_decoder_cfg=mask_cfg,
        add_cross_attn=True,
        torch_dtype="auto",
        device_map="cpu",
    )
    if not model.is_train_mask_decode:
        raise SystemExit("[ERROR] baseline mask_decode_train is false — wrong checkpoint?")

    model.init_set_conditioning_modules(
        argparse.Namespace(
            use_set_conditioner=args.use_set_conditioner,
            use_set_count_loss=True,
            use_set_category_loss=True,
            set_conditioner_layers=1,
            set_conditioner_heads=4,
            set_conditioner_gate_init=1e-3,
            lambda_set_count=0.05,
            lambda_set_category=0.1,
            set_max_count=10,
            lasers_category_vocab_path=args.lasers_category_vocab_path,
        )
    )

    if args.a3_train_only_set_modules:
        apply_a3_frozen_training(model, strict_forbidden=True)
        print_trainable_parameter_report(model)
        trainable = collect_trainable_names(model)
        for kw in A3_TRAINABLE_KEYWORDS:
            if not any(kw in n for n in trainable):
                raise SystemExit(f"[ERROR] No trainable param matched keyword: {kw}")
        assert_no_forbidden_trainables(model, forbidden_keywords=A3_FORBIDDEN_KEYWORDS, strict=True)
        print("[OK] A3-frozen: only set modules trainable")
    else:
        print("[INFO] a3_train_only_set_modules=False — skipped frozen check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
