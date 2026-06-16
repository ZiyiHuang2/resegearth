#!/usr/bin/env python3
"""RRSISD test inference for DGP Stage A checkpoint (base MLLM + dgp_stage_a_weights.bin)."""
from __future__ import annotations

import argparse
import gc
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch
from tifffile import imwrite as imsave
from tqdm import tqdm
from transformers import AutoTokenizer, SiglipImageProcessor

from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2, RRSISDDataset, get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.utils import conversation as conversation_lib

DATA_PATH = "/root/rivermind-data/huangziyi/data/RRSISD"


def load_model(model_path: str, dgp_weights: str | None = None, device: str = "cuda"):
    vt = "/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
    vtm = "/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl"
    mask_cfg = get_mask_config("segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")

    class MArgs:
        use_dgp_qdti = True
        use_qdti_bias = False
        dgp_training_stage = "a"
        dgp_version = "v6.1"
        gate_g_init = 0.01
        gate_l_init = 0.02
        dgp_fuse_dim = 256
        dgp_refiner_hidden_dim = 512
        dgp_pg_tokens = 1
        qdti_apply_layers = "disabled"
        qdti_scale_init = 0.0

    dtype = torch.bfloat16
    model = SegEarthR2.from_pretrained(model_path, mask_decoder_cfg=mask_cfg, torch_dtype=dtype)
    if not model.is_train_mask_decode:
        model.initial_mask_module(pretrained_path=vtm, model_args=MArgs())
    SegEarthR2.sync_dgp_config_from_args(model.config, MArgs())
    model.ensure_dgp_qdti_modules()
    model.to(device=device, dtype=dtype)

    class ModelArgs:
        vision_tower = vt
        vision_tower_mask = vtm
        train_clip_backbone = False
        train_swin_backbone = False
        version = "phi-2"

    model.get_model().initialize_vision_modules(model_args=ModelArgs(), fsdp=None)
    for m in [model.get_vision_tower(), model.model.get_vision_tower_mask()]:
        if m is not None:
            m.to(device=device, dtype=dtype)

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_tokens("[SEG]")
    model.resize_token_embeddings(len(tok))
    model.get_special_token(
        SEG=tok("[SEG]", return_tensors="pt", add_special_tokens=False)["input_ids"],
        EOS=tok.eos_token_id,
    )
    model.tokenizer = tok
    conversation_lib.default_conversation = conversation_lib.conv_templates.get(
        "phi-2", conversation_lib.conv_templates["vicuna_v1"]
    )

    if dgp_weights and os.path.isfile(dgp_weights):
        sd = torch.load(dgp_weights, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        print(f"[eval] loaded DGP weights: {dgp_weights}")
    return model, tok


def run_eval(
    model_path: str,
    ckpt_dir: str,
    out_dir: str,
    refined: bool,
    split: str = "test",
    max_samples: int = 0,
):
    dgp_w = os.path.join(ckpt_dir, "dgp_stage_a_weights.bin")
    if not os.path.isfile(dgp_w):
        raise FileNotFoundError(f"missing {dgp_w}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(model_path, dgp_weights=dgp_w, device=device)
    model.eval()

    clip_proc = SiglipImageProcessor.from_pretrained(
        "/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
    )

    class DArgs:
        base_data_path = DATA_PATH
        data_ratio = "1"
        switch_bs = 1
        segmentation = True
        dataset_name = "rrsisd"

    ds = RRSISDDataset(DATA_PATH, tok, DArgs(), split=split)
    collator = DataCollatorForCOCODatasetV2(tokenizer=tok, clip_image_processor=clip_proc)
    os.makedirs(out_dir, exist_ok=True)

    split_stem = f"{split}.json".split(".")[0]
    n = len(ds) if max_samples <= 0 else min(max_samples, len(ds))
    dtype = torch.bfloat16

    with torch.no_grad():
        for idx in tqdm(range(n), desc=f"refined={refined}"):
            batch = collator([ds[idx]])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch["token_refer_id"] = [x.to(device) for x in batch["token_refer_id"]]
            outputs = model.eval_seg(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                images=batch["images"].to(dtype=dtype),
                images_clip=batch["images_clip"].to(dtype=dtype),
                seg_info=batch["seg_info"],
                token_refer_id=batch["token_refer_id"],
                SEG_token_embedding_indices=batch["SEG_token_embedding_indices"],
                mask_num=batch["mask_num"],
                dgp_use_refined_query=refined,
            )
            for output in outputs:
                pred_mask = output["pred"]
                image_name = output["image_name"]
                sample_id = output["id"]
                mask_id = output["mask_id"]
                if pred_mask.ndim > 2:
                    pred_mask = np.squeeze(pred_mask)
                name = f"{image_name}_{sample_id}_{split_stem}_{mask_id}.tif"
                out_path = os.path.join(out_dir, name)
                if os.path.isfile(out_path):
                    continue
                imsave(out_path, pred_mask.astype(np.uint8))

    print(f"[eval] saved {n} samples -> {out_dir}")
    _free_model(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--model-path",
        default="/root/rivermind-data/huangziyi/reseg/output/base/standard-base-siglip1-28w-gd4/merged_model",
        help="SegEarth baseline merged model (not raw Mipha-3B)",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--mode", choices=("both", "model", "base"), default="both")
    args = parser.parse_args()

    model_dir = os.path.join(args.out_root, "test_results_model")
    base_dir = os.path.join(args.out_root, "test_results_base_qseg")
    if args.mode in ("both", "model"):
        run_eval(args.model_path, args.ckpt_dir, model_dir, refined=True, split=args.split, max_samples=args.max_samples)
    if args.mode in ("both", "base"):
        run_eval(args.model_path, args.ckpt_dir, base_dir, refined=False, split=args.split, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
