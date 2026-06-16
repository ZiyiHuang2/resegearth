#!/usr/bin/env python3
"""Probe DGP v6.1 Stage A health metrics from a checkpoint dgp_stage_a_weights.bin."""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from transformers import AutoTokenizer, SiglipImageProcessor

from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2, RRSISDDataset, get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.utils import conversation as conversation_lib

KEYS = [
    "cos_pg_pl", "cos_pg_qseg", "cos_pl_qseg", "cos_qdetail_qseg", "cos_qref_qseg",
    "entropy_pg_attention_norm", "entropy_pl_attention_norm",
    "delta_g_norm", "delta_l_norm", "refiner_delta_norm", "seg_query_norm",
    "query_refiner_gate_g", "query_refiner_gate_l",
    "pg_top_token_idx_mean", "pl_top_token_idx_mean", "detail_prompt_source",
]
PROBLEM = ["bridge", "vehicle", "ship", "tennis", "expressway_toll_station", "chimney"]


def load_model(dgp_weights: str | None, model_path: str | None = None):
    model_path = model_path or "/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
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
    model.initial_mask_module(pretrained_path=vtm, model_args=MArgs())
    SegEarthR2.sync_dgp_config_from_args(model.config, MArgs())
    model.ensure_dgp_qdti_modules()
    model.to(device="cuda", dtype=dtype)

    class ModelArgs:
        vision_tower = vt
        vision_tower_mask = vtm
        train_clip_backbone = False
        train_swin_backbone = False
        version = "phi-2"

    model.get_model().initialize_vision_modules(model_args=ModelArgs(), fsdp=None)
    for m in [model.get_vision_tower(), model.model.get_vision_tower_mask()]:
        if m is not None:
            m.to(device="cuda", dtype=dtype)

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
    return model


def probe(
    dgp_weights: str | None,
    n_batches: int = 8,
    data: str | None = None,
    model_path: str | None = None,
) -> dict:
    model = load_model(dgp_weights, model_path=model_path)
    data = data or "/root/rivermind-data/huangziyi/data/RRSISD"
    clip_proc = SiglipImageProcessor.from_pretrained(
        "/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384"
    )

    class DArgs:
        base_data_path = data
        data_ratio = "1"
        switch_bs = 1
        segmentation = True
        dataset_name = "rrsisd"

    ds = RRSISDDataset(data, model.tokenizer, DArgs(), split="train")
    collator = DataCollatorForCOCODatasetV2(tokenizer=model.tokenizer, clip_image_processor=clip_proc)
    acc = {k: [] for k in KEYS}
    cls_acc = {c: [] for c in PROBLEM}

    for i in range(n_batches):
        batch = collator([ds[i]])
        batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["token_refer_id"] = [x.to("cuda") for x in batch["token_refer_id"]]
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                images=batch["images"].to(dtype=torch.bfloat16),
                images_clip=batch["images_clip"].to(dtype=torch.bfloat16),
                seg_info=batch["seg_info"],
                token_refer_id=batch["token_refer_id"],
                SEG_token_embedding_indices=batch["SEG_token_embedding_indices"],
                mask_num=batch["mask_num"],
            )
        h = getattr(model, "_dgp_last_health", {})
        for k in KEYS:
            if k in h:
                acc[k].append(float(h[k]))
        for c in PROBLEM:
            ck = f"class_{c}_cos_pl_qseg"
            if ck in h:
                cls_acc[c].append(float(h[ck]))

    def avg(xs):
        return sum(xs) / len(xs) if xs else None

    row = {k: avg(acc[k]) for k in KEYS}
    row["detail_prompt_source_name"] = "refer_id"
    row["problem_class_cos_pl_qseg"] = {c: avg(cls_acc[c]) for c in cls_acc}
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None, help="dgp_stage_a_weights.bin or checkpoint dir")
    parser.add_argument(
        "--model-path",
        default="/root/rivermind-data/huangziyi/reseg/output/base/standard-base-siglip1-28w-gd4/merged_model",
        help="SegEarth baseline merged model for from-base Stage A",
    )
    parser.add_argument("--n-batches", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    w = args.weights
    if w and os.path.isdir(w):
        for cand in (
            os.path.join(w, "dgp_stage_a_weights.bin"),
            os.path.join(w, "dgp_stage_a_weights.safetensors"),
        ):
            if os.path.isfile(cand):
                w = cand
                break

    row = probe(w, n_batches=args.n_batches, model_path=args.model_path)
    text = json.dumps(row, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
