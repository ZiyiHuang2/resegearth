#!/usr/bin/env python3
"""One-off real LaSeRS batch forward debug (not imported by training)."""

import os
import sys
from types import SimpleNamespace

import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2, LaSeRSDataset, get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.model.set_conditioner import regroup_seg_embeddings
from transformers import SiglipImageProcessor, AutoTokenizer
import transformers


def main():
    model_path = os.environ.get(
        "MODEL_NAME_OR_PATH",
        "/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B",
    )
    base_data_path = os.environ.get(
        "BASE_DATA_PATH",
        "/root/rivermind-data/huangziyi/data/LaSeRS",
    )
    vision_tower = os.environ.get(
        "VISION_TOWER",
        "/root/rivermind-data/huangziyi/reseg/pretrained_model/CLIP/siglip-so400m-patch14-384",
    )
    mask_config = os.environ.get(
        "MASK_CONFIG",
        "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml",
    )
    mask2former_ckpt = os.environ.get(
        "VISION_TOWER_MASK",
        "/root/rivermind-data/huangziyi/reseg/pretrained_model/mask2former/model_final_54b88a.pkl",
    )

    model_args = SimpleNamespace(
        use_set_conditioner=True,
        use_set_count_loss=True,
        use_set_category_loss=True,
        set_conditioner_layers=1,
        set_conditioner_heads=4,
        set_conditioner_gate_init=1e-3,
        lambda_set_count=0.1,
        lambda_set_category=0.05,
        set_max_count=10,
        lasers_category_vocab_path=None,
        vision_tower=vision_tower,
        vision_tower_mask=mask2former_ckpt,
        load_mask2former=True,
        swin_type="base",
    )

    mask_cfg = get_mask_config(mask_config)
    model = SegEarthR2.from_pretrained(model_path, mask_decoder_cfg=mask_cfg, add_cross_attn=True)
    if not model.is_train_mask_decode:
        model.initial_mask_module(mask2former_ckpt, model_args)
    else:
        model.init_set_conditioning_modules(model_args)

    model.get_model().initialize_vision_modules(model_args=model_args)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.add_tokens("[SEG]")
    model.resize_token_embeddings(len(tokenizer))
    model.get_special_token(
        SEG=tokenizer("[SEG]", return_tensors="pt", add_special_tokens=False)["input_ids"],
        EOS=tokenizer.eos_token_id,
    )

    data_args = SimpleNamespace(
        base_data_path=base_data_path,
        lasers_holdout_seed=42,
        lasers_category_vocab_path=model_args.lasers_category_vocab_path,
    )
    dataset = LaSeRSDataset(
        base_data_path=base_data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        split="train_data.json",
        holdout_mode="train",
        holdout_seed=42,
    )
    clip_processor = SiglipImageProcessor.from_pretrained(vision_tower)
    collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_processor)

    batch_items = [dataset[i] for i in range(min(2, len(dataset)))]
    for i, item in enumerate(batch_items):
        print(f"sample[{i}] category_phrases:", item.get("category_phrases"))
    batch = collator(batch_items)
    vocab = dataset.lasers_category_vocab

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype)
    model.get_vision_tower().to(device=device, dtype=dtype)
    model.get_model().get_vision_tower_mask().to(device=device, dtype=dtype)
    model.eval()

    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "images": batch["images"].to(device=device, dtype=dtype),
        "images_clip": batch["images_clip"].to(device=device, dtype=dtype),
        "seg_info": batch["seg_info"],
        "token_refer_id": [x.to(device) for x in batch["token_refer_id"]],
        "SEG_token_embedding_indices": batch["SEG_token_embedding_indices"].to(device),
        "labels": batch["labels"].to(device),
        "mask_num": batch["mask_num"],
        "dataset_type": batch.get("dataset_type"),
        "category_set_labels": batch["category_set_labels"].to(device),
    }

    mask_num = inputs["mask_num"]
    cat_labels = batch["category_set_labels"]
    print("mask_num:", mask_num)
    print("sum(mask_num):", sum(mask_num))
    print("category_set_labels shape:", tuple(cat_labels.shape))
    for i in range(cat_labels.shape[0]):
        idxs = (cat_labels[i] > 0).nonzero(as_tuple=False).flatten().tolist()
        words = [vocab[j] for j in idxs]
        print(f"  sample[{i}] nonzero category indices:", idxs)
        print(f"  sample[{i}] category vocab words:", words)
    print("category_label_stats:", dataset.category_label_stats.summary())

    captured = {}

    orig_forward = model.set_conditioner.forward

    def hook_set_conditioner(seg_embedding, mask_num_in, *args, **kwargs):
        seg_group, valid_mask, _ = regroup_seg_embeddings(seg_embedding, mask_num_in)
        captured["before_shape"] = tuple(seg_embedding.shape)
        captured["regroup_shape"] = tuple(seg_group.shape)
        captured["valid_mask_shape"] = tuple(valid_mask.shape)
        captured["valid_mask"] = valid_mask.detach().cpu()
        captured["valid_counts"] = [int(valid_mask[b].sum()) for b in range(valid_mask.shape[0])]
        out = orig_forward(seg_embedding, mask_num_in, *args, **kwargs)
        captured["q_set_shape"] = tuple(out[1].shape)
        captured["refined_shape"] = tuple(out[0].shape)
        return out

    model.set_conditioner.forward = hook_set_conditioner

    with torch.no_grad():
        outputs = model(**inputs)

    model.set_conditioner.forward = orig_forward

    # mask feature shapes via a lightweight re-run of pixel path is expensive; approximate via batch size
    bs = len(mask_num)
    print("SEG_embedding before set conditioner shape:", captured.get("before_shape"))
    print("regroup shape:", captured.get("regroup_shape"))
    print("valid_mask shape:", captured.get("valid_mask_shape"))
    print("valid_mask content:\n", captured.get("valid_mask"))
    print("valid count per sample:", captured.get("valid_counts"))
    print("Q_set shape:", captured.get("q_set_shape"))
    print("refined SEG_embedding shape:", captured.get("refined_shape"))
    print("mask_features before repeat_interleave shape: [B, ...] with B=", bs)
    print("mask_features after repeat_interleave shape: [sum(mask_num), ...] with sum=", sum(mask_num))
    print("loss:", float(outputs.loss.detach().cpu()) if outputs.loss is not None else None)
    print("loss_set_count:", float(outputs.loss_set_count) if outputs.loss_set_count is not None else None)
    print(
        "loss_set_category:",
        float(outputs.loss_set_category) if outputs.loss_set_category is not None else None,
    )
    assert outputs.loss_set_category is not None, "A3 requires finite loss_set_category"
    assert torch.isfinite(outputs.loss_set_category), "loss_set_category not finite"
    print("set_gate_value:", outputs.set_gate_value)
    print("target_count_acc:", outputs.target_count_acc)


if __name__ == "__main__":
    main()
