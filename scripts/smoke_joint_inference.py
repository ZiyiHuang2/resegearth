#!/usr/bin/env python3
"""Phase-1 inference + tif smoke: single/multi [SEG] via eval_seg on LaSeRS samples."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tifffile import imwrite as imsave
from transformers import SiglipImageProcessor

from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2, LaSeRSDataset
from segearth_r2.eval.eval import DataArguments
from segearth_r2.utils.builder import load_pretrained_model


def pick_samples(json_path):
    data = json.loads(Path(json_path).read_text())
    single_idx = multi_idx = None
    for i, item in enumerate(data):
        conv = item.get("conversations", [])
        ans = conv[-1]["value"] if conv else item.get("answer", "")
        k = str(ans).count("[SEG]")
        if k == 1 and single_idx is None:
            single_idx = i
        if k >= 2 and multi_idx is None:
            multi_idx = (i, k)
        if single_idx is not None and multi_idx is not None:
            break
    return single_idx, multi_idx


@torch.no_grad()
def run_one(model, collator, dataset, idx, device, tag, save_dir=None, split_stem="test_multi_cate"):
    batch = collator([dataset[idx]])
    mask_num = batch["mask_num"]
    sum_k = sum(mask_num)
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    batch["token_refer_id"] = [ids.to(device) for ids in batch["token_refer_id"]]
    infer_dtype = next(model.parameters()).dtype
    outputs = model.eval_seg(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        images=batch["images"].to(device=device, dtype=infer_dtype),
        images_clip=batch["images_clip"].to(device=device, dtype=infer_dtype),
        seg_info=batch["seg_info"],
        token_refer_id=batch["token_refer_id"],
        SEG_token_embedding_indices=batch["SEG_token_embedding_indices"],
        labels=batch["labels"],
        mask_num=batch["mask_num"],
    )
    assert len(outputs) == sum_k, f"{tag}: got {len(outputs)} masks, expected sumK={sum_k}"
    print(f"[PASS] {tag}: mask_num={mask_num}, outputs={len(outputs)}, seg_info={len(batch['seg_info'])}")
    saved = []
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for output in outputs:
            pred_mask = np.squeeze(output["pred"])
            name = f"{output['image_name']}_{output['id']}_{split_stem}_{output['mask_id']}.tif"
            path = os.path.join(save_dir, name)
            imsave(path, pred_mask.astype(np.uint8))
            saved.append(path)
            print(f"  saved {path}")
    return outputs, batch, saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_data_path", required=True)
    parser.add_argument("--vision_tower", required=True)
    parser.add_argument("--vision_tower_mask", required=True)
    parser.add_argument("--mask_config", required=True)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = DataArguments(
        vision_tower=args.vision_tower,
        vision_tower_mask=args.vision_tower_mask,
        mask_config=args.mask_config,
        base_data_path=args.base_data_path,
        model_path=args.model_path,
    )
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path,
        model_args=model_args,
        mask_config=args.mask_config,
        device=str(device),
    )
    model.to(dtype=torch.float16, device=device)
    model.eval()

    clip_processor = SiglipImageProcessor.from_pretrained(args.vision_tower)

    class DA:
        lazy_preprocess = False
        image_aspect_ratio = "square"
        image_grid_pinpoints = None
        is_multimodal = True

    data_args = DA()
    collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_processor)

    ann_short = os.path.join(args.base_data_path, "test", "annotations", "test_short_query.json")
    ann_multi = os.path.join(args.base_data_path, "test", "annotations", "test_multi_cate.json")
    single_idx, _ = pick_samples(ann_short)
    _, multi_info = pick_samples(ann_multi)
    if single_idx is None:
        raise SystemExit(f"No single-[SEG] sample in {ann_short}")
    if multi_info is None:
        raise SystemExit(f"No multi-[SEG] sample in {ann_multi}")

    out_single = out_multi = None
    if args.output_dir:
        out_single = os.path.join(args.output_dir, "eval_single_seg")
        out_multi = os.path.join(args.output_dir, "eval_multi_seg")

    ds_single = LaSeRSDataset(
        base_data_path=args.base_data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        split="test_short_query.json",
    )
    run_one(
        model, collator, ds_single, single_idx, device,
        f"single-SEG idx={single_idx}",
        save_dir=out_single,
        split_stem="test_short_query",
    )

    multi_idx, multi_k = multi_info
    ds_multi = LaSeRSDataset(
        base_data_path=args.base_data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        split="test_multi_cate.json",
    )
    outs, _, saved = run_one(
        model, collator, ds_multi, multi_idx, device,
        f"multi-SEG idx={multi_idx} K={multi_k}",
        save_dir=out_multi,
        split_stem="test_multi_cate",
    )
    assert len(outs) == multi_k
    mask_ids = sorted(int(o["mask_id"]) for o in outs)
    assert mask_ids == list(range(multi_k)), f"mask_id order {mask_ids}"
    print(f"[PASS] multi-SEG mask_ids={mask_ids}")

    if out_multi:
        t0 = list(Path(out_multi).glob("*_0.tif"))
        t1 = list(Path(out_multi).glob("*_1.tif"))
        if multi_k >= 2:
            assert t0 and t1, f"missing _0/_1 tif under {out_multi}"
            print(f"[PASS] tif smoke: _0={len(t0)} _1={len(t1)} under {out_multi}")
        if out_single:
            assert list(Path(out_single).glob("*.tif")), f"no tif under {out_single}"
            print(f"[PASS] tif smoke: single tifs under {out_single}")

    print("INFERENCE_SMOKE_OK")


if __name__ == "__main__":
    main()
