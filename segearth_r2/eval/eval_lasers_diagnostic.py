#!/usr/bin/env python3
"""
LaSeRS diagnostic eval: generate model answers + per-[SEG] masks, save rich per-sample fields.

Unlike standard eval.py (teacher-forced GT answer), this script:
  1) autoregressively generates the assistant reply
  2) counts [SEG] in the generated text
  3) runs mask head on generated [SEG] positions
  4) computes pred x GT IoU matrix and merged-mask IoU
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import torch
import transformers
from pycocotools import mask as mask_utils
from tifffile import imwrite as imsave
from tqdm import tqdm
from transformers import SiglipImageProcessor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from segearth_r2.utils import conversation as conversation_lib
from segearth_r2.utils.builder import load_pretrained_model
from segearth_r2.datasets.dataset import (
    DataCollatorForCOCODatasetV2,
    LaSeRSDataset,
    preprocess_image,
)
from segearth_r2.utils.constants import IGNORE_INDEX


@dataclass
class DiagnosticArgs:
    local_rank: int = 0
    vision_tower: str = "pretrained_model/CLIP/siglip-so400m-patch14-384"
    vision_tower_mask: str = "pretrained_model/mask2former/model_final_54b88a.pkl"
    base_data_path: str = "your_data_path"
    model_path: str = "your_model_path"
    mask_config: str = (
        "../segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
    )
    version: str = "llava_phi"
    output_dir: str = "lasers_diagnostic_eval"
    eval_batch_size: int = 1
    dataloader_num_workers: int = 4
    max_eval_samples: int = 0
    max_new_tokens: int = 256
    load_8bit: bool = False
    load_4bit: bool = False
    resume: bool = True
    lasers_benchmark: str = "all"


def decode_rle_mask(mask_item):
    rle = {"size": mask_item["size"], "counts": mask_item["counts"]}
    m = mask_utils.decode(rle)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def mask_iou(pred, gt, eps=1e-7):
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return float(inter / (union + eps))


def resize_pred_to_gt(pred_mask, gt_mask):
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(
            pred_mask.astype(np.uint8),
            (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        pred_mask = (pred_mask > 0).astype(np.uint8)
    return pred_mask


def build_prompt_sources(description: str):
    prefix_inst = (
        "This is an image <|vision_bos|> <image> <|vision_eos|> <|sep|> <|user|>, "
        "please doing Reasoning Segmentation according to the following instruction:"
    )
    return [[
        {"from": "human", "value": prefix_inst + "\n<refer> <|assistant|>"},
        {"from": "gpt", "value": "\n"},
    ]]


def build_full_sources(description: str, answer_text: str):
    prefix_inst = (
        "This is an image <|vision_bos|> <image> <|vision_eos|> <|sep|> <|user|>, "
        "please doing Reasoning Segmentation according to the following instruction:"
    )
    if not answer_text.startswith("\n"):
        answer_text = "\n" + answer_text
    return [[
        {"from": "human", "value": prefix_inst + "\n<refer> <|assistant|>"},
        {"from": "gpt", "value": answer_text},
    ]]


@torch.inference_mode()
def greedy_generate_answer(
    model,
    tokenizer,
    dataset: LaSeRSDataset,
    prompt_input_ids: torch.Tensor,
    images: torch.Tensor,
    images_clip: torch.Tensor,
    token_refer_id: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
):
    token_dtype = prompt_input_ids.dtype
    input_ids = prompt_input_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    seg_indices = torch.zeros_like(input_ids)
    images = images.unsqueeze(0).to(device=device, dtype=next(model.parameters()).dtype)
    images_clip = images_clip.unsqueeze(0).to(device=device, dtype=next(model.parameters()).dtype)
    token_refer_id = [token_refer_id.to(device)]

    input_ids, attention_mask, past_key_values, inputs_embeds, _, seg_indices, _ = (
        model.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            None,
            None,
            images_clip,
            token_refer_id=token_refer_id,
            SEG_token_embedding_indices=seg_indices,
        )
    )

    generated_ids: List[int] = []
    cur_inputs_embeds = inputs_embeds
    cur_input_ids = input_ids
    cur_attention_mask = attention_mask
    past_key_values = None

    stop_ids = {tokenizer.eos_token_id}
    for stop_tok in ("<|endoftext|>", "<|end|>"):
        tid = tokenizer.convert_tokens_to_ids(stop_tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)

    for _ in range(max_new_tokens):
        outputs = model.model(
            input_ids=cur_input_ids if cur_inputs_embeds is None else None,
            attention_mask=cur_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=cur_inputs_embeds,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])
        next_token = int(logits.argmax(dim=-1).item())
        generated_ids.append(next_token)
        if next_token in stop_ids:
            break

        next_tensor = torch.tensor([[next_token]], device=device, dtype=token_dtype)
        cur_input_ids = next_tensor
        cur_inputs_embeds = None
        cur_attention_mask = torch.cat(
            [cur_attention_mask, torch.ones((1, 1), device=device, dtype=cur_attention_mask.dtype)],
            dim=1,
        )

    # Drop trailing stop tokens from decoded text / ids
    while generated_ids and generated_ids[-1] in stop_ids:
        generated_ids.pop()

    raw_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    model_answer = raw_generated_text.lstrip("\n").strip()
    return model_answer, raw_generated_text, generated_ids


def build_seg_batch(
    dataset: LaSeRSDataset,
    tokenizer,
    description: str,
    model_answer: str,
    sample_meta: dict,
    seg_token_id: int,
    set_token_id: Optional[int] = None,  # C-lite-v2: [SET] token id
):
    sources = build_full_sources(description, model_answer)
    text_dict = dataset.preprocess_llama2(sources, tokenizer)
    input_ids = text_dict["input_ids"][0]
    labels = text_dict["labels"][0]
    seg_indices = torch.zeros_like(input_ids)
    seg_indices[input_ids == seg_token_id] = 1
    generated_seg_count = model_answer.count("[SEG]")
    
    # C-lite-v2: 统计 [SET] token
    set_indices = None
    generated_set_count = 0
    if set_token_id is not None:
        set_indices = torch.zeros_like(input_ids)
        set_indices[input_ids == set_token_id] = 1
        generated_set_count = model_answer.count("[SET]")

    seg_info = []
    for i in range(generated_seg_count):
        seg_info.append({
            "data_id": sample_meta["id"],
            "mask_id": i,
            "image_id": os.path.splitext(sample_meta["image_name"])[0],
        })

    result = {
        "input_ids": input_ids.unsqueeze(0),
        "labels": labels.unsqueeze(0),
        "attention_mask": input_ids.unsqueeze(0).ne(tokenizer.pad_token_id),
        "SEG_token_embedding_indices": seg_indices.unsqueeze(0),
        "mask_num": [generated_seg_count],
        "seg_info": seg_info,
        "generated_SET_count": generated_set_count,  # C-lite-v2: 记录 [SET] 数量
    }
    if set_indices is not None:
        result["SET_token_embedding_indices"] = set_indices.unsqueeze(0)
    return result


def compute_iou_diagnostics(gt_masks: List[np.ndarray], pred_masks: List[np.ndarray]):
    gt_count = len(gt_masks)
    pred_count = len(pred_masks)
    if gt_count == 0 or pred_count == 0:
        return [], None, []

    iou_matrix = []
    for pred in pred_masks:
        row = []
        for gt in gt_masks:
            p = resize_pred_to_gt(pred, gt)
            row.append(mask_iou(p, gt))
        iou_matrix.append(row)

    merged_gt = np.zeros_like(gt_masks[0], dtype=np.uint8)
    for gt in gt_masks:
        merged_gt = np.logical_or(merged_gt, gt).astype(np.uint8)
    merged_pred = np.zeros_like(gt_masks[0], dtype=np.uint8)
    for pred in pred_masks:
        p = resize_pred_to_gt(pred, merged_gt)
        merged_pred = np.logical_or(merged_pred, p).astype(np.uint8)
    merged_iou = mask_iou(merged_pred, merged_gt)
    best_gt_idx = [int(np.argmax(row)) if row else -1 for row in iou_matrix]
    return iou_matrix, merged_iou, best_gt_idx


def load_done_keys(jsonl_path: str):
    done = set()
    if not os.path.isfile(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add((rec.get("subset"), rec.get("sample_id")))
    return done


def list_lasers_benchmarks(base_data_path: str, lasers_benchmark: str):
    json_dir = os.path.join(base_data_path, "test", "annotations")
    if not os.path.isdir(json_dir):
        raise FileNotFoundError(f"LaSeRS test annotations not found: {json_dir}")
    all_json = sorted(f for f in os.listdir(json_dir) if f.endswith(".json"))
    if lasers_benchmark.lower() == "all":
        return all_json
    name = lasers_benchmark if lasers_benchmark.endswith(".json") else f"{lasers_benchmark}.json"
    if name not in all_json:
        raise FileNotFoundError(f"Benchmark not found: {name}")
    return [name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vision_tower", default=DiagnosticArgs.vision_tower)
    parser.add_argument("--vision_tower_mask", default=DiagnosticArgs.vision_tower_mask)
    parser.add_argument("--mask_config", default=DiagnosticArgs.mask_config)
    parser.add_argument("--version", default=DiagnosticArgs.version)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--lasers_benchmark", default="all")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    mask_dir = os.path.join(args.output_dir, "per_seg_masks")
    os.makedirs(mask_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "lasers_per_sample_diagnostic.jsonl")
    done_keys = load_done_keys(jsonl_path) if args.resume else set()

    data_args = DiagnosticArgs(
        base_data_path=args.base_data_path,
        model_path=args.model_path,
        vision_tower=args.vision_tower,
        vision_tower_mask=args.vision_tower_mask,
        mask_config=args.mask_config,
        version=args.version,
    )
    data_args.is_multimodal = True
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]

    tokenizer, model, _, _ = load_pretrained_model(
        os.path.expanduser(args.model_path),
        model_args=data_args,
        mask_config=args.mask_config,
        load_8bit=False,
        load_4bit=False,
        device="cuda",
    )
    infer_dtype = next(model.parameters()).dtype
    model.to(dtype=infer_dtype, device=device)
    model.eval()

    clip_image_processor = SiglipImageProcessor.from_pretrained(args.vision_tower)
    seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    # C-lite-v2: 获取 [SET] token id
    set_token_id = tokenizer.convert_tokens_to_ids("[SET]")
    if set_token_id is None or set_token_id == tokenizer.unk_token_id:
        set_token_id = None  # 模型未使用 [SET] token

    benchmarks = list_lasers_benchmarks(args.base_data_path, args.lasers_benchmark)
    processed = 0
    out_f = open(jsonl_path, "a", encoding="utf-8")

    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    try:
        for bench_json in benchmarks:
            subset = os.path.splitext(bench_json)[0]
            dataset = LaSeRSDataset(
                base_data_path=args.base_data_path,
                tokenizer=tokenizer,
                data_args=data_args,
                split=bench_json,
            )
            bench_mask_dir = os.path.join(mask_dir, subset)
            os.makedirs(bench_mask_dir, exist_ok=True)

            for idx in tqdm(range(len(dataset)), desc=f"diag:{subset}"):
                sample = dataset.reason_file[idx]
                sample_id = sample["id"]
                key = (subset, sample_id)
                if key in done_keys:
                    continue
                if args.max_eval_samples > 0 and processed >= args.max_eval_samples:
                    break

                description = sample.get("description", "")
                gt_answer = sample.get("answer", "")
                image_name = sample["image_name"]
                image_path = os.path.join(dataset.LaSeRS_image_path, image_name)

                gt_masks = [decode_rle_mask(r) for r in sample.get("mask", [])]
                gt_mask_count = len(gt_masks)

                prompt_sources = build_prompt_sources(description)
                prompt_dict = dataset.preprocess_llama2(prompt_sources, tokenizer)
                prompt_input_ids = prompt_dict["input_ids"][0]
                token_refer_id = dataset.preprocess_referring_instruction(description)

                image_RGB = preprocess_image(image_path)
                image_tensor = torch.as_tensor(np.ascontiguousarray(image_RGB.transpose(2, 0, 1)))
                images = (image_tensor - pixel_mean) / pixel_std
                image_bgr = cv2.imread(image_path)
                image_clip = clip_image_processor.preprocess(
                    cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), return_tensors="pt"
                )["pixel_values"][0]

                model_answer, raw_generated_text, generated_token_ids = greedy_generate_answer(
                    model,
                    tokenizer,
                    dataset,
                    prompt_input_ids,
                    images,
                    image_clip,
                    token_refer_id,
                    device,
                    args.max_new_tokens,
                )
                generated_seg_count = model_answer.count("[SEG]")
                generated_set_count = model_answer.count("[SET]")

                pred_masks = []
                per_seg_paths = []
                per_seg_areas = []

                if generated_seg_count > 0:
                    seg_batch = build_seg_batch(
                        dataset, tokenizer, description, model_answer, sample, seg_token_id,
                        set_token_id=set_token_id,  # C-lite-v2
                    )
                    set_indices = seg_batch.get("SET_token_embedding_indices")
                    if set_indices is not None:
                        set_indices = set_indices.to(device)
                    outputs = model.eval_seg(
                        input_ids=seg_batch["input_ids"].to(device),
                        attention_mask=seg_batch["attention_mask"].to(device),
                        images=images.unsqueeze(0).to(device=device, dtype=infer_dtype),
                        images_clip=image_clip.unsqueeze(0).to(device=device, dtype=infer_dtype),
                        seg_info=seg_batch["seg_info"],
                        token_refer_id=[token_refer_id.to(device)],
                        SEG_token_embedding_indices=seg_batch["SEG_token_embedding_indices"].to(device),
                        SET_token_embedding_indices=set_indices,
                        labels=seg_batch["labels"].to(device),
                        mask_num=seg_batch["mask_num"],
                    )
                    for out in outputs:
                        pred = out["pred"]
                        if pred.ndim > 2:
                            pred = np.squeeze(pred)
                        pred_bin = (pred > 0).astype(np.uint8)
                        pred_masks.append(pred_bin)
                        mask_id = out["mask_id"]
                        mask_name = f"{os.path.splitext(image_name)[0]}_{sample_id}_{subset}_seg{mask_id}.tif"
                        mask_path = os.path.join(bench_mask_dir, mask_name)
                        imsave(mask_path, pred_bin * 255)
                        per_seg_paths.append(mask_path)
                        per_seg_areas.append(int(pred_bin.sum()))

                iou_matrix, merged_iou, best_gt_idx = compute_iou_diagnostics(gt_masks, pred_masks)

                record = {
                    "sample_id": sample_id,
                    "subset": subset,
                    "query": description,
                    "gt_answer": gt_answer,
                    "model_answer": model_answer,
                    "raw_generated_text": raw_generated_text,
                    "generated_token_ids": generated_token_ids,
                    "generated_SEG_count": generated_seg_count,
                    "generated_SET_count": generated_set_count,
                    "gt_mask_count": gt_mask_count,
                    "pred_mask_count": len(pred_masks),
                    "per_SEG_mask_path": per_seg_paths,
                    "per_SEG_mask_area": per_seg_areas,
                    "per_SEG_IoU_with_each_GT": iou_matrix,
                    "merged_mask_IoU": merged_iou,
                    "best_matching_GT_index": best_gt_idx,
                    "image_name": image_name,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                done_keys.add(key)
                processed += 1

            if args.max_eval_samples > 0 and processed >= args.max_eval_samples:
                break
    finally:
        out_f.close()

    print(f"Done. Wrote {processed} samples to {jsonl_path}")


if __name__ == "__main__":
    main()
