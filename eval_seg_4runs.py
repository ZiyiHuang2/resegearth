import os
import json
import random
import argparse
from types import SimpleNamespace

import torch
import numpy as np
from PIL import Image

from peft import PeftModel
from transformers import AutoTokenizer

from segearth_r2.model.language_model.llava_qwen import SegEarthR2Qwen
from segearth_r2.train.train import get_mask_config
from segearth_r2.datasets.dataset import RRSISDDataset


def to_device(x, device):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return [to_device(t, device) for t in x]
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def pick(sample, *keys, default=None):
    for k in keys:
        if k in sample and sample[k] is not None:
            return sample[k]
    return default


def tensor_to_mask_png(mask_tensor: torch.Tensor) -> np.ndarray:
    """
    Accepts:
      - [H,W] or [1,H,W] or [N,H,W]
    Returns uint8 [H,W] 0/255 for visualization.
    """
    m = mask_tensor
    if m.ndim == 3:
        # pick first mask for quick visualization
        m = m[0]
    if m.ndim == 4:
        m = m[0, 0]
    m = m.float()
    if m.max() > 1.0:
        # sometimes logits; threshold at 0
        m = (m > 0).float()
    else:
        m = (m > 0.5).float()
    return (m.cpu().numpy().astype(np.uint8) * 255)


@torch.inference_mode()
def run_one_adapter(tag, adapter_dir, args, samples):
    out_dir = os.path.join(args.out_dir, f"run_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    mask_cfg = get_mask_config(config=args.mask_config)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # load base
    base = SegEarthR2Qwen.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        mask_decoder_cfg=mask_cfg,
    ).to(args.device)
    base.eval()

    # attach adapter (no merge)
    model = PeftModel.from_pretrained(base, adapter_dir).to(args.device)
    model.eval()

    # init vision modules (must match training)
    model_args = SimpleNamespace(
        vision_tower=args.vision_tower,
        vision_tower_mask=args.vision_tower_mask,
        mm_vision_select_layer=-2,
        mm_vision_select_feature="patch",
        mm_projector_type="linear",
        swin_type="base",
    )
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)

    # iterate samples
    for k, (idx, sample) in enumerate(samples):
        # ---- IMPORTANT: adapt these keys if your dataset uses different names ----
        input_ids = pick(sample, "input_ids")
        attention_mask = pick(sample, "attention_mask")
        images = pick(sample, "images")               # should be list of tensors OR tensor [B,3,H,W]
        images_clip = pick(sample, "images_clip")     # usually tensor [B,3,H,W] or similar
        seg_info = pick(sample, "seg_info", default=None)
        token_refer_id = pick(sample, "token_refer_id", "refer_ids", default=None)
        seg_embed_idx = pick(sample, "SEG_token_embedding_indices", "seg_token_embedding_indices", default=None)
        mask_num = pick(sample, "mask_num", default=None)

        if input_ids is None or attention_mask is None or images is None or images_clip is None:
            print(f"[{tag}] sample keys = {list(sample.keys())}")
            raise RuntimeError(
                f"[{tag}] Missing required keys. Need input_ids/attention_mask/images/images_clip. "
                f"Please map dataset keys in eval script."
            )
        if mask_num is None:
            print(f"[{tag}] sample keys = {list(sample.keys())}")
            raise RuntimeError(f"[{tag}] Missing mask_num (required by eval_seg).")

        # move to device
        input_ids = to_device(input_ids, args.device)
        attention_mask = to_device(attention_mask, args.device)
        images = to_device(images, args.device)
        images_clip = to_device(images_clip, args.device)
        seg_embed_idx = to_device(seg_embed_idx, args.device)
        token_refer_id = to_device(token_refer_id, args.device)
        # mask_num is usually list[int] (keep on cpu)
        # -----------------------------------------------------------------------

        out = model.base_model.eval_seg(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            images_clip=images_clip,
            seg_info=seg_info,
            token_refer_id=token_refer_id,
            SEG_token_embedding_indices=seg_embed_idx,
            mask_num=mask_num,
            return_dict=True,
        )

        # try to find a mask tensor in outputs
        mask_tensor = None
        for cand in ["pred_masks", "mask_outputs", "masks", "pred_mask"]:
            if isinstance(out, dict) and cand in out and out[cand] is not None:
                mask_tensor = out[cand]
                break

        if mask_tensor is None:
            # print keys to help adapt
            if isinstance(out, dict):
                print(f"[{tag}] eval_seg output keys = {list(out.keys())}")
            raise RuntimeError(f"[{tag}] Cannot find mask tensor in eval_seg outputs.")

        # save one png per sample (first mask)
        if torch.is_tensor(mask_tensor):
            # If batch dimension exists, pick first item
            mt = mask_tensor
            if mt.ndim >= 4:
                mt = mt[0]   # [N,H,W] or [1,N,H,W]
            png = tensor_to_mask_png(mt)
        else:
            raise RuntimeError(f"[{tag}] mask tensor is not torch.Tensor, got {type(mask_tensor)}")

        Image.fromarray(png).save(os.path.join(out_dir, f"{k:04d}_idx{idx}.png"))

    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--base_data_path", type=str, required=True)
    ap.add_argument("--mask_config", type=str, required=True)
    ap.add_argument("--vision_tower", type=str, required=True)
    ap.add_argument("--vision_tower_mask", type=str, required=True)

    ap.add_argument("--adapter_A", type=str, required=True)
    ap.add_argument("--adapter_B", type=str, required=True)
    ap.add_argument("--adapter_C", type=str, required=True)
    ap.add_argument("--adapter_D", type=str, required=True)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # dataset: is_train=False for eval
    ds = RRSISDDataset(base_data_path=args.base_data_path, is_train=False)

    idxs = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(idxs)
    idxs = idxs[: args.num_samples]

    samples = []
    for i in idxs:
        samples.append((i, ds[i]))

    meta = {
        "base_model": args.base_model,
        "base_data_path": args.base_data_path,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "runs": {}
    }

    for tag, adapter in [("A", args.adapter_A), ("B", args.adapter_B), ("C", args.adapter_C), ("D", args.adapter_D)]:
        out_dir = run_one_adapter(tag, adapter, args, samples)
        meta["runs"][tag] = {"adapter_dir": adapter, "out_dir": out_dir}

    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()