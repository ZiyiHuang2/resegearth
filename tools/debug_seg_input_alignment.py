import argparse
import os
import sys
import types
import torch
import transformers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from segearth_r2.datasets.dataset import RRSISDDataset, DataCollatorForCOCODatasetV2, get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from transformers import SiglipImageProcessor


def _seg_id(tokenizer):
    tid = tokenizer.convert_tokens_to_ids("[SEG]")
    if not isinstance(tid, int) or tid < 0:
        raise RuntimeError("tokenizer has no [SEG] id; add_tokens('[SEG]') first")
    return tid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-data-path", required=True)
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--vision-tower", required=True)
    p.add_argument("--vision-tower-mask", default=None, help="Swin mask encoder weights (optional; None builds Swin without init_weights)")
    p.add_argument("--mask-config", default="segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--use-remoteclip-prior", action="store_true")
    p.add_argument("--remoteclip-weight-path", default=None)
    p.add_argument("--remoteclip-model-name", default="ViT-B-32")
    args = p.parse_args()

    if args.use_remoteclip_prior and not args.remoteclip_weight_path:
        raise RuntimeError("--use-remoteclip-prior requires --remoteclip-weight-path")

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, model_max_length=2048, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    mask_cfg = get_mask_config(args.mask_config)
    model = SegEarthR2.from_pretrained(args.model_name_or_path, mask_decoder_cfg=mask_cfg, add_cross_attn=True)
    if not model.is_train_mask_decode:
        model.initial_mask_module(None, None)
    model.runtime_tokenizer = tokenizer
    tokenizer.add_tokens("[SEG]")
    model.resize_token_embeddings(len(tokenizer))
    model.get_special_token(
        SEG=tokenizer("[SEG]", return_tensors="pt", add_special_tokens=False)["input_ids"],
        EOS=tokenizer.eos_token_id,
    )

    ma = types.SimpleNamespace(
        vision_tower=args.vision_tower,
        vision_tower_mask=args.vision_tower_mask,
        swin_type="base",
        mask_config=args.mask_config,
        use_remoteclip_prior=args.use_remoteclip_prior,
        remoteclip_weight_path=args.remoteclip_weight_path,
        remoteclip_model_name=args.remoteclip_model_name,
        remoteclip_device=args.device,
        remoteclip_fail_fast=True,
        use_confidence_scaling=True,
        use_weak_residual=True,
        prior_alpha=0.05,
        prior_beta=0.05,
        remoteclip_clip_input_size=224,
        unfreeze_remoteclip_last_layer=False,
        remoteclip_temperature=1.0,
    )
    model.get_model().initialize_vision_modules(ma, None)
    vision_tower = model.get_vision_tower()
    vision_tower_mask = model.model.get_vision_tower_mask()
    vision_tower.to(device=args.device, dtype=torch.float32)
    vision_tower_mask.to(device=args.device, dtype=torch.float32)

    if hasattr(model, "initialize_remoteclip_prior"):
        model.initialize_remoteclip_prior(ma)

    model.eval().to(args.device)

    data_args = types.SimpleNamespace(base_data_path=args.base_data_path)
    ds = RRSISDDataset(base_data_path=args.base_data_path, tokenizer=tokenizer, data_args=data_args, split=args.split)
    proc = SiglipImageProcessor.from_pretrained(args.vision_tower)
    collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=proc)

    n = min(args.num_samples, len(ds))
    if n < 1:
        raise RuntimeError("dataset empty")
    samples = [ds[i] for i in range(n)]
    batch = collator(samples)
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(args.device)
    if "token_refer_id" in batch:
        batch["token_refer_id"] = [x.to(args.device) for x in batch["token_refer_id"]]

    seg_tok_id = _seg_id(tokenizer)

    with torch.no_grad():
        _, _, _, inputs_embeds, _, seg_idx, img_idx = model.prepare_inputs_labels_for_multimodal(
            batch["input_ids"],
            batch["attention_mask"],
            None,
            batch["labels"],
            batch["images_clip"],
            token_refer_id=batch["token_refer_id"],
            SEG_token_embedding_indices=batch["SEG_token_embedding_indices"],
        )

    print("=== SEG input alignment check (>=3 samples when available) ===")
    for i in range(inputs_embeds.shape[0]):
        ids = batch["input_ids"][i]
        seg_pos_ids = torch.where(ids == seg_tok_id)[0].tolist()
        refer_txt = model._decode_refer_text(batch["token_refer_id"][i])
        seg_pos_emb = torch.where(seg_idx[i].bool())[0].tolist()
        seg_mask_true = torch.where(batch["SEG_token_embedding_indices"][i].bool())[0].tolist()
        img_pos = torch.where(img_idx[i].bool())[0].tolist()
        if len(seg_pos_ids) > 1:
            raise RuntimeError(f"sample={i}: multiple [SEG] in input_ids unsupported in controlled blast (positions={seg_pos_ids})")
        print(f"--- sample={i} ---")
        print(f"  input_ids_len={int(batch['input_ids'].shape[1])} inputs_embeds_len={int(inputs_embeds.shape[1])}")
        print(f"  input_ids [SEG] positions: {seg_pos_ids}")
        print(f"  token_refer_id decoded text: {refer_txt!r}")
        print(f"  SEG_token_embedding_indices true positions (pre-padding cols): {seg_mask_true}")
        print(f"  after prepare: seg_idx true positions in inputs_embeds: {seg_pos_emb}")
        if img_pos:
            print(f"  image_features_indices true count={len(img_pos)} min_idx={min(img_pos)} max_idx={max(img_pos)}")
        else:
            print("  image_features_indices true count=0 (empty)")

        if batch["input_ids"].shape[1] != inputs_embeds.shape[1]:
            left_pad = inputs_embeds.shape[1] - batch["input_ids"].shape[1]
            print(f"  padding_note: inputs_embeds is longer by {left_pad} (left multimodal padding as in training)")

        if len(seg_pos_emb) == 0:
            raise RuntimeError(f"sample {i} has no [SEG] position in inputs_embeds")
        for s in seg_pos_emb:
            if s < 0 or s >= inputs_embeds.shape[1]:
                raise RuntimeError(f"sample {i} invalid seg position {s}")

    if args.use_remoteclip_prior:
        emb0 = inputs_embeds.clone()
        emb1, diag = model._apply_remoteclip_prior_to_inputs_embeds(
            inputs_embeds,
            seg_idx,
            img_idx,
            batch["token_refer_id"],
            batch["seg_info"],
        )
        delta = (emb1 - emb0).float()
        print("=== RemoteCLIP prior injection delta ===")
        for i in range(inputs_embeds.shape[0]):
            seg_pos_emb = torch.where(seg_idx[i].bool())[0]
            img_pos = torch.where(img_idx[i].bool())[0]
            s0 = int(seg_pos_emb[0].item())
            d_seg = delta[i, s0, :].norm().item()
            d_vis = delta[i, img_pos, :].norm().item()
            seg_norm = emb0[i, s0, :].float().norm().item()
            vis_norm = emb0[i, img_pos, :].float().norm().item()
            image_path = batch["seg_info"][i]["image_path"]
            text = model._decode_refer_text(batch["token_refer_id"][i])
            po = model.remoteclip_prior.forward_image_expression(
                image_path=image_path, expression=text, clip_input_size=model.prior_clip_input_size
            )
            gconf = float(po.global_conf.view(-1)[0].item())
            print(
                f"sample={i} global_conf={gconf:.6g} seg_delta_norm={d_seg:.6g} visual_delta_norm={d_vis:.6g} "
                f"ratio_seg_delta_to_seg_norm={d_seg / max(seg_norm, 1e-8):.6g} "
                f"ratio_visual_delta_to_visual_norm={d_vis / max(vis_norm, 1e-8):.6g}"
            )
        print(f"aggregate_diag_keys (subset): {sorted([k for k in diag.keys() if k.startswith('remoteclip/') or k.startswith('gate/')])[:20]}")

    print("PASS: SEG input alignment checks completed.")


if __name__ == "__main__":
    main()
