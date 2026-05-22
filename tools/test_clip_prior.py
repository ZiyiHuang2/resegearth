import argparse
import os
import sys
from dataclasses import dataclass

import torch
import transformers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2, RRSISDDataset
from segearth_r2.model.prior.clip_prior import CLIPPriorConfig, CLIPPriorGenerator
from segearth_r2.model.mipha.model.multimodal_encoder.siglip_encoder import SiglipVisionTower


@dataclass
class DummyDataArgs:
    use_static_kb: bool = False
    use_ss_kb: bool = False
    use_semantic_kb: bool = False
    semantic_kb_inject_mode: str = "hard"
    semantic_kb_hard_query_max_tokens: int = 7
    semantic_kb_max_prefix_chars: int = 72
    semantic_kb_include_fields: str = "cat,rel,ctx,shape,scale"
    semantic_kb_category_priors_path: str = None
    semantic_kb_relation_priors_path: str = None
    clip_prior_text_source: str = "semantic_cat"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_path", type=str, required=True)
    parser.add_argument("--llm_path", type=str, required=True)
    parser.add_argument("--vision_tower", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--map_size", type=int, default=27)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.llm_path,
        use_fast=False,
        model_max_length=2048,
    )
    tokenizer.add_tokens("[SEG]")

    data_args = DummyDataArgs()
    dataset = RRSISDDataset(
        base_data_path=args.base_data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        split=args.split,
    )
    sample = dataset[0]
    print("raw_instruction:", sample.get("raw_instruction"))
    print("clip_prior_text:", sample.get("clip_prior_text"))

    clip_image_processor = transformers.SiglipImageProcessor.from_pretrained(args.vision_tower)
    collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_image_processor)
    batch = collator([sample])

    from segearth_r2.model.mipha.model.language_model.configuration_mipha import MiphaVisionConfig
    from transformers import SiglipVisionConfig

    siglip_cfg = SiglipVisionConfig.from_pretrained(args.vision_tower)
    clip_model = SiglipVisionTower.from_pretrained(
        args.vision_tower,
        config=MiphaVisionConfig(**siglip_cfg.to_dict()),
    )
    clip_model.to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    clip_model.eval()

    prior = CLIPPriorGenerator(
        clip_model=clip_model,
        tokenizer=tokenizer,
        config=CLIPPriorConfig(map_size=args.map_size),
        clip_model_name_or_path=args.vision_tower,
    ).to(device)
    prior.eval()

    images_clip = batch["images_clip"].to(device)
    clip_prior_text = batch["clip_prior_text"]
    with torch.no_grad():
        prior_map = prior(images_clip, clip_prior_text)

    print("prior_map.shape:", tuple(prior_map.shape))
    print("prior_map.has_nan:", bool(torch.isnan(prior_map).any().item()))
    print("prior_map.sum:", float(prior_map.sum().item()))
    print("prior_map.max:", float(prior_map.max().item()))
    print("prior_map.min:", float(prior_map.min().item()))


if __name__ == "__main__":
    main()
