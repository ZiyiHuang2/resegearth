import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import argparse
import copy

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, BitsAndBytesConfig

from segearth_r2.model import *
from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.utils.hf_checkpoint_state import (
    count_gate_keys,
    gate_keys_in_state_dict,
    load_flat_state_dict_from_hf_folder,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).lower()
    if s in ("yes", "true", "t", "1", "y"):
        return True
    if s in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="merge lora weights and save model with hf format"
    )
    parser.add_argument(
        "--model_path", default="./save_model/SegEarth-R2"
    )

    parser.add_argument(
        "--vision_tower", default="./pretrained_model/siglip-so400m-patch14-384"
    )
    parser.add_argument(
        "--vision_tower_mask", default="./pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl"
    )
    parser.add_argument(
        "--mask_config", default="./segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
    )

    parser.add_argument("--lora_enable", default=True, type=bool)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_weight_path", default="", type=str)
    parser.add_argument("--lora_bias", default="none", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")

    parser.add_argument("--save_path", default="./InstructSeg_model", type=str, required=True)

    parser.add_argument("--use_remoteclip_prior", type=str2bool, default=False)
    parser.add_argument("--remoteclip_fail_fast", type=str2bool, default=True)
    parser.add_argument("--remoteclip_weight_path", type=str, default="")
    parser.add_argument("--remoteclip_model_name", type=str, default="ViT-B-32")
    parser.add_argument("--remoteclip_device", type=str, default="cuda")
    parser.add_argument("--remoteclip_temperature", type=float, default=1.0)
    parser.add_argument("--remoteclip_clip_input_size", type=int, default=224)
    parser.add_argument("--unfreeze_remoteclip_last_layer", type=str2bool, default=False)
    parser.add_argument("--use_confidence_scaling", type=str2bool, default=True)
    parser.add_argument("--use_weak_residual", type=str2bool, default=True)
    parser.add_argument("--prior_alpha", type=float, default=0.05)
    parser.add_argument("--prior_beta", type=float, default=0.05)
    parser.add_argument("--debug_seg_input_alignment", type=str2bool, default=False)

    return parser.parse_args(args)


def _validate_merge_remoteclip(args):
    if not args.use_remoteclip_prior:
        return
    if not args.remoteclip_fail_fast:
        raise RuntimeError(
            "[MERGE RemoteCLIP] remoteclip_fail_fast=False is not allowed when use_remoteclip_prior=True"
        )
    wp = (args.remoteclip_weight_path or "").strip()
    if not wp:
        raise ValueError("[MERGE RemoteCLIP] use_remoteclip_prior=True but remoteclip_weight_path is empty")
    if not os.path.isfile(os.path.expanduser(wp)):
        raise FileNotFoundError(f"[MERGE RemoteCLIP] remoteclip_weight_path is not a file: {wp}")


def find_linear_layers(model, lora_target_modules=['q_proj', 'v_proj'], train_module_list=[]):
    cur_train_module_list = copy.deepcopy(train_module_list)
    cur_train_module_list.extend(["vision_tower", "vision_tower_mask"])
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (isinstance(module, cls)
            and all(
                        [
                            x not in name
                            for x in cur_train_module_list
                        ]
                    )
                    and any([x in name for x in lora_target_modules])):

            lora_module_names.add(name)

    return sorted(list(lora_module_names))


def load_pretrained_model(model_path, model_args, mask_config='/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml', load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):

    kwargs = {"device_map": 'cpu'}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    mask_cfg = get_mask_config(mask_config)
    mask_cfg.MODEL.MASK_FORMER.SEG_TASK = model_args.seg_task if hasattr(model_args, 'seg_task') else 'instance'

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = SegEarthR2.from_pretrained(model_path, mask_decoder_cfg=mask_cfg, **kwargs)

    model.use_temporal_query = model_args.use_temporal_query if hasattr(model_args, 'use_temporal_query') else False
    model.use_vmtf = model_args.use_vmtf if hasattr(model_args, 'use_vmtf') else False

    mask2former_ckpt = model_args.vision_tower_mask
    model.initial_mask_module(mask2former_ckpt, model_args)

    model.get_model().initialize_vision_modules(model_args)

    vision_tower = model.get_model().get_vision_tower_mask()

    vision_tower.to(device=device)

    train_module_list = [
        "lm_head", "pixel_decoder", "predictor", "SEG_token_projector",
    ]

    if model_args.lora_enable:
        lora_r = model_args.lora_r
        lora_alpha = model_args.lora_alpha
        lora_dropout = model_args.lora_dropout
        lora_target_modules = find_linear_layers(model, train_module_list=train_module_list)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.resize_token_embeddings(len(tokenizer))

    use_rc = bool(getattr(model_args, "use_remoteclip_prior", False))
    if use_rc:
        print("[MERGE RemoteCLIP] use_remoteclip_prior=True")
        model.get_special_token(
            SEG=tokenizer("[SEG]", return_tensors="pt", add_special_tokens=False)["input_ids"],
            EOS=tokenizer.eos_token_id,
        )
        model.use_remoteclip_prior = True
        model.remoteclip_fail_fast = True
        model.initialize_remoteclip_prior(model_args)
        print("[MERGE RemoteCLIP] initialized prior before zero checkpoint load")

    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
    model = load_state_dict_from_zero_checkpoint(model, model_path)

    sd_after_zero = model.state_dict()
    gate_keys = [k for k in sd_after_zero if "seg_visual_prior_gate" in k]
    if use_rc and len(gate_keys) == 0:
        raise RuntimeError(
            "[MERGE RemoteCLIP] after ZeRO load, model has no seg_visual_prior_gate parameters; "
            "checkpoint may be missing gate weights or prefix mismatch."
        )
    if use_rc:
        print(f"[MERGE RemoteCLIP] seg_visual_prior_gate keys restored: {len(gate_keys)}")

    model = model.merge_and_unload()

    sd_after_merge = model.state_dict()
    gate_keys_m = [k for k in sd_after_merge if "seg_visual_prior_gate" in k]
    if use_rc and len(gate_keys_m) == 0:
        raise RuntimeError(
            "[MERGE RemoteCLIP] after merge_and_unload(), seg_visual_prior_gate parameters missing."
        )

    return tokenizer, model


def main(args):
    args = parse_args(args)

    _validate_merge_remoteclip(args)

    tokenizer, model = load_pretrained_model(args.model_path, model_args=args, mask_config=args.mask_config, device='cuda')

    state_dict = {}
    for k, v in model.state_dict().items():
        print(k)
        state_dict[k] = v

    if args.use_remoteclip_prior:
        gk = gate_keys_in_state_dict(state_dict)
        if len(gk) == 0:
            raise RuntimeError(
                "[MERGE RemoteCLIP] save_pretrained aborted: state_dict has no seg_visual_prior_gate.* keys"
            )
        print("[MERGE RemoteCLIP] seg_visual_prior_gate sample keys (up to 5):")
        for x in gk[:5]:
            print(f"  {x}")

    model._hf_peft_config_loaded = False
    model.save_pretrained(args.save_path, state_dict=state_dict)

    tokenizer.save_pretrained(args.save_path)

    if args.use_remoteclip_prior:
        try:
            sd_saved = load_flat_state_dict_from_hf_folder(args.save_path)
        except FileNotFoundError as e:
            raise RuntimeError(
                "[MERGE RemoteCLIP] merged output has no readable weight files to verify gate keys"
            ) from e
        if count_gate_keys(sd_saved) == 0:
            raise RuntimeError(
                "[MERGE RemoteCLIP] merged_model on disk has no seg_visual_prior_gate.* tensors after save_pretrained"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
