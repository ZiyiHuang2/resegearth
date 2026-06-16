import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import argparse
import glob
import copy

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

from segearth_r2.model import *
from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2


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

    parser.add_argument("--lora_enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_weight_path", default="", type=str)
    parser.add_argument("--lora_bias", default="none", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--save_path", default="./InstructSeg_model", type=str, required=True)
    parser.add_argument(
        "--spot_check_base",
        default="",
        type=str,
        help="Optional base HF model dir; compare a weight in unfrozen LLM layers after export.",
    )
    parser.add_argument(
        "--spot_check_key",
        default="model.model.layers.30.self_attn.q_proj.weight",
        type=str,
        help="State-dict key suffix for spot-check; Test 4 unfreezes layers 30-31 on 32-layer Mipha.",
    )
    parser.add_argument(
        "--use_tcpd",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Preserve TCPD forward path in exported config (default: read from checkpoint config).",
    )
    parser.add_argument(
        "--tcpd_condition_source",
        default=None,
        type=str,
        help="TCPD condition source; only 'seg' is supported (default: read from checkpoint).",
    )
    parser.add_argument(
        "--tcpd_spatial_mode",
        default=None,
        type=str,
        help="TCPD MSDeformAttn mode: global or spatial (default: read from checkpoint).",
    )
    parser.add_argument(
        "--tcpd_condition_msdeform",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply TCPD in MSDeformAttn (default: read from checkpoint).",
    )
    parser.add_argument(
        "--tcpd_condition_fpn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply TCPD on FPN fusion (default: read from checkpoint).",
    )
    parser.add_argument(
        "--tcpd_condition_output_scale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply TCPD output scale fusion (default: read from checkpoint).",
    )

    parser.add_argument(
        "--init_model_path",
        default=None,
        type=str,
        help="HF init weights for architecture (default: same as --model_path). "
        "For DeepSpeed checkpoints, set to the training base merged_model.",
    )

    return parser.parse_args(args)


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

def load_pretrained_model(
    model_path,
    model_args,
    mask_config="/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml",
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    init_model_path=None,
):

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
    init_path = init_model_path or model_path
    model = SegEarthR2.from_pretrained(init_path, mask_decoder_cfg=mask_cfg, **kwargs)

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

    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
    model = load_state_dict_from_zero_checkpoint(model, model_path)
    if model_args.lora_enable:
        model = model.merge_and_unload()

    use_tcpd = (
        model_args.use_tcpd
        if getattr(model_args, "use_tcpd", None) is not None
        else getattr(model.config, "use_tcpd", False)
    )
    tcpd_source = (
        model_args.tcpd_condition_source
        if getattr(model_args, "tcpd_condition_source", None) is not None
        else getattr(model.config, "tcpd_condition_source", "seg")
    )
    if tcpd_source != "seg":
        raise ValueError(f"Unsupported tcpd_condition_source: {tcpd_source}")
    model.config.use_tcpd = use_tcpd
    model.config.tcpd_condition_source = tcpd_source

    def _cfg_or_arg(arg_val, cfg_key, default):
        if arg_val is not None:
            return arg_val
        return getattr(model.config, cfg_key, default)

    model.config.tcpd_spatial_mode = _cfg_or_arg(
        getattr(model_args, "tcpd_spatial_mode", None), "tcpd_spatial_mode", "spatial"
    )
    model.config.tcpd_condition_msdeform = _cfg_or_arg(
        getattr(model_args, "tcpd_condition_msdeform", None), "tcpd_condition_msdeform", True
    )
    model.config.tcpd_condition_fpn = _cfg_or_arg(
        getattr(model_args, "tcpd_condition_fpn", None), "tcpd_condition_fpn", True
    )
    model.config.tcpd_condition_output_scale = _cfg_or_arg(
        getattr(model_args, "tcpd_condition_output_scale", None), "tcpd_condition_output_scale", True
    )
    print(
        "[merge] TCPD config: "
        f"use_tcpd={use_tcpd}, tcpd_condition_source={tcpd_source}, "
        f"tcpd_spatial_mode={model.config.tcpd_spatial_mode}, "
        f"tcpd_condition_msdeform={model.config.tcpd_condition_msdeform}, "
        f"tcpd_condition_fpn={model.config.tcpd_condition_fpn}, "
        f"tcpd_condition_output_scale={model.config.tcpd_condition_output_scale}"
    )

    return tokenizer, model


def spot_check_weights(base_path, merged_state, key_suffix, mask_config=None):
    if not base_path:
        return
    try:
        mask_cfg = get_mask_config(mask_config) if mask_config else None
        base_model = SegEarthR2.from_pretrained(
            base_path, mask_decoder_cfg=mask_cfg, torch_dtype=torch.float16, device_map="cpu"
        )
        base_state = base_model.state_dict()
        merged_key = next((k for k in merged_state if k.endswith(key_suffix) or k == key_suffix), None)
        base_key = next((k for k in base_state if k.endswith(key_suffix) or k == key_suffix), None)
        if merged_key is None or base_key is None:
            print(f"[merge spot-check] key not found: {key_suffix}")
            return
        diff = (merged_state[merged_key].float() - base_state[base_key].float()).abs().mean().item()
        changed = diff > 0
        print(
            f"[merge spot-check] key={merged_key} mean_abs_diff={diff:.6e} changed={changed}"
        )
    except Exception as exc:
        print(f"[merge spot-check] skipped: {exc}")

def main(args):
    args = parse_args(args)

    tokenizer, model = load_pretrained_model(
        args.model_path,
        model_args=args,
        mask_config=args.mask_config,
        device="cuda",
        init_model_path=args.init_model_path,
    )

    state_dict = {}
    for k, v in model.state_dict().items():
        state_dict[k] = v
    spot_check_weights(args.spot_check_base, state_dict, args.spot_check_key, args.mask_config)
    model._hf_peft_config_loaded = False
    model.save_pretrained(args.save_path, state_dict=state_dict)
    print(
        f"[merge] saved config use_tcpd={getattr(model.config, 'use_tcpd', False)}, "
        f"tcpd_condition_source={getattr(model.config, 'tcpd_condition_source', 'seg')}, "
        f"tcpd_spatial_mode={getattr(model.config, 'tcpd_spatial_mode', 'spatial')}, "
        f"tcpd_condition_msdeform={getattr(model.config, 'tcpd_condition_msdeform', True)}, "
        f"tcpd_condition_fpn={getattr(model.config, 'tcpd_condition_fpn', True)}, "
        f"tcpd_condition_output_scale={getattr(model.config, 'tcpd_condition_output_scale', True)}"
    )

    tokenizer.save_pretrained(args.save_path)
    
if __name__ == "__main__":
    main(sys.argv[1:])
