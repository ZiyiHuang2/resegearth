import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import argparse
import glob
import copy
import json

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
    parser.add_argument(
        "--a3_only",
        action="store_true",
        help="Merge A3-frozen checkpoint (no LoRA; only set modules were trained).",
    )
    parser.add_argument(
        "--baseline_model_path",
        default="",
        type=str,
        help="LaSeRS baseline merged_model path (required for --a3_only).",
    )
    parser.add_argument(
        "--lasers_category_vocab_path",
        default="",
        type=str,
        help="Path to lasers_category_vocab.json (required for A3 category_set_head shape).",
    )
    parser.add_argument(
        "--use_explicit_set_token",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load/merge C-lite-v2 SET_token_projector and q_set_fusion modules.",
    )
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_weight_path", default="", type=str)
    parser.add_argument("--lora_bias", default="none", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    
    parser.add_argument("--save_path", default="./InstructSeg_model", type=str, required=True)
    
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


def resolve_init_path(model_path):
    """DeepSpeed checkpoints often lack config.json; load architecture from LoRA base."""
    if os.path.isfile(os.path.join(model_path, "config.json")):
        return model_path
    adapter_path = os.path.join(model_path, "adapter_config.json")
    if os.path.isfile(adapter_path):
        with open(adapter_path, encoding="utf-8") as f:
            base = json.load(f).get("base_model_name_or_path")
        if base and os.path.isdir(base):
            return base
    return model_path


def apply_set_training_config(init_args, model_args):
    vocab_path = getattr(model_args, "lasers_category_vocab_path", "") or ""
    use_explicit = getattr(model_args, "use_explicit_set_token", False)
    a3_only = getattr(model_args, "a3_only", False)
    if not (vocab_path or use_explicit or a3_only):
        return
    if not getattr(init_args, "use_set_conditioner", False):
        init_args.use_set_conditioner = True
        init_args.use_set_count_loss = True
        init_args.use_set_category_loss = True
    if vocab_path and not getattr(init_args, "lasers_category_vocab_path", ""):
        init_args.lasers_category_vocab_path = vocab_path
    if use_explicit:
        init_args.use_explicit_set_token = True


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

    init_path = model_path
    if getattr(model_args, "a3_only", False):
        baseline = getattr(model_args, "baseline_model_path", "") or ""
        if not baseline:
            raise ValueError("--a3_only requires --baseline_model_path (LaSeRS baseline merged_model).")
        init_path = baseline
    else:
        init_path = resolve_init_path(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = SegEarthR2.from_pretrained(init_path, mask_decoder_cfg=mask_cfg, **kwargs)

    model.use_temporal_query = model_args.use_temporal_query if hasattr(model_args, 'use_temporal_query') else False
    model.use_vmtf = model_args.use_vmtf if hasattr(model_args, 'use_vmtf') else False

    mask2former_ckpt = model_args.vision_tower_mask
    init_args = model.config
    apply_set_training_config(init_args, model_args)
    if model.is_train_mask_decode:
        model.init_set_conditioning_modules(init_args)
    else:
        model.initial_mask_module(mask2former_ckpt, model_args=init_args)

    model.get_model().initialize_vision_modules(model_args)

    vision_tower = model.get_model().get_vision_tower_mask()

    vision_tower.to(device=device)

    if getattr(model_args, "a3_only", False):
        model_args.lora_enable = False
        train_module_list = [
            "set_conditioner", "count_head", "category_set_head",
        ]
    else:
        train_module_list = [
            "lm_head", "pixel_decoder", "predictor", "SEG_token_projector",
            "set_conditioner", "count_head", "category_set_head",
        ]
    # C-lite-v2: 显式 [SET] token 相关模块
    if getattr(model_args, "use_explicit_set_token", False):
        train_module_list.extend(["SET_token_projector", "q_set_fusion"])

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
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()

    return tokenizer, model


def persist_set_config(model, args):
    """Write SET / C-lite-v2 flags into config.json so eval can reload modules."""
    set_config_fields = (
        "use_set_conditioner",
        "use_set_count_loss",
        "use_set_category_loss",
        "set_conditioner_layers",
        "set_conditioner_heads",
        "set_conditioner_gate_init",
        "lambda_set_count",
        "lambda_set_category",
        "set_max_count",
        "lasers_category_vocab_path",
        "use_explicit_set_token",
        "q_set_fusion_hidden",
    )
    for name in set_config_fields:
        value = getattr(model.config, name, None)
        if value is None and hasattr(args, name):
            value = getattr(args, name)
        if value is not None:
            setattr(model.config, name, value)

    vocab_path = getattr(args, "lasers_category_vocab_path", "") or ""
    if vocab_path and not getattr(model.config, "lasers_category_vocab_path", ""):
        model.config.lasers_category_vocab_path = vocab_path

    if getattr(model, "set_conditioner", None) is not None:
        model.config.use_set_conditioner = True
        if not getattr(model.config, "use_set_count_loss", False):
            model.config.use_set_count_loss = True
        if not getattr(model.config, "use_set_category_loss", False):
            model.config.use_set_category_loss = True


def main(args):
    args = parse_args(args)

    tokenizer, model = load_pretrained_model(args.model_path, model_args=args, mask_config=args.mask_config, device='cuda')

    state_dict = {}
    for k, v in model.state_dict().items():
        print(k)
        state_dict[k] = v
    model._hf_peft_config_loaded = False
    persist_set_config(model, args)
    model.save_pretrained(args.save_path, state_dict=state_dict)

    tokenizer.save_pretrained(args.save_path)
    
if __name__ == "__main__":
    main(sys.argv[1:])
