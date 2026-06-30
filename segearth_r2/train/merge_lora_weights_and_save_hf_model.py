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

    parser.add_argument("--lora_enable", default=True, type=bool)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_weight_path", default="", type=str)
    parser.add_argument("--lora_bias", default="none", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    
    parser.add_argument("--save_path", default="./InstructSeg_model", type=str, required=True)

    parser.add_argument("--setpp_enable", default=True, type=lambda x: str(x).lower() in ('1', 'true', 'yes'))
    parser.add_argument("--setpp_regroup_set_loss", default=True, type=lambda x: str(x).lower() in ('1', 'true', 'yes'))
    parser.add_argument("--setpp_closed_loop", default=True, type=lambda x: str(x).lower() in ('1', 'true', 'yes'))
    parser.add_argument("--setpp_csqr_enable", default=True, type=lambda x: str(x).lower() in ('1', 'true', 'yes'))
    parser.add_argument("--setpp_csqr_fusion_alpha_init", default=0.99, type=float)
    
    return parser.parse_args(args)


def resolve_merge_arch_path(checkpoint_path):
    """DeepSpeed checkpoints often lack config.json; use PEFT base_model for architecture init."""
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.isfile(config_path):
        return checkpoint_path
    adapter_cfg = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            base = json.load(f).get("base_model_name_or_path")
        if base and os.path.isdir(base):
            print(f"[merge] checkpoint has no config.json; init architecture from base: {base}")
            return base
    return checkpoint_path


def ensure_seg_set_tokens(tokenizer, model):
    """Match train.py tokenizer setup for [SEG]/[SET] and lm_head safety."""
    added = tokenizer.add_tokens(["[SEG]", "[SET]"])
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    lm_head_size = model.lm_head.out_features
    tokenizer_len = len(tokenizer)
    seg_id = tokenizer("[SEG]", return_tensors='pt', add_special_tokens=False)['input_ids'].item()
    set_id = tokenizer("[SET]", return_tensors='pt', add_special_tokens=False)['input_ids'].item()
    if tokenizer_len > lm_head_size or seg_id >= lm_head_size or set_id >= lm_head_size:
        raise RuntimeError(
            f"Tokenizer/lm_head mismatch after merge init: "
            f"tokenizer_len={tokenizer_len}, lm_head={lm_head_size}, SEG_id={seg_id}, SET_id={set_id}"
        )
    print(f"[merge] tokenizer_len={tokenizer_len}, lm_head={lm_head_size}, SEG_id={seg_id}, SET_id={set_id}")
    return tokenizer


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

    # TG-Swin scalar params break meta init with device_map; match tgswin merge script.
    kwargs = {"low_cpu_mem_usage": False}

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

    arch_path = resolve_merge_arch_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = SegEarthR2.from_pretrained(arch_path, mask_decoder_cfg=mask_cfg, **kwargs)

    model.use_temporal_query = model_args.use_temporal_query if hasattr(model_args, 'use_temporal_query') else False
    model.use_vmtf = model_args.use_vmtf if hasattr(model_args, 'use_vmtf') else False
    

    mask2former_ckpt = model_args.vision_tower_mask
    model.initial_mask_module(mask2former_ckpt, model_args)
    model.ensure_setpp_predictor(model_args)
    model.persist_setpp_config(model_args)

    model.get_model().initialize_vision_modules(model_args)

    vision_tower = model.get_model().get_vision_tower_mask()

    vision_tower.to(device=device)
    if not load_8bit and not load_4bit:
        model.to(device=device)

    train_module_list = [
        "lm_head", "pixel_decoder", "predictor", "SEG_token_projector", "SET_token_projector",
    ]
    if getattr(mask_cfg, "TG_SWIN", None) and getattr(mask_cfg.TG_SWIN, "ENABLED", False):
        train_module_list.extend(["tg_swin_tcf", "tg_swin_controller"])
        if getattr(mask_cfg.TG_SWIN, "USE_SET_TGSWIN_CONTROL", False):
            train_module_list.append("tg_swin_set_control")

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

    tokenizer = ensure_seg_set_tokens(tokenizer, model)

    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
    print(f"[merge] loading DeepSpeed ZeRO weights from: {model_path}")
    model = load_state_dict_from_zero_checkpoint(model, model_path)
    model = model.merge_and_unload()

    return tokenizer, model

def main(args):
    args = parse_args(args)

    tokenizer, model = load_pretrained_model(args.model_path, model_args=args, mask_config=args.mask_config, device='cuda')

    state_dict = {}
    for k, v in model.state_dict().items():
        print(k)
        state_dict[k] = v
    model._hf_peft_config_loaded = False
    model.save_pretrained(args.save_path, state_dict=state_dict)

    tokenizer.save_pretrained(args.save_path)
    
if __name__ == "__main__":
    main(sys.argv[1:])
