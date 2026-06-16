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

    parser.add_argument("--lora_enable", default=True, type=bool)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_weight_path", default="", type=str)
    parser.add_argument("--lora_bias", default="none", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    
    parser.add_argument("--save_path", default="./InstructSeg_model", type=str, required=True)

    parser.add_argument("--use_dgp_qdti", default=False, type=lambda x: str(x).lower() in ("1", "true", "yes"))
    parser.add_argument("--use_qdti_bias", default=None, type=lambda x: None if str(x).lower() in ("none", "null") else str(x).lower() in ("1", "true", "yes"))
    parser.add_argument("--dgp_fuse_dim", default=256, type=int)
    parser.add_argument("--dgp_refiner_hidden_dim", default=512, type=int)
    parser.add_argument("--dgp_pg_tokens", default=1, type=int)
    parser.add_argument("--qdti_bias_dim", default=128, type=int)
    parser.add_argument("--qdti_init_std", default=1e-3, type=float)
    parser.add_argument("--qdti_max_abs", default=0.01, type=float)
    parser.add_argument("--qdti_apply_layers", default=None, type=str)
    parser.add_argument("--qdti_scale_init", default=None, type=float)
    parser.add_argument("--scale_hard_loss_weight", default=0.0, type=float)
    parser.add_argument("--dgp_version", default=None, type=str)
    parser.add_argument("--dgp_training_stage", default=None, type=str)
    parser.add_argument("--gate_init", default=None, type=float)
    
    return parser.parse_args(args)


def _apply_merge_dgp_defaults(args):
    stage = (getattr(args, "dgp_training_stage", None) or "").strip().lower()
    version = (getattr(args, "dgp_version", None) or "").strip().lower()
    if version == "v6.1" or stage in ("a", "b"):
        SegEarthR2.apply_v61_stage_defaults(args, stage or "b")
    if getattr(args, "use_dgp_qdti", False) and getattr(args, "use_qdti_bias", None) is None:
        args.use_qdti_bias = True


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

    hf_config = None
    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        SegEarthR2.merge_dgp_config_from_hf(hf_config, model_args)

    use_dgp_qdti = bool(getattr(model_args, "use_dgp_qdti", False))
    require_qdti_bias = SegEarthR2.resolve_use_qdti_bias(model_args)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = SegEarthR2.from_pretrained(model_path, mask_decoder_cfg=mask_cfg, **kwargs)

    SegEarthR2.sync_dgp_config_from_args(model.config, model_args)

    model.use_temporal_query = model_args.use_temporal_query if hasattr(model_args, 'use_temporal_query') else False
    model.use_vmtf = model_args.use_vmtf if hasattr(model_args, 'use_vmtf') else False
    

    mask2former_ckpt = model_args.vision_tower_mask
    model.initial_mask_module(mask2former_ckpt, model_args)
    model.ensure_dgp_qdti_modules()

    model.get_model().initialize_vision_modules(model_args)

    vision_tower = model.get_model().get_vision_tower_mask()

    vision_tower.to(device=device)

    train_module_list = [
        "lm_head", "pixel_decoder", "predictor", "SEG_token_projector",
    ]
    if use_dgp_qdti:
        train_module_list.extend(["prompt_adapter", "query_refiner"])

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
    if use_dgp_qdti:
        state_dict = model.state_dict()
        SegEarthR2.validate_dgp_qdti_checkpoint(
            state_dict,
            context=f"ZeRO checkpoint {model_path}",
            require_qdti_bias=require_qdti_bias,
        )
    model = model.merge_and_unload()
    SegEarthR2.sync_dgp_config_from_args(model.config, model_args)

    return tokenizer, model

def main(args):
    args = parse_args(args)
    _apply_merge_dgp_defaults(args)

    tokenizer, model = load_pretrained_model(args.model_path, model_args=args, mask_config=args.mask_config, device='cuda')

    SegEarthR2.sync_dgp_config_from_args(model.config, args)

    state_dict = {}
    for k, v in model.state_dict().items():
        state_dict[k] = v
    model._hf_peft_config_loaded = False
    model.save_pretrained(args.save_path, state_dict=state_dict)

    tokenizer.save_pretrained(args.save_path)
    
if __name__ == "__main__":
    main(sys.argv[1:])
