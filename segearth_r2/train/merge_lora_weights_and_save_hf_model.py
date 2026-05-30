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


def _str2bool(v):
    """argparse-safe bool: ``type=bool`` treats non-empty strings (including 'False') as True."""
    if isinstance(v, bool):
        return v
    s = str(v).lower().strip()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off", ""):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v!r}")


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
    parser.add_argument("--use_mstva", default=False, type=_str2bool)
    parser.add_argument("--mstva_align_dim", default=256, type=int)
    parser.add_argument("--use_mstva_loss", default=False, type=_str2bool)
    parser.add_argument("--mstva_loss_weight", default=0.0, type=float)
    parser.add_argument("--mstva_scale_weights", default="0.5,0.3,0.2", type=str)

    parser.add_argument("--use_text_film", default=False, type=_str2bool)
    parser.add_argument("--text_film_init_std", default=1e-3, type=float)
    parser.add_argument("--text_film_branch_alpha", default=1.0, type=float)
    parser.add_argument("--text_film_visual_dim", default=512, type=int)
    parser.add_argument("--text_film_eval_mode", default="normal", type=str)
    parser.add_argument("--text_film_force_alpha", default=1.0, type=float)

    parser.add_argument("--use_decoder_attn_bias", default=False, type=_str2bool)
    parser.add_argument("--decoder_attn_bias_dim", default=128, type=int)
    parser.add_argument("--decoder_attn_bias_init_std", default=1e-3, type=float)
    parser.add_argument("--decoder_attn_bias_max_abs", default=0.01, type=float)
    parser.add_argument("--decoder_attn_bias_apply_layers", default="last3", type=str)
    parser.add_argument("--decoder_attn_bias_eval_mode", default="normal", type=str)
    parser.add_argument("--decoder_attn_bias_force_scale", default=1.0, type=float)
    parser.add_argument("--use_decoder_attn_bias_rank_loss", default=False, type=_str2bool)
    parser.add_argument("--decoder_attn_bias_rank_margin", default=0.1, type=float)
    parser.add_argument("--decoder_attn_bias_rank_loss_weight", default=0.001, type=float)

    parser.add_argument("--use_query_aware_decoder_bias", default=False, type=_str2bool)
    parser.add_argument("--use_qdti_mask_feedback", default=False, type=_str2bool)
    parser.add_argument("--use_qdti_rank_loss", default=False, type=_str2bool)
    parser.add_argument("--use_qdti_neg_loss", default=False, type=_str2bool)
    parser.add_argument("--use_qdti_div_loss", default=False, type=_str2bool)
    parser.add_argument("--qdti_gate_init", default=0.0, type=float)
    parser.add_argument("--qdti_warmup_steps", default=500, type=int)
    parser.add_argument("--allow_random_qdti_init", default=False, type=_str2bool)

    parser.add_argument("--lora_enable", default=True, type=_str2bool)
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


def _apply_qdti_config(model, model_args):
    use_qdti = bool(getattr(model_args, "use_query_aware_decoder_bias", False))
    model.config.use_query_aware_decoder_bias = use_qdti
    model.config.use_qdti_mask_feedback = bool(getattr(model_args, "use_qdti_mask_feedback", False))
    model.config.use_qdti_rank_loss = bool(getattr(model_args, "use_qdti_rank_loss", False))
    model.config.use_qdti_neg_loss = bool(getattr(model_args, "use_qdti_neg_loss", False))
    model.config.use_qdti_div_loss = bool(getattr(model_args, "use_qdti_div_loss", False))
    model.config.qdti_gate_init = float(getattr(model_args, "qdti_gate_init", 0.0))
    model.config.qdti_warmup_steps = int(getattr(model_args, "qdti_warmup_steps", 500))
    model.config.allow_random_qdti_init = bool(getattr(model_args, "allow_random_qdti_init", False))
    model.config.decoder_attn_bias_dim = int(getattr(model_args, "decoder_attn_bias_dim", 128))
    model.config.decoder_attn_bias_init_std = float(getattr(model_args, "decoder_attn_bias_init_std", 1e-3))
    model.config.decoder_attn_bias_max_abs = float(getattr(model_args, "decoder_attn_bias_max_abs", 0.02))
    model.config.decoder_attn_bias_apply_layers = str(
        getattr(model_args, "decoder_attn_bias_apply_layers", "last3")
    )
    if use_qdti:
        model.config.use_decoder_attn_bias = False
        model.config.use_mstva = False
        model.config.use_mstva_loss = False
        model.config.use_text_film = False
        setattr(model_args, "use_decoder_attn_bias", False)
        setattr(model_args, "use_mstva", False)
        setattr(model_args, "use_mstva_loss", False)
        setattr(model_args, "use_text_film", False)
    return use_qdti


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
    use_tf = bool(getattr(model_args, "use_text_film", False))
    use_mv = bool(getattr(model_args, "use_mstva", False))
    if use_tf and use_mv:
        use_mv = False
    model.config.use_text_film = use_tf
    model.config.text_film_init_std = float(getattr(model_args, "text_film_init_std", 1e-3))
    model.config.text_film_branch_alpha = float(getattr(model_args, "text_film_branch_alpha", 1.0))
    model.config.text_film_visual_dim = int(getattr(model_args, "text_film_visual_dim", 512))
    model.config.text_film_eval_mode = str(getattr(model_args, "text_film_eval_mode", "normal"))
    model.config.text_film_force_alpha = float(getattr(model_args, "text_film_force_alpha", 1.0))
    model.config.use_mstva = use_mv
    model.config.mstva_align_dim = int(getattr(model_args, "mstva_align_dim", 256))
    model.config.use_mstva_loss = bool(getattr(model_args, "use_mstva_loss", False))
    model.config.mstva_loss_weight = float(getattr(model_args, "mstva_loss_weight", 0.0))
    model.config.mstva_scale_weights = str(getattr(model_args, "mstva_scale_weights", "0.5,0.3,0.2"))
    if hasattr(model_args, "mstva_max_spatial_tokens"):
        model.config.mstva_max_spatial_tokens = int(getattr(model_args, "mstva_max_spatial_tokens"))
    if hasattr(model_args, "mstva_pool_large_scale"):
        model.config.mstva_pool_large_scale = bool(getattr(model_args, "mstva_pool_large_scale"))

    use_qdti = _apply_qdti_config(model, model_args)

    use_dac = bool(getattr(model_args, "use_decoder_attn_bias", False)) and not use_qdti
    model.config.use_decoder_attn_bias = use_dac
    if not use_qdti:
        model.config.decoder_attn_bias_dim = int(getattr(model_args, "decoder_attn_bias_dim", 128))
        model.config.decoder_attn_bias_init_std = float(getattr(model_args, "decoder_attn_bias_init_std", 1e-3))
        model.config.decoder_attn_bias_max_abs = float(getattr(model_args, "decoder_attn_bias_max_abs", 0.01))
        model.config.decoder_attn_bias_apply_layers = str(
            getattr(model_args, "decoder_attn_bias_apply_layers", "last3")
        )
    model.config.decoder_attn_bias_eval_mode = str(getattr(model_args, "decoder_attn_bias_eval_mode", "normal"))
    model.config.decoder_attn_bias_force_scale = float(getattr(model_args, "decoder_attn_bias_force_scale", 1.0))
    model.config.use_decoder_attn_bias_rank_loss = bool(getattr(model_args, "use_decoder_attn_bias_rank_loss", False))
    model.config.decoder_attn_bias_rank_margin = float(getattr(model_args, "decoder_attn_bias_rank_margin", 0.1))
    model.config.decoder_attn_bias_rank_loss_weight = float(
        getattr(model_args, "decoder_attn_bias_rank_loss_weight", 0.001)
    )
    if use_dac:
        model.config.use_mstva = False
        model.config.use_text_film = False
        setattr(model_args, "use_mstva", False)
        setattr(model_args, "use_text_film", False)
        setattr(model_args, "use_mstva_loss", False)

    model.use_temporal_query = model_args.use_temporal_query if hasattr(model_args, 'use_temporal_query') else False
    model.use_vmtf = model_args.use_vmtf if hasattr(model_args, 'use_vmtf') else False

    mask2former_ckpt = model_args.vision_tower_mask
    model.initial_mask_module(mask2former_ckpt, model_args)

    model.get_model().initialize_vision_modules(model_args)
    model.ensure_text_film_branch()
    if use_qdti and hasattr(model, "ensure_qdti_core_branch"):
        model.ensure_qdti_core_branch(allow_init=bool(getattr(model_args, "allow_random_qdti_init", False)))
    elif hasattr(model, "ensure_decoder_attn_bias_branch"):
        model.ensure_decoder_attn_bias_branch()

    vision_tower = model.get_model().get_vision_tower_mask()

    vision_tower.to(device=device)

    train_module_list = [
        "lm_head", "pixel_decoder", "predictor", "SEG_token_projector", "mid_stage_text_recalibration", "mstva",
        "text_film_branch",
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
    model = model.merge_and_unload()

    _apply_qdti_config(model, model_args)
    if use_qdti:
        has_qdti = any("qdti_core" in n for n, _ in model.named_parameters())
        if not has_qdti:
            raise RuntimeError(
                "[QDTI][merge] use_query_aware_decoder_bias=True but merged model has no qdti_core parameters."
            )

    return tokenizer, model

def main(args):
    args = parse_args(args)

    tokenizer, model = load_pretrained_model(args.model_path, model_args=args, mask_config=args.mask_config, device='cuda')

    state_dict = {}
    qdti_keys = []
    for k, v in model.state_dict().items():
        state_dict[k] = v
        if "qdti_core" in k:
            qdti_keys.append(k)
    if bool(getattr(args, "use_query_aware_decoder_bias", False)):
        print(f"[QDTI][merge] state_dict qdti_core keys={len(qdti_keys)}")
        if qdti_keys:
            print(f"[QDTI][merge] sample keys: {qdti_keys[:3]}")
        if len(qdti_keys) == 0:
            raise RuntimeError("[QDTI][merge] no predictor.qdti_core.* in merged state_dict.")
    model._hf_peft_config_loaded = False
    model.save_pretrained(args.save_path, state_dict=state_dict)

    tokenizer.save_pretrained(args.save_path)
    print(f"[OK] saved merged model to {args.save_path}")
    print(f"[OK] config use_query_aware_decoder_bias={getattr(model.config, 'use_query_aware_decoder_bias', False)}")
    
if __name__ == "__main__":
    main(sys.argv[1:])
