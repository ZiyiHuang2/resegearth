#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

from peft import LoraConfig, get_peft_model

from transformers import AutoTokenizer, BitsAndBytesConfig
import torch
from segearth_r2.model import *

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.utils.hf_checkpoint_state import (
    filter_state_dict_by_prefix,
    hf_folder_contains_trained_gate,
    load_flat_state_dict_from_hf_folder,
)

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
    
    vision_tower = model.get_model().get_vision_tower_mask()
    vision_tower.to(device=device)
    image_processor = vision_tower.image_processor

    model.resize_token_embeddings(len(tokenizer))

    use_rc = bool(getattr(model_args, "use_remoteclip_prior", False))
    if use_rc:
        wp = getattr(model_args, "remoteclip_weight_path", None) or ""
        if not wp.strip():
            raise ValueError("[Eval RemoteCLIP] use_remoteclip_prior=True but remoteclip_weight_path is empty")
        if not os.path.isfile(os.path.expanduser(wp)):
            raise FileNotFoundError(f"[Eval RemoteCLIP] remoteclip_weight_path is not a file: {wp}")
        ff = bool(getattr(model_args, "remoteclip_fail_fast", True))
        if not ff:
            raise RuntimeError(
                "[Eval RemoteCLIP] remoteclip_fail_fast=False is not allowed when use_remoteclip_prior=True"
            )
        _log_rc = getattr(model_args, "local_rank", 0) == 0
        if _log_rc:
            print("[Eval RemoteCLIP] use_remoteclip_prior=True")
            print(f"[Eval RemoteCLIP] remoteclip_fail_fast={ff}")
            print(f"[Eval RemoteCLIP] remoteclip_weight_path={wp}")
        model_path_resolved = os.path.abspath(os.path.expanduser(model_path))
        if not hf_folder_contains_trained_gate(model_path_resolved):
            raise RuntimeError(
                "[Eval RemoteCLIP] merged model has no seg_visual_prior_gate.* weights on disk; "
                "re-merge with merge_lora_weights_and_save_hf_model.py --use_remoteclip_prior True."
            )
        model.remoteclip_fail_fast = True
        model.use_remoteclip_prior = True
        model.runtime_tokenizer = tokenizer
        model.get_special_token(
            SEG=tokenizer("[SEG]", return_tensors="pt", add_special_tokens=False)["input_ids"],
            EOS=tokenizer.eos_token_id,
        )
        if model.seg_visual_prior_gate is not None:
            if _log_rc:
                print("[Eval RemoteCLIP] seg_visual_prior_gate present after from_pretrained; skipping gate re-init")
            model.initialize_remoteclip_prior(model_args)
        else:
            model.initialize_remoteclip_prior(model_args)
            full_sd = load_flat_state_dict_from_hf_folder(model_path_resolved)
            gdict = filter_state_dict_by_prefix(full_sd, "seg_visual_prior_gate.")
            gate_model_keys = {k for k in model.state_dict() if k.startswith("seg_visual_prior_gate.")}
            if set(gdict.keys()) != gate_model_keys:
                raise RuntimeError(
                    "[Eval RemoteCLIP] seg_visual_prior_gate checkpoint keys do not match model parameters: "
                    f"only_in_model={sorted(gate_model_keys - set(gdict.keys()))} "
                    f"only_in_ckpt={sorted(set(gdict.keys()) - gate_model_keys)}"
                )
            model.load_state_dict(gdict, strict=False)
            if _log_rc:
                print(
                    f"[Eval RemoteCLIP] loaded trained seg_visual_prior_gate from merged checkpoint "
                    f"({len(gdict)} tensors)"
                )
        if _log_rc:
            print("[RemoteCLIP] remoteclip_weight_loaded=True")
        if model.remoteclip_prior is None or model.seg_visual_prior_gate is None:
            raise RuntimeError("[Eval RemoteCLIP] initialize_remoteclip_prior failed (branch or gate is None)")

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
