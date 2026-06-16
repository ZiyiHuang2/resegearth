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

import glob
import os
from pathlib import Path

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoConfig
import torch

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.model.mipha.model.language_model.configuration_mipha import MiphaPhiConfig


def _read_hf_state_dict(model_path):
    st_paths = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if st_paths:
        from safetensors.torch import load_file
        state_dict = {}
        for path in st_paths:
            state_dict.update(load_file(path))
        return state_dict
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No HF weights found under {model_path}")


def _dgp_keys_in_state(state_dict) -> dict:
    pa = [k for k in state_dict if "prompt_adapter" in k]
    qr = [k for k in state_dict if "query_refiner" in k]
    qdti = [k for k in state_dict if "query_specific_text_memory_bias" in k or "qdti" in k.lower()]
    return {"prompt_adapter": pa, "query_refiner": qr, "qdti": qdti}


def _merge_stage_a_sidecar(model_path: str, state_dict: dict) -> dict:
    """Merge dgp_stage_a_weights.bin from output root or latest checkpoint if HF dict lacks DGP keys."""
    groups = _dgp_keys_in_state(state_dict)
    if groups["prompt_adapter"] and groups["query_refiner"]:
        return state_dict

    root = model_path
    candidates = [root]
    ckpts = sorted(glob.glob(os.path.join(root, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))
    if ckpts:
        candidates.insert(0, ckpts[-1])

    for base in candidates:
        for fname in (SegEarthR2.DGP_STAGE_A_WEIGHTS_BIN, SegEarthR2.DGP_STAGE_A_WEIGHTS_SAFE):
            path = os.path.join(base, fname)
            if os.path.isfile(path):
                sidecar = SegEarthR2._load_stage_a_file(Path(path))
                merged = dict(state_dict)
                merged.update({SegEarthR2._normalize_dgp_param_key(k): v for k, v in sidecar.items()})
                print(f"[Builder][DGP] Merged Stage A sidecar from {path}")
                return merged
    return state_dict


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

    try:
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except (ValueError, KeyError):
        hf_config = MiphaPhiConfig.from_pretrained(model_path)
    SegEarthR2.merge_dgp_config_from_hf(hf_config, model_args)

    use_dgp_qdti = bool(getattr(model_args, "use_dgp_qdti", False))
    require_qdti_bias = SegEarthR2.resolve_use_qdti_bias(model_args)
    dgp_stage = (getattr(model_args, "dgp_training_stage", None) or "").strip().lower()
    fresh_stage_a = use_dgp_qdti and dgp_stage == "a" and not require_qdti_bias

    hf_state_dict = _read_hf_state_dict(model_path)
    hf_state_dict = _merge_stage_a_sidecar(model_path, hf_state_dict)
    groups_before = _dgp_keys_in_state(hf_state_dict)
    has_dgp_weights = bool(groups_before["prompt_adapter"] and groups_before["query_refiner"])

    if use_dgp_qdti and has_dgp_weights:
        SegEarthR2.validate_dgp_qdti_checkpoint(
            hf_state_dict,
            context=f"HF model {model_path}",
            require_qdti_bias=require_qdti_bias,
        )
    elif use_dgp_qdti and fresh_stage_a and not has_dgp_weights:
        print(
            f"[Builder][DGP] Fresh Stage A init from {model_path}: "
            "no prompt_adapter/query_refiner in checkpoint; using module defaults."
        )
    elif use_dgp_qdti:
        SegEarthR2.validate_dgp_qdti_checkpoint(
            hf_state_dict,
            context=f"HF model {model_path}",
            require_qdti_bias=require_qdti_bias,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = SegEarthR2.from_pretrained(model_path, mask_decoder_cfg=mask_cfg, **kwargs)

    SegEarthR2.sync_dgp_config_from_args(model.config, model_args)
    model.ensure_dgp_qdti_modules()
    if use_dgp_qdti:
        print(
            f"[Builder][DGP config] version={getattr(model.config, 'dgp_version', None)} "
            f"stage={getattr(model.config, 'dgp_training_stage', None)} "
            f"qdti_apply_layers={getattr(model.config, 'qdti_apply_layers', None)} "
            f"qdti_scale_init={getattr(model.config, 'qdti_scale_init', None)} "
            f"use_qdti_bias={getattr(model.config, 'use_qdti_bias', None)}"
        )

    groups_before = _dgp_keys_in_state(hf_state_dict)
    print(f"[Builder][DGP load] loaded prompt_adapter keys: {len(groups_before['prompt_adapter'])}")
    print(f"[Builder][DGP load] loaded query_refiner keys: {len(groups_before['query_refiner'])}")
    if require_qdti_bias:
        print(f"[Builder][DGP load] loaded qdti keys: {len(groups_before['qdti'])}")

    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
    if use_dgp_qdti and not (fresh_stage_a and not has_dgp_weights):
        missing_dgp = [
            k for k in missing
            if any(marker in k for marker in SegEarthR2.DGP_PROMPT_KEY_MARKERS)
            or (require_qdti_bias and "query_specific_text_memory_bias" in k)
        ]
        unexpected_dgp = [
            k for k in unexpected
            if any(marker in k for marker in SegEarthR2.DGP_QDTI_STATE_KEY_MARKERS)
        ]
        print(f"[Builder][DGP load] missing DGP keys: {missing_dgp[:20]}")
        print(f"[Builder][DGP load] unexpected DGP keys: {unexpected_dgp[:20]}")
        if missing_dgp:
            raise RuntimeError(
                f"use_dgp_qdti=True but failed to load DGP weights from {model_path}. "
                f"Missing keys: {missing_dgp[:20]}. "
                f"If Stage A weights live in checkpoint-*/dgp_stage_a_weights.bin, point --model_path "
                f"to that checkpoint dir or ensure sidecar was merged."
            )
        if not groups_before["prompt_adapter"] or not groups_before["query_refiner"]:
            raise RuntimeError(
                f"use_dgp_qdti=True but no prompt_adapter/query_refiner tensors found under {model_path} "
                f"or checkpoint sidecars."
            )
    
    vision_tower = model.get_model().get_vision_tower_mask()
    vision_tower.to(device=device)
    image_processor = vision_tower.image_processor

    model.resize_token_embeddings(len(tokenizer))

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
