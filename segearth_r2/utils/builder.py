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

from peft import LoraConfig, get_peft_model

from transformers import AutoTokenizer, BitsAndBytesConfig
import torch
from segearth_r2.model import *

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2


def _read_hf_state_dict(model_path):
    import os
    import glob
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

    use_dgp_qdti = bool(getattr(model_args, "use_dgp_qdti", False))
    require_qdti_bias = SegEarthR2.resolve_use_qdti_bias(model_args)
    hf_state_dict = _read_hf_state_dict(model_path)
    if use_dgp_qdti:
        SegEarthR2.validate_dgp_qdti_checkpoint(
            hf_state_dict,
            context=f"HF model {model_path}",
            require_qdti_bias=require_qdti_bias,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = SegEarthR2.from_pretrained(model_path, mask_decoder_cfg=mask_cfg, **kwargs)

    SegEarthR2.sync_dgp_config_from_args(model.config, model_args)
    model.ensure_dgp_qdti_modules()
    missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
    if use_dgp_qdti:
        missing_dgp = [
            k for k in missing
            if any(marker in k for marker in SegEarthR2.DGP_PROMPT_KEY_MARKERS)
            or (require_qdti_bias and "query_specific_text_memory_bias" in k)
        ]
        if missing_dgp:
            raise RuntimeError(
                f"use_dgp_qdti=True but failed to load DGP-QDTI weights from {model_path}. "
                f"Missing keys: {missing_dgp[:20]}"
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
