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
import os

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_qwen import SegEarthR2Qwen as SegEarthR2

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

    base_model_path = getattr(model_args, "base_model_path", None) or model_path

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    model = SegEarthR2.from_pretrained(base_model_path, mask_decoder_cfg=mask_cfg, **kwargs)
    
    if not getattr(model, "is_train_mask_decode", False):
        mask2former_ckpt = getattr(model_args, "vision_tower_mask", None)
        model.initial_mask_module(mask2former_ckpt, model_args)

    model.get_model().initialize_vision_modules(model_args)
    
    vision_tower = model.get_model().get_vision_tower_mask()
    vision_tower.to(device=device)
    image_processor = vision_tower.image_processor

    if "[SEG]" not in tokenizer.get_vocab():
        tokenizer.add_tokens("[SEG]")
    model.resize_token_embeddings(len(tokenizer))

    # Optional: load merged/custom local state dict from model_path when base_model_path is provided.
    if model_path != base_model_path:
        state_dict = None
        safe_path = os.path.join(model_path, "model.safetensors")
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.isfile(safe_path):
            from safetensors.torch import load_file as load_safetensors
            state_dict = load_safetensors(safe_path, device="cpu")
        elif os.path.isfile(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")

        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
