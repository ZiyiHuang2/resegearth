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
    if not hasattr(model.config, "use_mstva"):
        model.config.use_mstva = False
    if not hasattr(model.config, "mstva_align_dim"):
        model.config.mstva_align_dim = 256
    if not hasattr(model.config, "use_mstva_loss"):
        model.config.use_mstva_loss = False
    if not hasattr(model.config, "mstva_loss_weight"):
        model.config.mstva_loss_weight = 0.0
    if not hasattr(model.config, "mstva_scale_weights"):
        model.config.mstva_scale_weights = "0.5,0.3,0.2"
    if not hasattr(model.config, "use_text_film"):
        model.config.use_text_film = False
    if not hasattr(model.config, "text_film_init_std"):
        model.config.text_film_init_std = 1e-3
    if not hasattr(model.config, "text_film_branch_alpha"):
        model.config.text_film_branch_alpha = 1.0
    if not hasattr(model.config, "text_film_visual_dim"):
        model.config.text_film_visual_dim = 512
    if not hasattr(model.config, "text_film_eval_mode"):
        model.config.text_film_eval_mode = "normal"
    if not hasattr(model.config, "text_film_force_alpha"):
        model.config.text_film_force_alpha = 1.0

    if not hasattr(model.config, "use_query_aware_decoder_bias"):
        model.config.use_query_aware_decoder_bias = False
    if not hasattr(model.config, "use_qdti_mask_feedback"):
        model.config.use_qdti_mask_feedback = False
    if not hasattr(model.config, "use_qdti_rank_loss"):
        model.config.use_qdti_rank_loss = False
    if not hasattr(model.config, "use_qdti_neg_loss"):
        model.config.use_qdti_neg_loss = False
    if not hasattr(model.config, "use_qdti_div_loss"):
        model.config.use_qdti_div_loss = False
    if not hasattr(model.config, "qdti_rank_loss_weight"):
        model.config.qdti_rank_loss_weight = 0.001
    if not hasattr(model.config, "qdti_rank_margin"):
        model.config.qdti_rank_margin = 0.1
    if not hasattr(model.config, "qdti_neg_loss_weight"):
        model.config.qdti_neg_loss_weight = 0.001
    if not hasattr(model.config, "qdti_div_loss_weight"):
        model.config.qdti_div_loss_weight = 0.001
    if not hasattr(model.config, "qdti_neg_iou_thresh"):
        model.config.qdti_neg_iou_thresh = 0.3
    if not hasattr(model.config, "qdti_gate_init"):
        model.config.qdti_gate_init = 0.0
    if not hasattr(model.config, "qdti_warmup_steps"):
        model.config.qdti_warmup_steps = 500
    if not hasattr(model.config, "allow_random_qdti_init"):
        model.config.allow_random_qdti_init = False
    if not hasattr(model.config, "use_decoder_attn_bias"):
        model.config.use_decoder_attn_bias = False
    if not hasattr(model.config, "decoder_attn_bias_dim"):
        model.config.decoder_attn_bias_dim = 128
    if not hasattr(model.config, "decoder_attn_bias_init_std"):
        model.config.decoder_attn_bias_init_std = 1e-3
    if not hasattr(model.config, "decoder_attn_bias_max_abs"):
        model.config.decoder_attn_bias_max_abs = 0.02
    if not hasattr(model.config, "decoder_attn_bias_apply_layers"):
        model.config.decoder_attn_bias_apply_layers = "last3"
    if not hasattr(model.config, "decoder_attn_bias_eval_mode"):
        model.config.decoder_attn_bias_eval_mode = "normal"
    if not hasattr(model.config, "decoder_attn_bias_force_scale"):
        model.config.decoder_attn_bias_force_scale = 1.0
    if not hasattr(model.config, "use_decoder_attn_bias_rank_loss"):
        model.config.use_decoder_attn_bias_rank_loss = False
    if not hasattr(model.config, "decoder_attn_bias_rank_margin"):
        model.config.decoder_attn_bias_rank_margin = 0.1
    if not hasattr(model.config, "decoder_attn_bias_rank_loss_weight"):
        model.config.decoder_attn_bias_rank_loss_weight = 0.001

    vision_tower = model.get_model().get_vision_tower_mask()
    vision_tower.to(device=device)
    image_processor = vision_tower.image_processor

    model.resize_token_embeddings(len(tokenizer))
    allow_random_qdti = bool(getattr(model_args, "allow_random_qdti_init", False)) if model_args is not None else False
    if getattr(model.config, "use_query_aware_decoder_bias", False):
        model.validate_qdti_core_weights(
            checkpoint_path=model_path,
            allow_random_init=allow_random_qdti,
            context="load_pretrained",
            fresh_training_init=False,
        )
    if hasattr(model, "ensure_text_film_branch"):
        model.ensure_text_film_branch()
    if hasattr(model, "ensure_qdti_core_branch"):
        model.ensure_qdti_core_branch(allow_init=allow_random_qdti)
    if hasattr(model, "ensure_decoder_attn_bias_branch"):
        model.ensure_decoder_attn_bias_branch()

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
