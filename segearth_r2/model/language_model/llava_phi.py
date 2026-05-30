import os
import json
from typing import List, Optional, Tuple, Union
from addict import Dict
from dataclasses import dataclass
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom

from ..mipha.model.language_model.mipha_phi import (MiphaPhiForCausalLM, MiphaPhiModel)

from segearth_r2.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, REFER_TOKEN_INDEX

from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MultiScaleMaskedTransformerDecoderForOPTPreTrain,
    DecoderTokenAttnBias,
)
from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.qdti_core import QDTICore
from ..mask_decoder.Mask2Former_Simplify.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from ..mask_encoder.swin_trans import build_swin_b, build_swin_l

from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine

from ..datasets_mapper.IVS_mapper import IVSDatasetMapper
from segearth_r2.model.mask_decoder.mask_criterion.Mask_Criterion import Criterion, hungarian_matcher_InstructSeg
from transformers import PhiModel, PhiForCausalLM, PhiConfig
from fvcore.nn import FlopCountAnalysis


class TextFiLMBranch(nn.Module):
    """Channel-wise FiLM on spatial features: out = x + alpha * ((x * (1+gamma) + beta) - x)."""

    def __init__(self, text_dim: int, visual_channels: int, init_std: float, branch_alpha_init: float):
        super().__init__()
        self.visual_channels = int(visual_channels)
        self.gamma_linear = nn.Linear(int(text_dim), self.visual_channels)
        self.beta_linear = nn.Linear(int(text_dim), self.visual_channels)
        nn.init.normal_(self.gamma_linear.weight, mean=0.0, std=float(init_std))
        nn.init.zeros_(self.gamma_linear.bias)
        nn.init.zeros_(self.beta_linear.weight)
        nn.init.zeros_(self.beta_linear.bias)
        self.branch_alpha = nn.Parameter(torch.tensor(float(branch_alpha_init), dtype=torch.float32))
        self._last_gamma_norm: Optional[torch.Tensor] = None
        self._last_beta_norm: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        text_cond: torch.Tensor,
        eval_mode: str = "normal",
        force_alpha: float = 1.0,
    ) -> torch.Tensor:
        self._last_gamma_norm = None
        self._last_beta_norm = None
        if text_cond is None:
            return x
        w_dtype = self.gamma_linear.weight.dtype
        tc = text_cond.to(dtype=w_dtype, device=self.gamma_linear.weight.device)
        gamma_vec = self.gamma_linear(tc)
        beta_vec = self.beta_linear(tc)
        B, C, _, _ = x.shape
        if C != self.visual_channels or gamma_vec.shape[0] != B or gamma_vec.shape[1] != C:
            return x
        gamma = gamma_vec.to(dtype=x.dtype, device=x.device).view(B, C, 1, 1)
        beta = beta_vec.to(dtype=x.dtype, device=x.device).view(B, C, 1, 1)
        film = x * (1.0 + gamma) + beta
        delta = film - x
        if eval_mode not in ("normal", "bypass", "force_alpha"):
            eval_mode = "normal"
        if eval_mode == "bypass":
            out = x
        elif eval_mode == "force_alpha":
            fa = torch.tensor(float(force_alpha), device=x.device, dtype=x.dtype)
            out = x + fa * delta
        else:
            alpha_eff = self.branch_alpha.to(device=x.device, dtype=x.dtype)
            out = x + alpha_eff * delta
        with torch.no_grad():
            gnm = gamma_vec.detach().float().norm(p=2, dim=-1).mean()
            bnm = beta_vec.detach().float().norm(p=2, dim=-1).mean()
            self._last_gamma_norm = gnm
            self._last_beta_norm = bnm
        return out


@dataclass
class CausalOutputWithMask(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_mask: Optional[torch.FloatTensor] = None
    loss_dice: Optional[torch.FloatTensor] = None
    loss_llm: Optional[torch.FloatTensor] = None
    loss_attention: Optional[torch.FloatTensor] = None
    loss_midstage_gate: Optional[torch.FloatTensor] = None
    midstage_gate_alpha: Optional[torch.FloatTensor] = None
    loss_mstva_align: Optional[torch.FloatTensor] = None
    mstva_alpha3: Optional[torch.FloatTensor] = None
    mstva_alpha4: Optional[torch.FloatTensor] = None
    mstva_alpha5: Optional[torch.FloatTensor] = None
    text_film_gamma_norm: Optional[torch.FloatTensor] = None
    text_film_beta_norm: Optional[torch.FloatTensor] = None
    text_film_branch_alpha: Optional[torch.FloatTensor] = None
    decoder_attn_bias_abs_mean: Optional[torch.FloatTensor] = None
    decoder_attn_bias_raw_std: Optional[torch.FloatTensor] = None
    decoder_attn_bias_max: Optional[torch.FloatTensor] = None
    decoder_attn_bias_min: Optional[torch.FloatTensor] = None
    decoder_attn_bias_enabled: Optional[torch.FloatTensor] = None
    loss_decoder_attn_bias_rank: Optional[torch.FloatTensor] = None
    decoder_attn_bias_rank_loss_raw: Optional[torch.FloatTensor] = None
    decoder_attn_bias_inside_mean: Optional[torch.FloatTensor] = None
    decoder_attn_bias_outside_mean: Optional[torch.FloatTensor] = None
    decoder_attn_bias_inside_outside_gap: Optional[torch.FloatTensor] = None
    decoder_attn_bias_rank_fg_access_ratio: Optional[torch.FloatTensor] = None
    decoder_attn_bias_rank_bg_access_ratio: Optional[torch.FloatTensor] = None
    decoder_attn_bias_rank_valid_count: Optional[torch.FloatTensor] = None
    decoder_attn_bias_rank_num_layers: Optional[torch.FloatTensor] = None
    decoder_attn_bias_rank_layer_indices: Optional[str] = None
    loss_qdti_rank: Optional[torch.FloatTensor] = None
    qdti_rank_loss_raw: Optional[torch.FloatTensor] = None
    loss_qdti_neg: Optional[torch.FloatTensor] = None
    loss_qdti_div: Optional[torch.FloatTensor] = None
    qdti_bias_abs_mean: Optional[torch.FloatTensor] = None
    qdti_enabled: Optional[torch.FloatTensor] = None
    qdti_rank_valid_count: Optional[torch.FloatTensor] = None
    qdti_alpha_l: Optional[torch.FloatTensor] = None
    qdti_gate_eff: Optional[torch.FloatTensor] = None
    qdti_warmup_factor: Optional[torch.FloatTensor] = None

class AttentionLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(AttentionLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, model_attention_logits: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        device = model_attention_logits.device
        
        # Initialize loss
        loss = torch.tensor(0.0, device=device)  # Make sure the tensor is on the correct device
        epsilon = 1e-8  # To avoid log(0)
        for idx in range(model_attention_logits.shape[0]):
            # Extract the attention map values based on the mask
            attention_map_target = model_attention_logits[idx][gt_mask[idx] == 1]
            attention_map_else = model_attention_logits[idx][gt_mask[idx] == 0]
            if attention_map_target.numel() == 0:
                continue
            mean = torch.mean(attention_map_else) if attention_map_else.numel() > 0 else torch.tensor(0.0, device=device)
            mse = torch.mean((attention_map_target - mean) ** 2)
            loss += -torch.log(mse + epsilon)
        if self.reduction == 'batchmean':
            loss = loss / model_attention_logits.shape[0]
        elif self.reduction == 'sum':
            pass  # Use the raw sum of losses
        elif self.reduction == 'mean':
            loss = loss / model_attention_logits.numel()  # Overall mean loss
        return loss

class SegEarthR2Model(MiphaPhiModel):

    def __init__(self, config: PhiConfig, mask_decoder_cfg=None):
        super(SegEarthR2Model, self).__init__(config)
        self.cfg = mask_decoder_cfg
        self.projector_outdim = config.hidden_size

        if hasattr(config, "mm_vision_tower"):
            use_mstva = getattr(config, "use_mstva", False)
            if getattr(config, "use_text_film", False) and use_mstva:
                use_mstva = False
            if getattr(config, "use_query_aware_decoder_bias", False) or getattr(config, "use_decoder_attn_bias", False):
                if use_mstva or getattr(config, "use_text_film", False):
                    print(
                        "[SegEarthR2Model] use_decoder_attn_bias=True: forcing use_mstva=False and use_text_film=False "
                        "for Swin (mutually exclusive)."
                    )
                use_mstva = False
                config.use_mstva = False
                config.use_text_film = False
            mstva_align_dim = getattr(config, "mstva_align_dim", 256)
            mstva_scale_weights = getattr(config, "mstva_scale_weights", "0.5,0.3,0.2")
            if isinstance(mstva_scale_weights, str):
                mstva_scale_weights = tuple(float(x.strip()) for x in mstva_scale_weights.split(",") if x.strip() != "")
            if len(mstva_scale_weights) != 3:
                mstva_scale_weights = (0.5, 0.3, 0.2)
            swin_type = getattr(config,'swin_type','base')
            if swin_type == 'base':
                self.vision_tower_mask = build_swin_b(
                    None,
                    text_cond_dim=config.hidden_size,
                    use_mstva=use_mstva,
                    mstva_align_dim=mstva_align_dim,
                    mstva_scale_weights=mstva_scale_weights,
                )
            else:
                self.vision_tower_mask = build_swin_l(
                    None,
                    text_cond_dim=config.hidden_size,
                    use_mstva=use_mstva,
                    mstva_align_dim=mstva_align_dim,
                    mstva_scale_weights=mstva_scale_weights,
                )

            self.vision_tower_mask.image_processor = IVSDatasetMapper(self.cfg)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_vision_tower_mask(self):
        vision_tower = getattr(self, 'vision_tower_mask', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower if hasattr(model_args, 'vision_tower') else model_args.mm_vision_tower
        vision_tower_mask = model_args.vision_tower_mask if hasattr(model_args, 'vision_tower_mask') else model_args.mm_vision_tower_mask

        self.config.mm_vision_tower = vision_tower
        self.config.use_text_film = getattr(model_args, "use_text_film", False)
        self.config.text_film_init_std = float(getattr(model_args, "text_film_init_std", 1e-3))
        self.config.text_film_branch_alpha = float(getattr(model_args, "text_film_branch_alpha", 1.0))
        self.config.text_film_visual_dim = int(getattr(model_args, "text_film_visual_dim", 512))
        use_qdti = getattr(model_args, "use_query_aware_decoder_bias", False)
        use_dac = getattr(model_args, "use_decoder_attn_bias", False) and not use_qdti
        self.config.use_query_aware_decoder_bias = bool(use_qdti)
        self.config.use_decoder_attn_bias = bool(use_dac)
        use_mstva_arg = getattr(model_args, "use_mstva", False)
        if self.config.use_text_film and use_mstva_arg:
            print("[initialize_vision_modules] use_text_film=True: forcing use_mstva=False for Swin.")
            use_mstva_arg = False
        if use_qdti or use_dac:
            if use_mstva_arg or getattr(model_args, "use_text_film", False):
                print(
                    "[initialize_vision_modules] query-aware decoder bias enabled: forcing use_mstva=False and "
                    "use_text_film=False for Swin."
                )
            use_mstva_arg = False
            self.config.use_text_film = False
        self.config.use_mstva = use_mstva_arg
        self.config.mstva_align_dim = getattr(model_args, "mstva_align_dim", 256)
        self.config.mstva_scale_weights = getattr(model_args, "mstva_scale_weights", "0.5,0.3,0.2")
        mstva_scale_weights = self.config.mstva_scale_weights
        if isinstance(mstva_scale_weights, str):
            mstva_scale_weights = tuple(float(x.strip()) for x in mstva_scale_weights.split(",") if x.strip() != "")
        if len(mstva_scale_weights) != 3:
            mstva_scale_weights = (0.5, 0.3, 0.2)
        swin_type = getattr(model_args,'swin_type','base')
        self.config.swin_type = swin_type
        if swin_type == 'base':
            vision_tower_mask = build_swin_b(
                vision_tower_mask,
                text_cond_dim=self.config.hidden_size,
                use_mstva=use_mstva_arg,
                mstva_align_dim=self.config.mstva_align_dim,
                mstva_scale_weights=mstva_scale_weights,
            )
        else:
            print('current visual encoder is swin large')
            vision_tower_mask = build_swin_l(
                vision_tower_mask,
                text_cond_dim=self.config.hidden_size,
                use_mstva=use_mstva_arg,
                mstva_align_dim=self.config.mstva_align_dim,
                mstva_scale_weights=mstva_scale_weights,
            )

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower_mask = [vision_tower_mask]
        else:
            self.vision_tower_mask = vision_tower_mask

        self.config.use_mm_proj = True
        vision_tower_mask.hidden_size = 256
        vision_tower_mask.image_processor = IVSDatasetMapper(self.cfg)

class SegEarthR2(MiphaPhiForCausalLM):
    def __init__(self, config, model_args=None, mask_decoder_cfg=None, add_cross_attn=True, cross_attn_index=None):
        if getattr(config, "use_text_film", False) and getattr(config, "use_mstva", False):
            print(
                "[SegEarthR2] use_text_film=True and use_mstva=True: forcing config.use_mstva=False "
                "(mutually exclusive; MSTVA module not built in Swin when use_text_film is on)."
            )
            config.use_mstva = False
        if getattr(config, "use_query_aware_decoder_bias", False) or getattr(config, "use_decoder_attn_bias", False):
            if getattr(config, "use_mstva", False) or getattr(config, "use_text_film", False):
                print(
                    "[SegEarthR2] query-aware decoder bias enabled: forcing config.use_mstva=False and "
                    "config.use_text_film=False (mutually exclusive)."
                )
            config.use_mstva = False
            config.use_text_film = False
        super(SegEarthR2, self).__init__(config)

        self.model = SegEarthR2Model(config, mask_decoder_cfg)
        self.init_config = config
        self.mask_decoder_cfg = mask_decoder_cfg
        self.cross_attn_index = cross_attn_index

        self.text_film_branch = None
        self._warn_text_film_no_res4_key = False
        self._warn_text_film_no_text_cond = False
        self._warn_text_film_ch_mismatch = False
        if getattr(self.config, "use_text_film", False):
            self._init_text_film_branch_from_config()

        self.lm_head = nn.Linear(config.hidden_size, 51200, bias=False)

        is_train_mask_decode = getattr(config, 'mask_decode_train', False)
        self.is_train_mask_decode = is_train_mask_decode

        if is_train_mask_decode:
            print('Mask Decoder has been trained, init directly')
            self.initial_mask_module()
        self.post_init()

    def _init_text_film_branch_from_config(self):
        if getattr(self, "text_film_branch", None) is not None:
            return
        td = int(self.config.hidden_size)
        vd = int(getattr(self.config, "text_film_visual_dim", 512))
        init_std = float(getattr(self.config, "text_film_init_std", 1e-3))
        ba = float(getattr(self.config, "text_film_branch_alpha", 1.0))
        self.text_film_branch = TextFiLMBranch(
            text_dim=td, visual_channels=vd, init_std=init_std, branch_alpha_init=ba
        )

    def ensure_text_film_branch(self):
        """If config enables TextFiLM but branch was not built at __init__ (e.g. config set after from_pretrained), create it."""
        if not getattr(self.config, "use_text_film", False):
            return
        self._init_text_film_branch_from_config()

    def _use_qdti_core(self) -> bool:
        return bool(getattr(self.config, "use_query_aware_decoder_bias", False))

    @staticmethod
    def checkpoint_contains_qdti_weights(checkpoint_path: str) -> bool:
        """Return True if checkpoint files contain predictor.qdti_core.* tensors."""
        if not checkpoint_path or not os.path.isdir(checkpoint_path):
            return False
        index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            weight_map = idx.get("weight_map", idx)
            return any("qdti_core" in str(k) for k in weight_map.keys())
        single = os.path.join(checkpoint_path, "model.safetensors")
        if os.path.isfile(single):
            try:
                from safetensors import safe_open
                with safe_open(single, framework="pt") as f:
                    return any("qdti_core" in k for k in f.keys())
            except Exception:
                pass
        pytorch_bin = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.isfile(pytorch_bin):
            try:
                sd = torch.load(pytorch_bin, map_location="cpu")
                return any("qdti_core" in k for k in sd.keys())
            except Exception:
                pass
        return False

    def validate_qdti_core_weights(
        self,
        *,
        checkpoint_path: Optional[str] = None,
        allow_random_init: bool = False,
        context: str = "load",
        fresh_training_init: bool = False,
    ) -> None:
        """
        Fail-fast when QDTI is enabled but checkpoint lacks qdti_core weights.
        fresh_training_init=True skips file check (predictor_init already built qdti_core).
        """
        if not self._use_qdti_core():
            return
        if fresh_training_init:
            if not hasattr(self, "predictor") or getattr(self.predictor, "qdti_core", None) is None:
                raise RuntimeError(
                    f"[QDTI][{context}] fresh_training_init=True but predictor.qdti_core is missing."
                )
            return
        if checkpoint_path and os.path.isdir(str(checkpoint_path)):
            if not self.checkpoint_contains_qdti_weights(str(checkpoint_path)):
                if not allow_random_init:
                    raise RuntimeError(
                        f"[QDTI][{context}] use_query_aware_decoder_bias=True but checkpoint "
                        f"'{checkpoint_path}' has no predictor.qdti_core.* weights. "
                        "Set allow_random_qdti_init=True to override (not recommended for eval)."
                    )
                print(
                    f"[WARNING][QDTI][{context}] checkpoint missing qdti_core weights; "
                    "allow_random_qdti_init=True — random QDTI init will be used.",
                    flush=True,
                )
        has_module = hasattr(self, "predictor") and getattr(self.predictor, "qdti_core", None) is not None
        if not has_module:
            if not allow_random_init:
                raise RuntimeError(
                    f"[QDTI][{context}] use_query_aware_decoder_bias=True but predictor.qdti_core "
                    "is not present. Load a QDTI checkpoint or set allow_random_qdti_init=True."
                )
            return
        in_model = any("qdti_core" in n for n, _ in self.named_parameters())
        if not in_model and not allow_random_init:
            raise RuntimeError(
                f"[QDTI][{context}] qdti_core submodule exists but has no registered parameters."
            )

    def ensure_qdti_core_branch(self, allow_init: bool = False):
        """Attach QDTI-Core when enabled. Random init only if allow_init or allow_random_qdti_init."""
        if not self._use_qdti_core():
            return
        if not hasattr(self, "predictor") or self.predictor is None:
            return
        spec = str(getattr(self.config, "decoder_attn_bias_apply_layers", "last3"))
        if getattr(self.predictor, "decoder_attn_bias_apply_layers", None) is None:
            self.predictor.decoder_attn_bias_apply_layers = spec
        self.predictor.use_qdti_mask_feedback = bool(getattr(self.config, "use_qdti_mask_feedback", False))
        if getattr(self.predictor, "qdti_core", None) is not None:
            return
        allow_random = bool(getattr(self.config, "allow_random_qdti_init", False))
        if not allow_init and not allow_random:
            raise RuntimeError(
                "[QDTI] predictor.qdti_core missing and random init not allowed. "
                "Load a checkpoint with qdti_core weights or set allow_random_qdti_init=True."
            )
        dec_layers = int(self.predictor.num_layers)
        hidden_dim = int(self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
        self.predictor.register_module(
            "qdti_core",
            QDTICore(
                text_dim=int(self.config.hidden_size),
                memory_dim=hidden_dim,
                query_dim=hidden_dim,
                bias_dim=int(getattr(self.config, "decoder_attn_bias_dim", 128)),
                init_std=float(getattr(self.config, "decoder_attn_bias_init_std", 1e-3)),
                max_abs=float(getattr(self.config, "decoder_attn_bias_max_abs", 0.02)),
                num_decoder_layers=dec_layers,
                alpha_init=float(getattr(self.config, "qdti_gate_init", 0.0)),
            ),
        )
        print("[WARNING][QDTI] Initialized new random predictor.qdti_core (allow_init/allow_random).", flush=True)

    def _sync_qdti_runtime_to_predictor(self, global_step: Optional[int] = None) -> None:
        if not self._use_qdti_core() or not hasattr(self, "predictor") or self.predictor is None:
            return
        self.predictor._qdti_global_step = global_step
        self.predictor._qdti_warmup_steps = int(getattr(self.config, "qdti_warmup_steps", 0))

    def ensure_decoder_attn_bias_branch(self):
        """If config enables decoder attn bias but predictor has no submodule (e.g. config toggled after load), attach it."""
        if self._use_qdti_core():
            self.ensure_qdti_core_branch(allow_init=False)
            return
        if not getattr(self.config, "use_decoder_attn_bias", False):
            return
        if not hasattr(self, "predictor") or self.predictor is None:
            return
        spec = str(getattr(self.config, "decoder_attn_bias_apply_layers", "last3"))
        if getattr(self.predictor, "decoder_attn_bias_apply_layers", None) is None:
            self.predictor.decoder_attn_bias_apply_layers = spec
        if getattr(self.predictor, "decoder_token_attn_bias", None) is not None:
            return
        hidden_dim = int(self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
        self.predictor.register_module(
            "decoder_token_attn_bias",
            DecoderTokenAttnBias(
                text_dim=int(self.config.hidden_size),
                memory_dim=hidden_dim,
                bias_dim=int(getattr(self.config, "decoder_attn_bias_dim", 128)),
                init_std=float(getattr(self.config, "decoder_attn_bias_init_std", 1e-3)),
                max_abs=float(getattr(self.config, "decoder_attn_bias_max_abs", 0.01)),
            ),
        )

    def initial_mask_module(self, pretrained_path=None, model_args=None):
        if not self.is_train_mask_decode:
            print('Initialize mask modules...')
            self.config.mask_decode_train = True

        self.attention_loss = AttentionLoss()
        
        self.test_topk_per_image = self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        input_shape = self.output_shape()
        self.pixel_decoder = self.pixel_decoder_init(cfg=self.mask_decoder_cfg, input_shape=input_shape)
        self.predictor = self.predictor_init(cfg=self.mask_decoder_cfg)

        self.SEG_token_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
            
        self.mask_decoder_training_init(self.mask_decoder_cfg)
        if pretrained_path is not None:
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            def change_w(weights, old_name, new_name):
                weights[new_name] = weights[old_name]
                weights.pop(old_name)

            if pretrained_path.endswith('.pkl'):
                with open(pretrained_path, 'rb') as f:
                    ckpt = pickle.load(f)
            else:
                ckpt = torch.load(pretrained_path)
            pixel_decoder_weights = get_w(ckpt['model'],'sem_seg_head.pixel_decoder')
            predictor_weights = get_w(ckpt['model'],'sem_seg_head.predictor')
            pixel_decoder_weights = {k: torch.tensor(v) for k, v in pixel_decoder_weights.items()}
            predictor_weights = {k: torch.tensor(v) for k, v in predictor_weights.items()}

            #deal some diff keys
            change_w(pixel_decoder_weights,'adapter_1.weight','adapter_1.0.weight')
            change_w(pixel_decoder_weights,'adapter_1.norm.weight','adapter_1.1.weight')
            change_w(pixel_decoder_weights,'adapter_1.norm.bias','adapter_1.1.bias')
            change_w(pixel_decoder_weights,'layer_1.weight','layer_1.0.weight')
            change_w(pixel_decoder_weights,'layer_1.norm.weight','layer_1.1.weight')
            change_w(pixel_decoder_weights,'layer_1.norm.bias','layer_1.1.bias')
            if 'static_query.weight' in predictor_weights:
                change_w(predictor_weights,'static_query.weight','query_feat.weight')
            if predictor_weights['query_embed.weight'].shape[0] == 200:
                predictor_weights['query_embed.weight'] = predictor_weights['query_embed.weight'][:100,:]
            diff_pixel_msg = self.pixel_decoder.load_state_dict(pixel_decoder_weights,strict=False)
            diff_predictor_msg = self.predictor.load_state_dict(predictor_weights,strict=False)
            print(diff_predictor_msg)
            print(diff_pixel_msg)

    def _apply_text_film_res4(self, features_dict, text_cond):
        if not getattr(self.config, "use_text_film", False):
            return
        br = getattr(self, "text_film_branch", None)
        if br is None:
            return
        if "res4" not in features_dict:
            if not self._warn_text_film_no_res4_key:
                self._warn_text_film_no_res4_key = True
                print("[WARNING][TextFiLM] features_dict has no 'res4' key; bypass TextFiLM.")
            return
        if text_cond is None:
            if not self._warn_text_film_no_text_cond:
                self._warn_text_film_no_text_cond = True
                print("[WARNING][TextFiLM] text_cond is None; bypass TextFiLM on res4.")
            return
        x = features_dict["res4"]
        if x.shape[1] != br.visual_channels:
            if not self._warn_text_film_ch_mismatch:
                self._warn_text_film_ch_mismatch = True
                print(
                    f"[WARNING][TextFiLM] res4 channels {x.shape[1]} != text_film_visual_dim {br.visual_channels}; "
                    "bypass TextFiLM."
                )
            return
        eval_mode = str(getattr(self.config, "text_film_eval_mode", "normal"))
        force_alpha = float(getattr(self.config, "text_film_force_alpha", 1.0))
        features_dict["res4"] = br(x, text_cond, eval_mode=eval_mode, force_alpha=force_alpha)

    def get_vision_tower_feature(
        self,
        images,
        text_cond=None,
        text_tokens=None,
        text_mask=None,
        return_midstage_gate=False,
        return_mstva_maps=False,
    ):
        if return_midstage_gate or return_mstva_maps:
            features, gate_info = self.get_model().get_vision_tower_mask()(
                images,
                text_cond=text_cond,
                text_tokens=text_tokens,
                text_mask=text_mask,
                return_midstage_gate=return_midstage_gate,
                return_mstva_maps=return_mstva_maps,
            )
        else:
            features = self.get_model().get_vision_tower_mask()(
                images,
                text_cond=text_cond,
                text_tokens=text_tokens,
                text_mask=text_mask,
            )
            gate_info = None

        features_dict = {
            'res2': features[0], # bs, 128, 256, 256
            'res3': features[1], # bs, 256, 128, 128
            'res4': features[2], # bs, 512, 64, 64  (Swin-B default; verify with x.shape[1])
            'res5': features[3], # bs, 1024, 32, 32
        }
        self._apply_text_film_res4(features_dict, text_cond)
        if return_midstage_gate or return_mstva_maps:
            return features_dict, gate_info
        return features_dict
    def mask_decoder_training_init(self, cfg):
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        # boundary_weight = cfg.MODEL.MASK_FORMER.BOUNDARY_WEIGHT
        
        matcher = hungarian_matcher_InstructSeg(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        
        weight_dict = {"loss_SEG_class": class_weight,  "loss_mask": mask_weight,
                       "loss_dice": dice_weight, }

        self.weight_dict = weight_dict
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["SEG_labels", "masks",]
        self.criterion = Criterion(
            matcher=matcher,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            device=self.device
        )
        self.size_divisibility = 32
        self.sem_seg_postprocess_before_inference = True
    
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images) # image_features: [4, 729, 1152]
        image_features = self.get_model().mm_projector(image_features) # image_features: [4, 729, 2560]
        
        return image_features
    
    def get_text_image_tokens(self, images):
        image_features = self.get_model().get_vision_tower()(images) 
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def _repeat_text_tokens_for_mask_num(self, text_tokens, text_mask, mask_num_tensor):
        if text_tokens is None or text_mask is None:
            return None, None
        mn = torch.as_tensor(mask_num_tensor, device=text_tokens.device, dtype=torch.long).flatten()
        if mn.numel() == 0:
            return text_tokens, text_mask
        return torch.repeat_interleave(text_tokens, mn, dim=0), torch.repeat_interleave(text_mask, mn, dim=0)

    def predictor_init(self, cfg):
        in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        nheads = cfg.MODEL.MASK_FORMER.NHEADS
        dim_feedforward = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        pre_norm = cfg.MODEL.MASK_FORMER.PRE_NORM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        enforce_input_project = False
        seg_norm = cfg.MODEL.MASK_FORMER.SEG_NORM
        seg_proj = cfg.MODEL.MASK_FORMER.SEG_PROJ
        seg_fuse_score = cfg.MODEL.MASK_FORMER.FUSE_SCORE

        use_qdti = getattr(self.config, "use_query_aware_decoder_bias", False)
        use_dac = getattr(self.config, "use_decoder_attn_bias", False) and not use_qdti
        dac_apply = str(getattr(self.config, "decoder_attn_bias_apply_layers", "last3")) if (use_dac or use_qdti) else None

        predictor = MultiScaleMaskedTransformerDecoderForOPTPreTrain(
            in_channels,
            hidden_dim,
            num_queries,
            nheads,
            dim_feedforward,
            dec_layers,
            pre_norm,
            mask_dim,
            enforce_input_project,
            seg_norm,
            seg_proj,
            seg_fuse_score,
            decoder_attn_bias_apply_layers=dac_apply,
        )
        if use_qdti:
            predictor.use_qdti_mask_feedback = bool(getattr(self.config, "use_qdti_mask_feedback", False))
            predictor.register_module(
                "qdti_core",
                QDTICore(
                    text_dim=int(self.config.hidden_size),
                    memory_dim=int(hidden_dim),
                    query_dim=int(hidden_dim),
                    bias_dim=int(getattr(self.config, "decoder_attn_bias_dim", 128)),
                    init_std=float(getattr(self.config, "decoder_attn_bias_init_std", 1e-3)),
                    max_abs=float(getattr(self.config, "decoder_attn_bias_max_abs", 0.02)),
                    num_decoder_layers=int(dec_layers),
                    alpha_init=float(getattr(self.config, "qdti_gate_init", 0.0)),
                ),
            )
        elif use_dac:
            predictor.register_module(
                "decoder_token_attn_bias",
                DecoderTokenAttnBias(
                    text_dim=int(self.config.hidden_size),
                    memory_dim=int(hidden_dim),
                    bias_dim=int(getattr(self.config, "decoder_attn_bias_dim", 128)),
                    init_std=float(getattr(self.config, "decoder_attn_bias_init_std", 1e-3)),
                    max_abs=float(getattr(self.config, "decoder_attn_bias_max_abs", 0.01)),
                ),
            )
        return predictor


    def get_model(self):
        return self.model
    def output_shape(self):
        out_features = self.mask_decoder_cfg.MODEL.SWIN.OUT_FEATURES
        out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        num_features = [int(self.mask_decoder_cfg.MODEL.SWIN.EMBED_DIM * 2 ** i) for i in
                        range(len(self.mask_decoder_cfg.MODEL.SWIN.DEPTHS))]
        out_feature_channels = {
            "res2": num_features[0],
            "res3": num_features[1],
            "res4": num_features[2],
            "res5": num_features[3],
        }
        backbone_feature_shape = dict()
        for name in out_features:
            backbone_feature_shape[name] = Dict(
                {'channel': out_feature_channels[name], 'stride': out_feature_strides[name]})
        return backbone_feature_shape

    def get_encoder_image(self, images):
        encode_image_features = self.get_model().get_vision_tower()(images)
        return encode_image_features

    def pixel_decoder_init(self, cfg, input_shape):
        common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        transformer_dropout = cfg.MODEL.MASK_FORMER.DROPOUT
        transformer_nheads = cfg.MODEL.MASK_FORMER.NHEADS
        transformer_dim_feedforward = 1024
        transformer_enc_layers = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS
        conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        transformer_in_features = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES  # ["res3", "res4", "res5"]

        pixel_decoder = MSDeformAttnPixelDecoder(input_shape,
                                                 transformer_dropout,
                                                 transformer_nheads,
                                                 transformer_dim_feedforward,
                                                 transformer_enc_layers,
                                                 conv_dim,
                                                 mask_dim,
                                                 transformer_in_features,
                                                 common_stride)
        return pixel_decoder
    
    def prepare_targets(self, targets, images):
        
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        has_gt_ids = False
        if hasattr(targets[0], 'gt_ids'):
            has_gt_ids = True
        for targets_per_image in targets:
            if has_gt_ids:
                inst_ids = targets_per_image.gt_ids
                valid_id = inst_ids!=-1
            else:
                inst_ids = None
                valid_id = None
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "valid": valid_id,
                    "inst_id": inst_ids,
                }
            )
        return new_targets

    def get_special_token(self, SEG, EOS):
        self.SEG_id = SEG
        self.EOS_id = EOS

    def embed_refer_ids(self, refer_ids):
        if refer_ids is None:
            return None
        embedded_refer = self.get_model().embed_tokens(refer_ids)
        return embedded_refer

    def _resolve_pad_token_id(self):
        pad_id = getattr(self.config, "pad_token_id", None)
        if pad_id is not None:
            return pad_id
        model_cfg = getattr(self.get_model(), "config", None)
        if model_cfg is not None:
            pad_id = getattr(model_cfg, "pad_token_id", None)
            if pad_id is not None:
                return pad_id
        generation_cfg = getattr(self, "generation_config", None)
        if generation_cfg is not None:
            pad_id = getattr(generation_cfg, "pad_token_id", None)
            if pad_id is not None:
                return pad_id
        embed_tokens = self.get_model().embed_tokens
        pad_id = getattr(embed_tokens, "padding_idx", None)
        return pad_id

    def build_text_condition(self, token_refer_id, batch_size=None, device=None):
        if token_refer_id is None:
            return None

        embed_tokens = self.get_model().embed_tokens
        text_dim = self.config.hidden_size
        dtype = embed_tokens.weight.dtype
        if device is None:
            device = embed_tokens.weight.device
        pad_id = self._resolve_pad_token_id()
        zero_vec = torch.zeros(text_dim, device=device, dtype=dtype)

        if torch.is_tensor(token_refer_id):
            if token_refer_id.dim() == 0:
                refer_items = [token_refer_id.view(1)]
            elif token_refer_id.dim() == 1:
                refer_items = [token_refer_id]
            elif token_refer_id.dim() == 2:
                refer_items = [token_refer_id[i] for i in range(token_refer_id.shape[0])]
            else:
                raise ValueError(f"Unsupported token_refer_id tensor dim: {token_refer_id.dim()}")
        elif isinstance(token_refer_id, (list, tuple)):
            refer_items = list(token_refer_id)
        else:
            refer_items = [None]

        if batch_size is None:
            batch_size = len(refer_items)

        if len(refer_items) < batch_size:
            refer_items.extend([None] * (batch_size - len(refer_items)))
        elif len(refer_items) > batch_size:
            refer_items = refer_items[:batch_size]

        text_conditions = []
        for refer_ids in refer_items:
            if refer_ids is None:
                text_conditions.append(zero_vec.clone())
                continue

            if isinstance(refer_ids, (list, tuple)):
                valid_tensors = [x.view(-1) for x in refer_ids if torch.is_tensor(x) and x.numel() > 0]
                if not valid_tensors:
                    text_conditions.append(zero_vec.clone())
                    continue
                refer_ids = torch.cat(valid_tensors, dim=0)

            if not torch.is_tensor(refer_ids):
                text_conditions.append(zero_vec.clone())
                continue

            refer_ids = refer_ids.to(device=device, dtype=torch.long).view(-1)
            if refer_ids.numel() == 0:
                text_conditions.append(zero_vec.clone())
                continue

            if pad_id is not None:
                refer_ids = refer_ids[refer_ids.ne(pad_id)]
            # Fallback when pad_id is unavailable: keep all tokens to avoid dropping real refer tokens.

            if refer_ids.numel() == 0:
                text_conditions.append(zero_vec.clone())
                continue

            refer_embed = self.embed_refer_ids(refer_ids)
            if refer_embed is None or refer_embed.numel() == 0:
                text_conditions.append(zero_vec.clone())
                continue

            if refer_embed.dim() == 1:
                refer_embed = refer_embed.unsqueeze(0)
            pooled = refer_embed.mean(dim=0)
            pooled = F.layer_norm(pooled.float(), (pooled.shape[-1],)).to(device=device, dtype=dtype)
            text_conditions.append(pooled)

        return torch.stack(text_conditions, dim=0)

    def build_text_tokens(self, token_refer_id, batch_size=None, device=None):
        if token_refer_id is None:
            return None, None
        embed_tokens = self.get_model().embed_tokens
        if device is None:
            device = embed_tokens.weight.device
        pad_id = self._resolve_pad_token_id()

        if torch.is_tensor(token_refer_id):
            if token_refer_id.dim() == 0:
                refer_items = [token_refer_id.view(1)]
            elif token_refer_id.dim() == 1:
                refer_items = [token_refer_id]
            elif token_refer_id.dim() == 2:
                token_refer_id = token_refer_id.to(device=device, dtype=torch.long)
                if batch_size is None:
                    batch_size = token_refer_id.shape[0]
                if token_refer_id.shape[0] != batch_size:
                    token_refer_id = token_refer_id[:batch_size]
                if pad_id is not None:
                    mask = token_refer_id.ne(pad_id)
                else:
                    mask = torch.ones_like(token_refer_id, dtype=torch.bool, device=token_refer_id.device)
                text_tokens = self.embed_refer_ids(token_refer_id)
                if text_tokens is None:
                    return None, None
                return text_tokens, mask
            else:
                raise ValueError(f"Unsupported token_refer_id tensor dim: {token_refer_id.dim()}")
        elif isinstance(token_refer_id, (list, tuple)):
            refer_items = list(token_refer_id)
        else:
            refer_items = [None]

        if batch_size is None:
            batch_size = len(refer_items)
        if len(refer_items) < batch_size:
            refer_items.extend([None] * (batch_size - len(refer_items)))
        elif len(refer_items) > batch_size:
            refer_items = refer_items[:batch_size]

        normalized = []
        max_len = 0
        for refer_ids in refer_items:
            if isinstance(refer_ids, (list, tuple)):
                valid_tensors = [x.view(-1) for x in refer_ids if torch.is_tensor(x) and x.numel() > 0]
                refer_ids = torch.cat(valid_tensors, dim=0) if valid_tensors else None
            if refer_ids is None or (torch.is_tensor(refer_ids) and refer_ids.numel() == 0):
                t = torch.empty(0, dtype=torch.long, device=device)
            elif torch.is_tensor(refer_ids):
                t = refer_ids.to(device=device, dtype=torch.long).view(-1)
            else:
                t = torch.empty(0, dtype=torch.long, device=device)
            max_len = max(max_len, int(t.numel()))
            normalized.append(t)

        if max_len == 0:
            return None, None

        if pad_id is None:
            pad_id = 0
        token_ids = torch.full((batch_size, max_len), fill_value=int(pad_id), dtype=torch.long, device=device)
        text_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        for i, t in enumerate(normalized):
            if t.numel() == 0:
                continue
            token_ids[i, : t.numel()] = t
            text_mask[i, : t.numel()] = True

        text_tokens = self.embed_refer_ids(token_ids)
        if text_tokens is None:
            return None, None
        return text_tokens, text_mask

    def _build_midstage_gate_targets(self, seg_info, gate_spatial):
        if seg_info is None or len(seg_info) == 0:
            return None
        target_masks = []
        if 'mask' in seg_info[0]:
            for item in seg_info:
                mask_item = item.get('mask', None)
                if mask_item is None or not torch.is_tensor(mask_item):
                    return None
                mask_item = mask_item.float().to(gate_spatial.device)
                if mask_item.ndim == 2:
                    mask_item = mask_item.unsqueeze(0)
                elif mask_item.ndim == 3 and mask_item.shape[0] != 1:
                    mask_item = mask_item[:1]
                target_masks.append(mask_item)
            gt = torch.stack(target_masks, dim=0)
        elif 'padding_mask' in seg_info[0]:
            for item in seg_info:
                instances = item.get("instances", None)
                if isinstance(instances, list):
                    instances = instances[0] if len(instances) > 0 else None
                if instances is None or not hasattr(instances, "gt_masks"):
                    return None
                gt_masks = instances.gt_masks
                if hasattr(gt_masks, "tensor"):
                    gt_masks = gt_masks.tensor
                if not torch.is_tensor(gt_masks) or gt_masks.numel() == 0:
                    return None
                gt_masks = gt_masks.float().to(gate_spatial.device)
                if gt_masks.ndim == 3:
                    gt_mask = gt_masks.any(dim=0, keepdim=True).float()
                elif gt_masks.ndim == 2:
                    gt_mask = gt_masks.unsqueeze(0).float()
                else:
                    return None
                target_masks.append(gt_mask)
            gt = torch.stack(target_masks, dim=0)
        else:
            return None

        if gt.ndim == 3:
            gt = gt.unsqueeze(1)
        gt_small = F.interpolate(gt, size=gate_spatial.shape[-2:], mode="nearest")
        return gt_small

    def _parse_mstva_scale_weights(self):
        weights = getattr(self.config, "mstva_scale_weights", "0.5,0.3,0.2")
        if isinstance(weights, str):
            try:
                parsed = [float(x.strip()) for x in weights.split(",") if x.strip() != ""]
            except ValueError:
                parsed = [0.5, 0.3, 0.2]
        elif isinstance(weights, (list, tuple)):
            parsed = [float(x) for x in weights]
        else:
            parsed = [0.5, 0.3, 0.2]
        if len(parsed) != 3:
            parsed = [0.5, 0.3, 0.2]
        return parsed

    def _build_binary_gt_mask(self, seg_info, ref_tensor):
        if seg_info is None or len(seg_info) == 0:
            return None
        target_masks = []
        if 'mask' in seg_info[0]:
            for item in seg_info:
                mask_item = item.get('mask', None)
                if mask_item is None or not torch.is_tensor(mask_item):
                    return None
                mask_item = mask_item.float().to(ref_tensor.device)
                if mask_item.ndim == 2:
                    mask_item = mask_item.unsqueeze(0)
                elif mask_item.ndim == 3 and mask_item.shape[0] != 1:
                    mask_item = mask_item[:1]
                target_masks.append(mask_item)
        elif 'padding_mask' in seg_info[0]:
            for item in seg_info:
                instances = item.get("instances", None)
                if isinstance(instances, list):
                    instances = instances[0] if len(instances) > 0 else None
                if instances is None or not hasattr(instances, "gt_masks"):
                    return None
                gt_masks = instances.gt_masks
                if hasattr(gt_masks, "tensor"):
                    gt_masks = gt_masks.tensor
                if not torch.is_tensor(gt_masks) or gt_masks.numel() == 0:
                    return None
                gt_masks = gt_masks.float().to(ref_tensor.device)
                if gt_masks.ndim == 3:
                    gt_mask = gt_masks.any(dim=0, keepdim=True).float()
                elif gt_masks.ndim == 2:
                    gt_mask = gt_masks.unsqueeze(0).float()
                else:
                    return None
                target_masks.append(gt_mask)
        else:
            return None

        gt = torch.stack(target_masks, dim=0)
        if gt.ndim == 3:
            gt = gt.unsqueeze(1)
        return gt

    def _decoder_attn_bias_ranking_loss(
        self,
        bias_maps,
        seg_info,
        margin: float,
        zero_base: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        逐层（如 last3）在 P_bias [B,H,W] 上：仅在 cross-attn hard mask 允许 attend 的 memory 格点上
        比较 GT 前景/背景内 P 的均值；relu(margin - inside + outside)；最后对层平均。

        inside/outside/gap 在 float32 下统计（日志用，已 detach）。

        返回：loss_mean, fg_access, bg_access, valid_layer_count, inside_mean, outside_mean, gap_mean
        """
        z0 = zero_base * 0.0
        if bias_maps is None or len(bias_maps) == 0 or seg_info is None:
            return z0, z0, z0, z0, z0, z0, z0
        ref = bias_maps[0]["P"]
        gt = self._build_binary_gt_mask(seg_info=seg_info, ref_tensor=ref)
        if gt is None:
            return (ref * 0.0).sum(), z0, z0, z0, z0, z0, z0
        B = ref.shape[0]
        if gt.shape[0] != B:
            return (ref * 0.0).sum(), z0, z0, z0, z0, z0, z0
        m = float(margin)
        layer_losses = []
        fg_ratio_layers = []
        bg_ratio_layers = []
        inside_means = []
        outside_means = []
        gap_means = []
        eps = 1e-6
        for entry in bias_maps:
            P = entry["P"]
            H, W = int(entry["H"]), int(entry["W"])
            if P.shape[0] != B or P.shape[1] != H or P.shape[2] != W:
                continue
            acc = entry.get("spatial_allowed", None)
            if acc is None or acc.shape != P.shape:
                if self.training:
                    raise ValueError(
                        "[DecoderAttnBiasRank] bias_maps entry missing spatial_allowed or shape mismatch "
                        f"layer_idx={entry.get('layer_idx', '?')}: acc="
                        f"{None if acc is None else tuple(acc.shape)} P={tuple(P.shape)}"
                    )
                acc = torch.ones(P.shape, device=P.device, dtype=torch.bool)
            else:
                acc = acc.to(device=P.device, dtype=torch.bool)
            gt_s = F.interpolate(gt.float(), size=(H, W), mode="nearest")
            fg = (gt_s > 0.5).squeeze(1)
            if fg.sum() < eps:
                continue
            P32 = P.float()
            fg32 = fg.float()
            acc32 = acc.float()
            fg_eff = fg32 * acc32
            bg_eff = (1.0 - fg32) * acc32
            denom_in = fg_eff.sum(dim=(1, 2)) + eps
            denom_out = bg_eff.sum(dim=(1, 2)) + eps
            inside = (P32 * fg_eff).sum(dim=(1, 2)) / denom_in
            outside = (P32 * bg_eff).sum(dim=(1, 2)) / denom_out
            gap = inside - outside
            per_sample = F.relu(torch.tensor(m, device=gap.device, dtype=gap.dtype) - gap)
            valid = (fg_eff.sum(dim=(1, 2)) > eps) & (bg_eff.sum(dim=(1, 2)) > eps)
            if not valid.any():
                continue
            layer_losses.append(per_sample[valid].mean())
            inside_means.append(inside[valid].mean().detach())
            outside_means.append(outside[valid].mean().detach())
            gap_means.append(gap[valid].mean().detach())
            with torch.no_grad():
                fg_ratio_layers.append((fg_eff.sum() / (fg32.sum() + eps)).detach())
                bg_ratio_layers.append((bg_eff.sum() / ((1.0 - fg32).sum() + eps)).detach())
        if not layer_losses:
            return (ref * 0.0).sum(), z0, z0, z0, z0, z0, z0
        loss_mean = sum(layer_losses) / float(len(layer_losses))
        fg_r = torch.stack(fg_ratio_layers).mean() if fg_ratio_layers else z0
        bg_r = torch.stack(bg_ratio_layers).mean() if bg_ratio_layers else z0
        vn = ref.new_tensor(float(len(layer_losses)))
        in_m = torch.stack(inside_means).mean() if inside_means else z0
        out_m = torch.stack(outside_means).mean() if outside_means else z0
        gap_m = torch.stack(gap_means).mean() if gap_means else z0
        return (
            loss_mean,
            fg_r.to(dtype=loss_mean.dtype, device=loss_mean.device),
            bg_r.to(dtype=loss_mean.dtype, device=loss_mean.device),
            vn,
            in_m.to(dtype=torch.float32, device=loss_mean.device),
            out_m.to(dtype=torch.float32, device=loss_mean.device),
            gap_m.to(dtype=torch.float32, device=loss_mean.device),
        )

    @staticmethod
    def _mask_iou(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pred_b = (pred.sigmoid() if pred.dtype.is_floating_point else pred.float()) > 0.5
        gt_b = gt > 0.5
        inter = (pred_b & gt_b).sum(dim=(-2, -1)).float()
        union = (pred_b | gt_b).sum(dim=(-2, -1)).float() + eps
        return inter / union

    def _resolve_positive_negative_queries(
        self,
        mask_outputs: dict,
        targets: list,
        iou_thresh: float,
    ) -> Tuple[List[Optional[int]], List[List[int]]]:
        """Positive: Hungarian match else max-IoU query. Negative: unmatched with IoU < thresh."""
        pred_masks = mask_outputs.get("pred_masks")
        if pred_masks is None or targets is None:
            return [], []
        B, Q = pred_masks.shape[0], pred_masks.shape[1]
        positives: List[Optional[int]] = []
        negatives: List[List[int]] = []
        with torch.no_grad():
            indices = self.criterion.matcher(mask_outputs, targets)
        for b in range(B):
            pos_q = None
            src_idx, _ = indices[b]
            if len(src_idx) > 0:
                pos_q = int(src_idx[0].item())
            tgt_masks = targets[b].get("masks")
            if tgt_masks is None or not torch.is_tensor(tgt_masks) or tgt_masks.numel() == 0:
                positives.append(None)
                negatives.append([])
                continue
            gt = tgt_masks.float()
            if gt.ndim == 3:
                gt = gt.max(dim=0, keepdim=True).values
            if pos_q is None:
                ious = []
                for q in range(Q):
                    pm = F.interpolate(
                        pred_masks[b : b + 1, q : q + 1].float(),
                        size=gt.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    ious.append(float(self._mask_iou(pm.squeeze(0), gt.squeeze(0)).item()))
                if ious:
                    pos_q = int(max(range(len(ious)), key=lambda i: ious[i]))
            neg_qs = []
            if pos_q is not None:
                for q in range(Q):
                    if q == pos_q:
                        continue
                    pm = F.interpolate(
                        pred_masks[b : b + 1, q : q + 1].float(),
                        size=gt.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    iou = float(self._mask_iou(pm.squeeze(0), gt.squeeze(0)).item())
                    if iou < float(iou_thresh):
                        neg_qs.append(q)
            positives.append(pos_q)
            negatives.append(neg_qs)
        return positives, negatives

    def _qdti_positive_query_rank_loss(
        self,
        bias_maps,
        seg_info,
        margin: float,
        positive_queries: List[Optional[int]],
        zero_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rank loss on positive query channels only; skip sample/layer when no valid positive."""
        z0 = zero_base * 0.0
        if bias_maps is None or len(bias_maps) == 0 or seg_info is None:
            return z0, z0, z0
        ref = bias_maps[0]["P"]
        gt = self._build_binary_gt_mask(seg_info=seg_info, ref_tensor=ref)
        if gt is None:
            return (ref * 0.0).sum(), z0, z0
        B = ref.shape[0]
        if gt.shape[0] != B:
            return (ref * 0.0).sum(), z0, z0
        m = float(margin)
        layer_losses = []
        valid_n = 0
        eps = 1e-6
        for entry in bias_maps:
            P = entry["P"]
            H, W = int(entry["H"]), int(entry["W"])
            per_query = bool(entry.get("per_query", P.dim() == 4))
            acc = entry.get("spatial_allowed", None)
            if acc is None or acc.shape != (B, H, W):
                if self.training:
                    raise ValueError("[QDTIRank] missing spatial_allowed for bias map entry.")
                acc = torch.ones((B, H, W), device=P.device, dtype=torch.bool)
            gt_s = F.interpolate(gt.float(), size=(H, W), mode="nearest")
            fg = (gt_s > 0.5).squeeze(1)
            for b in range(B):
                pos_q = positive_queries[b] if b < len(positive_queries) else None
                if pos_q is None:
                    continue
                if per_query:
                    if pos_q >= P.shape[1]:
                        continue
                    P_b = P[b, pos_q].float()
                else:
                    P_b = P[b].float()
                fg_b = fg[b].float()
                acc_b = acc[b].float()
                fg_eff = fg_b * acc_b
                bg_eff = (1.0 - fg_b) * acc_b
                if fg_eff.sum() < eps or bg_eff.sum() < eps:
                    continue
                inside = (P_b * fg_eff).sum() / (fg_eff.sum() + eps)
                outside = (P_b * bg_eff).sum() / (bg_eff.sum() + eps)
                layer_losses.append(F.relu(P_b.new_tensor(m) - inside + outside))
                valid_n += 1
        if not layer_losses:
            return (ref * 0.0).sum(), z0, ref.new_tensor(0.0)
        loss_mean = sum(layer_losses) / float(len(layer_losses))
        return loss_mean, ref.new_tensor(float(valid_n)), loss_mean.detach()

    def _qdti_negative_query_loss(
        self,
        bias_maps,
        seg_info,
        margin: float,
        positive_queries: List[Optional[int]],
        negative_queries: List[List[int]],
        zero_base: torch.Tensor,
    ) -> torch.Tensor:
        z0 = zero_base * 0.0
        if bias_maps is None or len(bias_maps) == 0:
            return z0
        ref = bias_maps[0]["P"]
        gt = self._build_binary_gt_mask(seg_info=seg_info, ref_tensor=ref)
        if gt is None:
            return z0
        B = ref.shape[0]
        losses = []
        eps = 1e-6
        m = float(margin)
        for entry in bias_maps:
            P = entry["P"]
            if not entry.get("per_query", P.dim() == 4):
                continue
            H, W = int(entry["H"]), int(entry["W"])
            acc = entry.get("spatial_allowed")
            gt_s = F.interpolate(gt.float(), size=(H, W), mode="nearest")
            fg = (gt_s > 0.5).squeeze(1)
            for b in range(B):
                pos_q = positive_queries[b] if b < len(positive_queries) else None
                neg_qs = negative_queries[b] if b < len(negative_queries) else []
                if pos_q is None or not neg_qs:
                    continue
                acc_b = acc[b].float() if acc is not None else torch.ones((H, W), device=P.device)
                fg_eff = fg[b].float() * acc_b
                if fg_eff.sum() < eps:
                    continue
                pos_inside = (P[b, pos_q].float() * fg_eff).sum() / (fg_eff.sum() + eps)
                for nq in neg_qs:
                    if nq >= P.shape[1]:
                        continue
                    neg_inside = (P[b, nq].float() * fg_eff).sum() / (fg_eff.sum() + eps)
                    losses.append(F.relu(P.new_tensor(m) + neg_inside - pos_inside))
        if not losses:
            return z0
        return sum(losses) / float(len(losses))

    def _qdti_diversity_loss(self, bias_maps, negative_queries: List[List[int]], zero_base: torch.Tensor) -> torch.Tensor:
        z0 = zero_base * 0.0
        if bias_maps is None or len(bias_maps) == 0:
            return z0
        ref = bias_maps[0]["P"]
        losses = []
        for entry in bias_maps:
            P = entry["P"]
            if not entry.get("per_query", P.dim() == 4) or P.shape[1] < 2:
                continue
            B = P.shape[0]
            for b in range(B):
                neg_qs = negative_queries[b] if b < len(negative_queries) else []
                if len(neg_qs) < 2:
                    continue
                neg_idx = [q for q in neg_qs if q < P.shape[1]]
                if len(neg_idx) < 2:
                    continue
                vecs = P[b, neg_idx].float().flatten(1)
                vecs = F.normalize(vecs, dim=-1, eps=1e-6)
                sim = torch.matmul(vecs, vecs.transpose(0, 1))
                n_neg = sim.shape[0]
                eye = torch.eye(n_neg, device=sim.device, dtype=torch.bool)
                off_diag = sim.masked_select(~eye)
                if off_diag.numel() > 0:
                    losses.append(off_diag.abs().mean())
        if not losses:
            return z0
        return sum(losses) / float(len(losses))

    def concat_image_seg_cls_embeds(self, input_id, img_feature, label, SEG_token_embedding_indices=None, refer_embedding=None):
        image_token_indices = torch.where(input_id == IMAGE_TOKEN_INDEX)[0]
        assert len(image_token_indices) == 1, 'not supporting multi image index'
        
        image_features_indices = []
        cur_new_input_embeds = []
        if label is not None:
            cur_new_label = []
            assert label.shape == input_id.shape
        else:
            cur_new_label = None
        
        cur_SEG_token_embedding_indices = [] if SEG_token_embedding_indices is not None else None
        
        chunks = []
        current_chunk = []

        for id in input_id:
            if id >= 0:
                current_chunk.append(id.item())
            else:
                if current_chunk:
                    chunks.append(torch.tensor(current_chunk, device=input_id.device))
                    current_chunk = []
                chunks.append([id])
        if current_chunk:
            chunks.append(torch.tensor(current_chunk, device=input_id.device))

       
        for chunk in chunks:
            chunk_len = len(chunk)
            if chunk_len == 1 and chunk[0] == IMAGE_TOKEN_INDEX:
                cur_new_input_embeds.append(img_feature)
                image_features_indices.append(torch.ones(img_feature.shape[0]))
                if SEG_token_embedding_indices is not None:
                    cur_SEG_token_embedding_indices.append(torch.full((img_feature.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(
                        torch.full((img_feature.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype)
                    )
                  
            elif chunk_len == 1 and chunk[0] == REFER_TOKEN_INDEX:
                refer_embed = refer_embedding
                if len(refer_embed.shape) == 1:
                    refer_embed = refer_embed.unsqueeze(0)
                cur_new_input_embeds.append(refer_embed)
                image_features_indices.append(torch.zeros(refer_embed.shape[0]))
                
                if SEG_token_embedding_indices is not None:
                    cur_SEG_token_embedding_indices.append(
                        torch.full((refer_embed.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if label is not None:
                    cur_new_label.append(
                        torch.full((refer_embed.shape[0],), IGNORE_INDEX, device=label.device,
                                   dtype=label.dtype)
                    )
            
            else:
                cur_new_input_embeds.append(self.get_model().embed_tokens(input_id[:chunk_len]))
                image_features_indices.append(torch.zeros(chunk_len))
                
                if SEG_token_embedding_indices is not None:
                    cur_SEG_token_embedding_indices.append(SEG_token_embedding_indices[:chunk_len])
                if label is not None:
                    cur_new_label.append(label[:chunk_len])

            input_id = input_id[chunk_len:]
            
            if SEG_token_embedding_indices is not None:
                SEG_token_embedding_indices = SEG_token_embedding_indices[chunk_len:]
            if label is not None:
                label = label[chunk_len:]

        cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
        if label is not None:
            cur_new_label = [x.to(device=self.device) for x in cur_new_label]
            cur_new_label = torch.cat(cur_new_label, dim=0)
        
        if SEG_token_embedding_indices is not None:
            cur_SEG_token_embedding_indices = [x.to(device=self.device) for x in cur_SEG_token_embedding_indices]
            cur_SEG_token_embedding_indices = torch.cat(cur_SEG_token_embedding_indices, dim=0)
        
        if image_features_indices:
            image_features_indices = [x.to(device=self.device) for x in image_features_indices]
            image_features_indices = torch.cat(image_features_indices, dim=0)

        return cur_new_input_embeds, cur_new_label, cur_SEG_token_embedding_indices, image_features_indices

    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images, token_refer_id=None, SEG_token_embedding_indices=None):

        vision_tower = self.get_vision_tower()
        
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, None, None

        image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_image_features_indices = []
        
        new_SEG_token_embedding_indices = [] if SEG_token_embedding_indices is not None else None
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_image_feature = image_features[batch_idx]
            
            cur_SEG_token_embedding_indices = SEG_token_embedding_indices[batch_idx] if SEG_token_embedding_indices is not None else None
            
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # ensure gradients back propagation, not changing cur_input_embeds
                cur_input_embeds = cur_input_embeds + (
                        0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                continue

            if labels is not None:
                cur_label = labels[batch_idx]
            else:
                cur_label = None

            if token_refer_id is not None:
                cur_token_refer_id = token_refer_id[batch_idx]
            else:
                cur_token_refer_id = None

            cur_refer_embedding = self.embed_refer_ids(cur_token_refer_id)

            cur_input_embeds, cur_label, cur_SEG_token_embedding_indices, cur_image_features_indices= self.concat_image_seg_cls_embeds(
                input_id=cur_input_ids,
                img_feature=cur_image_feature,
                label=cur_label,
                SEG_token_embedding_indices=cur_SEG_token_embedding_indices,
                refer_embedding=cur_refer_embedding
            )

            new_input_embeds.append(cur_input_embeds)
            if labels is not None:
                new_labels.append(cur_label)

            if SEG_token_embedding_indices is not None:
                new_SEG_token_embedding_indices.append(cur_SEG_token_embedding_indices)

            if new_image_features_indices is not None:
                new_image_features_indices.append(cur_image_features_indices)
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)),
                                          dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)
            
            if SEG_token_embedding_indices is not None:
                new_SEG_token_embedding_indices_align = []
                for new_SEG_token_embedding_indice in new_SEG_token_embedding_indices:
                    new_SEG_token_embedding_indice = torch.cat(
                        (new_SEG_token_embedding_indice,
                         torch.zeros((max_len - new_SEG_token_embedding_indice.shape[0]),dtype=new_SEG_token_embedding_indice.dtype, device=new_SEG_token_embedding_indice.device)),
                        dim=0)
                    new_SEG_token_embedding_indices_align.append(new_SEG_token_embedding_indice)
                new_SEG_token_embedding_indices = torch.stack(new_SEG_token_embedding_indices_align, dim=0)
            
            if new_image_features_indices is not None:
                new_image_features_indices_align = []
                for new_image_features_indice in new_image_features_indices:
                    new_image_features_indice = torch.cat(
                        (new_image_features_indice,
                         torch.zeros((max_len - new_image_features_indice.shape[0]),dtype=new_image_features_indice.dtype, device=new_image_features_indice.device)),
                        dim=0)
                    new_image_features_indices_align.append(new_image_features_indice)
                new_image_features_indices = torch.stack(new_image_features_indices_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
            
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if SEG_token_embedding_indices is not None:
                new_SEG_token_embedding_indices = torch.stack(new_SEG_token_embedding_indices, dim=0)

            if new_image_features_indices is not None:
                new_image_features_indices = torch.stack(new_image_features_indices, dim=0)
            
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
   
        return None, attention_mask, past_key_values, new_input_embeds, new_labels, new_SEG_token_embedding_indices, new_image_features_indices
    
    def get_SEG_embedding(self, hidden_states, SEG_embedding_indices):
        SEG_embedding_list = []
        for current_hidden_state, current_token_indice in zip(hidden_states, SEG_embedding_indices):
            current_refer_state = current_hidden_state[current_token_indice.bool()]
            SEG_embedding_list.append(current_refer_state)
        return torch.cat(SEG_embedding_list, dim=0).unsqueeze(1)
           
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            images_clip: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            seg_info=None,
            token_refer_id=None,
            SEG_token_embedding_indices=None,
            global_step=None,
            mask_num=None,
            dataset_type=None,) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if dataset_type is not None:
            assert all(item == dataset_type[0] for item in dataset_type), f'this batch contain different dataset_type: {dataset_type}'
            batch_dataset_type = dataset_type[0]
        else:
            batch_dataset_type = []
        output_attentions = True

        output_hidden_states = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (SEG_token_embedding_indices == 1).sum() != 0:

            # for generative mode only the 1th stage need
            if input_ids.shape[1] != 1:
                use_midstage_gate_loss = getattr(self.config, "use_midstage_gate_loss", False)
                use_mstva = getattr(self.config, "use_mstva", False)
                use_mstva_loss = getattr(self.config, "use_mstva_loss", False)
                use_qdti = self._use_qdti_core()
                use_decoder_attn_bias = getattr(self.config, "use_decoder_attn_bias", False)
                text_cond = self.build_text_condition(
                    token_refer_id=token_refer_id,
                    batch_size=input_ids.shape[0],
                    device=images.device,
                )
                text_tokens = None
                text_mask = None
                if use_mstva or use_mstva_loss or use_decoder_attn_bias or use_qdti:
                    text_tokens, text_mask = self.build_text_tokens(
                        token_refer_id=token_refer_id,
                        batch_size=input_ids.shape[0],
                        device=images.device,
                    )
                if use_midstage_gate_loss or use_mstva_loss:
                    image_features, extra_info = self.get_vision_tower_feature(
                        images,
                        text_cond=text_cond,
                        text_tokens=text_tokens,
                        text_mask=text_mask,
                        return_midstage_gate=use_midstage_gate_loss,
                        return_mstva_maps=use_mstva_loss,
                    )
                else:
                    image_features = self.get_vision_tower_feature(
                        images,
                        text_cond=text_cond,
                        text_tokens=text_tokens,
                        text_mask=text_mask,
                    )
                    extra_info = None
                bs = input_ids.shape[0]
            
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, SEG_token_embedding_indices, image_features_indices = self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images_clip,
                token_refer_id=token_refer_id, SEG_token_embedding_indices=SEG_token_embedding_indices)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        attentions = [attention_item.sum(dim=1) for attention_item in outputs.attentions]
        SEG_embedding = self.SEG_token_projector(self.get_SEG_embedding(hidden_states, SEG_token_embedding_indices))
        
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features)
        mask_num = torch.tensor(mask_num, device=mask_features.device)
        mask_features = torch.repeat_interleave(mask_features, repeats=mask_num, dim=0)
        multi_scale_features = [
            torch.repeat_interleave(feat, repeats=mask_num, dim=0)
            for feat in multi_scale_features
        ]

        dac_mode = str(getattr(self.config, "decoder_attn_bias_eval_mode", "normal"))
        dac_fscale = float(getattr(self.config, "decoder_attn_bias_force_scale", 1.0))
        self._sync_qdti_runtime_to_predictor(global_step=global_step)
        tt_rep, tm_rep = self._repeat_text_tokens_for_mask_num(text_tokens, text_mask, mask_num)
        if tt_rep is not None:
            if tt_rep.shape[0] != SEG_embedding.shape[0]:
                raise AssertionError(
                    f"[DecoderAttnBias] text batch {tt_rep.shape[0]} != SEG_embedding batch {SEG_embedding.shape[0]}"
                )
            if tt_rep.shape[0] != mask_features.shape[0]:
                raise AssertionError(
                    f"[DecoderAttnBias] text batch {tt_rep.shape[0]} != mask_features batch {mask_features.shape[0]}"
                )
        mask_outputs = self.predictor(
            multi_scale_features,
            mask_features,
            None,
            None,
            SEG_embedding,
            text_tokens=tt_rep,
            text_mask=tm_rep,
            dac_eval_mode=dac_mode,
            dac_force_scale=dac_fscale,
        )

        # 开始计算loss
        loss = None

        llm_loss = None
        if labels is not None:
            # if seg_query_mask is None or batch_dataset_type in seg_llm_loss_dataset:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            vocab_size = shift_logits.shape[-1]
            shift_logits = shift_logits.view(-1, vocab_size)  # self.config.vocab_size
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            llm_loss = loss_fct(shift_logits, shift_labels)
            
        mask_loss = None
        targets = None
        if seg_info is not None:
            if 'padding_mask' in seg_info[0]:
                if isinstance(seg_info[0]["instances"], list):
                    gt_instances = [x["instances"][0].to(self.device) for x in seg_info]
                else:
                    gt_instances = [x["instances"].to(self.device) for x in seg_info]

                targets = self.prepare_targets(gt_instances, images)
            elif 'mask' in seg_info[0]:
                targets = []
                for gt_mask in seg_info:
                    targets.append(
                        {
                            'labels': torch.tensor([0]).to(mask_outputs['pred_masks'].device),
                            'masks': gt_mask['mask'].to(mask_outputs['pred_masks'].device),
                            'valid': None,
                            'inst_id': None
                        }
                    )
            else:
                targets = None
            mask_losses = self.criterion(mask_outputs, targets)
            weight_dict = self.weight_dict

            loss_mask = 0.0
            loss_dice = 0.0
        
            for k in list(mask_losses.keys()):
                if k in weight_dict:
                    if mask_losses[k] is not None:
                        mask_losses[k] *= weight_dict[k]
                    
                    if '_mask' in k:
                        loss_mask += mask_losses[k]
                    
                    elif '_dice' in k:
                        loss_dice += mask_losses[k]
                else:
                    mask_losses.pop(k)
            mask_loss = loss_mask + loss_dice

        use_attention_loss = getattr(self.config, "use_attention_loss", True)
        use_midstage_gate_loss = getattr(self.config, "use_midstage_gate_loss", False)
        midstage_gate_loss_weight = getattr(self.config, "midstage_gate_loss_weight", 0.0)
        use_mstva_loss = getattr(self.config, "use_mstva_loss", False)
        mstva_loss_weight = float(getattr(self.config, "mstva_loss_weight", 0.0))
        zero_base = llm_loss if llm_loss is not None else mask_loss
        if zero_base is None:
            zero_base = logits.sum() * 0.0
        loss_midstage_gate = torch.zeros_like(zero_base)
        midstage_gate_alpha = torch.zeros_like(zero_base)
        loss_mstva_align = torch.zeros_like(zero_base)
        mstva_alpha3 = torch.zeros_like(zero_base)
        mstva_alpha4 = torch.zeros_like(zero_base)
        mstva_alpha5 = torch.zeros_like(zero_base)
        if use_attention_loss:
            masks = [_seg_info['mask'] for _seg_info in seg_info]
            masks_resized = [
                F.interpolate(m.unsqueeze(0).float(), size=(800, 800), mode="nearest").squeeze(0)
                for m in masks
            ]
            masks = torch.stack(masks_resized, dim=0) # [4, 1, 800, 800]
            masks_down = F.interpolate(masks, size=(27, 27), mode="bilinear", align_corners=False)
            masks_down = masks_down.view(masks_down.size(0), -1)
            masks_down[masks_down > 0] = 1
            
            loss_attention = torch.tensor(0.0, device=mask_loss.device)
            for full_attention_map in attentions:
                batch_attentions_list = []
                for batch_idx in range(bs):
                    attention_map = full_attention_map[batch_idx]
                    SEG_mask = SEG_token_embedding_indices[batch_idx].bool()
                    image_features_mask = image_features_indices[batch_idx].bool()
                    attention = attention_map[SEG_mask][:, image_features_mask] # [1, 729]
                    batch_attentions_list.append(attention)
                batch_attentions = torch.cat(batch_attentions_list, dim=0) # [4, 729]
                loss_attention += self.attention_loss(batch_attentions, masks_down)
        else:
            loss_attention = torch.zeros_like(zero_base)

        if use_midstage_gate_loss and seg_info is not None:
            current_gate_info = locals().get("extra_info", None)
            mid_stage_gate = None
            if isinstance(current_gate_info, dict):
                mid_stage_gate = current_gate_info.get("mid_stage_gate", None)
                alpha_item = current_gate_info.get("mid_stage_alpha", None)
                if alpha_item is not None:
                    midstage_gate_alpha = alpha_item.detach().float().view(-1)[0].to(zero_base.device)
            if mid_stage_gate is None:
                print("[WARNING] use_midstage_gate_loss=True but mid_stage_gate is None, set gate loss to 0.")
            else:
                gate_spatial = mid_stage_gate.mean(dim=1, keepdim=True)
                gt_small = self._build_midstage_gate_targets(seg_info=seg_info, gate_spatial=gate_spatial)
                if gt_small is None or gt_small.shape != gate_spatial.shape:
                    print("[WARNING] midstage gate supervision target unavailable or shape mismatch, set gate loss to 0.")
                elif gt_small.max().item() <= 0:
                    print("[WARNING] midstage gate supervision GT is all-zero, set gate loss to 0.")
                else:
                    loss_midstage_gate = F.binary_cross_entropy(
                        gate_spatial.float().clamp(1e-4, 1 - 1e-4),
                        gt_small.float(),
                    )

        if use_mstva_loss and seg_info is not None:
            mstva_info = None
            current_extra_info = locals().get("extra_info", None)
            if isinstance(current_extra_info, dict):
                mstva_info = current_extra_info.get("mstva", None)
            if not isinstance(mstva_info, dict):
                print("[WARNING] use_mstva_loss=True but mstva info is missing, set MSTVA loss to 0.")
            else:
                r3 = mstva_info.get("R3", None)
                r4 = mstva_info.get("R4", None)
                r5 = mstva_info.get("R5", None)
                if r3 is None or r4 is None or r5 is None:
                    print("[WARNING] MSTVA maps incomplete (R3/R4/R5 missing), set MSTVA loss to 0.")
                else:
                    gt_mask = self._build_binary_gt_mask(seg_info=seg_info, ref_tensor=r3)
                    if gt_mask is None:
                        print("[WARNING] MSTVA GT target unavailable, set MSTVA loss to 0.")
                    elif gt_mask.max().item() <= 0:
                        print("[WARNING] MSTVA GT is all-zero, skip MSTVA loss.")
                    else:
                        scale_weights = self._parse_mstva_scale_weights()
                        loss_items = []
                        skipped_scales = []
                        for ridx, r_map in enumerate([r3, r4, r5]):
                            if r_map is None:
                                continue
                            if r_map.shape[0] != gt_mask.shape[0]:
                                print(f"[WARNING] MSTVA batch mismatch at scale {ridx+3}, skip this scale.")
                                skipped_scales.append(f"R{ridx+3}:batch_mismatch")
                                continue
                            gt_s = F.interpolate(gt_mask.float(), size=r_map.shape[-2:], mode="nearest")
                            if gt_s.sum().item() <= 0:
                                print(f"[WARNING] MSTVA scale R{ridx+3} gt_s.sum()==0 after resize, skip this scale.")
                                skipped_scales.append(f"R{ridx+3}:gt_disappear")
                                continue
                            loss_s = F.binary_cross_entropy(
                                r_map.float().clamp(1e-4, 1 - 1e-4),
                                gt_s.float(),
                            )
                            loss_items.append(scale_weights[ridx] * loss_s)
                        if loss_items:
                            loss_mstva_align = sum(loss_items)
                        else:
                            if skipped_scales:
                                print(f"[WARNING] MSTVA all scales skipped: {skipped_scales}")
                            print("[WARNING] No valid MSTVA scale loss computed, set MSTVA loss to 0.")

                alpha3_item = mstva_info.get("alpha3", None)
                alpha4_item = mstva_info.get("alpha4", None)
                alpha5_item = mstva_info.get("alpha5", None)
                if alpha3_item is not None:
                    mstva_alpha3 = alpha3_item.detach().float().view(-1)[0].to(zero_base.device)
                if alpha4_item is not None:
                    mstva_alpha4 = alpha4_item.detach().float().view(-1)[0].to(zero_base.device)
                if alpha5_item is not None:
                    mstva_alpha5 = alpha5_item.detach().float().view(-1)[0].to(zero_base.device)

        loss_dac_rank_weighted = torch.zeros_like(zero_base)
        dac_rank_lr_raw = torch.zeros_like(zero_base)
        dac_rank_fg_acc = torch.zeros_like(zero_base)
        dac_rank_bg_acc = torch.zeros_like(zero_base)
        dac_rank_valid = torch.zeros_like(zero_base)
        dac_rank_num_layers = torch.zeros_like(zero_base)
        dac_rank_inside = torch.zeros_like(zero_base)
        dac_rank_outside = torch.zeros_like(zero_base)
        dac_rank_gap = torch.zeros_like(zero_base)
        dac_rank_layer_indices_str = ""
        use_qdti = self._use_qdti_core()
        use_dac_rank = (
            bool(getattr(self.config, "use_decoder_attn_bias_rank_loss", False))
            and bool(getattr(self.config, "use_decoder_attn_bias", False))
            and not use_qdti
            and self.training
        )
        use_qdti_rank = (
            bool(getattr(self.config, "use_qdti_rank_loss", False))
            and use_qdti
            and self.training
        )
        dac_rank_w = float(getattr(self.config, "decoder_attn_bias_rank_loss_weight", 0.001))
        qdti_rank_w = float(getattr(self.config, "qdti_rank_loss_weight", 0.001))
        dac_rank_margin = float(getattr(self.config, "decoder_attn_bias_rank_margin", 0.1))
        qdti_rank_margin = float(getattr(self.config, "qdti_rank_margin", 0.1))
        bias_maps = getattr(self.predictor, "_last_qdti_maps_for_rank", None) if use_qdti else getattr(
            self.predictor, "_last_decoder_attn_bias_maps_for_rank", None
        )
        loss_qdti_rank_weighted = torch.zeros_like(zero_base)
        qdti_rank_lr_raw = torch.zeros_like(zero_base)
        qdti_rank_valid = torch.zeros_like(zero_base)
        loss_qdti_neg_weighted = torch.zeros_like(zero_base)
        loss_qdti_div_weighted = torch.zeros_like(zero_base)
        pos_queries: List[Optional[int]] = []
        neg_queries: List[List[int]] = []
        if bias_maps:
            dac_rank_num_layers = zero_base + float(len(bias_maps))
            dac_rank_layer_indices_str = ",".join(str(int(x["layer_idx"])) for x in bias_maps)
        if use_dac_rank and dac_rank_w > 0.0 and seg_info is not None:
            lr, dac_rank_fg_acc, dac_rank_bg_acc, dac_rank_valid, dac_rank_inside, dac_rank_outside, dac_rank_gap = (
                self._decoder_attn_bias_ranking_loss(bias_maps, seg_info, dac_rank_margin, zero_base)
            )
            dac_rank_lr_raw = lr.detach().float().to(zero_base.device)
            dac_rank_inside = zero_base + dac_rank_inside.detach().float().to(zero_base.device)
            dac_rank_outside = zero_base + dac_rank_outside.detach().float().to(zero_base.device)
            dac_rank_gap = zero_base + dac_rank_gap.detach().float().to(zero_base.device)
            loss_dac_rank_weighted = dac_rank_w * lr
            if not torch.isfinite(lr).all() or not torch.isfinite(loss_dac_rank_weighted).all():
                raise ValueError("[DecoderAttnBiasRank] non-finite ranking loss (NaN/Inf).")
            if not getattr(self, "_decoder_attn_bias_rank_meta_logged", False):
                if bias_maps and len(bias_maps) > 0 and float(dac_rank_valid.detach().item()) > 0.0:
                    print(
                        f"[DecoderAttnBiasRank] rank_num_layers={len(bias_maps)} "
                        f"layer_indices={dac_rank_layer_indices_str} "
                        f"decoder_attn_bias_rank_valid_count={float(dac_rank_valid.detach().item())}",
                        flush=True,
                    )
                    self._decoder_attn_bias_rank_meta_logged = True
        if use_qdti and seg_info is not None and targets is not None:
            iou_thresh = float(getattr(self.config, "qdti_neg_iou_thresh", 0.3))
            pos_queries, neg_queries = self._resolve_positive_negative_queries(
                mask_outputs, targets, iou_thresh
            )
        if use_qdti_rank and qdti_rank_w > 0.0 and seg_info is not None and bias_maps:
            lr_q, qdti_rank_valid, qdti_rank_lr_raw = self._qdti_positive_query_rank_loss(
                bias_maps, seg_info, qdti_rank_margin, pos_queries, zero_base
            )
            qdti_rank_lr_raw = qdti_rank_lr_raw.detach().float().to(zero_base.device)
            qdti_rank_valid = zero_base + qdti_rank_valid.detach().float().to(zero_base.device)
            loss_qdti_rank_weighted = qdti_rank_w * lr_q
            if not torch.isfinite(lr_q).all():
                raise ValueError("[QDTIRank] non-finite positive-query rank loss.")
            if not getattr(self, "_qdti_rank_meta_logged", False) and float(qdti_rank_valid.detach().item()) > 0:
                print(
                    f"[QDTIRank] layers={len(bias_maps)} valid={float(qdti_rank_valid.detach().item())}",
                    flush=True,
                )
                self._qdti_rank_meta_logged = True
        if (
            use_qdti
            and bool(getattr(self.config, "use_qdti_neg_loss", False))
            and self.training
            and bias_maps
        ):
            neg_w = float(getattr(self.config, "qdti_neg_loss_weight", 0.001))
            if neg_w > 0:
                loss_qdti_neg_weighted = neg_w * self._qdti_negative_query_loss(
                    bias_maps, seg_info, qdti_rank_margin, pos_queries, neg_queries, zero_base
                )
        if (
            use_qdti
            and bool(getattr(self.config, "use_qdti_div_loss", False))
            and self.training
            and bias_maps
        ):
            div_w = float(getattr(self.config, "qdti_div_loss_weight", 0.001))
            if div_w > 0:
                loss_qdti_div_weighted = div_w * self._qdti_diversity_loss(bias_maps, neg_queries, zero_base)

        loss = llm_loss + mask_loss
        if use_attention_loss:
            loss = loss + 0.01 * loss_attention
        if use_midstage_gate_loss and midstage_gate_loss_weight > 0:
            loss = loss + midstage_gate_loss_weight * loss_midstage_gate
        if use_mstva_loss and mstva_loss_weight > 0:
            loss = loss + mstva_loss_weight * loss_mstva_align
        loss = loss + loss_dac_rank_weighted + loss_qdti_rank_weighted + loss_qdti_neg_weighted + loss_qdti_div_weighted

        text_film_gamma_norm = torch.zeros_like(zero_base)
        text_film_beta_norm = torch.zeros_like(zero_base)
        text_film_branch_alpha_log = torch.zeros_like(zero_base)
        if getattr(self.config, "use_text_film", False) and getattr(self, "text_film_branch", None) is not None:
            br = self.text_film_branch
            if getattr(br, "_last_gamma_norm", None) is not None:
                text_film_gamma_norm = br._last_gamma_norm.detach().float().to(zero_base.device)
            if getattr(br, "_last_beta_norm", None) is not None:
                text_film_beta_norm = br._last_beta_norm.detach().float().to(zero_base.device)
            text_film_branch_alpha_log = br.branch_alpha.detach().float().to(zero_base.device)

        dac_log = getattr(self.predictor, "_last_qdti_log", None) if use_qdti else getattr(
            self.predictor, "_last_decoder_attn_bias_log", None
        )
        z0 = zero_base * 0
        dac_abs_mean = z0
        dac_raw_std = z0
        dac_max = z0
        dac_min = z0
        dac_enabled = z0
        qdti_abs_mean = z0
        qdti_enabled = z0
        qdti_alpha_log = z0
        qdti_gate_eff_log = z0
        qdti_warmup_log = z0
        if dac_log is not None:
            if use_qdti:
                qdti_abs_mean = z0 + dac_log.get("qdti_bias_abs_mean", z0).detach().float().to(zero_base.device)
                qdti_enabled = z0 + dac_log.get("qdti_enabled", z0).detach().float().to(zero_base.device)
                if dac_log.get("qdti_alpha_l") is not None:
                    qdti_alpha_log = z0 + dac_log["qdti_alpha_l"].detach().float().to(zero_base.device)
                if dac_log.get("qdti_gate_eff") is not None:
                    qdti_gate_eff_log = z0 + dac_log["qdti_gate_eff"].detach().float().to(zero_base.device)
                if dac_log.get("qdti_warmup_factor") is not None:
                    qdti_warmup_log = z0 + dac_log["qdti_warmup_factor"].detach().float().to(zero_base.device)
            else:
                dac_abs_mean = z0 + dac_log["decoder_attn_bias_abs_mean"].detach().float().to(zero_base.device)
                dac_raw_std = z0 + dac_log["decoder_attn_bias_raw_std"].detach().float().to(zero_base.device)
                dac_max = z0 + dac_log["decoder_attn_bias_max"].detach().float().to(zero_base.device)
                dac_min = z0 + dac_log["decoder_attn_bias_min"].detach().float().to(zero_base.device)
                dac_enabled = z0 + dac_log["decoder_attn_bias_enabled"].detach().float().to(zero_base.device)

        return CausalOutputWithMask(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_mask=loss_mask.detach(),
            loss_dice=loss_dice.detach(),
            loss_llm=llm_loss.detach(),
            loss_attention=(0.01 * loss_attention.detach()) if use_attention_loss else torch.zeros_like(loss_attention.detach()),
            loss_midstage_gate=(midstage_gate_loss_weight * loss_midstage_gate.detach()) if use_midstage_gate_loss else torch.zeros_like(loss_midstage_gate.detach()),
            midstage_gate_alpha=midstage_gate_alpha.detach(),
            loss_mstva_align=(mstva_loss_weight * loss_mstva_align.detach()) if use_mstva_loss else torch.zeros_like(loss_mstva_align.detach()),
            mstva_alpha3=mstva_alpha3.detach(),
            mstva_alpha4=mstva_alpha4.detach(),
            mstva_alpha5=mstva_alpha5.detach(),
            text_film_gamma_norm=text_film_gamma_norm.detach(),
            text_film_beta_norm=text_film_beta_norm.detach(),
            text_film_branch_alpha=text_film_branch_alpha_log.detach(),
            decoder_attn_bias_abs_mean=dac_abs_mean.detach(),
            decoder_attn_bias_raw_std=dac_raw_std.detach(),
            decoder_attn_bias_max=dac_max.detach(),
            decoder_attn_bias_min=dac_min.detach(),
            decoder_attn_bias_enabled=dac_enabled.detach(),
            loss_decoder_attn_bias_rank=loss_dac_rank_weighted.detach(),
            decoder_attn_bias_rank_loss_raw=dac_rank_lr_raw.detach(),
            decoder_attn_bias_inside_mean=dac_rank_inside.detach(),
            decoder_attn_bias_outside_mean=dac_rank_outside.detach(),
            decoder_attn_bias_inside_outside_gap=dac_rank_gap.detach(),
            decoder_attn_bias_rank_fg_access_ratio=dac_rank_fg_acc.detach(),
            decoder_attn_bias_rank_bg_access_ratio=dac_rank_bg_acc.detach(),
            decoder_attn_bias_rank_valid_count=dac_rank_valid.detach(),
            decoder_attn_bias_rank_num_layers=dac_rank_num_layers.detach(),
            decoder_attn_bias_rank_layer_indices=dac_rank_layer_indices_str,
            loss_qdti_rank=loss_qdti_rank_weighted.detach(),
            qdti_rank_loss_raw=qdti_rank_lr_raw.detach(),
            loss_qdti_neg=loss_qdti_neg_weighted.detach(),
            loss_qdti_div=loss_qdti_div_weighted.detach(),
            qdti_bias_abs_mean=qdti_abs_mean.detach(),
            qdti_enabled=qdti_enabled.detach(),
            qdti_rank_valid_count=qdti_rank_valid.detach(),
            qdti_alpha_l=qdti_alpha_log.detach(),
            qdti_gate_eff=qdti_gate_eff_log.detach(),
            qdti_warmup_factor=qdti_warmup_log.detach(),
        )
    
    def eval_seg(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            images_clip: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            seg_info=None,
            token_refer_id=None,
            SEG_token_embedding_indices=None,
            mask_num = None):
        
        output_attentions = False
        output_hidden_states = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_cond = self.build_text_condition(
            token_refer_id=token_refer_id,
            batch_size=input_ids.shape[0],
            device=images.device,
        )

        use_mstva = getattr(self.config, "use_mstva", False)
        use_qdti = self._use_qdti_core()
        use_decoder_attn_bias = getattr(self.config, "use_decoder_attn_bias", False)
        text_tokens = None
        text_mask = None
        if use_mstva or use_decoder_attn_bias or use_qdti:
            text_tokens, text_mask = self.build_text_tokens(
                token_refer_id,
                batch_size=input_ids.shape[0] if input_ids is not None else None,
                device=images.device if images is not None else None,
            )
            if text_tokens is None and use_mstva:
                if not getattr(self, "_eval_seg_mstva_bypass_warned", False):
                    self._eval_seg_mstva_bypass_warned = True
                    print(
                        "[WARNING][eval_seg] use_mstva=True but text_tokens is None "
                        "(token_refer_id is None or yielded no valid tokens); MSTVA bypassed in eval."
                    )
            elif text_tokens is None and (use_decoder_attn_bias or use_qdti):
                if not getattr(self, "_eval_seg_qdti_bypass_warned", False):
                    self._eval_seg_qdti_bypass_warned = True
                    print(
                        "[WARNING][eval_seg][QDTI] query-aware decoder bias enabled but text_tokens is None "
                        "(token_refer_id is None or yielded no valid tokens); QDTI-Core bypassed."
                    )
            elif use_mstva and text_tokens is not None and not getattr(self, "_eval_seg_mstva_debug_logged", False):
                self._eval_seg_mstva_debug_logged = True
                print("[DEBUG][eval_seg] use_mstva=True")
                print(f"[DEBUG][eval_seg] text_tokens shape={tuple(text_tokens.shape)}")
                if text_mask is not None:
                    valid_per_sample = text_mask.sum(dim=-1).float()
                    print(
                        "[DEBUG][eval_seg] text_mask valid count "
                        f"mean={valid_per_sample.mean().item():.2f} "
                        f"min={int(valid_per_sample.min().item())} "
                        f"max={int(valid_per_sample.max().item())}"
                    )
                else:
                    print("[DEBUG][eval_seg] text_mask is None")

        image_features = self.get_vision_tower_feature(
            images,
            text_cond=text_cond,
            text_tokens=text_tokens,
            text_mask=text_mask,
        )

        input_ids, attention_mask, past_key_values, inputs_embeds, labels, SEG_token_embedding_indices, image_features_indices = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images_clip,
            token_refer_id=token_refer_id, SEG_token_embedding_indices=SEG_token_embedding_indices)
    
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs.last_hidden_state   

        SEG_embedding = self.SEG_token_projector(self.get_SEG_embedding(hidden_states, SEG_token_embedding_indices))

        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features)
    
        images = [image.repeat((num, 1, 1, 1)) for image, num in zip(images, mask_num)]
        images = [s[0] for image_repeat in images for s in torch.split(image_repeat, 1, dim=0)]
        mask_num = torch.tensor(mask_num, device=mask_features.device)
        mask_features = torch.repeat_interleave(mask_features, repeats=mask_num, dim=0)
        multi_scale_features = [
            torch.repeat_interleave(feat, repeats=mask_num, dim=0)
            for feat in multi_scale_features
        ]

        dac_mode = str(getattr(self.config, "decoder_attn_bias_eval_mode", "normal"))
        dac_fscale = float(getattr(self.config, "decoder_attn_bias_force_scale", 1.0))
        if use_qdti:
            # Eval: warmup is complete so trained alpha_l applies (avoid global_step=None -> factor=0).
            qdti_wu = int(getattr(self.config, "qdti_warmup_steps", 0) or 0)
            self._sync_qdti_runtime_to_predictor(
                global_step=qdti_wu if qdti_wu > 0 else 0
            )
        tt_rep, tm_rep = self._repeat_text_tokens_for_mask_num(text_tokens, text_mask, mask_num)
        if tt_rep is not None:
            if tt_rep.shape[0] != SEG_embedding.shape[0]:
                raise AssertionError(
                    f"[DecoderAttnBias] text batch {tt_rep.shape[0]} != SEG_embedding batch {SEG_embedding.shape[0]}"
                )
            if tt_rep.shape[0] != mask_features.shape[0]:
                raise AssertionError(
                    f"[DecoderAttnBias] text batch {tt_rep.shape[0]} != mask_features batch {mask_features.shape[0]}"
                )
        mask_outputs = self.predictor(
            multi_scale_features,
            mask_features,
            None,
            None,
            SEG_embedding,
            text_tokens=tt_rep,
            text_mask=tm_rep,
            dac_eval_mode=dac_mode,
            dac_force_scale=dac_fscale,
        )

        
        mask_pred_results = mask_outputs["pred_masks"]
        images = ImageList.from_tensors(images, self.size_divisibility)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        
        processed_results = []
        for _seg_info, mask_pred_result in zip(seg_info, mask_pred_results):
            instance_r = {
                'pred': ((mask_pred_result.detach().float().cpu().numpy() > 0) * 255).astype(np.uint8),
                'image_name': _seg_info['image_id'],
                'id': _seg_info['data_id'],
                'mask_id': _seg_info['mask_id'],
            }
            processed_results.append(instance_r)
        return processed_results
