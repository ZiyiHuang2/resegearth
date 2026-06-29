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

from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoderForOPTPreTrain
from ..mask_decoder.Mask2Former_Simplify.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from ..mask_encoder.swin_trans import build_swin_b, build_swin_l
from ..mask_encoder.tg_swin import TextConditionFactory, TGSwimController, SETControlHead

from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine

from ..datasets_mapper.IVS_mapper import IVSDatasetMapper
from segearth_r2.model.mask_decoder.mask_criterion.Mask_Criterion import Criterion, hungarian_matcher_InstructSeg
from transformers import PhiModel, PhiForCausalLM, PhiConfig
from fvcore.nn import FlopCountAnalysis

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
    loss_union: Optional[torch.FloatTensor] = None
    loss_setpp_coverage: Optional[torch.FloatTensor] = None
    loss_setpp_consistency: Optional[torch.FloatTensor] = None

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
            swin_type = getattr(config,'swin_type','base')
            if swin_type == 'base':
                self.vision_tower_mask = build_swin_b(None)
            else:
                self.vision_tower_mask = build_swin_l(None)

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
        swin_type = getattr(model_args,'swin_type','base')
        self.config.swin_type = swin_type
        if swin_type == 'base':
            vision_tower_mask = build_swin_b(vision_tower_mask)
        else:
            print('current visual encoder is swin large')
            vision_tower_mask = build_swin_l(vision_tower_mask)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower_mask = [vision_tower_mask]
        else:
            self.vision_tower_mask = vision_tower_mask

        self.config.use_mm_proj = True
        vision_tower_mask.hidden_size = 256
        vision_tower_mask.image_processor = IVSDatasetMapper(self.cfg)

class SegEarthR2(MiphaPhiForCausalLM):
    @classmethod
    def from_pretrained(cls, *model_args, **kwargs):
        requested_loading_info = kwargs.pop('output_loading_info', False)
        kwargs['output_loading_info'] = True
        model, loading_info = super().from_pretrained(*model_args, **kwargs)
        if any(k.startswith('SET_token_projector') for k in loading_info.get('missing_keys', [])):
            model._copy_set_projector_from_seg()
        if requested_loading_info:
            return model, loading_info
        return model

    def __init__(self, config, model_args=None, mask_decoder_cfg=None, add_cross_attn=True, cross_attn_index=None):
        super(SegEarthR2, self).__init__(config)

        self.model = SegEarthR2Model(config, mask_decoder_cfg)
        self.init_config = config
        self.mask_decoder_cfg = mask_decoder_cfg
        self.cross_attn_index = cross_attn_index

        # --- lm_head size decision logic ---
        # Priority: 1) config.lm_head_size (explicit override),
        #           2) max(config.vocab_size, 51200) to be compatible with:
        #              - Mipha-3B pretrained (vocab_size=51200)
        #              - merged_model whose config.vocab_size may be 50296
        #                but actual lm_head.weight is [51200, hidden]
        lm_head_size = getattr(config, 'lm_head_size', None)
        if lm_head_size is None:
            lm_head_size = max(config.vocab_size, 51200)
        self.lm_head = nn.Linear(config.hidden_size, lm_head_size, bias=False)
        self.config.lm_head_size = lm_head_size

        is_train_mask_decode = getattr(config, 'mask_decode_train', False)
        self.is_train_mask_decode = is_train_mask_decode

        if is_train_mask_decode:
            print('Mask Decoder has been trained, init directly')
            self.initial_mask_module()
        self.post_init()

    def initial_mask_module(self, pretrained_path=None, model_args=None):
        if not self.is_train_mask_decode:
            print('Initialize mask modules...')
            self.config.mask_decode_train = True

        self.attention_loss = AttentionLoss()
        
        self.test_topk_per_image = self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        input_shape = self.output_shape()
        self.pixel_decoder = self.pixel_decoder_init(cfg=self.mask_decoder_cfg, input_shape=input_shape)
        self.predictor = self.predictor_init(cfg=self.mask_decoder_cfg, model_args=model_args)

        self.SEG_token_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
        self.SET_token_projector = nn.Linear(
            self.config.hidden_size,
            self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            bias=self.SEG_token_projector.bias is not None,
        )
        self._copy_set_projector_from_seg()

        self._init_tg_swin_modules()

        self.mask_decoder_training_init(self.mask_decoder_cfg, model_args=model_args)
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

    def _copy_set_projector_from_seg(self):
        if not (hasattr(self, 'SET_token_projector') and hasattr(self, 'SEG_token_projector')):
            return
        seg = self.SEG_token_projector
        setp = self.SET_token_projector
        if getattr(seg.weight, 'is_meta', seg.weight.device.type == 'meta'):
            return
        with torch.no_grad():
            setp.weight.copy_(seg.weight)
            if seg.bias is not None and setp.bias is not None:
                setp.bias.copy_(seg.bias)

    def _normalize_csqr_state_dict(self, state_dict):
        normalized = False
        for key, alpha in list(state_dict.items()):
            if not key.endswith('csqr_block.fusion_alpha_logit'):
                continue
            if getattr(alpha, 'ndim', None) == 0:
                if not normalized:
                    state_dict = dict(state_dict)
                    normalized = True
                state_dict[key] = alpha.reshape(1)
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = self._normalize_csqr_state_dict(state_dict)
        has_set_in_ckpt = any(k.startswith('SET_token_projector') for k in state_dict)
        if not has_set_in_ckpt:
            incompatible = super().load_state_dict(state_dict, strict=False, assign=assign)
            self._copy_set_projector_from_seg()
            missing = [k for k in incompatible.missing_keys if not k.startswith('SET_token_projector.')]
            if strict and (missing or incompatible.unexpected_keys):
                raise RuntimeError(
                    f"Error(s) in loading state_dict:\n"
                    f"\tMissing key(s): {missing}\n"
                    f"\tUnexpected key(s): {incompatible.unexpected_keys}"
                )
            from torch.nn.modules.module import _IncompatibleKeys
            return _IncompatibleKeys(missing, incompatible.unexpected_keys)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _get_tg_swin_cfg(self):
        return getattr(self.mask_decoder_cfg, "TG_SWIN", None)

    def _tg_swin_enabled(self):
        cfg = self._get_tg_swin_cfg()
        return bool(cfg and getattr(cfg, "ENABLED", False))

    def _use_coarse_evidence(self):
        cfg = self._get_tg_swin_cfg()
        return bool(cfg and getattr(cfg, "USE_COARSE_EVIDENCE", False))

    def _is_dr_ewti(self):
        cfg = self._get_tg_swin_cfg()
        return bool(cfg and str(getattr(cfg, "VERSION", "")).lower() == "dr-ewti")

    def _is_enhanced_wti_v2(self):
        cfg = self._get_tg_swin_cfg()
        if not cfg:
            return False
        version = str(getattr(cfg, "VERSION", "")).lower()
        return version == "enhanced-wti-v2" or bool(getattr(cfg, "ENHANCED_WTI", False))

    def _init_tg_swin_modules(self):
        cfg = self._get_tg_swin_cfg()
        if not cfg or not getattr(cfg, "ENABLED", False):
            self.tg_swin_tcf = None
            self.tg_swin_controller = None
            self.tg_swin_set_control = None
            return

        text_dim = getattr(cfg, "TEXT_DIM", None) or self.config.hidden_size
        cond_dim = getattr(cfg, "COND_DIM", 256)
        swin_type = getattr(self.config, "swin_type", "base")
        window_size = getattr(self.mask_decoder_cfg.MODEL.SWIN, "WINDOW_SIZE", 12)
        version = getattr(cfg, "VERSION", "v1")
        wti_stages = list(getattr(cfg, "WTI_STAGES", [1, 2, 3]))
        num_stages = int(getattr(cfg, "NUM_STAGES", len(wti_stages)))
        enhanced_wti = self._is_enhanced_wti_v2()
        use_dr_ewti = bool(
            getattr(cfg, "USE_DR_EWTI", self._is_dr_ewti() or bool(getattr(cfg, "USE_COARSE_EVIDENCE", False)))
        )
        if enhanced_wti:
            use_dr_ewti = False

        tcf_version = version if version not in ("dr-ewti", "enhanced-wti-v2") else "v1.5"
        stage_router_default = version in ("v1.5", "v1.6", "dr-ewti", "enhanced-wti-v2")

        self.tg_swin_tcf = TextConditionFactory(
            text_dim=text_dim,
            cond_dim=cond_dim,
            reliability_init=getattr(cfg, "RELIABILITY_INIT", 0.0),
            use_phrase_pool=getattr(cfg, "USE_PHRASE_POOL", True),
            version=tcf_version,
            num_stages=num_stages,
            stage_router=bool(getattr(cfg, "STAGE_ROUTER", stage_router_default)),
            router_hidden_dim=int(getattr(cfg, "ROUTER_HIDDEN_DIM", 512)),
            use_relation_pool=bool(getattr(cfg, "USE_RELATION_AWARE_POOL", True)),
            use_stage_phrase=bool(getattr(cfg, "USE_STAGE_PHRASE", False)),
            use_hybrid_reliability=bool(getattr(cfg, "USE_HYBRID_RELIABILITY", False)),
            stage_phrase_temp=float(getattr(cfg, "STAGE_PHRASE_TEMP", 1.0)),
            hybrid_reliability_temp=float(getattr(cfg, "HYBRID_RELIABILITY_TEMP", 1.0)),
        )
        self.tg_swin_controller = TGSwimController(
            cond_dim=cond_dim,
            wti_rank=getattr(cfg, "WTI_RANK", 16),
            wti_stages=wti_stages,
            wti_start_layer=getattr(cfg, "WTI_START_LAYER", 0),
            bias_max=getattr(cfg, "BIAS_MAX", 4.0),
            alpha_init=getattr(cfg, "ALPHA_INIT", 0.0),
            window_size=window_size,
            swin_type=swin_type,
            log_stats=getattr(cfg, "LOG_STATS", False),
            head_aware=bool(getattr(cfg, "HEAD_AWARE", version in ("v1.5", "v1.6", "dr-ewti", "enhanced-wti-v2"))),
            num_text_stages=num_stages,
            use_evidence_state=bool(getattr(cfg, "USE_EVIDENCE_STATE", False)),
            evidence_state_dim=int(getattr(cfg, "EVIDENCE_STATE_DIM", 64)),
            use_dr_ewti=use_dr_ewti,
            evidence_dim=int(getattr(cfg, "EVIDENCE_DIM", 32)),
            evidence_eps=float(getattr(cfg, "EVIDENCE_EPS", 1e-4)),
            evidence_logit_clip=float(getattr(cfg, "EVIDENCE_LOGIT_CLIP", 4.0)),
            evidence_relation_gate_init=float(getattr(cfg, "EVIDENCE_RELATION_GATE_INIT", 0.0)),
            enhanced_wti=enhanced_wti,
            wti_projector_ratio=float(getattr(cfg, "WTI_PROJECTOR_RATIO", 2.0)),
            wti_projector_max_hidden=int(getattr(cfg, "WTI_PROJECTOR_MAX_HIDDEN", 512)),
            wti_head_mixer=bool(getattr(cfg, "WTI_HEAD_MIXER", True)),
            wti_head_mixer_ratio=float(getattr(cfg, "WTI_HEAD_MIXER_RATIO", 2.0)),
            wti_head_mixer_gamma_init=float(getattr(cfg, "WTI_HEAD_MIXER_GAMMA_INIT", 0.0)),
        )
        use_set_control = bool(getattr(cfg, "USE_SET_TGSWIN_CONTROL", False))
        if use_set_control and enhanced_wti:
            set_dim = int(self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)
            self.tg_swin_set_control = SETControlHead(
                text_dim=set_dim,
                num_stages=num_stages,
                init_bias=float(getattr(cfg, "SET_CONTROL_INIT_BIAS", 4.0)),
            )
        else:
            self.tg_swin_set_control = None
        print(
            f"[TG_SWIN] Initialized v={version} TCF + WTI "
            f"(cond_dim={cond_dim}, enhanced_wti={enhanced_wti}, "
            f"head_aware={getattr(cfg, 'HEAD_AWARE', version in ('v1.5', 'v1.6', 'dr-ewti', 'enhanced-wti-v2'))}, "
            f"dr_ewti={use_dr_ewti}, coarse_evidence={getattr(cfg, 'USE_COARSE_EVIDENCE', False)}, "
            f"set_control={use_set_control and enhanced_wti}, "
            f"stage_phrase={getattr(cfg, 'USE_STAGE_PHRASE', False)}, "
            f"hybrid_rho={getattr(cfg, 'USE_HYBRID_RELIABILITY', False)}, "
            f"evidence_state={getattr(cfg, 'USE_EVIDENCE_STATE', False)}, "
            f"stages={list(cfg.WTI_STAGES)}, num_stages={num_stages})"
        )

    def _repeat_images_per_target(self, images, mask_num):
        if isinstance(mask_num, torch.Tensor):
            mask_num_list = mask_num.tolist()
        else:
            mask_num_list = list(mask_num)

        if isinstance(images, torch.Tensor):
            chunks = []
            for i, n in enumerate(mask_num_list):
                n = int(n)
                if n > 0:
                    chunks.append(images[i:i + 1].expand(n, -1, -1, -1))
            if not chunks:
                return images[:0]
            return torch.cat(chunks, dim=0)

        repeated = []
        for img, n in zip(images, mask_num_list):
            n = int(n)
            repeated.extend([img] * n)
        if not repeated:
            return torch.stack([], dim=0) if isinstance(images[0], torch.Tensor) else []
        return torch.stack(repeated, dim=0)

    def _gather_refer_phrase_hidden(self, hidden_states, seg_indices, refer_span_mask):
        if refer_span_mask is None:
            return None, None

        phrase_pieces = []
        for seq_hidden, seg_mask, ref_mask in zip(hidden_states, seg_indices, refer_span_mask):
            seg_mask = seg_mask.bool()
            ref_mask = ref_mask.bool()
            seq_len = seq_hidden.shape[0]
            if ref_mask.shape[0] != seq_len:
                if ref_mask.shape[0] < seq_len:
                    pad = torch.zeros(
                        seq_len - ref_mask.shape[0],
                        dtype=ref_mask.dtype,
                        device=ref_mask.device,
                    )
                    ref_mask = torch.cat([ref_mask, pad])
                else:
                    ref_mask = ref_mask[:seq_len]
            seg_positions = seg_mask.nonzero(as_tuple=False).squeeze(-1)
            if seg_positions.numel() == 0:
                continue
            if seg_positions.dim() == 0:
                seg_positions = seg_positions.unsqueeze(0)

            prev_boundary = 0
            for seg_pos in seg_positions.tolist():
                seg_pos = int(seg_pos)
                region = ref_mask.clone()
                region[:prev_boundary] = False
                region[seg_pos:] = False
                if region.any():
                    phrase_pieces.append(seq_hidden[region])
                else:
                    phrase_pieces.append(
                        torch.zeros(0, seq_hidden.shape[-1], device=seq_hidden.device, dtype=seq_hidden.dtype)
                    )
                prev_boundary = seg_pos + 1

        if not phrase_pieces:
            return None, None

        max_len = max(p.shape[0] for p in phrase_pieces)
        max_len = max(max_len, 1)
        n_target = len(phrase_pieces)
        hidden_dim = hidden_states.shape[-1]
        device = hidden_states.device
        dtype = hidden_states.dtype
        phrase_hidden = torch.zeros(n_target, max_len, hidden_dim, device=device, dtype=dtype)
        phrase_mask = torch.zeros(n_target, max_len, device=device, dtype=torch.bool)
        for i, piece in enumerate(phrase_pieces):
            plen = piece.shape[0]
            if plen > 0:
                phrase_hidden[i, :plen] = piece
                phrase_mask[i, :plen] = True
        return phrase_hidden, phrase_mask

    def build_text_cond(self, hidden_states, seg_indices, refer_span_mask=None):
        seg_hidden = self.get_SEG_embedding(hidden_states, seg_indices).squeeze(1)
        phrase_hidden, phrase_mask = self._gather_refer_phrase_hidden(
            hidden_states, seg_indices, refer_span_mask
        )
        text_cond, reliability = self.tg_swin_tcf(
            seg_hidden,
            hidden_states=hidden_states,
            seg_indices=seg_indices,
            phrase_hidden=phrase_hidden,
            phrase_mask=phrase_mask,
        )
        return seg_hidden, text_cond, reliability

    def build_set_control(self, set_embedding_t):
        """set_embedding_t: [T, 1, C] after repeat_interleave. Returns [T, S, 1] or None."""
        if self.tg_swin_set_control is None:
            return None
        set_hidden = set_embedding_t.squeeze(1)
        return self.tg_swin_set_control(set_hidden)

    def get_vision_tower_feature(
        self,
        images,
        text_cond=None,
        reliability=None,
        set_control=None,
        coarse_evidence=None,
        enable_tg_swin=True,
    ):
        swin = self.get_model().get_vision_tower_mask()
        tg_kwargs = {}
        if (
            self._tg_swin_enabled()
            and enable_tg_swin
            and text_cond is not None
            and self.tg_swin_controller is not None
        ):
            tg_kwargs = {
                "text_cond": text_cond,
                "reliability": reliability,
                "tg_swin_controller": self.tg_swin_controller,
                "coarse_evidence": coarse_evidence,
                "enable_tg_swin": True,
                "set_control": set_control,
            }
        elif self._tg_swin_enabled():
            tg_kwargs = {
                "text_cond": None,
                "reliability": None,
                "tg_swin_controller": self.tg_swin_controller,
                "coarse_evidence": None,
                "enable_tg_swin": False,
            }
        features = swin(images, **tg_kwargs)

        features_dict = {
            'res2': features[0],
            'res3': features[1],
            'res4': features[2],
            'res5': features[3],
        }
        return features_dict

    def _repeat_features_per_target(self, features_dict, mask_num):
        if not features_dict:
            raise ValueError("features_dict must not be empty")
        if isinstance(mask_num, torch.Tensor):
            mask_num_list = [int(n) for n in mask_num.tolist()]
        else:
            mask_num_list = [int(n) for n in mask_num]

        first_feature = next(iter(features_dict.values()))
        batch_size = int(first_feature.shape[0])
        if len(mask_num_list) != batch_size:
            raise ValueError(
                f"mask_num length={len(mask_num_list)} != feature batch={batch_size}"
            )
        mask_num_tensor = torch.as_tensor(
            mask_num_list,
            device=first_feature.device,
            dtype=torch.long,
        )
        repeated = {}
        for key, feat in features_dict.items():
            if int(feat.shape[0]) != batch_size:
                raise ValueError(
                    f"feature '{key}' batch={feat.shape[0]} != expected batch={batch_size}"
                )
            repeated[key] = torch.repeat_interleave(feat, repeats=mask_num_tensor, dim=0)
        return repeated

    @staticmethod
    def _build_target_to_image(mask_num, device):
        repeats = torch.as_tensor(list(mask_num), device=device, dtype=torch.long)
        return torch.repeat_interleave(torch.arange(len(mask_num), device=device, dtype=torch.long), repeats)

    @staticmethod
    def _repeat_set_embedding_per_target(set_embedding_b, mask_num):
        """Expand image-level SET [B, 1, C] to target-level [T, 1, C]."""
        if mask_num is None:
            return set_embedding_b
        total = int(sum(int(n) for n in mask_num))
        if set_embedding_b.shape[0] == total:
            return set_embedding_b
        repeats = torch.as_tensor(list(mask_num), device=set_embedding_b.device, dtype=torch.long)
        return torch.repeat_interleave(set_embedding_b, repeats, dim=0)

    def _build_flat_seg_targets(self, seg_info, mask_num, device, pred_masks):
        targets = []
        offset = 0
        for k in mask_num:
            for _j in range(int(k)):
                raw_mask = seg_info[offset]['mask'].to(device)
                if raw_mask.ndim == 3 and raw_mask.shape[0] == 1:
                    raw_mask = raw_mask.squeeze(0)
                elif raw_mask.ndim == 3:
                    raw_mask = raw_mask.squeeze(0)
                targets.append({
                    'labels': torch.zeros(1, dtype=torch.long, device=device),
                    'masks': raw_mask.unsqueeze(0),
                    'valid': None,
                    'inst_id': None,
                })
                offset += 1
        return targets

    def predict_coarse_masks(self, multi_scale_features, mask_features, seg_embeddings, set_embedding_t):
        """Pass A: detached coarse SEG masks [T, 1, H, W] via SET++ per-target predictor."""
        with torch.no_grad():
            mask_outputs = self.predictor(
                multi_scale_features,
                mask_features,
                None,
                None,
                SEG_embedding=seg_embeddings,
                SET_embedding=set_embedding_t,
                mask_num=None,
                per_target_mode=True,
            )
            return mask_outputs["pred_seg_masks"].sigmoid().detach()

    def get_shared_coarse_evidence(
        self,
        images,
        seg_embeddings,
        mask_num,
        set_embedding_b=None,
    ) -> torch.Tensor:
        """Shared Pass A: detached coarse probabilities [T, 1, Hm, Wm]."""
        if isinstance(mask_num, torch.Tensor):
            mask_num_list = mask_num.tolist()
        else:
            mask_num_list = list(mask_num)
        n_target = int(sum(int(n) for n in mask_num_list))
        assert seg_embeddings.shape[0] == n_target, (
            f"coarse evidence: SEG count={seg_embeddings.shape[0]} != sum(mask_num)={n_target}"
        )

        seg_emb_det = seg_embeddings.detach()
        with torch.no_grad():
            image_features = self.get_vision_tower_feature(images, enable_tg_swin=False)
            mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(image_features)
            mask_features_t = self._repeat_features_per_target(
                {"mask_features": mask_features}, mask_num
            )["mask_features"]
            multi_scale_t = self._repeat_features_per_target(
                {f"ms{i}": f for i, f in enumerate(multi_scale_features)}, mask_num
            )
            multi_scale_list = [multi_scale_t[f"ms{i}"] for i in range(len(multi_scale_features))]
            assert mask_features_t.shape[0] == n_target, (
                f"coarse Pass A mask_features batch={mask_features_t.shape[0]} != targets={n_target}"
            )
            set_embedding_t = self._repeat_set_embedding_per_target(set_embedding_b, mask_num)
            coarse_prob = self.predict_coarse_masks(
                multi_scale_list, mask_features_t, seg_emb_det, set_embedding_t
            )
        assert coarse_prob.shape[0] == n_target, (
            f"coarse_prob batch={coarse_prob.shape[0]} != targets={n_target}, shape={tuple(coarse_prob.shape)}"
        )
        assert not coarse_prob.requires_grad
        return coarse_prob
    def mask_decoder_training_init(self, cfg, model_args=None):
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        # boundary_weight = cfg.MODEL.MASK_FORMER.BOUNDARY_WEIGHT

        setpp_closed_loop = getattr(model_args, 'setpp_closed_loop', True) if model_args else True
        setpp_kwargs = {}
        if model_args is not None:
            setpp_kwargs = dict(
                setpp_closed_loop=setpp_closed_loop,
                lambda_union_single=getattr(model_args, 'setpp_lambda_union_single', 0.01),
                lambda_union_multi=getattr(model_args, 'setpp_lambda_union_multi', 0.05),
                lambda_coverage_single=getattr(model_args, 'setpp_lambda_coverage_single', 0.005),
                lambda_coverage_multi=getattr(model_args, 'setpp_lambda_coverage_multi', 0.02),
                lambda_consistency_single=getattr(model_args, 'setpp_lambda_consistency_single', 0.005),
                lambda_consistency_multi=getattr(model_args, 'setpp_lambda_consistency_multi', 0.02),
                closed_loop_warmup_steps=getattr(model_args, 'setpp_closed_loop_warmup_steps', 2000),
                consistency_mode=getattr(model_args, 'setpp_consistency_mode', 'seg_align_set'),
            )
        
        matcher = hungarian_matcher_InstructSeg(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        
        weight_dict = {
            "loss_SEG_class": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
            "loss_union_mask": 1.0,
            "loss_union_dice": 1.0,
            "loss_setpp_coverage": 1.0,
            "loss_setpp_consistency": 1.0,
        }
        if not setpp_closed_loop:
            weight_dict["loss_union_mask"] = mask_weight
            weight_dict["loss_union_dice"] = dice_weight

        self.weight_dict = weight_dict
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["SEG_labels", "masks", "union",]
        self.criterion = Criterion(
            matcher=matcher,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            device=self.device,
            **setpp_kwargs,
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

    def predictor_init(self, cfg, model_args=None):
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
        if model_args is not None:
            use_csqr = getattr(model_args, 'setpp_csqr_enable', True)
            csqr_fusion_alpha_init = getattr(model_args, 'setpp_csqr_fusion_alpha_init', 0.99)
        else:
            use_csqr = getattr(self.config, 'setpp_csqr_enable', False)
            csqr_fusion_alpha_init = getattr(self.config, 'setpp_csqr_fusion_alpha_init', 0.99)

        predictor = MultiScaleMaskedTransformerDecoderForOPTPreTrain(in_channels,
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
                                                                     use_csqr=use_csqr,
                                                                     csqr_fusion_alpha_init=csqr_fusion_alpha_init,)
        return predictor

    def resolve_setpp_flags(self, model_args=None):
        if model_args is not None:
            use_csqr = getattr(model_args, 'setpp_csqr_enable', True)
            closed_loop = getattr(model_args, 'setpp_closed_loop', True)
        else:
            use_csqr = getattr(self.config, 'setpp_csqr_enable', True)
            closed_loop = getattr(self.config, 'setpp_closed_loop', True)
        return bool(use_csqr), bool(closed_loop)

    def persist_setpp_config(self, model_args=None):
        use_csqr, closed_loop = self.resolve_setpp_flags(model_args)
        self.config.setpp_csqr_enable = use_csqr
        self.config.setpp_closed_loop = closed_loop
        return use_csqr, closed_loop

    def ensure_setpp_predictor(self, model_args=None):
        """Rebuild predictor when setpp_csqr_enable disagrees with loaded predictor.use_csqr."""
        if not hasattr(self, 'mask_decoder_cfg') or self.mask_decoder_cfg is None:
            return

        target_use_csqr, _ = self.resolve_setpp_flags(model_args)

        if not hasattr(self, 'predictor'):
            self.predictor = self.predictor_init(self.mask_decoder_cfg, model_args=model_args)
            return

        current_use_csqr = getattr(self.predictor, 'use_csqr', False)
        if current_use_csqr == target_use_csqr:
            return

        old_state = self.predictor.state_dict()
        self.predictor = self.predictor_init(self.mask_decoder_cfg, model_args=model_args)
        incompatible = self.predictor.load_state_dict(old_state, strict=False)
        if target_use_csqr:
            print('[SET++] enabled CSQR block on predictor (loaded compatible weights, strict=False)')
        else:
            print('[SET++] disabled CSQR block on predictor (dropped csqr_block weights)')
        if incompatible.missing_keys:
            print(f"[SET++] predictor missing keys after rebuild: {incompatible.missing_keys[:8]}")
        if incompatible.unexpected_keys:
            print(f"[SET++] predictor unexpected keys after rebuild: {incompatible.unexpected_keys[:8]}")


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

    def get_special_token(self, SEG, SET, EOS):
        self.SEG_id = SEG
        self.SET_id = SET
        self.EOS_id = EOS

    def embed_refer_ids(self, refer_ids):
        if refer_ids is None:
            return None
        embedded_refer = self.get_model().embed_tokens(refer_ids)
        return embedded_refer

    def concat_image_seg_cls_embeds(self, input_id, img_feature, label, SEG_token_embedding_indices=None, SET_token_embedding_indices=None, refer_embedding=None):
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
        cur_SET_token_embedding_indices = [] if SET_token_embedding_indices is not None else None
        cur_refer_span_indices = []
        track_refer_span = SEG_token_embedding_indices is not None

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
                if SET_token_embedding_indices is not None:
                    cur_SET_token_embedding_indices.append(torch.full((img_feature.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if track_refer_span:
                    cur_refer_span_indices.append(torch.zeros(img_feature.shape[0], device=input_id.device, dtype=torch.bool))
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
                if SET_token_embedding_indices is not None:
                    cur_SET_token_embedding_indices.append(
                        torch.full((refer_embed.shape[0],), 0, device=input_id.device,
                                   dtype=input_id.dtype))
                if track_refer_span:
                    cur_refer_span_indices.append(torch.ones(refer_embed.shape[0], device=input_id.device, dtype=torch.bool))
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
                if SET_token_embedding_indices is not None:
                    cur_SET_token_embedding_indices.append(SET_token_embedding_indices[:chunk_len])
                if track_refer_span:
                    cur_refer_span_indices.append(torch.zeros(chunk_len, device=input_id.device, dtype=torch.bool))
                if label is not None:
                    cur_new_label.append(label[:chunk_len])

            input_id = input_id[chunk_len:]

            if SEG_token_embedding_indices is not None:
                SEG_token_embedding_indices = SEG_token_embedding_indices[chunk_len:]
            if SET_token_embedding_indices is not None:
                SET_token_embedding_indices = SET_token_embedding_indices[chunk_len:]
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

        if SET_token_embedding_indices is not None:
            cur_SET_token_embedding_indices = [x.to(device=self.device) for x in cur_SET_token_embedding_indices]
            cur_SET_token_embedding_indices = torch.cat(cur_SET_token_embedding_indices, dim=0)

        cur_refer_span_mask = None
        if track_refer_span:
            cur_refer_span_indices = [x.to(device=self.device) for x in cur_refer_span_indices]
            cur_refer_span_mask = torch.cat(cur_refer_span_indices, dim=0).bool()

        if image_features_indices:
            image_features_indices = [x.to(device=self.device) for x in image_features_indices]
            image_features_indices = torch.cat(image_features_indices, dim=0)

        return cur_new_input_embeds, cur_new_label, cur_SEG_token_embedding_indices, cur_SET_token_embedding_indices, image_features_indices, cur_refer_span_mask

    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images, token_refer_id=None, SEG_token_embedding_indices=None, SET_token_embedding_indices=None):

        vision_tower = self.get_vision_tower()
        
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, None, None, None

        image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        new_image_features_indices = []
        new_refer_span_mask = []
        
        new_SEG_token_embedding_indices = [] if SEG_token_embedding_indices is not None else None
        new_SET_token_embedding_indices = [] if SET_token_embedding_indices is not None else None
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_image_feature = image_features[batch_idx]
            
            cur_SEG_token_embedding_indices = SEG_token_embedding_indices[batch_idx] if SEG_token_embedding_indices is not None else None
            cur_SET_token_embedding_indices = SET_token_embedding_indices[batch_idx] if SET_token_embedding_indices is not None else None
            
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

            cur_input_embeds, cur_label, cur_SEG_token_embedding_indices, cur_SET_token_embedding_indices, cur_image_features_indices, cur_refer_span_mask = self.concat_image_seg_cls_embeds(
                input_id=cur_input_ids,
                img_feature=cur_image_feature,
                label=cur_label,
                SEG_token_embedding_indices=cur_SEG_token_embedding_indices,
                SET_token_embedding_indices=cur_SET_token_embedding_indices,
                refer_embedding=cur_refer_embedding
            )

            new_input_embeds.append(cur_input_embeds)
            if labels is not None:
                new_labels.append(cur_label)

            if SEG_token_embedding_indices is not None:
                new_SEG_token_embedding_indices.append(cur_SEG_token_embedding_indices)

            if SET_token_embedding_indices is not None:
                new_SET_token_embedding_indices.append(cur_SET_token_embedding_indices)

            if new_image_features_indices is not None:
                new_image_features_indices.append(cur_image_features_indices)
            new_refer_span_mask.append(cur_refer_span_mask)
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
            
            if SET_token_embedding_indices is not None:
                new_SET_token_embedding_indices_align = []
                for new_SET_token_embedding_indice in new_SET_token_embedding_indices:
                    new_SET_token_embedding_indice = torch.cat(
                        (new_SET_token_embedding_indice,
                         torch.zeros((max_len - new_SET_token_embedding_indice.shape[0]),dtype=new_SET_token_embedding_indice.dtype, device=new_SET_token_embedding_indice.device)),
                        dim=0)
                    new_SET_token_embedding_indices_align.append(new_SET_token_embedding_indice)
                new_SET_token_embedding_indices = torch.stack(new_SET_token_embedding_indices_align, dim=0)
            
            if new_image_features_indices is not None:
                new_image_features_indices_align = []
                for new_image_features_indice in new_image_features_indices:
                    new_image_features_indice = torch.cat(
                        (new_image_features_indice,
                         torch.zeros((max_len - new_image_features_indice.shape[0]),dtype=new_image_features_indice.dtype, device=new_image_features_indice.device)),
                        dim=0)
                    new_image_features_indices_align.append(new_image_features_indice)
                new_image_features_indices = torch.stack(new_image_features_indices_align, dim=0)

            if new_refer_span_mask:
                new_refer_span_mask_align = []
                for cur_refer_mask in new_refer_span_mask:
                    pad_len = max_len - cur_refer_mask.shape[0]
                    if pad_len > 0:
                        cur_refer_mask = torch.cat(
                            (
                                cur_refer_mask,
                                torch.zeros(
                                    pad_len,
                                    dtype=cur_refer_mask.dtype,
                                    device=cur_refer_mask.device,
                                ),
                            ),
                            dim=0,
                        )
                    new_refer_span_mask_align.append(cur_refer_mask)
                new_refer_span_mask = torch.stack(new_refer_span_mask_align, dim=0)

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

            if SET_token_embedding_indices is not None:
                new_SET_token_embedding_indices = torch.stack(new_SET_token_embedding_indices, dim=0)

            if new_image_features_indices is not None:
                new_image_features_indices = torch.stack(new_image_features_indices, dim=0)

            if new_refer_span_mask:
                new_refer_span_mask = torch.stack(new_refer_span_mask, dim=0)
            
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
   
        return None, attention_mask, past_key_values, new_input_embeds, new_labels, new_SEG_token_embedding_indices, new_SET_token_embedding_indices, new_image_features_indices, new_refer_span_mask
    
    def get_SEG_embedding(self, hidden_states, SEG_embedding_indices):
        assert SEG_embedding_indices is not None, "[BUG] SEG_embedding_indices is None in get_SEG_embedding"
        assert (SEG_embedding_indices == 1).sum() > 0, "[BUG] SEG_embedding_indices contains no 1s -- [SEG] token not found in sequence"
        SEG_embedding_list = []
        for current_hidden_state, current_token_indice in zip(hidden_states, SEG_embedding_indices):
            current_refer_state = current_hidden_state[current_token_indice.bool()]
            assert current_refer_state.shape[0] > 0, f"[BUG] No SEG token found in this sample. SEG_embedding_indices sum: {current_token_indice.sum().item()}"
            SEG_embedding_list.append(current_refer_state)
        return torch.cat(SEG_embedding_list, dim=0).unsqueeze(1)

    def get_SET_embedding(self, hidden_states, SET_embedding_indices):
        assert SET_embedding_indices is not None, "[BUG] SET_embedding_indices is None in get_SET_embedding -- [SET] token not registered or not passed through"
        assert (SET_embedding_indices == 1).sum() > 0, "[BUG] SET_embedding_indices contains no 1s -- [SET] token not found in sequence"
        SET_embedding_list = []
        for current_hidden_state, current_token_indice in zip(hidden_states, SET_embedding_indices):
            current_set_state = current_hidden_state[current_token_indice.bool()]
            assert current_set_state.shape[0] == 1, f"[BUG] Expected exactly 1 SET token per sample, got {current_set_state.shape[0]}. SET_embedding_indices sum: {current_token_indice.sum().item()}"
            SET_embedding_list.append(current_set_state)
        return torch.cat(SET_embedding_list, dim=0).unsqueeze(1)
           
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
            SET_token_embedding_indices=None,
            global_step=None,
            mask_num=None,
            dataset_type=None,) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if dataset_type is not None:
            assert all(item == dataset_type[0] for item in dataset_type), f'this batch contain different dataset_type: {dataset_type}'
            batch_dataset_type = dataset_type[0]
        else:
            batch_dataset_type = []
        enable_attention_loss = (
            self.training
            and seg_info is not None
            and bool(getattr(self.config, "enable_attention_loss", False))
        )
        if output_attentions is None:
            output_attentions = enable_attention_loss

        output_hidden_states = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        tg_swin_enabled = self._tg_swin_enabled()
        image_features = None
        bs = input_ids.shape[0]
        refer_span_mask = None

        per_target_swin = (
            tg_swin_enabled
            and mask_num is not None
            and seg_info is not None
            and (input_ids is None or input_ids.shape[1] != 1)
            and getattr(self._get_tg_swin_cfg(), "PER_TARGET_SWING_REPEAT", True)
        )

        if (SEG_token_embedding_indices == 1).sum() != 0:

            # for generative mode only the 1th stage need
            if input_ids.shape[1] != 1 and not per_target_swin:
                image_features = self.get_vision_tower_feature(images)
                bs = input_ids.shape[0]
            
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, SEG_token_embedding_indices, SET_token_embedding_indices, image_features_indices, refer_span_mask = self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images_clip,
                token_refer_id=token_refer_id, SEG_token_embedding_indices=SEG_token_embedding_indices,
                SET_token_embedding_indices=SET_token_embedding_indices)

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
        if enable_attention_loss and outputs.attentions is not None:
            attentions = [attention_item.sum(dim=1) for attention_item in outputs.attentions]
        else:
            attentions = None

        SET_embedding_b = self.SET_token_projector(self.get_SET_embedding(hidden_states, SET_token_embedding_indices))
        per_target_mode = False
        target_to_image = None

        if per_target_swin:
            seg_hidden, text_cond, reliability = self.build_text_cond(
                hidden_states, SEG_token_embedding_indices, refer_span_mask=refer_span_mask
            )
            SEG_embedding = self.SEG_token_projector(seg_hidden.unsqueeze(1))
            n_target = SEG_embedding.shape[0]
            mask_num_tensor = torch.tensor(mask_num, device=hidden_states.device)
            assert n_target == int(mask_num_tensor.sum().item()), (
                f"TG_SWIN alignment: SEG targets={n_target} != sum(mask_num)={int(mask_num_tensor.sum().item())}"
            )
            target_to_image = self._build_target_to_image(mask_num, hidden_states.device)
            SET_embedding = self._repeat_set_embedding_per_target(SET_embedding_b, mask_num)
            set_control = self.build_set_control(SET_embedding)
            assert text_cond.shape[0] == n_target, (
                f"TG_SWIN alignment: text_cond batch={text_cond.shape[0]} != n_target={n_target}"
            )
            if set_control is not None:
                assert set_control.shape[0] == n_target, (
                    f"TG_SWIN alignment: set_control batch={set_control.shape[0]} != n_target={n_target}"
                )
            per_target_mode = True
            coarse_evidence = None
            if self._use_coarse_evidence():
                coarse_evidence = self.get_shared_coarse_evidence(
                    images, SEG_embedding, mask_num, set_embedding_b=SET_embedding_b
                )
            images_expanded = self._repeat_images_per_target(images, mask_num)
            image_features = self.get_vision_tower_feature(
                images_expanded,
                text_cond=text_cond,
                reliability=reliability,
                set_control=set_control,
                coarse_evidence=coarse_evidence,
                enable_tg_swin=True,
            )
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
                image_features)
        else:
            SEG_embedding = self.SEG_token_projector(self.get_SEG_embedding(hidden_states, SEG_token_embedding_indices))
            SET_embedding = SET_embedding_b
            if image_features is None:
                image_features = self.get_vision_tower_feature(images)
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
                image_features)
            mask_num_tensor = torch.tensor(mask_num, device=mask_features.device)
            mask_features = torch.repeat_interleave(mask_features, repeats=mask_num_tensor, dim=0)
            multi_scale_features = [
                torch.repeat_interleave(feat, repeats=mask_num_tensor, dim=0)
                for feat in multi_scale_features
            ]

        mask_outputs = self.predictor(multi_scale_features, mask_features, None, None,
                                      SEG_embedding=SEG_embedding,
                                      SET_embedding=SET_embedding,
                                      mask_num=mask_num if not per_target_mode else None,
                                      per_target_mode=per_target_mode)
        if per_target_mode:
            mask_outputs["per_target_mode"] = True
            mask_outputs["target_to_image"] = target_to_image
            mask_outputs["mask_num"] = list(mask_num)

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
        loss_mask = torch.tensor(0.0, device=hidden_states.device)
        loss_dice = torch.tensor(0.0, device=hidden_states.device)
        loss_union = torch.tensor(0.0, device=hidden_states.device)
        loss_setpp_coverage = torch.tensor(0.0, device=hidden_states.device)
        loss_setpp_consistency = torch.tensor(0.0, device=hidden_states.device)
        if seg_info is not None:
            if 'padding_mask' in seg_info[0]:
                if isinstance(seg_info[0]["instances"], list):
                    gt_instances = [x["instances"][0].to(self.device) for x in seg_info]
                else:
                    gt_instances = [x["instances"].to(self.device) for x in seg_info]

                targets = self.prepare_targets(gt_instances, images)
            elif 'mask' in seg_info[0]:
                if per_target_mode:
                    targets = self._build_flat_seg_targets(
                        seg_info, mask_num, mask_outputs['pred_masks'].device, mask_outputs['pred_masks']
                    )
                else:
                    targets = []
                    # Regroup flat per-mask seg_info into per-image targets using mask_num
                    offset = 0
                    for k in mask_num:
                        image_masks = []
                        for j in range(k):
                            raw_mask = seg_info[offset + j]['mask'].to(mask_outputs['pred_masks'].device)
                            # Dataset stores masks as [1, H, W]; squeeze to [H, W]
                            if raw_mask.ndim == 3 and raw_mask.shape[0] == 1:
                                raw_mask = raw_mask.squeeze(0)
                            elif raw_mask.ndim == 3:
                                raw_mask = raw_mask.squeeze(0)  # fallback: squeeze first dim
                            image_masks.append(raw_mask)
                        offset += k
                        if len(image_masks) > 0:
                            stacked_masks = torch.stack(image_masks, dim=0)  # [K, H, W]
                            assert stacked_masks.ndim == 3, \
                                f"[BUG] target masks should be [K, H, W], got {stacked_masks.shape}"
                        else:
                            stacked_masks = torch.zeros(0, *mask_outputs['pred_masks'].shape[-2:],
                                                        device=mask_outputs['pred_masks'].device)
                        targets.append(
                            {
                                'labels': torch.zeros(k, dtype=torch.long, device=mask_outputs['pred_masks'].device),
                                'masks': stacked_masks,
                                'valid': None,
                                'inst_id': None
                            }
                        )
            else:
                targets = None
            # Warmup: set global step for union loss warmup
            if global_step is not None:
                self.criterion.set_global_step(global_step)

            mask_losses = self.criterion(mask_outputs, targets)
            weight_dict = self.weight_dict

            loss_mask = torch.tensor(0.0, device=hidden_states.device)
            loss_dice = torch.tensor(0.0, device=hidden_states.device)
            loss_union = torch.tensor(0.0, device=hidden_states.device)
            loss_setpp_coverage = torch.tensor(0.0, device=hidden_states.device)
            loss_setpp_consistency = torch.tensor(0.0, device=hidden_states.device)
        
            for k in list(mask_losses.keys()):
                if k in weight_dict:
                    if mask_losses[k] is not None:
                        mask_losses[k] *= weight_dict[k]
                    
                    if 'coverage' in k:
                        loss_setpp_coverage += mask_losses[k]
                    elif 'consistency' in k:
                        loss_setpp_consistency += mask_losses[k]
                    elif 'union' in k:
                        loss_union += mask_losses[k]
                    elif '_mask' in k:
                        loss_mask += mask_losses[k]
                    elif '_dice' in k:
                        loss_dice += mask_losses[k]
                else:
                    mask_losses.pop(k)
            mask_loss = loss_mask + loss_dice + loss_union + loss_setpp_coverage + loss_setpp_consistency

        loss_attention = None
        if enable_attention_loss and attentions is not None and seg_info is not None:
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
            loss_attention = torch.tensor(0.0, device=hidden_states.device)

        loss = llm_loss + mask_loss + 0.01 * loss_attention

        return CausalOutputWithMask(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_mask=loss_mask.detach(),
            loss_dice=loss_dice.detach(),
            loss_llm=llm_loss.detach(),
            loss_attention=0.01 * loss_attention.detach(),
            loss_union=loss_union.detach(),
            loss_setpp_coverage=loss_setpp_coverage.detach(),
            loss_setpp_consistency=loss_setpp_consistency.detach(),
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
            SET_token_embedding_indices=None,
            mask_num = None):
        
        output_attentions = False
        output_hidden_states = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        tg_swin_enabled = self._tg_swin_enabled()
        per_target_swin = (
            tg_swin_enabled
            and mask_num is not None
            and getattr(self._get_tg_swin_cfg(), "PER_TARGET_SWING_REPEAT", True)
        )

        if not per_target_swin:
            image_features = self.get_vision_tower_feature(images)

        input_ids, attention_mask, past_key_values, inputs_embeds, labels, SEG_token_embedding_indices, SET_token_embedding_indices, image_features_indices, refer_span_mask = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images_clip,
            token_refer_id=token_refer_id, SEG_token_embedding_indices=SEG_token_embedding_indices,
            SET_token_embedding_indices=SET_token_embedding_indices)
    
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
        SET_embedding_b = self.SET_token_projector(self.get_SET_embedding(hidden_states, SET_token_embedding_indices))
        per_target_mode = False

        if per_target_swin:
            seg_hidden, text_cond, reliability = self.build_text_cond(
                hidden_states, SEG_token_embedding_indices, refer_span_mask=refer_span_mask
            )
            SEG_embedding = self.SEG_token_projector(seg_hidden.unsqueeze(1))
            n_target = SEG_embedding.shape[0]
            SET_embedding = self._repeat_set_embedding_per_target(SET_embedding_b, mask_num)
            set_control = self.build_set_control(SET_embedding)
            assert text_cond.shape[0] == n_target, (
                f"TG_SWIN alignment: text_cond batch={text_cond.shape[0]} != n_target={n_target}"
            )
            if set_control is not None:
                assert set_control.shape[0] == n_target, (
                    f"TG_SWIN alignment: set_control batch={set_control.shape[0]} != n_target={n_target}"
                )
            per_target_mode = True
            coarse_evidence = None
            if self._use_coarse_evidence():
                coarse_evidence = self.get_shared_coarse_evidence(
                    images, SEG_embedding, mask_num, set_embedding_b=SET_embedding_b
                )
            images_expanded = self._repeat_images_per_target(images, mask_num)
            image_features = self.get_vision_tower_feature(
                images_expanded,
                text_cond=text_cond,
                reliability=reliability,
                set_control=set_control,
                coarse_evidence=coarse_evidence,
                enable_tg_swin=True,
            )
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
                image_features)
            images = [s[0] for s in torch.split(images_expanded, 1, dim=0)]
        else:
            SEG_embedding = self.SEG_token_projector(self.get_SEG_embedding(hidden_states, SEG_token_embedding_indices))
            SET_embedding = SET_embedding_b
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
                image_features)
            images = [image.repeat((num, 1, 1, 1)) for image, num in zip(images, mask_num)]
            images = [s[0] for image_repeat in images for s in torch.split(image_repeat, 1, dim=0)]
            mask_num_tensor = torch.tensor(mask_num, device=mask_features.device)
            mask_features = torch.repeat_interleave(mask_features, repeats=mask_num_tensor, dim=0)
            multi_scale_features = [
                torch.repeat_interleave(feat, repeats=mask_num_tensor, dim=0)
                for feat in multi_scale_features
            ]

        mask_outputs = self.predictor(multi_scale_features, mask_features, None, None,
                                      SEG_embedding=SEG_embedding,
                                      SET_embedding=SET_embedding,
                                      mask_num=mask_num if not per_target_mode else None,
                                      per_target_mode=per_target_mode)

        mask_pred_results = mask_outputs["pred_masks"]
        if per_target_mode:
            if isinstance(images, torch.Tensor):
                images_list = [images[i] for i in range(images.shape[0])]
            else:
                images_list = list(images)
            image_list = ImageList.from_tensors(images_list, self.size_divisibility)
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(image_list.tensor.shape[-2], image_list.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            offsets = [0] + list(torch.tensor(mask_num).cumsum(0).tolist())
            union_masks = []
            for b in range(len(mask_num)):
                row_slice = slice(offsets[b], offsets[b + 1])
                union_masks.append(mask_pred_results[row_slice, 0, :, :].max(dim=0).values)
            union_mask_info = {
                'union_preds': [(m.detach().float().cpu().numpy() > 0).astype(np.uint8) for m in union_masks],
                'image_ids': [seg_info[offsets[b]]['image_id'] for b in range(len(mask_num))],
                'data_ids': [seg_info[offsets[b]]['data_id'] for b in range(len(mask_num))],
            }
            mask_pred_results = mask_pred_results[:, 1, :, :]
            processed_results = []
            for _seg_info, mask_pred_result in zip(seg_info, mask_pred_results):
                instance_r = {
                    'pred': ((mask_pred_result.detach().float().cpu().numpy() > 0) * 255).astype(np.uint8),
                    'image_name': _seg_info['image_id'],
                    'id': _seg_info['data_id'],
                    'mask_id': _seg_info['mask_id'],
                }
                processed_results.append(instance_r)
            return processed_results, union_mask_info

        if isinstance(images, torch.Tensor):
            images_list = [images[b] for b in range(images.shape[0])]
        else:
            images_list = list(images)
        image_list = ImageList.from_tensors(images_list, self.size_divisibility)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(image_list.tensor.shape[-2], image_list.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        
        # Extract union mask (query 0) for diagnostic metrics
        union_masks = mask_pred_results[:, 0, :, :]  # [B, H, W]
        # seg_info is flat per-mask; extract per-image image_id/data_id using mask_num
        offsets = [0] + list(torch.tensor(mask_num).cumsum(0).tolist())
        per_image_seg_info = seg_info  # keep flat, but use offsets to index
        union_mask_info = {
            'union_preds': [(m.detach().float().cpu().numpy() > 0).astype(np.uint8) for m in union_masks],
            'image_ids': [per_image_seg_info[offsets[b]]['image_id'] for b in range(len(mask_num))],
            'data_ids': [per_image_seg_info[offsets[b]]['data_id'] for b in range(len(mask_num))],
        }
        
        # Strip query 0 (SET union mask) and remove padded SEG queries
        # [B, 1+Kmax, H, W] -> [B, Kmax, H, W] -> flatten to [sum(K_i), H, W]
        instance_masks = mask_pred_results[:, 1:, :, :]  # [B, Kmax, H, W]
        flattened_masks = []
        for b_idx, k in enumerate(mask_num):
            flattened_masks.append(instance_masks[b_idx, :k, :, :])
        mask_pred_results = torch.cat(flattened_masks, dim=0)  # [sum(K_i), H, W]
        
        processed_results = []
        for _seg_info, mask_pred_result in zip(seg_info, mask_pred_results):
            instance_r = {
                'pred': ((mask_pred_result.detach().float().cpu().numpy() > 0) * 255).astype(np.uint8),
                'image_name': _seg_info['image_id'],
                'id': _seg_info['data_id'],
                'mask_id': _seg_info['mask_id'],
            }
            processed_results.append(instance_r)
        return processed_results, union_mask_info