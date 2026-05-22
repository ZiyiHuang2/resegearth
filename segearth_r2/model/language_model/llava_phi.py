import os
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
    loss_attn_fg_bg: Optional[torch.FloatTensor] = None
    loss_attn_boundary_outer: Optional[torch.FloatTensor] = None
    precomputed_structured_count: Optional[torch.FloatTensor] = None
    fallback_structured_count: Optional[torch.FloatTensor] = None
    missing_precomputed_count: Optional[torch.FloatTensor] = None
    selected_attn_layers: Optional[torch.FloatTensor] = None
    structured_effective_weight: Optional[torch.FloatTensor] = None
    structured_loss_small: Optional[torch.FloatTensor] = None
    structured_loss_non_small: Optional[torch.FloatTensor] = None
    structured_small_count: Optional[torch.FloatTensor] = None
    structured_non_small_count: Optional[torch.FloatTensor] = None
    structured_skipped_batches: Optional[torch.FloatTensor] = None
    structured_skip_ratio: Optional[torch.FloatTensor] = None
    loss_attn_fg_bg_small: Optional[torch.FloatTensor] = None
    loss_attn_fg_bg_non_small: Optional[torch.FloatTensor] = None
    loss_attn_boundary_outer_small: Optional[torch.FloatTensor] = None
    loss_attn_boundary_outer_non_small: Optional[torch.FloatTensor] = None
    selected_attn_heads_total: Optional[torch.FloatTensor] = None
    selected_attn_heads_per_layer: Optional[torch.FloatTensor] = None
    structured_schedule_phase_id: Optional[torch.FloatTensor] = None
    structured_skip_reason_code: Optional[torch.FloatTensor] = None
    attention_audit: Optional[dict] = None
    # --- structured output-effect diagnostics (optional; see diagnose_structured_effect) ---
    structured_output_effect_mean: Optional[torch.FloatTensor] = None
    structured_output_effect_nonzero_ratio: Optional[torch.FloatTensor] = None
    structured_output_effect_gt_threshold_ratio: Optional[torch.FloatTensor] = None
    structured_output_effect_mean_small: Optional[torch.FloatTensor] = None
    structured_output_effect_mean_non_small: Optional[torch.FloatTensor] = None
    structured_output_effect_nonzero_ratio_small: Optional[torch.FloatTensor] = None
    structured_output_effect_nonzero_ratio_non_small: Optional[torch.FloatTensor] = None
    diag_mask_logits_mean_abs_diff: Optional[torch.FloatTensor] = None
    diag_binary_mask_disagree_ratio: Optional[torch.FloatTensor] = None
    diag_pred_mask_iou_between_runs: Optional[torch.FloatTensor] = None
    diag_pred_area_change_ratio: Optional[torch.FloatTensor] = None
    diag_iou_with_gt_run_a: Optional[torch.FloatTensor] = None
    diag_iou_with_gt_run_b: Optional[torch.FloatTensor] = None
    diag_delta_iou_vs_gt: Optional[torch.FloatTensor] = None
    diag_attention_sup_default_minus_unstructured: Optional[torch.FloatTensor] = None
    diag_structured_supervision_alters_mask_forward_path: Optional[torch.FloatTensor] = None
    # --- seg query refiner (top-k attention -> MLP -> fixed 0.5/0.5 blend) ---
    seg_query_delta_norm: Optional[torch.FloatTensor] = None
    seg_query_original_norm: Optional[torch.FloatTensor] = None
    seg_query_refined_norm: Optional[torch.FloatTensor] = None
    seg_query_cosine_q_delta: Optional[torch.FloatTensor] = None
    seg_query_cosine_q_qnew: Optional[torch.FloatTensor] = None
    # --- SEG explicit coarse spatial prior (q_sem / q_loc -> prior -> loc_code -> q_final) ---
    loss_seg_loc_prior: Optional[torch.FloatTensor] = None
    seg_loc_prior_mean: Optional[torch.FloatTensor] = None
    seg_loc_prior_entropy: Optional[torch.FloatTensor] = None
    seg_loc_prior_fg_mass: Optional[torch.FloatTensor] = None
    seg_loc_prior_bg_mass: Optional[torch.FloatTensor] = None
    seg_loc_code_norm: Optional[torch.FloatTensor] = None
    seg_loc_sem_norm: Optional[torch.FloatTensor] = None
    seg_loc_qfinal_norm: Optional[torch.FloatTensor] = None
    seg_loc_code_to_sem_norm_ratio: Optional[torch.FloatTensor] = None
    seg_loc_cosine_sem_qfinal: Optional[torch.FloatTensor] = None
    seg_loc_cosine_sem_loccode: Optional[torch.FloatTensor] = None
    # Per-batch dict of extra seg_loc prior diagnostics (percentiles, ratios, coarse IoU); trainer flattens to log.
    seg_loc_prior_extended_metrics: Optional[dict] = None
    # Optional list[dict] per SEG instance for jsonl export; trainer appends when enabled.
    seg_loc_prior_sample_rows: Optional[list] = None

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


class StructuredAttentionLoss(nn.Module):
    def __init__(
        self,
        reduction='batchmean',
        fg_bg_weight: float = 1.0,
        boundary_outer_weight: float = 1.0,
        margin: float = 0.0,
    ):
        super(StructuredAttentionLoss, self).__init__()
        self.reduction = reduction
        self.fg_bg_weight = fg_bg_weight
        self.boundary_outer_weight = boundary_outer_weight
        self.margin = margin

    def _reduce(self, loss: torch.Tensor, batch_size: int, numel: int) -> torch.Tensor:
        if self.reduction == 'batchmean':
            return loss / max(batch_size, 1)
        if self.reduction == 'mean':
            return loss / max(numel, 1)
        return loss

    def forward(
        self,
        model_attention_logits: torch.Tensor,
        fg_mask: torch.Tensor,
        boundary_map: Optional[torch.Tensor] = None,
        outer_ring_map: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        device = model_attention_logits.device
        loss = torch.tensor(0.0, device=device)
        loss_fg_bg = torch.tensor(0.0, device=device)
        loss_boundary_outer = torch.tensor(0.0, device=device)
        for idx in range(model_attention_logits.shape[0]):
            attn = model_attention_logits[idx]
            fg = fg_mask[idx] > 0
            bg = ~fg
            if fg.sum() == 0 or bg.sum() == 0:
                continue
            fg_mean = attn[fg].mean()
            bg_mean = attn[bg].mean()
            fg_bg_penalty = F.relu(self.margin - (fg_mean - bg_mean))
            sample_loss_fg_bg = self.fg_bg_weight * fg_bg_penalty
            sample_loss_boundary_outer = torch.tensor(0.0, device=device)

            if boundary_map is not None and outer_ring_map is not None:
                boundary = boundary_map[idx] > 0
                outer = outer_ring_map[idx] > 0
                if boundary.sum() > 0 and outer.sum() > 0:
                    boundary_mean = attn[boundary].mean()
                    outer_mean = attn[outer].mean()
                    bo_penalty = F.relu(self.margin - (boundary_mean - outer_mean))
                    sample_loss_boundary_outer = self.boundary_outer_weight * bo_penalty
            sample_loss = sample_loss_fg_bg + sample_loss_boundary_outer
            loss = loss + sample_loss
            loss_fg_bg = loss_fg_bg + sample_loss_fg_bg
            loss_boundary_outer = loss_boundary_outer + sample_loss_boundary_outer
        loss = self._reduce(loss, model_attention_logits.shape[0], model_attention_logits.numel())
        if not return_components:
            return loss
        loss_fg_bg = self._reduce(loss_fg_bg, model_attention_logits.shape[0], model_attention_logits.numel())
        loss_boundary_outer = self._reduce(loss_boundary_outer, model_attention_logits.shape[0], model_attention_logits.numel())
        return loss, loss_fg_bg, loss_boundary_outer

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
    def __init__(self, config, model_args=None, mask_decoder_cfg=None, add_cross_attn=True, cross_attn_index=None):
        super(SegEarthR2, self).__init__(config)

        self.model = SegEarthR2Model(config, mask_decoder_cfg)
        self.init_config = config
        self.mask_decoder_cfg = mask_decoder_cfg
        self.cross_attn_index = cross_attn_index

        self.lm_head = nn.Linear(config.hidden_size, 51200, bias=False)

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
        self.structured_attention_loss = StructuredAttentionLoss()
        
        self.test_topk_per_image = self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        input_shape = self.output_shape()
        self.pixel_decoder = self.pixel_decoder_init(cfg=self.mask_decoder_cfg, input_shape=input_shape)
        self.predictor = self.predictor_init(cfg=self.mask_decoder_cfg)

        self.SEG_token_projector = nn.Linear(self.config.hidden_size, self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM)

        self.seg_query_refiner_mlp = None
        if bool(getattr(self.config, "use_seg_query_refiner", False)):
            d = int(self.config.hidden_size)
            mid = int(getattr(self.config, "seg_query_refiner_hidden_dim", 512))
            self.seg_query_refiner_mlp = nn.Sequential(
                nn.Linear(d, mid),
                nn.GELU(),
                nn.Linear(mid, d),
            )
            print(f"[SEG][QueryRefiner] enabled: top_k={int(getattr(self.config, 'seg_query_topk_tokens', 16))}, mlp_hidden={mid}")

        self.seg_loc_w_sem = None
        self.seg_loc_w_loc = None
        self.seg_loc_prior_head = None
        self.seg_loc_code_linear = None
        self.seg_loc_fuse_linear = None
        if bool(getattr(self.config, "use_seg_loc_prior", False)):
            d = int(self.config.hidden_size)
            gh = int(getattr(self.config, "seg_loc_prior_grid_size", 14) or 14)
            gh = max(4, gh)
            g = gh * gh
            mid = int(getattr(self.config, "seg_loc_hidden_dim", 256) or 256)
            mid = max(32, mid)
            self.seg_loc_w_sem = nn.Linear(d, d, bias=True)
            self.seg_loc_w_loc = nn.Linear(d, d, bias=True)
            self.seg_loc_prior_head = nn.Sequential(
                nn.Linear(d, mid, bias=True),
                nn.ReLU(),
                nn.Linear(mid, g, bias=True),
            )
            self.seg_loc_code_linear = nn.Linear(g, d, bias=True)
            self.seg_loc_fuse_linear = nn.Linear(2 * d, d, bias=True)
            print(
                f"[SEG][LocPrior] enabled: grid={gh}x{gh} (G={g}), prior_mlp={d}->{mid}->{g}, "
                f"fuse=concat+Linear({2*d}->{d}), prior_weight={float(getattr(self.config, 'seg_loc_prior_weight', 0.1))}"
            )

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

    def get_vision_tower_feature(self, images):
        features = self.get_model().get_vision_tower_mask()(images)
        
        features_dict = {
            'res2': features[0], # bs, 128, 256, 256
            'res3': features[1], # bs, 256, 128, 128
            'res4': features[2], # bs, 512, 64, 64
            'res5': features[3], # bs, 1024, 32, 32
        }
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
                                                                     seg_fuse_score,)
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

    def _seg_query_refiner_enabled(self) -> bool:
        return bool(getattr(self.config, "use_seg_query_refiner", False)) and getattr(
            self, "seg_query_refiner_mlp", None
        ) is not None

    def ensure_seg_query_refiner_mlp(self):
        """Call after updating config on a loaded (e.g. merged) checkpoint so MLP exists when flag is turned on."""
        if not bool(getattr(self.config, "use_seg_query_refiner", False)):
            return
        if getattr(self, "seg_query_refiner_mlp", None) is not None:
            return
        d = int(self.config.hidden_size)
        mid = int(getattr(self.config, "seg_query_refiner_hidden_dim", 512))
        self.seg_query_refiner_mlp = nn.Sequential(
            nn.Linear(d, mid),
            nn.GELU(),
            nn.Linear(mid, d),
        )
        print(
            f"[SEG][QueryRefiner] lazy-init MLP (merged ckpt path): top_k="
            f"{int(getattr(self.config, 'seg_query_topk_tokens', 16))}, mlp_hidden={mid}"
        )

    def _seg_loc_prior_enabled(self) -> bool:
        return bool(getattr(self.config, "use_seg_loc_prior", False)) and getattr(
            self, "seg_loc_prior_head", None
        ) is not None and getattr(self, "seg_loc_fuse_linear", None) is not None

    def ensure_seg_loc_modules(self):
        if not bool(getattr(self.config, "use_seg_loc_prior", False)):
            return
        if getattr(self, "seg_loc_prior_head", None) is not None and getattr(self, "seg_loc_fuse_linear", None) is not None:
            return
        d = int(self.config.hidden_size)
        if getattr(self, "seg_loc_prior_head", None) is None:
            gh = int(getattr(self.config, "seg_loc_prior_grid_size", 14) or 14)
            gh = max(4, gh)
            g = gh * gh
            mid = int(getattr(self.config, "seg_loc_hidden_dim", 256) or 256)
            mid = max(32, mid)
            self.seg_loc_w_sem = nn.Linear(d, d, bias=True)
            self.seg_loc_w_loc = nn.Linear(d, d, bias=True)
            self.seg_loc_prior_head = nn.Sequential(
                nn.Linear(d, mid, bias=True),
                nn.ReLU(),
                nn.Linear(mid, g, bias=True),
            )
            self.seg_loc_code_linear = nn.Linear(g, d, bias=True)
            self.seg_loc_fuse_linear = nn.Linear(2 * d, d, bias=True)
            print(f"[SEG][LocPrior] lazy-init modules (merged ckpt path): grid={gh}x{gh}")
        else:
            self.seg_loc_fuse_linear = nn.Linear(2 * d, d, bias=True)
            print("[SEG][LocPrior] lazy-init seg_loc_fuse_linear only (older ckpt missing fuse)")

    def _gather_q_seg_matrix(self, hidden_states: torch.Tensor, SEG_token_embedding_indices: Optional[torch.Tensor]):
        if SEG_token_embedding_indices is None:
            return torch.empty(
                0,
                hidden_states.size(-1),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        rows: List[torch.Tensor] = []
        bs = hidden_states.size(0)
        for b in range(bs):
            h_b = hidden_states[b]
            seg_pos = torch.where(SEG_token_embedding_indices[b].bool())[0]
            for s in seg_pos:
                rows.append(h_b[int(s.item())])
        if len(rows) == 0:
            return torch.empty(
                0,
                hidden_states.size(-1),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        return torch.stack(rows, dim=0)

    def _seg_loc_prior_loss_and_metrics(
        self,
        prior_logits: torch.Tensor,
        targets,
        SEG_token_embedding_indices: torch.Tensor,
        seg_info=None,
        global_step=None,
    ):
        """BCE prior vs GT mask downsampled to coarse grid; no GT fed into q. Returns (loss, metrics, ext, sample_rows)."""
        gh = int(getattr(self.config, "seg_loc_prior_grid_size", 14) or 14)
        gh = max(4, gh)
        g = gh * gh
        device = prior_logits.device
        dtype = prior_logits.dtype
        rows: List[torch.Tensor] = []
        bs = SEG_token_embedding_indices.shape[0]
        for b in range(bs):
            seg_pos = torch.where(SEG_token_embedding_indices[b].bool())[0]
            n_seg = int(seg_pos.numel())
            masks_b = targets[b]["masks"]
            for j in range(n_seg):
                mj = min(j, int(masks_b.shape[0]) - 1)
                m = masks_b[mj].float()
                if m.dim() == 3:
                    m = m[0]
                down = F.interpolate(
                    m.unsqueeze(0).unsqueeze(0),
                    size=(gh, gh),
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0.0, 1.0)
                rows.append(down.reshape(-1))
        if len(rows) == 0:
            z = torch.tensor(0.0, device=device, dtype=dtype)
            empty_m = {
                "seg_loc_prior_mean": z.detach(),
                "seg_loc_prior_entropy": z.detach(),
                "seg_loc_prior_fg_mass": z.detach(),
                "seg_loc_prior_bg_mass": z.detach(),
            }
            return z, empty_m, {}, None
        tgt = torch.stack(rows, dim=0)
        if tgt.shape[0] != prior_logits.shape[0] or tgt.shape[1] != g:
            raise ValueError(
                f"seg loc prior shape mismatch: logits {tuple(prior_logits.shape)} vs target {tuple(tgt.shape)}"
            )
        loss = F.binary_cross_entropy_with_logits(prior_logits, tgt, reduction="mean")
        p_det = torch.sigmoid(prior_logits.detach())
        tgt_d = tgt.detach()
        eps = 1e-8
        fg_m = (tgt_d > 0.5).float()
        bg_m = 1.0 - fg_m
        fg_sum = fg_m.sum(dim=-1).clamp_min(1.0)
        bg_sum = bg_m.sum(dim=-1).clamp_min(1.0)
        fg_mass_per = (p_det * fg_m).sum(dim=-1) / fg_sum
        bg_mass_per = (p_det * bg_m).sum(dim=-1) / bg_sum
        ent_per = -(p_det * torch.log(p_det + eps) + (1.0 - p_det) * torch.log(1.0 - p_det + eps)).mean(dim=-1)
        mean_per = p_det.mean(dim=-1)
        prior_target_fg_ratio = fg_m.mean(dim=-1)

        p_bin = (p_det > 0.5).float()
        t_bin = (tgt_d > 0.5).float()
        inter = (p_bin * t_bin).sum(dim=-1)
        union = p_bin.sum(dim=-1) + t_bin.sum(dim=-1) - inter
        iou_per = inter / union.clamp_min(eps)
        top_idx = torch.argmax(p_det, dim=-1)
        top_in_fg = (torch.gather(t_bin, 1, top_idx.unsqueeze(1)).squeeze(1) > 0.5).float()

        ent_high_thr = float(getattr(self.config, "seg_loc_prior_entropy_high_thresh", 0.35) or 0.35)
        fg_low_thr = float(getattr(self.config, "seg_loc_prior_low_fg_mass_thresh", 0.05) or 0.05)

        fg_gt_bg_ratio = (fg_mass_per / (bg_mass_per + eps)).mean()
        fg_better_ratio = (fg_mass_per > bg_mass_per).float().mean()
        high_ent_ratio = (ent_per > ent_high_thr).float().mean()
        low_fg_ratio = (fg_mass_per < fg_low_thr).float().mean()

        qv = torch.tensor([0.25, 0.5, 0.75], device=device, dtype=torch.float32)

        def _pct(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            xf = x.float()
            return torch.quantile(xf, qv[0]), torch.quantile(xf, qv[1]), torch.quantile(xf, qv[2])

        fg_p25, fg_p50, fg_p75 = _pct(fg_mass_per)
        bg_p25, bg_p50, bg_p75 = _pct(bg_mass_per)
        en_p25, en_p50, en_p75 = _pct(ent_per)
        mn_p25, mn_p50, mn_p75 = _pct(mean_per)

        ent = ent_per.mean()
        fg_mass = fg_mass_per.mean()
        bg_mass = bg_mass_per.mean()

        ext = {
            "seg_loc_prior_fg_mass_p25": fg_p25.detach(),
            "seg_loc_prior_fg_mass_p50": fg_p50.detach(),
            "seg_loc_prior_fg_mass_p75": fg_p75.detach(),
            "seg_loc_prior_bg_mass_p25": bg_p25.detach(),
            "seg_loc_prior_bg_mass_p50": bg_p50.detach(),
            "seg_loc_prior_bg_mass_p75": bg_p75.detach(),
            "seg_loc_prior_entropy_p25": en_p25.detach(),
            "seg_loc_prior_entropy_p50": en_p50.detach(),
            "seg_loc_prior_entropy_p75": en_p75.detach(),
            "seg_loc_prior_mean_p25": mn_p25.detach(),
            "seg_loc_prior_mean_p50": mn_p50.detach(),
            "seg_loc_prior_mean_p75": mn_p75.detach(),
            "seg_loc_prior_fg_gt_bg_ratio": fg_gt_bg_ratio.detach(),
            "seg_loc_prior_fg_better_than_bg_ratio": fg_better_ratio.detach(),
            "seg_loc_prior_high_entropy_ratio": high_ent_ratio.detach(),
            "seg_loc_prior_low_fg_mass_ratio": low_fg_ratio.detach(),
            "seg_loc_prior_coarse_iou_mean": iou_per.mean().detach(),
            "seg_loc_prior_top1_in_fg_ratio": top_in_fg.mean().detach(),
        }

        metrics = {
            "seg_loc_prior_mean": p_det.mean().detach(),
            "seg_loc_prior_entropy": ent.detach(),
            "seg_loc_prior_fg_mass": fg_mass.detach(),
            "seg_loc_prior_bg_mass": bg_mass.detach(),
        }

        sample_rows = None
        if bool(getattr(self.config, "export_seg_loc_prior_samples", False)):
            small_th = float(getattr(self.config, "small_area_ratio_threshold", 0.01) or 0.01)
            n_inst = prior_logits.shape[0]
            sample_rows = []
            row_idx = 0
            for b in range(bs):
                seg_pos = torch.where(SEG_token_embedding_indices[b].bool())[0]
                n_seg = int(seg_pos.numel())
                si = {}
                if seg_info is not None and isinstance(seg_info, (list, tuple)) and b < len(seg_info):
                    raw = seg_info[b]
                    si = raw if isinstance(raw, dict) else {}
                for j in range(n_seg):
                    if row_idx >= n_inst:
                        break
                    image_id = str(si.get("image_id", si.get("image_name", "")))
                    data_id = str(si.get("data_id", si.get("id", "")))
                    mask_id = str(si.get("mask_id", ""))
                    sample_key = f"{image_id}_{data_id}_{mask_id}_seg{j}".strip("_")
                    ptf = float(prior_target_fg_ratio[row_idx].item())
                    is_small = int(ptf < small_th)
                    ti = int(top_idx[row_idx].item())
                    tr = ti // gh
                    tc = ti % gh
                    sample_rows.append(
                        {
                            "sample_key": sample_key,
                            "seg_idx": int(j),
                            "image_id": image_id,
                            "data_id": data_id,
                            "mask_id": mask_id,
                            "is_small": is_small,
                            "global_step": int(global_step) if global_step is not None else -1,
                            "seg_loc_prior_fg_mass": float(fg_mass_per[row_idx].item()),
                            "seg_loc_prior_bg_mass": float(bg_mass_per[row_idx].item()),
                            "seg_loc_prior_entropy": float(ent_per[row_idx].item()),
                            "seg_loc_prior_mean": float(mean_per[row_idx].item()),
                            "prior_target_fg_ratio": ptf,
                            "seg_loc_prior_coarse_iou": float(iou_per[row_idx].item()),
                            "seg_loc_prior_top1_in_fg": int(top_in_fg[row_idx].item()),
                            "seg_loc_prior_top1_row": tr,
                            "seg_loc_prior_top1_col": tc,
                        }
                    )
                    row_idx += 1
        return loss, metrics, ext, sample_rows

    def _compute_seg_embedding_for_mask(
        self,
        hidden_states: torch.Tensor,
        attentions: Optional[Tuple],
        SEG_token_embedding_indices: torch.Tensor,
        image_features_indices: torch.Tensor,
    ):
        """
        Returns (SEG_embedding_for_predictor, aux).

        aux is None, or dict with:
          - "loc_logits": [N, G] when use_seg_loc_prior (takes precedence over query refiner)
          - "loc_fuse_diag": dict of detached scalars for fusion strength logging
          - "query_diag": dict when use_seg_query_refiner (top-k attention blend path)
        """
        if self._seg_loc_prior_enabled():
            q_mat = self._gather_q_seg_matrix(hidden_states, SEG_token_embedding_indices)
            if q_mat.size(0) == 0:
                q_in = self.get_SEG_embedding(hidden_states, SEG_token_embedding_indices)
                return self.SEG_token_projector(q_in), None
            q_sem = self.seg_loc_w_sem(q_mat)
            q_loc = self.seg_loc_w_loc(q_mat)
            prior_logits = self.seg_loc_prior_head(q_loc)
            p = torch.sigmoid(prior_logits)
            loc_code = self.seg_loc_code_linear(p)
            q_fused_in = torch.cat([q_sem, loc_code], dim=-1)
            q_final = self.seg_loc_fuse_linear(q_fused_in)
            seg_emb = self.SEG_token_projector(q_final.unsqueeze(1))
            eps = 1e-8
            sem_n = q_sem.norm(dim=-1)
            loc_n = loc_code.norm(dim=-1)
            qf_n = q_final.norm(dim=-1)
            loc_fuse_diag = {
                "seg_loc_code_norm": loc_n.mean().detach(),
                "seg_loc_sem_norm": sem_n.mean().detach(),
                "seg_loc_qfinal_norm": qf_n.mean().detach(),
                "seg_loc_code_to_sem_norm_ratio": (loc_n / sem_n.clamp_min(eps)).mean().detach(),
                "seg_loc_cosine_sem_qfinal": F.cosine_similarity(q_sem, q_final, dim=-1, eps=eps).mean().detach(),
                "seg_loc_cosine_sem_loccode": F.cosine_similarity(q_sem, loc_code, dim=-1, eps=eps).mean().detach(),
            }
            return seg_emb, {"loc_logits": prior_logits, "loc_fuse_diag": loc_fuse_diag}

        if (
            self._seg_query_refiner_enabled()
            and attentions is not None
            and len(attentions) > 0
            and SEG_token_embedding_indices is not None
            and image_features_indices is not None
        ):
            top_k = int(getattr(self.config, "seg_query_topk_tokens", 16) or 16)
            top_k = max(1, top_k)
            last_attn = attentions[-1]
            bs = hidden_states.size(0)
            d_model = hidden_states.size(-1)
            device = hidden_states.device
            dtype = hidden_states.dtype

            q_rows: List[torch.Tensor] = []
            v_rows: List[torch.Tensor] = []

            for b in range(bs):
                h_b = hidden_states[b]
                seg_pos = torch.where(SEG_token_embedding_indices[b].bool())[0]
                img_pos = torch.where(image_features_indices[b].bool())[0]
                n_img = int(img_pos.numel())
                attn_b = last_attn[b]

                for s in seg_pos:
                    s_i = int(s.item())
                    q_row = h_b[s_i].to(dtype=dtype)
                    if n_img == 0:
                        v_row = torch.zeros(d_model, device=device, dtype=dtype)
                    else:
                        seg_to_img = attn_b[:, s_i, img_pos]
                        w = seg_to_img.mean(dim=0)
                        k_eff = min(top_k, n_img)
                        vals, rel_idx = torch.topk(w.float(), k=k_eff)
                        idx = img_pos[rel_idx]
                        w_top = torch.softmax(vals, dim=0).to(dtype=dtype)
                        vecs = h_b[idx]
                        v_row = (w_top.unsqueeze(-1) * vecs).sum(dim=0)
                    q_rows.append(q_row)
                    v_rows.append(v_row)

            if len(q_rows) == 0:
                q_in = self.get_SEG_embedding(hidden_states, SEG_token_embedding_indices)
                return self.SEG_token_projector(q_in), None

            q_mat = torch.stack(q_rows, dim=0)
            v_mat = torch.stack(v_rows, dim=0)
            delta_q = self.seg_query_refiner_mlp(v_mat)
            q_new = 0.5 * q_mat + 0.5 * delta_q
            seg_emb = self.SEG_token_projector(q_new.unsqueeze(1))

            eps = 1e-6
            diag = {
                "seg_query_delta_norm": delta_q.norm(dim=-1).mean().detach(),
                "seg_query_original_norm": q_mat.norm(dim=-1).mean().detach(),
                "seg_query_refined_norm": q_new.norm(dim=-1).mean().detach(),
                "seg_query_cosine_q_delta": F.cosine_similarity(q_mat, delta_q, dim=-1, eps=eps).mean().detach(),
                "seg_query_cosine_q_qnew": F.cosine_similarity(q_mat, q_new, dim=-1, eps=eps).mean().detach(),
            }
            return seg_emb, {"query_diag": diag}

        q_in = self.get_SEG_embedding(hidden_states, SEG_token_embedding_indices)
        return self.SEG_token_projector(q_in), None

    def set_attention_loss_config(self):
        self.attention_loss = AttentionLoss(
            reduction=getattr(self.config, "attention_loss_reduction", "batchmean")
        )
        self.structured_attention_loss = StructuredAttentionLoss(
            reduction=getattr(self.config, "attention_loss_reduction", "batchmean"),
            fg_bg_weight=getattr(self.config, "structured_fg_bg_weight", 1.0),
            boundary_outer_weight=getattr(self.config, "structured_boundary_outer_weight", 1.0),
            margin=getattr(self.config, "structured_attention_margin", 0.0),
        )

    def _build_structured_maps(self, fg_mask_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fg = (fg_mask_2d > 0).float().unsqueeze(1)
        dilated = (F.max_pool2d(fg, kernel_size=3, stride=1, padding=1) > 0).float()
        eroded = (1.0 - F.max_pool2d(1.0 - fg, kernel_size=3, stride=1, padding=1)).clamp(min=0.0, max=1.0)
        boundary = (fg - eroded).clamp(min=0.0, max=1.0).squeeze(1)
        outer_ring = (dilated - fg).clamp(min=0.0, max=1.0).squeeze(1)
        return boundary, outer_ring

    def _resize_binary_map(self, map_tensor, target_hw, device, mode="nearest"):
        if map_tensor is None:
            return None
        if not torch.is_tensor(map_tensor):
            map_tensor = torch.as_tensor(map_tensor)
        map_tensor = map_tensor.to(device=device, dtype=torch.float32)
        if map_tensor.ndim == 2:
            map_tensor = map_tensor.unsqueeze(0).unsqueeze(0)
        elif map_tensor.ndim == 3:
            if map_tensor.shape[0] == 1:
                map_tensor = map_tensor.unsqueeze(0)
            elif map_tensor.shape[-1] == 1:
                map_tensor = map_tensor.permute(2, 0, 1).unsqueeze(0)
            else:
                map_tensor = map_tensor[:1].unsqueeze(0)
        elif map_tensor.ndim == 4:
            pass
        else:
            return None
        if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            resized = F.interpolate(map_tensor, size=target_hw, mode=mode, align_corners=False)
        else:
            resized = F.interpolate(map_tensor, size=target_hw, mode=mode)
        resized = resized.squeeze(0).squeeze(0)
        return (resized > 0).float()

    def _prepare_attention_supervision(self, seg_info, target_hw=(27, 27), device=None):
        if seg_info is None or len(seg_info) == 0:
            return None, None, None, {
                "precomputed_structured_count": 0.0,
                "fallback_structured_count": 0.0,
                "missing_precomputed_count": 0.0,
            }
        if device is None:
            device = self.device
        use_precomputed_maps = getattr(self.config, "use_precomputed_structured_maps", False)
        h, w = target_hw
        fg_masks = []
        boundary_maps = []
        outer_maps = []
        precomputed_structured_count = 0.0
        fallback_structured_count = 0.0
        missing_precomputed_count = 0.0
        for item in seg_info:
            has_precomputed_triplet = (
                item.get("fg_map", None) is not None
                and item.get("boundary_map", None) is not None
                and item.get("outer_ring_map", None) is not None
            )
            if use_precomputed_maps:
                if has_precomputed_triplet:
                    precomputed_structured_count += 1.0
                else:
                    fallback_structured_count += 1.0
                    if item.get("structured_map_path", None) is not None:
                        missing_precomputed_count += 1.0
            fg = None
            if use_precomputed_maps and item.get("fg_map", None) is not None:
                fg = self._resize_binary_map(item["fg_map"], target_hw=(h, w), device=device, mode="nearest")
            if fg is None and item.get("mask", None) is not None:
                mask = item["mask"].to(device=device, dtype=torch.float32)
                mask = F.interpolate(mask.unsqueeze(0), size=(800, 800), mode="nearest").squeeze(0)
                mask_down = F.interpolate(mask.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
                fg = (mask_down > 0).float()
            if fg is None:
                fg = torch.zeros((h, w), device=device, dtype=torch.float32)
            fg_masks.append(fg)

            boundary = None
            outer = None
            if use_precomputed_maps and item.get("boundary_map", None) is not None and item.get("outer_ring_map", None) is not None:
                boundary = self._resize_binary_map(item["boundary_map"], target_hw=(h, w), device=device, mode="nearest")
                outer = self._resize_binary_map(item["outer_ring_map"], target_hw=(h, w), device=device, mode="nearest")
            boundary_maps.append(boundary)
            outer_maps.append(outer)

        fg_stack = torch.stack(fg_masks, dim=0)
        fg_flat = fg_stack.view(fg_stack.size(0), -1)

        boundary_final = []
        outer_final = []
        for idx in range(fg_stack.size(0)):
            if boundary_maps[idx] is not None and outer_maps[idx] is not None:
                boundary_final.append(boundary_maps[idx])
                outer_final.append(outer_maps[idx])
            else:
                b, o = self._build_structured_maps(fg_stack[idx:idx + 1])
                boundary_final.append(b.squeeze(0))
                outer_final.append(o.squeeze(0))
        boundary_flat = torch.stack(boundary_final, dim=0).view(fg_stack.size(0), -1)
        outer_flat = torch.stack(outer_final, dim=0).view(fg_stack.size(0), -1)
        if use_precomputed_maps and missing_precomputed_count > 0 and not hasattr(self, "_warned_missing_precomputed_maps"):
            print(f"[StructuredMap] missing precomputed maps in batch: {int(missing_precomputed_count)}; fallback to online maps.")
            self._warned_missing_precomputed_maps = True
        stats = {
            "precomputed_structured_count": precomputed_structured_count,
            "fallback_structured_count": fallback_structured_count,
            "missing_precomputed_count": missing_precomputed_count,
        }
        return fg_flat, boundary_flat, outer_flat, stats

    def _extract_seg_to_image_attentions(
        self,
        attentions,
        SEG_token_embedding_indices: torch.Tensor,
        image_features_indices: torch.Tensor,
    ):
        per_layer_head = []
        per_layer_baseline = []
        layer_indices = []
        bs = SEG_token_embedding_indices.shape[0]
        for layer_idx, layer_attention in enumerate(attentions):
            layer_head_rows = []
            layer_baseline_rows = []
            for batch_idx in range(bs):
                attn = layer_attention[batch_idx]  # [heads, seq, seq]
                seg_mask = SEG_token_embedding_indices[batch_idx].bool()
                image_mask = image_features_indices[batch_idx].bool()
                seg_to_img = attn[:, seg_mask][:, :, image_mask]  # [heads, seg, image]
                if seg_to_img.shape[1] == 0 or seg_to_img.shape[2] == 0:
                    continue
                seg_to_img = seg_to_img.mean(dim=1)  # [heads, image]
                layer_head_rows.append(seg_to_img)
                layer_baseline_rows.append(seg_to_img.sum(dim=0, keepdim=True))  # [1, image]
            if len(layer_head_rows) == 0:
                continue
            per_layer_head.append(torch.stack(layer_head_rows, dim=0))  # [bs, heads, image]
            per_layer_baseline.append(torch.cat(layer_baseline_rows, dim=0))  # [bs, image]
            layer_indices.append(layer_idx)
        return per_layer_head, per_layer_baseline, layer_indices

    def _compute_structured_loss_per_instance(
        self,
        attention_map: torch.Tensor,
        fg_mask: torch.Tensor,
        boundary_map: Optional[torch.Tensor] = None,
        outer_ring_map: Optional[torch.Tensor] = None,
    ):
        device = attention_map.device
        bs = attention_map.shape[0]
        total = torch.zeros(bs, device=device)
        fg_bg = torch.zeros(bs, device=device)
        boundary_outer = torch.zeros(bs, device=device)
        margin = float(getattr(self.structured_attention_loss, "margin", 0.0))
        fg_bg_weight = float(getattr(self.structured_attention_loss, "fg_bg_weight", 1.0))
        boundary_outer_weight = float(getattr(self.structured_attention_loss, "boundary_outer_weight", 1.0))

        for idx in range(bs):
            attn = attention_map[idx]
            fg = fg_mask[idx] > 0
            bg = ~fg
            if fg.sum() == 0 or bg.sum() == 0:
                continue
            fg_mean = attn[fg].mean()
            bg_mean = attn[bg].mean()
            fg_bg_penalty = F.relu(torch.tensor(margin, device=device) - (fg_mean - bg_mean))
            cur_fg_bg = fg_bg_weight * fg_bg_penalty

            cur_boundary_outer = torch.tensor(0.0, device=device)
            if boundary_map is not None and outer_ring_map is not None:
                boundary = boundary_map[idx] > 0
                outer = outer_ring_map[idx] > 0
                if boundary.sum() > 0 and outer.sum() > 0:
                    boundary_mean = attn[boundary].mean()
                    outer_mean = attn[outer].mean()
                    bo_penalty = F.relu(torch.tensor(margin, device=device) - (boundary_mean - outer_mean))
                    cur_boundary_outer = boundary_outer_weight * bo_penalty

            fg_bg[idx] = cur_fg_bg
            boundary_outer[idx] = cur_boundary_outer
            total[idx] = cur_fg_bg + cur_boundary_outer
        return total, fg_bg, boundary_outer

    def _resolve_target_heads_dict(self):
        raw = getattr(self.config, "target_heads_dict", None)
        if raw is None:
            return None
        top_k_heads = int(getattr(self.config, "top_k_heads", 0) or 0)
        if not isinstance(raw, dict):
            raise ValueError(f"target_heads_dict must be a dict, got {type(raw)}")
        normalized = {}
        for k, v in raw.items():
            try:
                layer_idx = int(k)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid layer key in target_heads_dict: {k}") from e
            if not isinstance(v, (list, tuple)) or len(v) == 0:
                raise ValueError(f"target_heads_dict[{layer_idx}] must be a non-empty list")
            try:
                head_indices = [int(x) for x in v]
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid head list for layer {layer_idx}: {v}") from e
            if any(x < 0 for x in head_indices):
                raise ValueError(f"Head indices must be non-negative for layer {layer_idx}: {head_indices}")
            if len(set(head_indices)) != len(head_indices):
                raise ValueError(f"Duplicated head index for layer {layer_idx}: {head_indices}")
            if top_k_heads > 0:
                head_indices = head_indices[:top_k_heads]
                if len(head_indices) == 0:
                    raise ValueError(f"Layer {layer_idx} has no valid heads after top_k_heads={top_k_heads}.")
            normalized[layer_idx] = head_indices
        return normalized

    def _log_attention_selection_once(
        self,
        requested_layers,
        selected_layer_indices,
        layer_to_head_tensor,
        target_heads_dict,
    ):
        if hasattr(self, "_printed_attention_supervision_alignment"):
            return
        self._printed_attention_supervision_alignment = True
        print(f"[SEG][Attention] requested target_layers = {requested_layers}")
        print(f"[SEG][Attention] internal supervised layer indices = {selected_layer_indices}")
        for layer_idx in selected_layer_indices:
            head_tensor = layer_to_head_tensor[layer_idx]
            total_heads = int(head_tensor.shape[1])
            selected_heads = target_heads_dict[layer_idx] if target_heads_dict is not None else list(range(total_heads))
            print(
                f"[SEG][Attention] layer={layer_idx}, total_heads={total_heads}, "
                f"selected_heads={selected_heads}, tensor_shape={tuple(head_tensor.shape)}"
            )

    def _get_structured_schedule_weight(self, global_step):
        step = int(global_step) if global_step is not None else 0
        warmup_steps = int(getattr(self.config, "structured_warmup_steps", 0) or 0)
        decay_start = int(getattr(self.config, "structured_decay_start_step", -1) or -1)
        decay_end = int(getattr(self.config, "structured_decay_end_step", -1) or -1)

        weight = 1.0
        if warmup_steps > 0 and step < warmup_steps:
            weight = float(step + 1) / float(warmup_steps)
        if decay_start >= 0 and decay_end > decay_start:
            if step >= decay_end:
                weight = 0.0
            elif step >= decay_start:
                remain = float(decay_end - step)
                total = float(decay_end - decay_start)
                weight = min(weight, max(0.0, remain / max(total, 1.0)))
        return float(max(0.0, min(1.0, weight)))

    def _get_structured_schedule_phase_id(self, global_step):
        # 0: off_or_constant, 1: warmup, 2: full, 3: decay, 4: finished
        step = int(global_step) if global_step is not None else 0
        warmup_steps = int(getattr(self.config, "structured_warmup_steps", 0) or 0)
        decay_start = int(getattr(self.config, "structured_decay_start_step", -1) or -1)
        decay_end = int(getattr(self.config, "structured_decay_end_step", -1) or -1)
        if warmup_steps > 0 and step < warmup_steps:
            return 1
        if decay_start >= 0 and decay_end > decay_start:
            if step >= decay_end:
                return 4
            if step >= decay_start:
                return 3
            return 2
        if warmup_steps > 0:
            return 2
        return 0

    def _should_log_structured_detail(self, global_step):
        interval = int(getattr(self.config, "structured_log_interval", 50) or 0)
        if interval <= 0:
            return False
        step = int(global_step) if global_step is not None else 0
        return (step % interval) == 0

    def _maybe_dump_attention_audit(
        self,
        per_layer_head,
        fg_mask,
        boundary_map,
        outer_ring_map,
        seg_info,
        global_step,
    ):
        if not getattr(self.config, "attention_audit_mode", False):
            return None
        payload = {
            "seg_to_image_attn": [x.detach().cpu() for x in per_layer_head],
            "fg_mask": fg_mask.detach().cpu() if fg_mask is not None else None,
            "boundary_map": boundary_map.detach().cpu() if boundary_map is not None else None,
            "outer_ring_map": outer_ring_map.detach().cpu() if outer_ring_map is not None else None,
            "meta": [
                {
                    "image_id": str(item.get("image_id", "")),
                    "data_id": str(item.get("data_id", "")),
                    "mask_id": str(item.get("mask_id", "")),
                }
                for item in (seg_info or [])
            ],
        }
        audit_dir = getattr(self.config, "attention_audit_dir", None)
        if audit_dir:
            os.makedirs(audit_dir, exist_ok=True)
            step = int(global_step) if global_step is not None else -1
            filename = f"attention_audit_step{step:08d}.pt" if step >= 0 else "attention_audit_step_unknown.pt"
            save_path = os.path.join(audit_dir, filename)
            if os.path.exists(save_path):
                stem, ext = os.path.splitext(save_path)
                uniq = int(torch.randint(low=0, high=10**9, size=(1,)).item())
                save_path = f"{stem}_{os.getpid()}_{uniq}{ext}"
            torch.save(payload, save_path)
            payload["saved_path"] = save_path
        return payload

    def _compute_attention_loss(
        self,
        attentions,
        SEG_token_embedding_indices,
        image_features_indices,
        seg_info,
        global_step=None,
        supervision_branch: Optional[str] = None,
        skip_side_effects: bool = False,
    ):
        if not getattr(self.config, "use_attention_loss", True):
            zero = torch.tensor(0.0, device=self.device)
            return zero, None, {
                "loss_attn_fg_bg": zero,
                "loss_attn_boundary_outer": zero,
                "precomputed_structured_count": zero,
                "fallback_structured_count": zero,
                "missing_precomputed_count": zero,
                "selected_attn_layers": zero,
                "structured_effective_weight": zero,
                "structured_loss_small": zero,
                "structured_loss_non_small": zero,
                "structured_small_count": zero,
                "structured_non_small_count": zero,
                "structured_skipped_batches": zero,
                "structured_skip_ratio": zero,
                "loss_attn_fg_bg_small": zero,
                "loss_attn_fg_bg_non_small": zero,
                "loss_attn_boundary_outer_small": zero,
                "loss_attn_boundary_outer_non_small": zero,
                "selected_attn_heads_total": zero,
                "selected_attn_heads_per_layer": zero,
                "structured_schedule_phase_id": zero,
                "structured_skip_reason_code": zero,
            }
        fg_mask, boundary_map, outer_ring_map, supervision_stats = self._prepare_attention_supervision(
            seg_info=seg_info, target_hw=(27, 27), device=self.device
        )
        if fg_mask is None:
            zero = torch.tensor(0.0, device=self.device)
            return zero, None, {
                "loss_attn_fg_bg": zero,
                "loss_attn_boundary_outer": zero,
                "precomputed_structured_count": zero,
                "fallback_structured_count": zero,
                "missing_precomputed_count": zero,
                "selected_attn_layers": zero,
                "structured_effective_weight": zero,
                "structured_loss_small": zero,
                "structured_loss_non_small": zero,
                "structured_small_count": zero,
                "structured_non_small_count": zero,
                "structured_skipped_batches": zero,
                "structured_skip_ratio": zero,
                "loss_attn_fg_bg_small": zero,
                "loss_attn_fg_bg_non_small": zero,
                "loss_attn_boundary_outer_small": zero,
                "loss_attn_boundary_outer_non_small": zero,
                "selected_attn_heads_total": zero,
                "selected_attn_heads_per_layer": zero,
                "structured_schedule_phase_id": zero,
                "structured_skip_reason_code": zero,
            }
        per_layer_head, per_layer_baseline, extracted_layer_indices = self._extract_seg_to_image_attentions(
            attentions=attentions,
            SEG_token_embedding_indices=SEG_token_embedding_indices,
            image_features_indices=image_features_indices,
        )
        if len(per_layer_baseline) == 0:
            zero = torch.tensor(0.0, device=self.device)
            return zero, None, {
                "loss_attn_fg_bg": zero,
                "loss_attn_boundary_outer": zero,
                "precomputed_structured_count": zero,
                "fallback_structured_count": zero,
                "missing_precomputed_count": zero,
                "selected_attn_layers": zero,
                "structured_effective_weight": zero,
                "structured_loss_small": zero,
                "structured_loss_non_small": zero,
                "structured_small_count": zero,
                "structured_non_small_count": zero,
                "structured_skipped_batches": zero,
                "structured_skip_ratio": zero,
                "loss_attn_fg_bg_small": zero,
                "loss_attn_fg_bg_non_small": zero,
                "loss_attn_boundary_outer_small": zero,
                "loss_attn_boundary_outer_non_small": zero,
                "selected_attn_heads_total": zero,
                "selected_attn_heads_per_layer": zero,
                "structured_schedule_phase_id": zero,
                "structured_skip_reason_code": zero,
            }
        total_internal_layers = len(attentions)
        layer_to_head_tensor = {
            layer_idx: layer_head
            for layer_idx, layer_head in zip(extracted_layer_indices, per_layer_head)
        }
        layer_to_baseline_tensor = {
            layer_idx: layer_base
            for layer_idx, layer_base in zip(extracted_layer_indices, per_layer_baseline)
        }
        use_structured = getattr(self.config, "use_structured_attention_loss", False)
        effective_use_structured = use_structured
        if supervision_branch == "unstructured_baseline":
            effective_use_structured = False
        requested_target_layers = getattr(self.config, "target_layers", None) if use_structured else None
        selected_layer_indices = list(extracted_layer_indices)
        if use_structured and requested_target_layers is not None and len(requested_target_layers) > 0:
            explicit_layers = [int(x) for x in requested_target_layers]
            if len(set(explicit_layers)) != len(explicit_layers):
                raise ValueError(f"Duplicated layer index in target_layers: {explicit_layers}")
            for layer_idx in explicit_layers:
                if layer_idx < 0 or layer_idx >= total_internal_layers:
                    raise ValueError(
                        f"Requested target layer {layer_idx} out of range [0, {total_internal_layers - 1}]"
                    )
            selected_layer_indices = explicit_layers
        else:
            last_k_layers = int(getattr(self.config, "attention_loss_last_k_layers", 0) or 0)
            if last_k_layers > 0 and len(selected_layer_indices) > last_k_layers:
                selected_layer_indices = selected_layer_indices[-last_k_layers:]

        target_heads_dict = self._resolve_target_heads_dict() if use_structured else None
        if use_structured and target_heads_dict is not None:
            for layer_idx in selected_layer_indices:
                if layer_idx not in target_heads_dict:
                    raise ValueError(
                        f"target_heads_dict missing layer {layer_idx}; selected layers={selected_layer_indices}"
                    )

        self._log_attention_selection_once(
            requested_layers=requested_target_layers,
            selected_layer_indices=selected_layer_indices,
            layer_to_head_tensor=layer_to_head_tensor,
            target_heads_dict=target_heads_dict,
        )

        selected_attn_layers = torch.tensor(float(len(selected_layer_indices)), device=fg_mask.device)
        selected_attn_heads_total = torch.tensor(0.0, device=fg_mask.device)
        selected_attn_heads_per_layer = torch.tensor(0.0, device=fg_mask.device)
        loss_attention = torch.tensor(0.0, device=fg_mask.device)
        loss_attn_fg_bg = torch.tensor(0.0, device=fg_mask.device)
        loss_attn_boundary_outer = torch.tensor(0.0, device=fg_mask.device)
        structured_effective_weight = torch.tensor(0.0, device=fg_mask.device)
        structured_loss_small = torch.tensor(0.0, device=fg_mask.device)
        structured_loss_non_small = torch.tensor(0.0, device=fg_mask.device)
        structured_small_count = torch.tensor(0.0, device=fg_mask.device)
        structured_non_small_count = torch.tensor(0.0, device=fg_mask.device)
        loss_attn_fg_bg_small = torch.tensor(0.0, device=fg_mask.device)
        loss_attn_fg_bg_non_small = torch.tensor(0.0, device=fg_mask.device)
        loss_attn_boundary_outer_small = torch.tensor(0.0, device=fg_mask.device)
        loss_attn_boundary_outer_non_small = torch.tensor(0.0, device=fg_mask.device)
        structured_schedule_phase_id = torch.tensor(0.0, device=fg_mask.device)
        structured_skip_reason_code = torch.tensor(0.0, device=fg_mask.device)
        if not hasattr(self, "_structured_total_batches"):
            self._structured_total_batches = 0
        if not hasattr(self, "_structured_skipped_batches"):
            self._structured_skipped_batches = 0
        if effective_use_structured:
            small_weight = float(getattr(self.config, "small_weight", 1.5))
            small_ratio_threshold = float(getattr(self.config, "small_area_ratio_threshold", 0.01))
            area_ratio = fg_mask.float().mean(dim=1)
            is_small = area_ratio < small_ratio_threshold
            structured_small_count = is_small.float().sum()
            structured_non_small_count = (~is_small).float().sum()
            sample_weights = torch.ones_like(area_ratio)
            sample_weights = sample_weights + (small_weight - 1.0) * is_small.float()

            per_instance_total = torch.zeros(fg_mask.shape[0], device=fg_mask.device)
            per_instance_fg_bg = torch.zeros(fg_mask.shape[0], device=fg_mask.device)
            per_instance_boundary_outer = torch.zeros(fg_mask.shape[0], device=fg_mask.device)
            num_terms = 0

            layer_contrib = {}
            layer_head_map = {}
            layer_head_contrib = {}
            skip_reason = None
            requested_top_k = int(getattr(self.config, "top_k_heads", 0) or 0)
            capped_layers = {}
            for layer_idx in selected_layer_indices:
                if layer_idx not in layer_to_head_tensor:
                    continue
                layer_head = layer_to_head_tensor[layer_idx]  # [bs, heads, image]
                if target_heads_dict is not None:
                    selected_heads = target_heads_dict[layer_idx]
                    head_count = int(layer_head.shape[1])
                    if any(h >= head_count for h in selected_heads):
                        raise ValueError(
                            f"Layer {layer_idx} head index out of range; max={head_count - 1}, requested={selected_heads}"
                        )
                    layer_head_map[layer_idx] = [int(h) for h in selected_heads]
                    if requested_top_k > 0 and len(selected_heads) < requested_top_k:
                        capped_layers[layer_idx] = {
                            "requested_top_k": requested_top_k,
                            "available_from_config": len(selected_heads),
                        }
                    layer_total_contrib = torch.tensor(0.0, device=fg_mask.device)
                    layer_head_contrib[layer_idx] = {}
                    for head_idx in selected_heads:
                        head_attention = layer_head[:, head_idx, :]  # [bs, image]
                        h_total, h_fg_bg, h_boundary_outer = self._compute_structured_loss_per_instance(
                            head_attention,
                            fg_mask,
                            boundary_map=boundary_map,
                            outer_ring_map=outer_ring_map,
                        )
                        per_instance_total = per_instance_total + h_total
                        per_instance_fg_bg = per_instance_fg_bg + h_fg_bg
                        per_instance_boundary_outer = per_instance_boundary_outer + h_boundary_outer
                        layer_total_contrib = layer_total_contrib + h_total.mean()
                        layer_head_contrib[layer_idx][int(head_idx)] = float(h_total.mean().detach().item())
                        num_terms += 1
                    if len(selected_heads) > 0:
                        layer_contrib[layer_idx] = float((layer_total_contrib / float(len(selected_heads))).detach().item())
                else:
                    layer_head_map[layer_idx] = list(range(int(layer_head.shape[1])))
                    baseline_attention = layer_to_baseline_tensor[layer_idx]
                    b_total, b_fg_bg, b_boundary_outer = self._compute_structured_loss_per_instance(
                        baseline_attention,
                        fg_mask,
                        boundary_map=boundary_map,
                        outer_ring_map=outer_ring_map,
                    )
                    per_instance_total = per_instance_total + b_total
                    per_instance_fg_bg = per_instance_fg_bg + b_fg_bg
                    per_instance_boundary_outer = per_instance_boundary_outer + b_boundary_outer
                    layer_contrib[layer_idx] = float(b_total.mean().detach().item())
                    layer_head_contrib[layer_idx] = {"aggregated_heads_sum": float(b_total.mean().detach().item())}
                    num_terms += 1

            if len(layer_head_map) > 0:
                total_heads = float(sum(len(v) for v in layer_head_map.values()))
                selected_attn_heads_total = torch.tensor(total_heads, device=fg_mask.device)
                selected_attn_heads_per_layer = torch.tensor(
                    total_heads / float(max(len(layer_head_map), 1)),
                    device=fg_mask.device,
                )
            structured_schedule_phase_id = torch.tensor(
                float(self._get_structured_schedule_phase_id(global_step)),
                device=fg_mask.device,
            )

            if num_terms == 0:
                if len(selected_layer_indices) == 0:
                    skip_reason = "no_valid_selected_layers"
                    structured_skip_reason_code = torch.tensor(1.0, device=fg_mask.device)
                else:
                    skip_reason = "no_valid_selected_heads_or_attention"
                    structured_skip_reason_code = torch.tensor(2.0, device=fg_mask.device)
                strict_attention_selection = bool(getattr(self.config, "strict_attention_selection", False))
                if not skip_side_effects:
                    self._structured_total_batches += 1
                    self._structured_skipped_batches += 1
                if strict_attention_selection:
                    raise ValueError(f"No valid structured attention supervision terms were selected ({skip_reason}).")
                if not skip_side_effects:
                    print(f"[StructuredAttention][warn] skip structured loss at step={global_step}, reason={skip_reason}")
                loss_attention = torch.tensor(0.0, device=fg_mask.device)
                loss_attn_fg_bg = torch.tensor(0.0, device=fg_mask.device)
                loss_attn_boundary_outer = torch.tensor(0.0, device=fg_mask.device)
            else:
                if not skip_side_effects:
                    self._structured_total_batches += 1
                per_instance_total = per_instance_total / float(num_terms)
                per_instance_fg_bg = per_instance_fg_bg / float(num_terms)
                per_instance_boundary_outer = per_instance_boundary_outer / float(num_terms)
                norm = sample_weights.sum().clamp_min(1e-6)
                raw_loss_attention = (per_instance_total * sample_weights).sum() / norm
                raw_loss_attn_fg_bg = (per_instance_fg_bg * sample_weights).sum() / norm
                raw_loss_attn_boundary_outer = (per_instance_boundary_outer * sample_weights).sum() / norm
                sched_weight = self._get_structured_schedule_weight(global_step)
                structured_effective_weight = torch.tensor(float(sched_weight), device=fg_mask.device)
                loss_attention = raw_loss_attention * structured_effective_weight
                loss_attn_fg_bg = raw_loss_attn_fg_bg * structured_effective_weight
                loss_attn_boundary_outer = raw_loss_attn_boundary_outer * structured_effective_weight

                small_mask = is_small
                non_small_mask = ~is_small
                if small_mask.any():
                    structured_loss_small = (per_instance_total[small_mask] * sample_weights[small_mask]).mean() * structured_effective_weight
                    loss_attn_fg_bg_small = (per_instance_fg_bg[small_mask] * sample_weights[small_mask]).mean() * structured_effective_weight
                    loss_attn_boundary_outer_small = (
                        per_instance_boundary_outer[small_mask] * sample_weights[small_mask]
                    ).mean() * structured_effective_weight
                if non_small_mask.any():
                    structured_loss_non_small = (per_instance_total[non_small_mask] * sample_weights[non_small_mask]).mean() * structured_effective_weight
                    loss_attn_fg_bg_non_small = (
                        per_instance_fg_bg[non_small_mask] * sample_weights[non_small_mask]
                    ).mean() * structured_effective_weight
                    loss_attn_boundary_outer_non_small = (
                        per_instance_boundary_outer[non_small_mask] * sample_weights[non_small_mask]
                    ).mean() * structured_effective_weight

                if self._should_log_structured_detail(global_step) and not skip_side_effects:
                    skip_ratio = float(self._structured_skipped_batches / max(self._structured_total_batches, 1))
                    print(
                        f"[StructuredAttention][diag] step={global_step}, selected_layers={selected_layer_indices}, "
                        f"layer_heads={layer_head_map}, layer_contrib={layer_contrib}, layer_head_contrib={layer_head_contrib}, "
                        f"small={int(structured_small_count.item())}, non_small={int(structured_non_small_count.item())}, "
                        f"small_loss={float(structured_loss_small.item()):.6f}, non_small_loss={float(structured_loss_non_small.item()):.6f}, "
                        f"fg_bg_s={float(loss_attn_fg_bg_small.item()):.6f}, fg_bg_ns={float(loss_attn_fg_bg_non_small.item()):.6f}, "
                        f"bo_s={float(loss_attn_boundary_outer_small.item()):.6f}, bo_ns={float(loss_attn_boundary_outer_non_small.item()):.6f}, "
                        f"fg_bg={float(loss_attn_fg_bg.item()):.6f}, boundary_outer={float(loss_attn_boundary_outer.item()):.6f}, "
                        f"sched_w={float(structured_effective_weight.item()):.4f}, sched_phase={int(structured_schedule_phase_id.item())}, "
                        f"selected_heads_total={float(selected_attn_heads_total.item()):.1f}, "
                        f"selected_heads_per_layer={float(selected_attn_heads_per_layer.item()):.2f}, "
                        f"skip_reason_code={int(structured_skip_reason_code.item())}, "
                        f"capped_layers={capped_layers}, "
                        f"skipped={self._structured_skipped_batches}/{self._structured_total_batches} ({skip_ratio:.4f})"
                    )
        else:
            for layer_idx in selected_layer_indices:
                baseline_attention = layer_to_baseline_tensor[layer_idx]
                loss_attention = loss_attention + self.attention_loss(baseline_attention, fg_mask)

        selected_layer_head_tensors = [layer_to_head_tensor[x] for x in selected_layer_indices]
        audit_payload = None
        if not skip_side_effects:
            audit_payload = self._maybe_dump_attention_audit(
                per_layer_head=selected_layer_head_tensors,
                fg_mask=fg_mask,
                boundary_map=boundary_map,
                outer_ring_map=outer_ring_map,
                seg_info=seg_info,
                global_step=global_step,
            )
        stats = {
            "loss_attn_fg_bg": loss_attn_fg_bg,
            "loss_attn_boundary_outer": loss_attn_boundary_outer,
            "precomputed_structured_count": torch.tensor(
                supervision_stats["precomputed_structured_count"], device=fg_mask.device, dtype=torch.float32
            ),
            "fallback_structured_count": torch.tensor(
                supervision_stats["fallback_structured_count"], device=fg_mask.device, dtype=torch.float32
            ),
            "missing_precomputed_count": torch.tensor(
                supervision_stats["missing_precomputed_count"], device=fg_mask.device, dtype=torch.float32
            ),
            "selected_attn_layers": selected_attn_layers,
            "structured_effective_weight": structured_effective_weight,
            "structured_loss_small": structured_loss_small,
            "structured_loss_non_small": structured_loss_non_small,
            "structured_small_count": structured_small_count,
            "structured_non_small_count": structured_non_small_count,
            "structured_skipped_batches": torch.tensor(float(self._structured_skipped_batches), device=fg_mask.device),
            "structured_skip_ratio": torch.tensor(
                float(self._structured_skipped_batches / max(self._structured_total_batches, 1)),
                device=fg_mask.device,
            ),
            "loss_attn_fg_bg_small": loss_attn_fg_bg_small,
            "loss_attn_fg_bg_non_small": loss_attn_fg_bg_non_small,
            "loss_attn_boundary_outer_small": loss_attn_boundary_outer_small,
            "loss_attn_boundary_outer_non_small": loss_attn_boundary_outer_non_small,
            "selected_attn_heads_total": selected_attn_heads_total,
            "selected_attn_heads_per_layer": selected_attn_heads_per_layer,
            "structured_schedule_phase_id": structured_schedule_phase_id,
            "structured_skip_reason_code": structured_skip_reason_code,
        }
        return loss_attention, audit_payload, stats

    def _should_run_structured_effect_diagnose(self, global_step, seg_info) -> bool:
        if not bool(getattr(self.config, "diagnose_structured_effect", False)):
            return False
        if seg_info is None:
            return False
        interval = int(getattr(self.config, "diagnose_structured_effect_interval", 0) or 0)
        if interval <= 0:
            return False
        if global_step is None:
            return False
        if int(global_step) % interval != 0:
            return False
        if len(seg_info) == 0:
            return False
        first = seg_info[0]
        if "padding_mask" not in first and "mask" not in first:
            return False
        return True

    def _forward_llm_and_mask_no_prepare(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        inputs_embeds: Optional[torch.FloatTensor],
        labels: Optional[torch.LongTensor],
        image_features,
        mask_num,
        SEG_token_embedding_indices: torch.Tensor,
        image_features_indices: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: bool = True,
        output_attentions: bool = False,
    ):
        eff_output_att = bool(output_attentions) or (
            self._seg_query_refiner_enabled() and not self._seg_loc_prior_enabled()
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=eff_output_att,
            output_hidden_states=False,
            return_dict=return_dict,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        SEG_embedding, _ = self._compute_seg_embedding_for_mask(
            hidden_states,
            outputs.attentions,
            SEG_token_embedding_indices,
            image_features_indices,
        )
        mask_features, _transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features
        )
        mask_num_t = torch.tensor(mask_num, device=mask_features.device)
        mask_features = torch.repeat_interleave(mask_features, repeats=mask_num_t, dim=0)
        multi_scale_features = [
            torch.repeat_interleave(feat, repeats=mask_num_t, dim=0) for feat in multi_scale_features
        ]
        mask_outputs = self.predictor(multi_scale_features, mask_features, None, None, SEG_embedding)
        return logits, mask_outputs, outputs

    @staticmethod
    def _structured_diag_subset_key(seg_item) -> str:
        subset = seg_item.get("subset", None)
        if subset is None:
            subset = seg_item.get("subset_name", None)
        if subset is None:
            subset = seg_item.get("eval_subset", None)
        if subset is None:
            return "ALL"
        text = str(subset).strip().upper()
        if text.startswith("B"):
            return "B"
        if text.startswith("R"):
            return "R"
        return "OTHER"

    def _structured_diag_best_query_flat(self, pred_logits_qhw: torch.Tensor, gt_hw: torch.Tensor, logit_thresh: float):
        """pred_logits_qhw [Q,Hp,Wp], gt_hw [H,W] bool or float mask."""
        q = int(pred_logits_qhw.shape[0])
        gt_bin = (gt_hw > 0.5).float()
        h, w = int(gt_bin.shape[-2]), int(gt_bin.shape[-1])
        best_iou = torch.tensor(-1.0, device=pred_logits_qhw.device, dtype=torch.float32)
        best_flat = pred_logits_qhw[0]
        for qi in range(q):
            up = F.interpolate(
                pred_logits_qhw[qi : qi + 1].unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
            pbin = (up > logit_thresh).float()
            inter = (pbin * gt_bin).sum()
            union = pbin.sum() + gt_bin.sum() - inter
            iou = inter / union.clamp_min(1e-6)
            if float(iou.item()) > float(best_iou.item()):
                best_iou = iou
                best_flat = up
        return best_flat, best_iou

    def _run_structured_effect_diagnostics(
        self,
        *,
        outputs,
        seg_info,
        targets,
        input_ids,
        attention_mask,
        past_key_values,
        inputs_embeds,
        labels,
        image_features,
        mask_num,
        SEG_token_embedding_indices,
        image_features_indices,
        use_cache,
        return_dict,
        global_step,
    ):
        device = self.device

        def _z(v):
            return torch.tensor(float(v), device=device, dtype=torch.float32)

        thresh = float(getattr(self.config, "diagnose_structured_effect_threshold", 0.0))
        small_ratio_threshold = float(getattr(self.config, "small_area_ratio_threshold", 0.01))
        effect_eps = float(getattr(self.config, "diagnose_structured_effect_eps", 1e-6))
        mean_abs_thr = float(getattr(self.config, "diagnose_structured_effect_mean_abs_threshold", 1e-3))

        loss_default, _, _ = self._compute_attention_loss(
            attentions=outputs.attentions,
            SEG_token_embedding_indices=SEG_token_embedding_indices,
            image_features_indices=image_features_indices,
            seg_info=seg_info,
            global_step=global_step,
            supervision_branch=None,
            skip_side_effects=True,
        )
        loss_unstructured, _, _ = self._compute_attention_loss(
            attentions=outputs.attentions,
            SEG_token_embedding_indices=SEG_token_embedding_indices,
            image_features_indices=image_features_indices,
            seg_info=seg_info,
            global_step=global_step,
            supervision_branch="unstructured_baseline",
            skip_side_effects=True,
        )
        diag_loss_delta = loss_default.detach().float() - loss_unstructured.detach().float()

        was_training = self.training
        self.eval()
        mean_abs_list = []
        disagree_list = []
        area_ratio_list = []
        iou_a_list = []
        iou_b_list = []
        is_small_list = []
        pred_a_list = []
        pred_b_list = []

        with torch.inference_mode():
            _logits_a, mo_a, _ = self._forward_llm_and_mask_no_prepare(
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_features,
                mask_num,
                SEG_token_embedding_indices,
                image_features_indices,
                use_cache,
                return_dict,
                output_attentions=False,
            )
            _logits_b, mo_b, _ = self._forward_llm_and_mask_no_prepare(
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_features,
                mask_num,
                SEG_token_embedding_indices,
                image_features_indices,
                use_cache,
                return_dict,
                output_attentions=False,
            )

        self.train(was_training)

        pred_a = mo_a["pred_masks"].detach().float()
        pred_b = mo_b["pred_masks"].detach().float()
        bs = int(pred_a.shape[0])

        for b in range(bs):
            gt_masks = targets[b]["masks"]
            if gt_masks.dim() != 3 or gt_masks.shape[0] == 0:
                continue
            gt0 = gt_masks[0].detach().float()
            pa, iou_a = self._structured_diag_best_query_flat(pred_a[b], gt0, thresh)
            pb, iou_b = self._structured_diag_best_query_flat(pred_b[b], gt0, thresh)
            pred_a_list.append(pa)
            pred_b_list.append(pb)
            iou_a_list.append(iou_a)
            iou_b_list.append(iou_b)
            diff = (pa - pb).abs()
            mean_abs_list.append(diff.mean())
            pa_bin = (pa > thresh).float()
            pb_bin = (pb > thresh).float()
            disagree_list.append((pa_bin != pb_bin).float().mean())
            denom = pb_bin.sum().clamp_min(torch.tensor(1.0, device=device, dtype=pb_bin.dtype))
            area_ratio_list.append((pa_bin.sum() - pb_bin.sum()).abs() / denom)
            gt_area_ratio = float(gt0.float().mean().item())
            is_small_list.append(gt_area_ratio < small_ratio_threshold)

        if len(mean_abs_list) == 0:
            return {
                "structured_output_effect_mean": _z(0.0),
                "structured_output_effect_nonzero_ratio": _z(0.0),
                "structured_output_effect_gt_threshold_ratio": _z(0.0),
                "structured_output_effect_mean_small": _z(0.0),
                "structured_output_effect_mean_non_small": _z(0.0),
                "structured_output_effect_nonzero_ratio_small": _z(0.0),
                "structured_output_effect_nonzero_ratio_non_small": _z(0.0),
                "diag_mask_logits_mean_abs_diff": _z(0.0),
                "diag_binary_mask_disagree_ratio": _z(0.0),
                "diag_pred_mask_iou_between_runs": _z(1.0),
                "diag_pred_area_change_ratio": _z(0.0),
                "diag_iou_with_gt_run_a": _z(0.0),
                "diag_iou_with_gt_run_b": _z(0.0),
                "diag_delta_iou_vs_gt": _z(0.0),
                "diag_attention_sup_default_minus_unstructured": diag_loss_delta.detach(),
                "diag_structured_supervision_alters_mask_forward_path": _z(0.0),
            }

        mean_abs = torch.stack(mean_abs_list)
        disagree = torch.stack(disagree_list)
        iou_a_t = torch.stack([x.view(()) for x in iou_a_list])
        iou_b_t = torch.stack([x.view(()) for x in iou_b_list])
        is_small_t = torch.tensor(is_small_list, device=device, dtype=torch.bool)

        nonzero_ratio = (mean_abs > effect_eps).float().mean()
        gt_ratio = (mean_abs > mean_abs_thr).float().mean()

        def _bucket_mean(mask_t, vals):
            if not mask_t.any():
                return _z(0.0)
            return vals[mask_t].mean()

        def _bucket_ratio(mask_t, vals):
            if not mask_t.any():
                return _z(0.0)
            return (vals[mask_t] > effect_eps).float().mean()

        inter_ab = []
        union_ab = []
        for pa, pb in zip(pred_a_list, pred_b_list):
            a_bin = (pa > thresh).float()
            b_bin = (pb > thresh).float()
            inter_ab.append((a_bin * b_bin).sum())
            union_ab.append(a_bin.sum() + b_bin.sum() - (a_bin * b_bin).sum())
        inter_t = torch.stack(inter_ab)
        union_t = torch.stack(union_ab).clamp_min(1e-6)
        iou_between = (inter_t / union_t).mean()
        area_change = torch.stack(area_ratio_list).mean()

        out = {
            "structured_output_effect_mean": mean_abs.mean(),
            "structured_output_effect_nonzero_ratio": nonzero_ratio,
            "structured_output_effect_gt_threshold_ratio": gt_ratio,
            "structured_output_effect_mean_small": _bucket_mean(is_small_t, mean_abs),
            "structured_output_effect_mean_non_small": _bucket_mean(~is_small_t, mean_abs),
            "structured_output_effect_nonzero_ratio_small": _bucket_ratio(is_small_t, mean_abs),
            "structured_output_effect_nonzero_ratio_non_small": _bucket_ratio(~is_small_t, mean_abs),
            "diag_mask_logits_mean_abs_diff": mean_abs.mean(),
            "diag_binary_mask_disagree_ratio": disagree.mean(),
            "diag_pred_mask_iou_between_runs": iou_between,
            "diag_pred_area_change_ratio": area_change,
            "diag_iou_with_gt_run_a": iou_a_t.mean(),
            "diag_iou_with_gt_run_b": iou_b_t.mean(),
            "diag_delta_iou_vs_gt": (iou_a_t - iou_b_t).mean(),
            "diag_attention_sup_default_minus_unstructured": diag_loss_delta.detach().float(),
            "diag_structured_supervision_alters_mask_forward_path": _z(0.0),
        }
        return out

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
        seg_query_diag = None
        loc_logits = None
        loc_fuse_diag = {}
        loss_seg_loc_tensor = None
        loc_prior_metrics = {}
        loc_prior_ext = {}
        seg_loc_prior_sample_rows = None

        if (SEG_token_embedding_indices == 1).sum() != 0:

            # for generative mode only the 1th stage need
            if input_ids.shape[1] != 1:
                image_features = self.get_vision_tower_feature(images)
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
        SEG_embedding, seg_aux = self._compute_seg_embedding_for_mask(
            hidden_states,
            outputs.attentions,
            SEG_token_embedding_indices,
            image_features_indices,
        )
        if seg_aux:
            seg_query_diag = seg_aux.get("query_diag")
            loc_logits = seg_aux.get("loc_logits")
            loc_fuse_diag = seg_aux.get("loc_fuse_diag") or {}
        else:
            seg_query_diag = None
            loc_logits = None
            loc_fuse_diag = {}

        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            image_features)
        mask_num = torch.tensor(mask_num, device=mask_features.device)
        mask_features = torch.repeat_interleave(mask_features, repeats=mask_num, dim=0)
        multi_scale_features = [
            torch.repeat_interleave(feat, repeats=mask_num, dim=0)
            for feat in multi_scale_features
        ]

        mask_outputs = self.predictor(multi_scale_features, mask_features, None, None, SEG_embedding)

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
        structured_effect_diag_pack = None
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
            if loc_logits is not None and targets is not None:
                loss_seg_loc_tensor, loc_prior_metrics, loc_prior_ext, seg_loc_prior_sample_rows = (
                    self._seg_loc_prior_loss_and_metrics(
                        loc_logits,
                        targets,
                        SEG_token_embedding_indices,
                        seg_info=seg_info,
                        global_step=global_step,
                    )
                )
            if targets is not None and self._should_run_structured_effect_diagnose(global_step, seg_info):
                structured_effect_diag_pack = self._run_structured_effect_diagnostics(
                    outputs=outputs,
                    seg_info=seg_info,
                    targets=targets,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    image_features=image_features,
                    mask_num=mask_num,
                    SEG_token_embedding_indices=SEG_token_embedding_indices,
                    image_features_indices=image_features_indices,
                    use_cache=use_cache,
                    return_dict=return_dict,
                    global_step=global_step,
                )

        loss_attention, attention_audit, attention_stats = self._compute_attention_loss(
            attentions=outputs.attentions,
            SEG_token_embedding_indices=SEG_token_embedding_indices,
            image_features_indices=image_features_indices,
            seg_info=seg_info,
            global_step=global_step,
        )
        attention_loss_weight = getattr(self.config, "attention_loss_weight", 0.01)
        w_loc = float(getattr(self.config, "seg_loc_prior_weight", 0.1))
        loc_term = (
            w_loc * loss_seg_loc_tensor
            if loss_seg_loc_tensor is not None
            else torch.tensor(0.0, device=loss_attention.device, dtype=loss_attention.dtype)
        )
        loss = llm_loss + mask_loss + loc_term + attention_loss_weight * loss_attention

        return CausalOutputWithMask(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_mask=loss_mask.detach(),
            loss_dice=loss_dice.detach(),
            loss_llm=llm_loss.detach(),
            loss_attention=attention_loss_weight * loss_attention.detach(),
            loss_attn_fg_bg=attention_stats["loss_attn_fg_bg"].detach(),
            loss_attn_boundary_outer=attention_stats["loss_attn_boundary_outer"].detach(),
            precomputed_structured_count=attention_stats["precomputed_structured_count"].detach(),
            fallback_structured_count=attention_stats["fallback_structured_count"].detach(),
            missing_precomputed_count=attention_stats["missing_precomputed_count"].detach(),
            selected_attn_layers=attention_stats["selected_attn_layers"].detach(),
            structured_effective_weight=attention_stats["structured_effective_weight"].detach(),
            structured_loss_small=attention_stats["structured_loss_small"].detach(),
            structured_loss_non_small=attention_stats["structured_loss_non_small"].detach(),
            structured_small_count=attention_stats["structured_small_count"].detach(),
            structured_non_small_count=attention_stats["structured_non_small_count"].detach(),
            structured_skipped_batches=attention_stats["structured_skipped_batches"].detach(),
            structured_skip_ratio=attention_stats["structured_skip_ratio"].detach(),
            loss_attn_fg_bg_small=attention_stats["loss_attn_fg_bg_small"].detach(),
            loss_attn_fg_bg_non_small=attention_stats["loss_attn_fg_bg_non_small"].detach(),
            loss_attn_boundary_outer_small=attention_stats["loss_attn_boundary_outer_small"].detach(),
            loss_attn_boundary_outer_non_small=attention_stats["loss_attn_boundary_outer_non_small"].detach(),
            selected_attn_heads_total=attention_stats["selected_attn_heads_total"].detach(),
            selected_attn_heads_per_layer=attention_stats["selected_attn_heads_per_layer"].detach(),
            structured_schedule_phase_id=attention_stats["structured_schedule_phase_id"].detach(),
            structured_skip_reason_code=attention_stats["structured_skip_reason_code"].detach(),
            attention_audit=attention_audit,
            structured_output_effect_mean=(
                structured_effect_diag_pack["structured_output_effect_mean"].detach()
                if structured_effect_diag_pack
                else None
            ),
            structured_output_effect_nonzero_ratio=(
                structured_effect_diag_pack["structured_output_effect_nonzero_ratio"].detach()
                if structured_effect_diag_pack
                else None
            ),
            structured_output_effect_gt_threshold_ratio=(
                structured_effect_diag_pack["structured_output_effect_gt_threshold_ratio"].detach()
                if structured_effect_diag_pack
                else None
            ),
            structured_output_effect_mean_small=(
                structured_effect_diag_pack["structured_output_effect_mean_small"].detach()
                if structured_effect_diag_pack
                else None
            ),
            structured_output_effect_mean_non_small=(
                structured_effect_diag_pack["structured_output_effect_mean_non_small"].detach()
                if structured_effect_diag_pack
                else None
            ),
            structured_output_effect_nonzero_ratio_small=(
                structured_effect_diag_pack["structured_output_effect_nonzero_ratio_small"].detach()
                if structured_effect_diag_pack
                else None
            ),
            structured_output_effect_nonzero_ratio_non_small=(
                structured_effect_diag_pack["structured_output_effect_nonzero_ratio_non_small"].detach()
                if structured_effect_diag_pack
                else None
            ),
            diag_mask_logits_mean_abs_diff=(
                structured_effect_diag_pack["diag_mask_logits_mean_abs_diff"].detach()
                if structured_effect_diag_pack
                else None
            ),
            diag_binary_mask_disagree_ratio=(
                structured_effect_diag_pack["diag_binary_mask_disagree_ratio"].detach()
                if structured_effect_diag_pack
                else None
            ),
            diag_pred_mask_iou_between_runs=(
                structured_effect_diag_pack["diag_pred_mask_iou_between_runs"].detach()
                if structured_effect_diag_pack
                else None
            ),
            diag_pred_area_change_ratio=(
                structured_effect_diag_pack["diag_pred_area_change_ratio"].detach()
                if structured_effect_diag_pack
                else None
            ),
            diag_iou_with_gt_run_a=(
                structured_effect_diag_pack["diag_iou_with_gt_run_a"].detach()
                if structured_effect_diag_pack
                else None
            ),
            diag_iou_with_gt_run_b=(
                structured_effect_diag_pack["diag_iou_with_gt_run_b"].detach()
                if structured_effect_diag_pack
                else None
            ),
            diag_delta_iou_vs_gt=(
                structured_effect_diag_pack["diag_delta_iou_vs_gt"].detach()
                if structured_effect_diag_pack
                else None
            ),
            diag_attention_sup_default_minus_unstructured=(
                structured_effect_diag_pack["diag_attention_sup_default_minus_unstructured"].detach()
                if structured_effect_diag_pack
                else None
            ),
            diag_structured_supervision_alters_mask_forward_path=(
                structured_effect_diag_pack["diag_structured_supervision_alters_mask_forward_path"].detach()
                if structured_effect_diag_pack
                else None
            ),
            seg_query_delta_norm=seg_query_diag["seg_query_delta_norm"] if seg_query_diag else None,
            seg_query_original_norm=seg_query_diag["seg_query_original_norm"] if seg_query_diag else None,
            seg_query_refined_norm=seg_query_diag["seg_query_refined_norm"] if seg_query_diag else None,
            seg_query_cosine_q_delta=seg_query_diag["seg_query_cosine_q_delta"] if seg_query_diag else None,
            seg_query_cosine_q_qnew=seg_query_diag["seg_query_cosine_q_qnew"] if seg_query_diag else None,
            loss_seg_loc_prior=loss_seg_loc_tensor.detach() if loss_seg_loc_tensor is not None else None,
            seg_loc_prior_mean=loc_prior_metrics.get("seg_loc_prior_mean") if loc_prior_metrics else None,
            seg_loc_prior_entropy=loc_prior_metrics.get("seg_loc_prior_entropy") if loc_prior_metrics else None,
            seg_loc_prior_fg_mass=loc_prior_metrics.get("seg_loc_prior_fg_mass") if loc_prior_metrics else None,
            seg_loc_prior_bg_mass=loc_prior_metrics.get("seg_loc_prior_bg_mass") if loc_prior_metrics else None,
            seg_loc_code_norm=loc_fuse_diag.get("seg_loc_code_norm") if loc_fuse_diag else None,
            seg_loc_sem_norm=loc_fuse_diag.get("seg_loc_sem_norm") if loc_fuse_diag else None,
            seg_loc_qfinal_norm=loc_fuse_diag.get("seg_loc_qfinal_norm") if loc_fuse_diag else None,
            seg_loc_code_to_sem_norm_ratio=loc_fuse_diag.get("seg_loc_code_to_sem_norm_ratio") if loc_fuse_diag else None,
            seg_loc_cosine_sem_qfinal=loc_fuse_diag.get("seg_loc_cosine_sem_qfinal") if loc_fuse_diag else None,
            seg_loc_cosine_sem_loccode=loc_fuse_diag.get("seg_loc_cosine_sem_loccode") if loc_fuse_diag else None,
            seg_loc_prior_extended_metrics=loc_prior_ext if loc_prior_ext else None,
            seg_loc_prior_sample_rows=seg_loc_prior_sample_rows,
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
            mask_num=None,
            return_mask_logits_all: bool = False):
        """
        return_mask_logits_all: if True, each output dict includes float32 'pred_logits_all' [Q,H,W]
        (after interpolate to image size). Default False preserves legacy eval behavior.
        """
        output_attentions = bool(self._seg_query_refiner_enabled()) and not self._seg_loc_prior_enabled()
        output_hidden_states = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_features = self.get_vision_tower_feature(images)

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

        SEG_embedding, _ = self._compute_seg_embedding_for_mask(
            hidden_states,
            outputs.attentions,
            SEG_token_embedding_indices,
            image_features_indices,
        )

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

        mask_outputs = self.predictor(multi_scale_features, mask_features, None, None, SEG_embedding) 

        
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
            if return_mask_logits_all:
                instance_r['pred_logits_all'] = mask_pred_result.detach().float().cpu().numpy()
            processed_results.append(instance_r)
        return processed_results
