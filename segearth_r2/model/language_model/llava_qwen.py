from typing import List, Optional, Tuple, Union
from addict import Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import fvcore.nn.weight_init as weight_init

from segearth_r2.model.mipha.model.language_model.configuration_mipha import MiphaVisionConfig, ProjectorConfig
from segearth_r2.model.mipha.model.multimodal_projector.builder import build_vision_projector
from segearth_r2.model.mipha.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from segearth_r2.model.mipha.model.multimodal_encoder.siglip_encoder import SiglipVisionTower
from segearth_r2.model.mipha.model.multimodal_encoder.dinov2_encoder import Dinov2VisionTower


from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration
# 兼容老一点的 VL 名字（你环境里也有 True）
from transformers import Qwen2VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
from fvcore.nn import FlopCountAnalysis

from ..mipha.model.language_model.mipha_qwen import (
    MiphaQwenForCausalLM,
    MiphaQwenModel,
)

try:
    from transformers import Qwen2_5_VLConfig as QwenVLConfig
except ImportError:
    from transformers import Qwen2VLConfig as QwenVLConfig

from segearth_r2.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, REFER_TOKEN_INDEX

from ..mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import (
    MultiScaleMaskedTransformerDecoderForOPTPreTrain,
)
from ..mask_decoder.Mask2Former_Simplify.modeling.pixel_decoder.msdeformattn import (
    MSDeformAttnPixelDecoder,
)
from ..mask_encoder.swin_trans import build_swin_b, build_swin_l
from ..datasets_mapper.IVS_mapper import IVSDatasetMapper
from segearth_r2.model.mask_decoder.mask_criterion.Mask_Criterion import (
    Criterion,
    hungarian_matcher_InstructSeg,
)


def _get_hidden_size(config):
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    raise AttributeError("Cannot find hidden_size in config or config.text_config")


def _get_vocab_size(config):
    if hasattr(config, "vocab_size"):
        return config.vocab_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "vocab_size"):
        return config.text_config.vocab_size
    raise AttributeError("Cannot find vocab_size in config or config.text_config")


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


class AttentionLoss(nn.Module):
    def __init__(self, reduction="batchmean"):
        super(AttentionLoss, self).__init__()
        self.reduction = reduction

    def forward(self, model_attention_logits: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        device = model_attention_logits.device
        loss = torch.tensor(0.0, device=device)
        epsilon = 1e-8

        for idx in range(model_attention_logits.shape[0]):
            attention_map_target = model_attention_logits[idx][gt_mask[idx] == 1]
            attention_map_else = model_attention_logits[idx][gt_mask[idx] == 0]

            if attention_map_target.numel() == 0:
                continue

            mean = (
                torch.mean(attention_map_else)
                if attention_map_else.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            mse = torch.mean((attention_map_target - mean) ** 2)
            loss += -torch.log(mse + epsilon)

        if self.reduction == "batchmean":
            loss = loss / model_attention_logits.shape[0]
        elif self.reduction == "mean":
            loss = loss / model_attention_logits.numel()

        return loss


class SegEarthR2ModelQwen(MiphaQwenModel):
    def __init__(self, config: QwenVLConfig, mask_decoder_cfg=None, qwen_model=None):
        super(SegEarthR2ModelQwen, self).__init__(config, qwen_model=qwen_model)
        # 让 MiphaMetaForCausalLM / prepare_inputs_labels_for_multimodal 能用 get_model().embed_tokens(...)
        qwen_core = getattr(self, "qwen_model", None)

        if qwen_core is not None and hasattr(qwen_core, "embed_tokens"):
            self.embed_tokens = qwen_core.embed_tokens
        elif qwen_core is not None and hasattr(qwen_core, "model") and hasattr(qwen_core.model, "embed_tokens"):
            self.embed_tokens = qwen_core.model.embed_tokens
        else:
            raise AttributeError("Cannot find embed_tokens in qwen_model (expected qwen_model.embed_tokens or qwen_model.model.embed_tokens)")
        self.cfg = mask_decoder_cfg
        self.projector_outdim = _get_hidden_size(config)

        # mask tower（swin）初始化（保持你原逻辑）
        if hasattr(config, "mm_vision_tower"):
            swin_type = getattr(config, "swin_type", "base")
            if swin_type == "base":
                self.vision_tower_mask = build_swin_b(None)
            else:
                self.vision_tower_mask = build_swin_l(None)

            self.vision_tower_mask.image_processor = IVSDatasetMapper(self.cfg)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_vision_tower_mask(self):
        vision_tower = getattr(self, "vision_tower_mask", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        # ---- 1) 取训练参数里的路径 ----
        vision_tower = (
            model_args.vision_tower
            if hasattr(model_args, "vision_tower")
            else model_args.mm_vision_tower
        )
        vision_tower_mask = (
            model_args.vision_tower_mask
            if hasattr(model_args, "vision_tower_mask")
            else model_args.mm_vision_tower_mask
        )

        self.config.mm_vision_tower = vision_tower
        swin_type = getattr(model_args, "swin_type", "base")
        self.config.swin_type = swin_type

        # ---- 2) 初始化 swin mask tower（保持你原逻辑）----
        if swin_type == "base":
            vision_tower_mask = build_swin_b(vision_tower_mask)
        else:
            print("current visual encoder is swin large")
            vision_tower_mask = build_swin_l(vision_tower_mask)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower_mask = [vision_tower_mask]
        else:
            self.vision_tower_mask = vision_tower_mask

        self.config.use_mm_proj = True
        self.vision_tower_mask.hidden_size = 256
        self.vision_tower_mask.image_processor = IVSDatasetMapper(self.cfg)

        # ---- 3) 关键新增：如果 MiphaMetaModel 没建出 vision_tower/mm_projector，就在这里补齐 ----
        if getattr(self, "vision_tower", None) is None or getattr(self, "mm_projector", None) is None:
            vt_path = vision_tower

            # 这里依赖：MiphaVisionConfig / ProjectorConfig / (CLIPVisionTower, SiglipVisionTower, Dinov2VisionTower) / build_vision_projector
            # 请把这些 import 放到 llava_qwen.py 文件顶部（顶格），不要写在类内部。
            vt_cfg = MiphaVisionConfig(
                vision_model_name_or_path=vt_path,
                mm_vision_select_layer=getattr(model_args, "mm_vision_select_layer", -2),
                mm_vision_select_feature=getattr(model_args, "mm_vision_select_feature", "patch"),
            )

            if "clip" in vt_path:
                self.vision_tower = CLIPVisionTower(vt_cfg)
            elif "siglip" in vt_path:
                self.vision_tower = SiglipVisionTower(vt_cfg)
            elif "dinov2" in vt_path:
                self.vision_tower = Dinov2VisionTower(vt_cfg)
            else:
                raise ValueError(f"Unsupported vision_tower path: {vt_path}")

            # projector 的输入维度 = 视觉塔输出 hidden_size；输出维度 = 语言模型 hidden_size
            mm_hidden = getattr(self.vision_tower, "hidden_size", None)
            if mm_hidden is None and hasattr(self.vision_tower, "config"):
                mm_hidden = getattr(self.vision_tower.config, "hidden_size", None)
            if mm_hidden is None:
                raise ValueError("Cannot infer vision tower hidden size for projector")

            lm_hidden = _get_hidden_size(self.config)
            proj_cfg = ProjectorConfig(
                mm_projector_type=getattr(model_args, "mm_projector_type", "linear"),
                mm_hidden_size=mm_hidden,
                hidden_size=lm_hidden,
            )
            self.mm_projector = build_vision_projector(proj_cfg)

class SegEarthR2Qwen(MiphaQwenForCausalLM):
    def __init__(self, config, model_args=None, mask_decoder_cfg=None, add_cross_attn=True, cross_attn_index=None,
             qwen_model=None, lm_head=None):
        super(SegEarthR2Qwen, self).__init__(config, qwen_model=qwen_model, lm_head=lm_head)

        self.model = SegEarthR2ModelQwen(config, mask_decoder_cfg, qwen_model=qwen_model)
        self.init_config = config
        self.mask_decoder_cfg = mask_decoder_cfg
        self.cross_attn_index = cross_attn_index

        if lm_head is None:
            self.lm_head = nn.Linear(_get_hidden_size(config), _get_vocab_size(config), bias=False)
        else:
            self.lm_head = lm_head

        is_train_mask_decode = getattr(config, "mask_decode_train", False)
        self.is_train_mask_decode = is_train_mask_decode

        if is_train_mask_decode:
            print("Mask Decoder has been trained, init directly")
            self.initial_mask_module()

        self.post_init()


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        cache_dir = kwargs.pop("cache_dir", None)
        mask_decoder_cfg = kwargs.pop("mask_decoder_cfg", None)
        add_cross_attn = kwargs.pop("add_cross_attn", True)
        cross_attn_index = kwargs.pop("cross_attn_index", None)
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        # 先读 config 判断是否 VL
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        is_vl = config.__class__.__name__ in ("Qwen2_5_VLConfig", "Qwen2VLConfig")

        # 1) 先加载 base 模型
        if is_vl:
            try:
                base_lm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    trust_remote_code=trust_remote_code,
                    *model_args,
                    **kwargs,
                )
            except Exception:
                base_lm = Qwen2VLForConditionalGeneration.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    trust_remote_code=trust_remote_code,
                    *model_args,
                    **kwargs,
                )
        else:
            base_lm = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                *model_args,
                **kwargs,
            )
            config = base_lm.config  # 非 VL 情况用 base_lm.config

        # 2) 构建你们的 wrapper
# VL：取 language_model 当作 qwen_model（文本主干）
        if is_vl:
            qwen_model = base_lm.model.language_model
            lm_head = base_lm.lm_head
        else:
            qwen_model = base_lm.model
            lm_head = base_lm.lm_head

        model = cls(
            config=config,
            mask_decoder_cfg=mask_decoder_cfg,
            add_cross_attn=add_cross_attn,
            cross_attn_index=cross_attn_index,
            qwen_model=qwen_model,
            lm_head=lm_head,
        )

    # 3) 灌权重（关键：VL 的文本主干在 base_lm.model.language_model）
    # 你的输出证明了：base_lm.model.children() = ['visual', 'language_model']，并且有 base_lm.lm_head :contentReference[oaicite:1]{index=1}
        if is_vl:
            # 只灌语言模型 + lm_head。visual 我们不需要（你训练里用的是 siglip2）
            src_lm = base_lm.model.language_model
            if hasattr(model.model, "qwen_model"):
                model.model.qwen_model.load_state_dict(src_lm.state_dict(), strict=False)
            elif hasattr(model.model, "language_model"):
                model.model.language_model.load_state_dict(src_lm.state_dict(), strict=False)
            else:
            # 兜底：直接往 model.model 灌（可能会有一堆 unexpected keys，但 strict=False 不会炸）
                model.model.load_state_dict(src_lm.state_dict(), strict=False)

            if hasattr(base_lm, "lm_head") and hasattr(model, "lm_head"):
                model.lm_head.load_state_dict(base_lm.lm_head.state_dict(), strict=False)

        # 可选：释放 base_lm 的视觉部分省显存/内存
            try:
                del base_lm.model.visual
            except Exception:
                pass

        else:
        # 非 VL（纯文本）保持你原来的灌法即可
            if hasattr(base_lm, "model") and hasattr(model.model, "qwen_model"):
                model.model.qwen_model.load_state_dict(base_lm.model.state_dict(), strict=False)
            if hasattr(base_lm, "lm_head") and hasattr(model, "lm_head"):
                model.lm_head.load_state_dict(base_lm.lm_head.state_dict(), strict=False)

        return model


    def initial_mask_module(self, pretrained_path=None, model_args=None):
        if not self.is_train_mask_decode:
            print("Initialize mask modules...")
            self.config.mask_decode_train = True

        self.attention_loss = AttentionLoss()

        self.test_topk_per_image = self.mask_decoder_cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        input_shape = self.output_shape()
        self.pixel_decoder = self.pixel_decoder_init(cfg=self.mask_decoder_cfg, input_shape=input_shape)
        self.predictor = self.predictor_init(cfg=self.mask_decoder_cfg)

        self.SEG_token_projector = nn.Linear(
            _get_hidden_size(self.config),
            self.mask_decoder_cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
        )

        self.mask_decoder_training_init(self.mask_decoder_cfg)

        if pretrained_path is not None:
            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            def change_w(weights, old_name, new_name):
                weights[new_name] = weights[old_name]
                weights.pop(old_name)

            if pretrained_path.endswith(".pkl"):
                with open(pretrained_path, "rb") as f:
                    ckpt = pickle.load(f)
            else:
                ckpt = torch.load(pretrained_path)

            pixel_decoder_weights = get_w(ckpt["model"], "sem_seg_head.pixel_decoder")
            predictor_weights = get_w(ckpt["model"], "sem_seg_head.predictor")
            pixel_decoder_weights = {k: torch.tensor(v) for k, v in pixel_decoder_weights.items()}
            predictor_weights = {k: torch.tensor(v) for k, v in predictor_weights.items()}

            change_w(pixel_decoder_weights, "adapter_1.weight", "adapter_1.0.weight")
            change_w(pixel_decoder_weights, "adapter_1.norm.weight", "adapter_1.1.weight")
            change_w(pixel_decoder_weights, "adapter_1.norm.bias", "adapter_1.1.bias")
            change_w(pixel_decoder_weights, "layer_1.weight", "layer_1.0.weight")
            change_w(pixel_decoder_weights, "layer_1.norm.weight", "layer_1.1.weight")
            change_w(pixel_decoder_weights, "layer_1.norm.bias", "layer_1.1.bias")

            if "static_query.weight" in predictor_weights:
                change_w(predictor_weights, "static_query.weight", "query_feat.weight")

            if predictor_weights["query_embed.weight"].shape[0] == 200:
                predictor_weights["query_embed.weight"] = predictor_weights["query_embed.weight"][:100, :]

            diff_pixel_msg = self.pixel_decoder.load_state_dict(pixel_decoder_weights, strict=False)
            diff_predictor_msg = self.predictor.load_state_dict(predictor_weights, strict=False)
            print(diff_predictor_msg)
            print(diff_pixel_msg)

    def get_vision_tower_feature(self, images):
        features = self.get_model().get_vision_tower_mask()(images)
        features_dict = {
            "res2": features[0],
            "res3": features[1],
            "res4": features[2],
            "res5": features[3],
        }
        return features_dict

    def mask_decoder_training_init(self, cfg):
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

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
        }

        self.weight_dict = weight_dict

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["SEG_labels", "masks"]
        self.criterion = Criterion(
            matcher=matcher,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            device=self.device,
        )
        self.size_divisibility = 32
        self.sem_seg_postprocess_before_inference = True

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
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
        )
        return predictor

    def get_model(self):
        return self.model
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
    # 让 Trainer / PEFT 能调用到
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}

    # 标记 config（有些代码会读这个）
        if hasattr(self, "config"):
            setattr(self.config, "gradient_checkpointing", True)

    # 尝试把底层语言模型真正打开 checkpointing
    # 你的底层主干通常在 self.model.qwen_model（MiphaQwenModel 内）
        qwen_core = None
        if hasattr(self, "model") and hasattr(self.model, "qwen_model"):
            qwen_core = self.model.qwen_model
        elif hasattr(self, "model"):
            qwen_core = self.model

        if qwen_core is not None and hasattr(qwen_core, "gradient_checkpointing_enable"):
            try:
                qwen_core.gradient_checkpointing_enable(**gradient_checkpointing_kwargs)
            except TypeError:
            # 有些实现不收 kwargs
                qwen_core.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        if hasattr(self, "config"):
            setattr(self.config, "gradient_checkpointing", False)

        qwen_core = None
        if hasattr(self, "model") and hasattr(self.model, "qwen_model"):
            qwen_core = self.model.qwen_model
        elif hasattr(self, "model"):
            qwen_core = self.model

        if qwen_core is not None and hasattr(qwen_core, "gradient_checkpointing_disable"):
            qwen_core.gradient_checkpointing_disable()
    def output_shape(self):
        out_features = self.mask_decoder_cfg.MODEL.SWIN.OUT_FEATURES
        out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        num_features = [
            int(self.mask_decoder_cfg.MODEL.SWIN.EMBED_DIM * 2 ** i)
            for i in range(len(self.mask_decoder_cfg.MODEL.SWIN.DEPTHS))
        ]
        out_feature_channels = {
            "res2": num_features[0],
            "res3": num_features[1],
            "res4": num_features[2],
            "res5": num_features[3],
        }
        backbone_feature_shape = dict()
        for name in out_features:
            backbone_feature_shape[name] = Dict(
                {"channel": out_feature_channels[name], "stride": out_feature_strides[name]}
            )
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
        transformer_in_features = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES

        pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape,
            transformer_dropout,
            transformer_nheads,
            transformer_dim_feedforward,
            transformer_enc_layers,
            conv_dim,
            mask_dim,
            transformer_in_features,
            common_stride,
        )
        return pixel_decoder

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        has_gt_ids = False
        if hasattr(targets[0], "gt_ids"):
            has_gt_ids = True

        for targets_per_image in targets:
            if has_gt_ids:
                inst_ids = targets_per_image.gt_ids
                valid_id = inst_ids != -1
            else:
                inst_ids = None
                valid_id = None

            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
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

    # ===== 以下三段尽量原样复用 llava_phi.py =====
    # concat_image_seg_cls_embeds
    # prepare_inputs_labels_for_multimodal
    # get_SEG_embedding
    #
    # 为了避免你手改出错，建议从 llava_phi.py 原样复制。
    # 我这里先把函数壳给出来，提醒你必须保留逻辑一致。

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
        inputs_embeds: Optional[torch.Tensor] = None,
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
        dataset_type=None,
    ) -> Union[Tuple, CausalOutputWithMask]:

        if dataset_type is not None:
            assert all(item == dataset_type[0] for item in dataset_type), \
                f"this batch contain different dataset_type: {dataset_type}"
            batch_dataset_type = dataset_type[0]
        else:
            batch_dataset_type = []

        output_attentions = True
        output_hidden_states = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_features = None
        bs = input_ids.shape[0] if input_ids is not None else 0

        if SEG_token_embedding_indices is not None and (SEG_token_embedding_indices == 1).sum() != 0:
            if input_ids.shape[1] != 1:
                image_features = self.get_vision_tower_feature(images)
                bs = input_ids.shape[0]

            input_ids, attention_mask, past_key_values, inputs_embeds, labels, \
            SEG_token_embedding_indices, image_features_indices = \
                self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images_clip,
                    token_refer_id=token_refer_id,
                    SEG_token_embedding_indices=SEG_token_embedding_indices,
                )
        else:
            image_features_indices = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # DEBUG: inspect raw attention shapes
        if global_step is None or global_step < 3:
            print("=== DEBUG raw attentions ===")
            print("num attention layers:", len(outputs.attentions))
            for i, att in enumerate(outputs.attentions[:3]):
                print(f"layer {i} raw attention shape:", att.shape)

        # 用 mean 比 sum 更稳，避免数值尺度过大
        attentions = [attention_item.mean(dim=1) for attention_item in outputs.attentions]

        if global_step is None or global_step < 3:
            print("=== DEBUG reduced attentions ===")
            for i, att in enumerate(attentions[:3]):
                print(f"layer {i} reduced attention shape:", att.shape)

        SEG_embedding = self.SEG_token_projector(
            self.get_SEG_embedding(hidden_states, SEG_token_embedding_indices)
        )

        mask_features, transformer_encoder_features, multi_scale_features = \
            self.pixel_decoder.forward_features(image_features)

        mask_num = torch.tensor(mask_num, device=mask_features.device)
        mask_features = torch.repeat_interleave(mask_features, repeats=mask_num, dim=0)
        multi_scale_features = [
            torch.repeat_interleave(feat, repeats=mask_num, dim=0)
            for feat in multi_scale_features
        ]

        mask_outputs = self.predictor(
            multi_scale_features,
            mask_features,
            None,
            None,
            SEG_embedding,
        )

        loss = None
        llm_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            vocab_size = shift_logits.shape[-1]
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            llm_loss = loss_fct(shift_logits, shift_labels)

        mask_loss = None
        loss_mask = torch.tensor(0.0, device=logits.device)
        loss_dice = torch.tensor(0.0, device=logits.device)

        if seg_info is not None:
            if "padding_mask" in seg_info[0]:
                if isinstance(seg_info[0]["instances"], list):
                    gt_instances = [x["instances"][0].to(self.device) for x in seg_info]
                else:
                    gt_instances = [x["instances"].to(self.device) for x in seg_info]
                targets = self.prepare_targets(gt_instances, images)
            elif "mask" in seg_info[0]:
                targets = []
                for gt_mask in seg_info:
                    targets.append(
                        {
                            "labels": torch.tensor([0]).to(mask_outputs["pred_masks"].device),
                            "masks": gt_mask["mask"].to(mask_outputs["pred_masks"].device),
                            "valid": None,
                            "inst_id": None,
                        }
                    )
            else:
                targets = None

            mask_losses = self.criterion(mask_outputs, targets)
            weight_dict = self.weight_dict

            for k in list(mask_losses.keys()):
                if k in weight_dict:
                    if mask_losses[k] is not None:
                        mask_losses[k] *= weight_dict[k]

                    if "_mask" in k:
                        loss_mask += mask_losses[k]
                    elif "_dice" in k:
                        loss_dice += mask_losses[k]
                else:
                    mask_losses.pop(k)

            mask_loss = loss_mask + loss_dice

        loss_attention = torch.tensor(0.0, device=logits.device)

        if seg_info is not None and image_features_indices is not None and SEG_token_embedding_indices is not None:
            masks = [_seg_info["mask"] for _seg_info in seg_info]
            masks_resized = [
                F.interpolate(m.unsqueeze(0).float(), size=(800, 800), mode="nearest").squeeze(0)
                for m in masks
            ]
            masks = torch.stack(masks_resized, dim=0)
            masks_down = F.interpolate(masks, size=(27, 27), mode="bilinear", align_corners=False)
            masks_down = masks_down.view(masks_down.size(0), -1)
            masks_down[masks_down > 0] = 1

            if global_step is None or global_step < 3:
                print("=== DEBUG masks_down ===")
                print("masks_down.shape:", masks_down.shape)
                print("masks_down positive counts:", (masks_down > 0).sum(dim=1)[:10])

            for layer_idx, full_attention_map in enumerate(attentions):
                batch_attentions_list = []

                if global_step is None or global_step < 3:
                    print(f"=== DEBUG attention layer {layer_idx} ===")
                    print("full_attention_map.shape:", full_attention_map.shape)

                for batch_idx in range(bs):
                    attention_map = full_attention_map[batch_idx]
                    SEG_mask = SEG_token_embedding_indices[batch_idx].bool()
                    image_features_mask = image_features_indices[batch_idx].bool()

                    attention = attention_map[SEG_mask][:, image_features_mask]
                    batch_attentions_list.append(attention)

                    if global_step is None or global_step < 3:
                        print(f"[layer {layer_idx}][batch {batch_idx}] attention_map.shape:", attention_map.shape)
                        print(f"[layer {layer_idx}][batch {batch_idx}] SEG count:", SEG_mask.sum().item())
                        print(f"[layer {layer_idx}][batch {batch_idx}] image token count:", image_features_mask.sum().item())
                        print(f"[layer {layer_idx}][batch {batch_idx}] sliced attention.shape:", attention.shape)

                if len(batch_attentions_list) > 0:
                    batch_attentions = torch.cat(batch_attentions_list, dim=0)

                    if global_step is None or global_step < 3:
                        print(f"[layer {layer_idx}] batch_attentions.shape:", batch_attentions.shape)
                        print(f"[layer {layer_idx}] masks_down.shape:", masks_down.shape)

                    if batch_attentions.shape[0] == masks_down.shape[0] and batch_attentions.shape[1] == masks_down.shape[1]:
                        att_loss_item = self.attention_loss(batch_attentions, masks_down)

                        if global_step is None or global_step < 3:
                            print(f"[layer {layer_idx}] att_loss_item:", att_loss_item)

                        loss_attention += att_loss_item
                    else:
                        if global_step is None or global_step < 3:
                            print(
                                f"[layer {layer_idx}] shape mismatch, skip attention loss: "
                                f"batch_attentions.shape={batch_attentions.shape}, "
                                f"masks_down.shape={masks_down.shape}"
                            )

        if llm_loss is None:
            llm_loss = torch.tensor(0.0, device=logits.device)
        if mask_loss is None:
            mask_loss = torch.tensor(0.0, device=logits.device)

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
            loss_attention=(0.01 * loss_attention).detach(),
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
    ):
        output_attentions = False
        output_hidden_states = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_features = self.get_vision_tower_feature(images)

        input_ids, attention_mask, past_key_values, inputs_embeds, labels, \
        SEG_token_embedding_indices, image_features_indices = \
            self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images_clip,
                token_refer_id=token_refer_id,
                SEG_token_embedding_indices=SEG_token_embedding_indices,
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state
        SEG_embedding = self.SEG_token_projector(
            self.get_SEG_embedding(hidden_states, SEG_token_embedding_indices)
        )

        mask_features, transformer_encoder_features, multi_scale_features = \
            self.pixel_decoder.forward_features(image_features)

        images = [image.repeat((num, 1, 1, 1)) for image, num in zip(images, mask_num)]
        images = [s[0] for image_repeat in images for s in torch.split(image_repeat, 1, dim=0)]

        mask_num = torch.tensor(mask_num, device=mask_features.device)
        mask_features = torch.repeat_interleave(mask_features, repeats=mask_num, dim=0)
        multi_scale_features = [
            torch.repeat_interleave(feat, repeats=mask_num, dim=0)
            for feat in multi_scale_features
        ]

        mask_outputs = self.predictor(
            multi_scale_features,
            mask_features,
            None,
            None,
            SEG_embedding,
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
                "pred": ((mask_pred_result.cpu().numpy() > 0) * 255).astype(np.uint8),
                "image_name": _seg_info["image_id"],
                "id": _seg_info["data_id"],
                "mask_id": _seg_info["mask_id"],
            }
            processed_results.append(instance_r)
        return processed_results