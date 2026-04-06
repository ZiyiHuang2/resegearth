from abc import ABC

import torch
import torch.nn as nn

from transformers.models.siglip import SiglipPreTrainedModel, SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer
from ..language_model.configuration_mipha import MiphaVisionConfig


class SiglipVisionTower(SiglipPreTrainedModel):
    config_class = MiphaVisionConfig

    def __init__(self, config):
        super().__init__(config)

        self.vision_model = SiglipVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()
        # ---- workaround: transformers PreTrainedModel.device 依赖 parameters() 非空 ----
        if len(list(self.vision_model.parameters())) == 0:
            self.vision_model.register_parameter(
                "_dummy_param", nn.Parameter(torch.empty(0), requires_grad=False)
            )
# 即便不为空也可以直接注册（不影响训练），想更简单就去掉 if，直接注册也行：
# self.vision_model.register_parameter("_dummy_param", nn.Parameter(torch.empty(0), requires_grad=False))
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def feature_select(self, image_forward_outs):
    # 有些 transformers 版本即使传了 output_hidden_states=True，hidden_states 仍可能为 None
        hidden_states = getattr(image_forward_outs, "hidden_states", None)
        if hidden_states is None:
        # fallback：用最后一层输出
            image_features = image_forward_outs.last_hidden_state
        else:
        # 正常路径：按配置选层
            image_features = hidden_states[self.config.mm_vision_select_layer]

    # 你原来的选择逻辑保持不变
        if self.config.mm_vision_select_feature == "patch":
            return image_features
        elif self.config.mm_vision_select_feature == "cls_patch":
            return image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.config.mm_vision_select_feature}")

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_model(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                    return_dict=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_model(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                return_dict=True,
                interpolate_pos_encoding=True,
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

@property
def dtype(self):
    p = next(self.vision_model.parameters(), None)
    if p is not None:
        return p.dtype
    b = next(self.vision_model.buffers(), None)
    if b is not None:
        return b.dtype
    # 兜底：避免崩溃
    return torch.float32

@property
def device(self):
    p = next(self.vision_model.parameters(), None)
    if p is not None:
        return p.device
    b = next(self.vision_model.buffers(), None)
    if b is not None:
        return b.device
    # 兜底：避免崩溃。训练一般在 CUDA 上，就优先返回 cuda
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
