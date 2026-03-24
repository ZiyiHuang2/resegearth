from transformers import AutoConfig

from ..language_model.configuration_mipha import MiphaVisionConfig
from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SiglipVisionTower
from .dinov2_encoder import Dinov2VisionTower


def _extract_vision_config_dict(config):
    # 对于像 SigLIP / CLIP 这种顶层 config 里嵌 vision_config 的情况，
    # 必须取真正的 vision 子配置；否则会退回 MiphaVisionConfig 默认值。
    if hasattr(config, "vision_config") and config.vision_config is not None:
        if hasattr(config.vision_config, "to_dict"):
            return config.vision_config.to_dict()
        return dict(config.vision_config)

    return config.to_dict()


def build_vision_tower(vision_tower_name_or_path: str):
    config = AutoConfig.from_pretrained(vision_tower_name_or_path)

    vision_config_dict = _extract_vision_config_dict(config)
    phi_vis_config = MiphaVisionConfig(**vision_config_dict)
    phi_vis_config.vision_model_name_or_path = vision_tower_name_or_path

    model_type = getattr(config, "model_type", "").lower()
    vision_model_type = vision_config_dict.get("model_type", "").lower()

    # 先按 vision 子配置判断
    if vision_model_type in ["siglip_vision_model", "siglip"]:
        return SiglipVisionTower(phi_vis_config)

    if vision_model_type in ["clip_vision_model", "clip"]:
        return CLIPVisionTower(phi_vis_config)

    if vision_model_type in ["dinov2", "dinov2_model"]:
        return Dinov2VisionTower(phi_vis_config)

    # 再按顶层 config 判断
    if model_type == "siglip":
        return SiglipVisionTower(phi_vis_config)

    if model_type == "clip":
        return CLIPVisionTower(phi_vis_config)

    if model_type == "dinov2":
        return Dinov2VisionTower(phi_vis_config)

    # 最后用路径名兜底
    vt = vision_tower_name_or_path.lower()

    if "siglip" in vt:
        return SiglipVisionTower(phi_vis_config)

    if "clip" in vt:
        return CLIPVisionTower(phi_vis_config)

    if "dinov2" in vt:
        return Dinov2VisionTower(phi_vis_config)

    raise ValueError(
        f"Unsupported vision tower: {vision_tower_name_or_path}. "
        f"model_type={model_type}, vision_model_type={vision_model_type}. "
        f"Currently supported: siglip, clip, dinov2."
    )