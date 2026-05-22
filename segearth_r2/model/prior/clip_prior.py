import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer


@dataclass
class CLIPPriorConfig:
    map_size: int = 27
    normalize: bool = True


class CLIPPriorGenerator(nn.Module):
    def __init__(
        self,
        clip_model,
        tokenizer,
        config: CLIPPriorConfig,
        clip_model_name_or_path: str,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.config = config
        self.clip_model_name_or_path = clip_model_name_or_path

        # 冻结 image encoder
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # ===== 关键修复：稳健初始化 text tokenizer =====
        self.text_tokenizer = self._build_text_tokenizer(clip_model_name_or_path)

        # ===== 关键修复：稳健初始化 text model =====
        self.text_model = AutoModel.from_pretrained(
            clip_model_name_or_path,
            trust_remote_code=True,
        )
        self.text_model.eval()
        for p in self.text_model.parameters():
            p.requires_grad = False

        # 只记录当前 device，真正迁移由外部调用 .to(device) 统一处理
        self._text_model_device: Optional[torch.device] = None

    def _build_text_tokenizer(self, model_path: str):
        # 优先尝试 processor，自带 tokenizer 时最稳
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
                return processor.tokenizer
        except Exception:
            pass

        # 再尝试慢速 tokenizer，避免 fast tokenizer 解析 tokenizer.json 崩掉
        try:
            return AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                trust_remote_code=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to build text tokenizer from {model_path}. "
                f"Please check whether this vision tower path contains a usable text tokenizer "
                f"or provide a compatible processor/tokenizer. Original error: {e}"
            ) from e

    @staticmethod
    def _to_spatial_map(patch_similarity: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens = patch_similarity.shape
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            if num_tokens > 1:
                candidate = int(math.sqrt(num_tokens - 1))
                if candidate * candidate == (num_tokens - 1):
                    # 去掉 cls token
                    patch_similarity = patch_similarity[:, 1:]
                    side = candidate
                else:
                    raise ValueError(
                        f"Cannot reshape patch tokens to square map: num_tokens={num_tokens}"
                    )
            else:
                raise ValueError(
                    f"Cannot reshape patch tokens to square map: num_tokens={num_tokens}"
                )
        return patch_similarity.reshape(batch_size, 1, side, side)

    def _move_text_model_once(self, device: torch.device):
        # 只在 device 变化时迁移一次，不要每个 step 重复搬
        if self._text_model_device != device:
            self.text_model.to(device=device)
            self._text_model_device = device

    def _encode_text(self, texts: List[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        encoded = self.text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        self._move_text_model_once(device)

        with torch.no_grad():
            # 优先使用 get_text_features
            if hasattr(self.text_model, "get_text_features"):
                text_features = self.text_model.get_text_features(**encoded)
            else:
                outputs = self.text_model(**encoded)

                if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                    text_features = outputs.text_embeds
                elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    text_features = outputs.pooler_output
                else:
                    text_features = outputs.last_hidden_state[:, 0]

        return text_features.to(dtype=dtype)

    def forward(self, images_clip: torch.Tensor, texts: List[str]) -> torch.Tensor:
        if images_clip is None or texts is None:
            raise ValueError("images_clip and texts must not be None for CLIP prior generation.")
        if len(texts) != images_clip.shape[0]:
            raise ValueError(
                f"Batch size mismatch between images and texts: {images_clip.shape[0]} vs {len(texts)}."
            )

        with torch.no_grad():
            patch_features = self.clip_model(images_clip)

        # 兼容 [B, C, H, W] -> [B, HW, C]
        if patch_features.dim() == 4:
            bsz, channels, h, w = patch_features.shape
            patch_features = (
                patch_features.reshape(bsz, channels, h * w)
                .transpose(1, 2)
                .contiguous()
            )

        patch_features = patch_features.float()
        text_features = self._encode_text(
            texts,
            patch_features.device,
            patch_features.dtype,
        ).float()

        patch_features = F.normalize(patch_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        similarity = torch.einsum("bnd,bd->bn", patch_features, text_features)
        prior_map = self._to_spatial_map(similarity)

        if self.config.map_size is not None:
            prior_map = F.interpolate(
                prior_map,
                size=(self.config.map_size, self.config.map_size),
                mode="bilinear",
                align_corners=False,
            )

        if self.config.normalize:
            map_min = prior_map.amin(dim=(2, 3), keepdim=True)
            map_max = prior_map.amax(dim=(2, 3), keepdim=True)
            denom = (map_max - map_min).clamp(min=1e-6)
            prior_map = (prior_map - map_min) / denom

        return prior_map.to(dtype=images_clip.dtype, device=images_clip.device)