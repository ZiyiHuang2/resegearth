import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageTextAlignmentAdapter(nn.Module):
<<<<<<< HEAD
    def __init__(self, image_dim=256, llm_dim=4096, hidden_dim=512, num_heads=8, dropout=0.0):
        super().__init__()
=======
    """
    Image-Text Alignment Adapter (ITAA)

    当前版本目标：
    1. 主前向在训练 / 推理保持一致：
       始终使用 image feature 的全局池化 token + dense visual tokens 做 cross-attention。
    2. GT mask 仅用于 align_loss，不直接改变主前向给 decoder 的 query。
    3. 兼容 seg_embedding 输入维度：
       - llm_dim（原始 LLM hidden）
       - image_dim（已经过 projector 的 query）
    """

    def __init__(
        self,
        image_dim=256,
        llm_dim=4096,
        hidden_dim=512,
        num_heads=8,
        dropout=0.0,
        seg_gate_init=0.5,
    ):
        super(ImageTextAlignmentAdapter, self).__init__()
>>>>>>> afa692edf9184821a18a71587130b4a93163d82e

        self.image_dim = image_dim
        self.llm_dim = llm_dim

<<<<<<< HEAD
        # text projection
        self.text_proj = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, image_dim)
        )

=======
        # seg 输入如果还是 llm_dim，用这个投到 image_dim
        self.text_proj = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, image_dim),
        )

        # seg 输入如果已经是 image_dim，再过一个轻量映射
>>>>>>> afa692edf9184821a18a71587130b4a93163d82e
        self.text_proj_image_dim = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, image_dim),
        )
<<<<<<< HEAD

        self.query_norm = nn.LayerNorm(image_dim)

        # image projection
        self.image_proj = nn.Conv2d(image_dim, image_dim, kernel_size=1)

        self.image_norm = nn.LayerNorm(image_dim)

=======
        self.query_norm = nn.LayerNorm(image_dim)

        # image/value branch
        self.image_proj = nn.Conv2d(image_dim, image_dim, kernel_size=1)
        self.image_norm = nn.LayerNorm(image_dim)

        # pooled token projection
>>>>>>> afa692edf9184821a18a71587130b4a93163d82e
        self.mask_proj = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.GELU(),
            nn.Linear(image_dim, image_dim),
        )

<<<<<<< HEAD
        # cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.attn_norm = nn.LayerNorm(image_dim)

        self.mlp = nn.Sequential(
            nn.Linear(image_dim, image_dim * 4),
            nn.GELU(),
            nn.Linear(image_dim * 4, image_dim),
        )

        self.mlp_norm = nn.LayerNorm(image_dim)

        self.seg_gate = nn.Parameter(torch.tensor(0.0))
=======
        # query <- image cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(image_dim)

        # MLP + residual + norm
        self.mlp = nn.Sequential(
            nn.Linear(image_dim, image_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(image_dim * 4, image_dim),
        )
        self.mlp_norm = nn.LayerNorm(image_dim)

        # 残差门控，初始不为 0，避免新分支前期几乎不生效
        self.seg_gate = nn.Parameter(torch.tensor(float(seg_gate_init)))
>>>>>>> afa692edf9184821a18a71587130b4a93163d82e

    def _normalize_seg_embedding(self, seg_embedding):
        if seg_embedding.dim() == 3:
            if seg_embedding.shape[1] == 1:
                seg_embedding = seg_embedding.squeeze(1)
            else:
                seg_embedding = seg_embedding.mean(dim=1)
<<<<<<< HEAD
        return seg_embedding

    def _pool_image_feature(self, image_feat, mask=None):
        if mask is None:
            return image_feat.mean(dim=(2, 3))

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        mask = F.interpolate(mask.float(), size=image_feat.shape[-2:], mode='nearest')
        masked_feat = image_feat * mask

        area = mask.sum(dim=(2, 3)).clamp_min(1e-6)
        return masked_feat.sum(dim=(2, 3)) / area

    def _build_query_to_image_index(self, Q, B, mask_num=None, device=None):
        if mask_num is not None:
            if not torch.is_tensor(mask_num):
                mask_num = torch.tensor(mask_num, device=device)
            return torch.repeat_interleave(torch.arange(B, device=device), mask_num)
        return torch.arange(Q, device=device)
=======
        elif seg_embedding.dim() != 2:
            raise ValueError(
                f"seg_embedding must be 2D or 3D, but got shape {tuple(seg_embedding.shape)}"
            )
        return seg_embedding

    def _project_text_feature(self, seg_embedding):
        """
        兼容两种输入：
        1. 原始 LLM hidden: [Q, llm_dim]
        2. 已经 projector 后的 query: [Q, image_dim]
        """
        if seg_embedding.shape[-1] == self.llm_dim:
            text_feat = self.text_proj(seg_embedding)
        elif seg_embedding.shape[-1] == self.image_dim:
            text_feat = self.text_proj_image_dim(seg_embedding)
        else:
            raise ValueError(
                f"Unsupported seg embedding dim {seg_embedding.shape[-1]}, "
                f"expected {self.llm_dim} or {self.image_dim}"
            )
        return self.query_norm(text_feat)

    def _pool_image_feature(self, image_feat, gt_mask=None):
        if gt_mask is not None:
            if gt_mask.dim() == 3:
                gt_mask = gt_mask.unsqueeze(1)

            if gt_mask.shape[-2:] != image_feat.shape[-2:]:
                gt_mask = F.interpolate(
                    gt_mask.to(dtype=image_feat.dtype),
                    size=image_feat.shape[-2:],
                    mode="nearest",
                )
            else:
                gt_mask = gt_mask.to(dtype=image_feat.dtype)

            masked_feat = image_feat * gt_mask
            mask_area = gt_mask.sum(dim=(2, 3)).clamp_min(1e-6)
            return masked_feat.sum(dim=(2, 3)) / mask_area

        # 无 GT 时统一使用全局平均池化
        return image_feat.mean(dim=(2, 3))

    def _build_query_to_image_index(
        self,
        num_queries,
        batch_size,
        mask_num=None,
        query_to_image_index=None,
        device=None,
    ):
        if query_to_image_index is not None:
            if not torch.is_tensor(query_to_image_index):
                query_to_image_index = torch.tensor(query_to_image_index, device=device)
            query_to_image_index = query_to_image_index.to(device=device, dtype=torch.long)
            if query_to_image_index.numel() != num_queries:
                raise ValueError(
                    f"query_to_image_index size mismatch, expected {num_queries}, got {query_to_image_index.numel()}"
                )
            return query_to_image_index

        if mask_num is not None:
            if not torch.is_tensor(mask_num):
                mask_num = torch.tensor(mask_num, device=device)
            mask_num = mask_num.to(device=device, dtype=torch.long)
            if mask_num.numel() == batch_size and mask_num.sum().item() == num_queries:
                return torch.repeat_interleave(
                    torch.arange(batch_size, device=device),
                    repeats=mask_num,
                )

        if num_queries == batch_size:
            return torch.arange(batch_size, device=device, dtype=torch.long)

        raise ValueError(
            f"Cannot infer query->image mapping: num_queries={num_queries}, batch_size={batch_size}"
        )
>>>>>>> afa692edf9184821a18a71587130b4a93163d82e

    def forward(
        self,
        image_embedding,
        seg_embedding,
        gt_mask=None,
        mask_num=None,
<<<<<<< HEAD
    ):
=======
        query_to_image_index=None,
        gt_masks_per_query=None,
    ):
        """
        Args:
            image_embedding: [B, C, H, W]
            seg_embedding: [Q, 1, D] or [Q, D]
            gt_mask: optional GT mask
            mask_num: queries per image
            query_to_image_index: [Q]
            gt_masks_per_query: optional per-query GT mask

        Returns:
            fused_seg_feat: [Q, 1, C]
            align_loss: scalar
        """
>>>>>>> afa692edf9184821a18a71587130b4a93163d82e
        B, C, H, W = image_embedding.shape

        seg_embedding = self._normalize_seg_embedding(seg_embedding)
        Q = seg_embedding.shape[0]

        query_to_image_index = self._build_query_to_image_index(
<<<<<<< HEAD
            Q, B, mask_num, image_embedding.device
        )

        # ---- text ----
        if seg_embedding.shape[-1] == self.llm_dim:
            text_feat = self.query_norm(self.text_proj(seg_embedding))
        else:
            text_feat = self.query_norm(self.text_proj_image_dim(seg_embedding))

        # ---- image ----
        img_feat = self.image_proj(image_embedding)
        per_query_feat = img_feat[query_to_image_index]

        # ---- align loss ----
        align_loss = torch.tensor(0.0, device=image_embedding.device)

        if self.training and gt_mask is not None:
            pooled = self._pool_image_feature(img_feat, gt_mask)
            align_loss = F.mse_loss(text_feat, pooled)

        # ---- cross attention ----
        value_tokens = per_query_feat.flatten(2).transpose(1, 2)

        query = text_feat.unsqueeze(1)

        attn_out, _ = self.cross_attn(query, value_tokens, value_tokens)
        fused = self.attn_norm(query + attn_out)
        fused = self.mlp_norm(fused + self.mlp(fused))

        out = text_feat + self.seg_gate * fused.squeeze(1)

        return out.unsqueeze(1), align_loss
=======
            num_queries=Q,
            batch_size=B,
            mask_num=mask_num,
            query_to_image_index=query_to_image_index,
            device=image_embedding.device,
        )
        query_to_image_index = query_to_image_index.clamp(min=0, max=B - 1)

        # text query
        text_feat = self._project_text_feature(seg_embedding)  # [Q, C]

        # image features
        img_feat_proj = self.image_proj(image_embedding)              # [B, C, H, W]
        per_query_image_feat = img_feat_proj[query_to_image_index]   # [Q, C, H, W]

        # 主前向统一使用全局 pooled token，保证 train / eval 一致
        global_feat = self._pool_image_feature(per_query_image_feat)     # [Q, C]
        global_feat = self.image_norm(self.mask_proj(global_feat))       # [Q, C]

        # GT mask 只用于 align loss
        per_query_mask = gt_masks_per_query if gt_masks_per_query is not None else gt_mask
        align_loss = torch.tensor(
            0.0,
            device=image_embedding.device,
            dtype=img_feat_proj.dtype,
        )

        if self.training and per_query_mask is not None:
            if per_query_mask.dim() == 3:
                per_query_mask = per_query_mask.unsqueeze(1)

            if per_query_mask.shape[0] != Q:
                raise ValueError(
                    f"gt mask size mismatch, expected first dim {Q}, got {per_query_mask.shape[0]}"
                )

            masked_feat = self._pool_image_feature(per_query_image_feat, per_query_mask)
            masked_feat = self.image_norm(self.mask_proj(masked_feat))
            align_loss = F.mse_loss(text_feat.float(), masked_feat.float())

        # cross attention: query <- image tokens + global token
        value_tokens = per_query_image_feat.flatten(2).transpose(1, 2)  # [Q, HW, C]
        value_tokens = self.image_norm(value_tokens)
        kv_tokens = torch.cat([value_tokens, global_feat.unsqueeze(1)], dim=1)

        query = text_feat.unsqueeze(1)  # [Q, 1, C]
        attn_out, _ = self.cross_attn(
            query=query,
            key=kv_tokens,
            value=kv_tokens,
            need_weights=False,
        )
        fused = self.attn_norm(query + attn_out)
        fused = self.mlp_norm(fused + self.mlp(fused))

        fused_seg_feat = text_feat + self.seg_gate * fused.squeeze(1)
        return fused_seg_feat.unsqueeze(1), align_loss
>>>>>>> afa692edf9184821a18a71587130b4a93163d82e
