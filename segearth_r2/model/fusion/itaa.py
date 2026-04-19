import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageTextAlignmentAdapter(nn.Module):
    def __init__(self, image_dim=256, llm_dim=4096, hidden_dim=512, num_heads=8, dropout=0.0):
        super().__init__()

        self.image_dim = image_dim
        self.llm_dim = llm_dim

        # llm query -> image dim
        self.text_proj = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, image_dim),
        )

        # 对已经是 image_dim 的 SEG embedding，不再重复重投影
        self.text_proj_image_dim = nn.Identity()

        self.query_norm = nn.LayerNorm(image_dim)

        # image projection
        self.image_proj = nn.Conv2d(image_dim, image_dim, kernel_size=1)
        self.image_norm = nn.LayerNorm(image_dim)

        # query <- image cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(image_dim)

        self.mlp = nn.Sequential(
            nn.Linear(image_dim, image_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(image_dim * 4, image_dim),
        )
        self.mlp_norm = nn.LayerNorm(image_dim)

        self.seg_gate = nn.Parameter(torch.tensor(0.0))

    def _normalize_seg_embedding(self, seg_embedding):
        if seg_embedding.dim() == 3:
            if seg_embedding.shape[1] == 1:
                seg_embedding = seg_embedding.squeeze(1)
            else:
                seg_embedding = seg_embedding.mean(dim=1)
        elif seg_embedding.dim() != 2:
            raise ValueError(
                f"seg_embedding must be 2D or 3D, but got shape {tuple(seg_embedding.shape)}"
            )
        return seg_embedding

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
                    torch.arange(batch_size, device=device, dtype=torch.long),
                    repeats=mask_num,
                )

        if num_queries == batch_size:
            return torch.arange(batch_size, device=device, dtype=torch.long)

        raise ValueError(
            f"Cannot infer query->image mapping: num_queries={num_queries}, batch_size={batch_size}"
        )

    def _dense_attention_align_loss(self, text_feat, per_query_image_feat, per_query_mask):
        """
        text_feat: [Q, C]
        per_query_image_feat: [Q, C, H, W]
        per_query_mask: [Q, 1, Hm, Wm] or [Q, Hm, Wm]
        """
        if per_query_mask.dim() == 3:
            per_query_mask = per_query_mask.unsqueeze(1)

        gt_mask_resized = F.interpolate(
            per_query_mask.float(),
            size=per_query_image_feat.shape[-2:],
            mode="nearest",
        )  # [Q,1,H,W]

        # query x dense feature map -> attention logit map
        attn_map = torch.sum(
            per_query_image_feat * text_feat.unsqueeze(-1).unsqueeze(-1),
            dim=1,
        )  # [Q,H,W]

        align_loss = F.binary_cross_entropy_with_logits(
            attn_map.float(),
            gt_mask_resized.squeeze(1).float(),
        )
        return align_loss

    def forward(
        self,
        image_embedding,
        seg_embedding,
        gt_mask=None,
        mask_num=None,
        query_to_image_index=None,
        gt_masks_per_query=None,
    ):
        """
        image_embedding: [B, C, H, W]
        seg_embedding: [Q, 1, D] or [Q, D]
        gt_mask / gt_masks_per_query:
            - [B, 1, H, W] / [Q, 1, H, W]
            - 用于 dense attention supervision
        """
        B, C, H, W = image_embedding.shape

        seg_embedding = self._normalize_seg_embedding(seg_embedding)  # [Q, D]
        Q = seg_embedding.shape[0]

        query_to_image_index = self._build_query_to_image_index(
            num_queries=Q,
            batch_size=B,
            mask_num=mask_num,
            query_to_image_index=query_to_image_index,
            device=image_embedding.device,
        )
        query_to_image_index = query_to_image_index.clamp(min=0, max=B - 1)

        # query branch
        if seg_embedding.shape[-1] == self.llm_dim:
            text_feat = self.query_norm(self.text_proj(seg_embedding))   # [Q, C]
        elif seg_embedding.shape[-1] == self.image_dim:
            text_feat = self.query_norm(self.text_proj_image_dim(seg_embedding))  # [Q, C]
        else:
            raise ValueError(
                f"Unsupported seg embedding dim {seg_embedding.shape[-1]}, "
                f"expected {self.llm_dim} or {self.image_dim}"
            )

        # image branch
        img_feat_proj = self.image_proj(image_embedding)                 # [B, C, H, W]
        per_query_image_feat = img_feat_proj[query_to_image_index]       # [Q, C, H, W]

        # Dense attention supervision
        per_query_mask = gt_masks_per_query if gt_masks_per_query is not None else gt_mask
        if per_query_mask is not None:
            if per_query_mask.dim() == 3:
                per_query_mask = per_query_mask.unsqueeze(1)

            # 如果给的是按 batch 组织的 mask，而不是按 query 组织的 mask，就按 query_to_image_index 展开
            if per_query_mask.shape[0] == B and Q != B:
                per_query_mask = per_query_mask[query_to_image_index]

            if per_query_mask.shape[0] != Q:
                raise ValueError(
                    f"gt mask size mismatch, expected first dim {Q}, got {per_query_mask.shape[0]}"
                )

        align_loss = torch.tensor(
            0.0,
            device=image_embedding.device,
            dtype=img_feat_proj.dtype,
        )

        if self.training and per_query_mask is not None:
            align_loss = self._dense_attention_align_loss(
                text_feat=text_feat,
                per_query_image_feat=per_query_image_feat,
                per_query_mask=per_query_mask,
            )

        # Cross-attention refinement
        value_tokens = per_query_image_feat.flatten(2).transpose(1, 2)   # [Q, HW, C]
        value_tokens = self.image_norm(value_tokens)

        query = text_feat.unsqueeze(1)                                   # [Q,1,C]
        attn_out, _ = self.cross_attn(
            query=query,
            key=value_tokens,
            value=value_tokens,
            need_weights=False,
        )

        fused = self.attn_norm(query + attn_out)
        fused = self.mlp_norm(fused + self.mlp(fused))
        fused_seg_feat = text_feat + self.seg_gate * fused.squeeze(1)

        return fused_seg_feat.unsqueeze(1), align_loss