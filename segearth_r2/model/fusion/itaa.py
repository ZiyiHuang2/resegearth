import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageTextAlignmentAdapter(nn.Module):
    def __init__(self, image_dim=256, llm_dim=4096, hidden_dim=512, num_heads=8, dropout=0.0):
        super().__init__()

        self.image_dim = image_dim
        self.llm_dim = llm_dim

        # text projection
        self.text_proj = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, image_dim)
        )

        self.text_proj_image_dim = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, image_dim),
        )

        self.query_norm = nn.LayerNorm(image_dim)

        # image projection
        self.image_proj = nn.Conv2d(image_dim, image_dim, kernel_size=1)

        self.image_norm = nn.LayerNorm(image_dim)

        self.mask_proj = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.GELU(),
            nn.Linear(image_dim, image_dim),
        )

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

    def _normalize_seg_embedding(self, seg_embedding):
        if seg_embedding.dim() == 3:
            if seg_embedding.shape[1] == 1:
                seg_embedding = seg_embedding.squeeze(1)
            else:
                seg_embedding = seg_embedding.mean(dim=1)
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

    def forward(
        self,
        image_embedding,
        seg_embedding,
        gt_mask=None,
        mask_num=None,
    ):
        B, C, H, W = image_embedding.shape

        seg_embedding = self._normalize_seg_embedding(seg_embedding)
        Q = seg_embedding.shape[0]

        query_to_image_index = self._build_query_to_image_index(
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