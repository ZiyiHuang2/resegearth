import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageTextAlignmentAdapter(nn.Module):
    """
    Image-Text Alignment Adapter (ITAA) 模块
    
    【模块用法与作用】：
     1. 前向特征融合：将 MLLM 提取的 <SEG> 文本特征与 SFM 提取的高分辨率图像特征进行交互融合，
         输出可直接送入 Transformer decoder 的 SEG embedding。
     2. 训练特征对齐：在训练阶段，利用 Ground Truth (GT) Mask 对图像特征进行掩码池化，
         计算其与文本 <SEG> 特征的对齐 Loss，以此监督语义向量更贴近目标区域。
       
    【调用方法】：
    # 初始化
    itaa = ImageTextAlignmentAdapter(image_dim=256, llm_dim=4096, hidden_dim=512)
    
    # 前向调用 (训练阶段，传入 gt_mask)
    seg_embedding, align_loss = itaa(image_embedding, seg_embedding, gt_mask)
    total_loss = decode_loss + 0.1 * align_loss # 将 align_loss 加入总损失
    
    # 前向调用 (推理阶段，无 gt_mask)
    seg_embedding, _ = itaa(image_embedding, seg_embedding)
    mask_pred = decoder(image_embedding, seg_embedding)
    """

    def __init__(self, image_dim=256, llm_dim=4096, hidden_dim=512):
        super(ImageTextAlignmentAdapter, self).__init__()
        
        # 1. text/query 分支
        self.text_proj = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, image_dim)
        )
        
        self.query_norm = nn.LayerNorm(image_dim)

        # 2. image/value 分支
        self.image_proj = nn.Conv2d(image_dim, image_dim, kernel_size=1)
        
        self.image_norm = nn.LayerNorm(image_dim)

        # 3. query-to-mask 对齐分支
        self.mask_proj = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.GELU(),
            nn.Linear(image_dim, image_dim),
        )
        # 4. query <- image cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(image_dim)

        # 5. MLP + residual + norm
        self.mlp = nn.Sequential(
            nn.Linear(image_dim, image_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(image_dim * 4, image_dim),
        )
        self.mlp_norm = nn.LayerNorm(image_dim)

        # 残差门控，初始保持与原始 query 一致

        # SEG 残差门控，初始时保持与原始文本向量一致
        self.seg_gate = nn.Parameter(torch.tensor(0.0))

    def _normalize_seg_embedding(self, seg_embedding):
        if seg_embedding.dim() == 3:
            if seg_embedding.shape[1] == 1:
                seg_embedding = seg_embedding.squeeze(1)
            else:
                seg_embedding = seg_embedding.mean(dim=1)
        elif seg_embedding.dim() != 2:
            raise ValueError(f"seg_embedding must be 2D or 3D, but got shape {tuple(seg_embedding.shape)}")
        return seg_embedding

    def _pool_image_feature(self, image_feat, gt_mask=None):
        if gt_mask is not None:
            if gt_mask.dim() == 3:
                gt_mask = gt_mask.unsqueeze(1)
            if gt_mask.shape[-2:] != image_feat.shape[-2:]:
                gt_mask = F.interpolate(gt_mask.float(), size=image_feat.shape[-2:], mode='nearest')
            else:
                gt_mask = gt_mask.float()

            masked_feat = image_feat * gt_mask
            mask_area = gt_mask.sum(dim=(2, 3)).clamp_min(1e-6)
            return masked_feat.sum(dim=(2, 3)) / mask_area

        return image_feat.mean(dim=(2, 3))

    def _build_query_to_image_index(self, num_queries, batch_size, mask_num=None, query_to_image_index=None, device=None):
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
                return torch.repeat_interleave(torch.arange(batch_size, device=device), repeats=mask_num)

        if num_queries == batch_size:
            return torch.arange(batch_size, device=device, dtype=torch.long)

        raise ValueError(
            f"Cannot infer query->image mapping: num_queries={num_queries}, batch_size={batch_size}"
        )

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
        参数:
        - image_embedding: SFM Vision Encoder 提取的图像特征, 形状 [B, C, H, W]
        - seg_embedding: MLLM 对应 <SEG> token 的输出特征, 形状 [B, L, D] 或 [B, D]
        - gt_mask: (仅训练阶段使用) Ground Truth 标注, 形状 [B, 1, H, W]
        - mask_num: 每个样本对应的 [SEG] 数量，用于和 decoder 的 batch 维度对齐
        
        返回:
        - aligned_seg_embedding: 可直接送入 decoder 的 SEG embedding，形状 [B, 1, C] 或 [sum(mask_num), 1, C]
        - align_loss: 图像区域与文本特征的对齐损失（推理时为 0.0）
        """
        B, C, H, W = image_embedding.shape
        
        # --- 步骤 1: 处理 LLM 特征 ---
        seg_embedding = self._normalize_seg_embedding(seg_embedding)
        Q = seg_embedding.shape[0]
        query_to_image_index = self._build_query_to_image_index(
            num_queries=Q,
            batch_size=B,
            mask_num=mask_num,
            query_to_image_index=query_to_image_index,
            device=image_embedding.device,
        )
        query_to_image_index = query_to_image_index.clamp(min=0, max=B - 1)

        # 将 LLM 维度映射到 图像特征维度
        text_feat = self.query_norm(self.text_proj(seg_embedding))
        
        # --- 步骤 2: 计算对齐 Loss (仅训练时, 且存在 GT mask 时) ---
        img_feat_proj = self.image_proj(image_embedding)  # [B, C, H, W]
        per_query_image_feat = img_feat_proj[query_to_image_index]  # [Q, C, H, W]
        dense_feat = self._pool_image_feature(per_query_image_feat)

        per_query_mask = gt_masks_per_query if gt_masks_per_query is not None else gt_mask
        sparse_feat = dense_feat
        if per_query_mask is not None:
            if per_query_mask.dim() == 3:
                per_query_mask = per_query_mask.unsqueeze(1)
            if per_query_mask.shape[0] != Q:
                raise ValueError(
                    f"gt mask size mismatch, expected first dim {Q}, got {per_query_mask.shape[0]}"
                )
            sparse_feat = self._pool_image_feature(per_query_image_feat, per_query_mask)
        sparse_feat = self.mask_proj(sparse_feat)
        sparse_feat = self.image_norm(sparse_feat)
        align_loss = torch.tensor(0.0, device=image_embedding.device, dtype=img_feat_proj.dtype)
        if self.training and per_query_mask is not None:
            align_loss = F.mse_loss(text_feat, sparse_feat)

        # --- 步骤 3: query <- image cross attention ---
        value_tokens = per_query_image_feat.flatten(2).transpose(1, 2)  # [Q, HW, C]
        value_tokens = self.image_norm(value_tokens)
        kv_tokens = torch.cat([value_tokens, sparse_feat.unsqueeze(1)], dim=1)

        query = text_feat.unsqueeze(1)  # [Q, 1, C]
        attn_out, _ = self.cross_attn(query=query, key=kv_tokens, value=kv_tokens, need_weights=False)
        fused = self.attn_norm(query + attn_out)
        fused = self.mlp_norm(fused + self.mlp(fused))
        fused_seg_feat = text_feat + self.seg_gate * fused.squeeze(1)

        return fused_seg_feat.unsqueeze(1), align_loss

# ================= 模拟测试用例 =================
if __name__ == "__main__":
    # 模拟输入参数
    B, C, H, W = 2, 256, 32, 32
    llm_dim = 4096
    
    model = ImageTextAlignmentAdapter(image_dim=C, llm_dim=llm_dim)
    
    # 模拟数据
    img_emb = torch.randn(B, C, H, W)
    llm_out = torch.randn(B, 1, llm_dim) # 取出 <SEG> 的 embedding
    gt_mask = torch.randint(0, 2, (B, 1, 256, 256)).float() # 原始高分辨率 mask
    
    # 1. 训练阶段测试
    model.train()
    out_feat, loss = model(img_emb, llm_out, gt_mask)
    print(f"【训练模式】输出特征尺寸: {out_feat.shape}, 对齐 Loss: {loss.item():.4f}")
    
    # 2. 推理阶段测试 (不传 Mask)
    model.eval()
    with torch.no_grad():
        out_feat_eval, loss_eval = model(img_emb, llm_out)
        print(f"【推理模式】输出特征尺寸: {out_feat_eval.shape}, 对齐 Loss: {loss_eval.item():.4f}")