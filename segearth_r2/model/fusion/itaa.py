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
        
        # 1. 文本特征降维/映射层：将 LLM 庞大的输出维度映射到 adapter 的隐藏层维度
        self.text_proj = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, image_dim)
        )
        
        # 2. 图像特征映射层（可选，用于与文本特征在统一空间做交互）
        self.image_proj = nn.Conv2d(image_dim, image_dim, kernel_size=1)
        
        # 3. 融合后的特征处理
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(image_dim * 2, image_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(image_dim),
            nn.ReLU(inplace=True)
        )

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

    def forward(self, image_embedding, seg_embedding, gt_mask=None, mask_num=None):
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

        # 将 LLM 维度映射到 图像特征维度
        text_feat = self.text_proj(seg_embedding)
        
        # --- 步骤 2: 计算对齐 Loss (仅训练时, 且存在 GT mask 时) ---
        img_feat_proj = self.image_proj(image_embedding)
        align_loss = torch.tensor(0.0, device=image_embedding.device, dtype=img_feat_proj.dtype)
        if self.training and gt_mask is not None:
            pooled_img_feat = self._pool_image_feature(img_feat_proj, gt_mask)
            align_loss = F.mse_loss(text_feat, pooled_img_feat)

        # --- 步骤 3: 图像-文本特征融合并生成送给 Decoder 的 SEG embedding ---
        text_feat_spatial = text_feat.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
        concat_feat = torch.cat([img_feat_proj, text_feat_spatial], dim=1)
        aligned_feature = self.fusion_conv(concat_feat)
        fused_seg_feat = text_feat + self.seg_gate * self._pool_image_feature(aligned_feature)

        if mask_num is not None:
            if not torch.is_tensor(mask_num):
                mask_num = torch.tensor(mask_num, device=fused_seg_feat.device)
            else:
                mask_num = mask_num.to(device=fused_seg_feat.device)
            fused_seg_feat = torch.repeat_interleave(fused_seg_feat, repeats=mask_num, dim=0)

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