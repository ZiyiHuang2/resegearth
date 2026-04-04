import torch
import torch.nn as nn
import torch.nn.functional as F

class BHFMLayer(nn.Module):
    """
    Bi-directional Hybrid Fusion Module (BHFM) Layer
    
    【模块用法与作用】：
    插入到 SAM2 Image Encoder 的 Transformer Block 中。
    实现视觉与文本的双向跨模态交互：
    1. Vision queries Text: 视觉特征融合文本提示。
    2. Text queries Vision: 文本特征感知当前视觉状态并更新。
    同时采用并行 Adapter 设计（冻结主干 MLP，训练旁路 MLP），实现参数高效微调。
    
    【调用示例】：
    bhfm = BHFMLayer(embed_dim=256)
    # f_in 通常是该 Transformer block 的最原始输入
    f_out, t_out = bhfm(f_i=visual_feat, t_i=text_feat, f_in=original_visual_feat)
    """

    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0):
        super(BHFMLayer, self).__init__()
        self.embed_dim = embed_dim
        hidden_dim = int(embed_dim * mlp_ratio)

        # ================= 上方：视觉分支 (Vision queries Text) =================
        # 1. 视觉 Q 映射 (Linear -> GeLU)
        self.vis_q_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        # 2. 文本 K,V 映射
        self.text_kv_proj = nn.Linear(embed_dim, embed_dim * 2)
        # 3. 第一次 MHCA (Vision queries Text)
        self.mhca_v = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # 4. MHCA 后的 Linear 映射
        self.vis_out_proj = nn.Linear(embed_dim, embed_dim)

        # ================= 并行适配器 (Frozen & Tunable) =================
        # 冻结的主分支 (对应图上带 ❄️ 的 LN & MLP)
        self.frozen_ln = nn.LayerNorm(embed_dim)
        self.frozen_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self._freeze_module(self.frozen_ln)
        self._freeze_module(self.frozen_mlp)

        # 可微调的旁路分支 (对应图上带 🔥 的 Linear->GeLU->Linear)
        self.tunable_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        # 视觉加权融合权重 (对应图上 Ⓦ)
        self.w_v = nn.Parameter(torch.tensor(0.0)) # 初始化为0，等效于初始时不改变原网络

        # ================= 下方：文本分支 (Text queries Vision) =================
        # 1. 文本 Q 映射
        self.text_q_proj = nn.Linear(embed_dim, embed_dim)
        # 2. 第二次 MHCA (Text queries Vision)
        # K, V 来源于上方 MHCA 的输出，无需额外投影层（图中无标识）
        self.mhca_t = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # 3. MHCA 后的 Linear 映射
        self.text_out_proj = nn.Linear(embed_dim, embed_dim)
        # 文本加权融合权重 (对应图上 Ⓦ)
        self.w_t = nn.Parameter(torch.tensor(0.0))

    def _freeze_module(self, module):
        """冻结指定模块的参数"""
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, f_i, t_i, f_in):
        """
        参数：
        - f_i: 当前 Transformer 块前半部分处理后的视觉特征 [B, N, C]
        - t_i: 上一层传过来的文本特征 [B, L, C]
        - f_in: 外部注入的残差视觉特征 [B, N, C] (通常是进入该 Block 前的特征)
        """
        # ================= 1. 视觉特征更新分支 =================
        # 生成 Q_v
        q_v = self.vis_q_proj(f_i) # [B, N, C]
        
        # 生成 K_t, V_t
        k_t, v_t = self.text_kv_proj(t_i).chunk(2, dim=-1) # [B, L, C]
        
        # 视觉查询文本 MHCA & Add
        attn_v_out, _ = self.mhca_v(query=q_v, key=k_t, value=v_t) 
        attn_v_out = q_v + attn_v_out # MHCA & Add 模块的内部残差
        
        # 经过 Linear 后与 f_i, f_in 进行 Pointwise Add
        f_mid = self.vis_out_proj(attn_v_out)
        f_added = f_i + f_mid + f_in # 对应图中三个箭头的聚合 ⊕

        # 并行 Adapter：冻结分支 vs 可训练分支
        # 冻结分支 (由于我们手动设置了requires_grad=False，这里无需用torch.no_grad())
        frozen_out = self.frozen_mlp(self.frozen_ln(f_added))
        # 微调分支
        tune_out = self.tunable_mlp(f_added)
        
        # 加权融合生成最终的 f_out
        f_out = frozen_out + self.w_v * tune_out # 如果 w_v 初始化为 0，初始状态退化为纯冻结网络

        # ================= 2. 文本特征更新分支 =================
        # 生成 Q_t
        q_t = self.text_q_proj(t_i) # [B, L, C]
        
        # 视觉端传来的 K_v, V_v (使用上面 MHCA_v 的输出作为键值)
        k_v = v_v = attn_v_out # [B, N, C]
        
        # 文本查询视觉 MHCA & Add
        attn_t_out, _ = self.mhca_t(query=q_t, key=k_v, value=v_v)
        attn_t_out = q_t + attn_t_out # MHCA & Add 内部残差
        
        # 经过 Linear 映射
        t_mid = self.text_out_proj(attn_t_out)
        
        # 文本特征加权残差更新 Ⓦ
        t_out = t_i + self.w_t * t_mid # 传递给下一层
        
        return f_out, t_out

# ================= 模拟测试用例 =================
if __name__ == "__main__":
    B, N, L, C = 2, 1024, 16, 256  # N是图像序列长度, L是文本序列长度
    
    bhfm = BHFMLayer(embed_dim=C)
    
    # 模拟输入：假设图像被展平为序列 [B, N, C]
    f_in = torch.randn(B, N, C)  # 整个Block的输入残差
    f_i = torch.randn(B, N, C)   # MHA之后的输入
    t_i = torch.randn(B, L, C)   # 当前层传入的文本特征
    
    f_out, t_out = bhfm(f_i, t_i, f_in)
    
    print(f"输入视觉尺寸: {f_i.shape}, 输入文本尺寸: {t_i.shape}")
    print(f"输出视觉尺寸: {f_out.shape}, 输出文本尺寸: {t_out.shape}")