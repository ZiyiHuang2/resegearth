"""
Lightweight query-to-feature bridge: refines SEG_embedding before Mask2Former predictor.

Does not modify Pixel Decoder, Mask2Former cross-attention, or nn.MultiheadAttention.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tensor_stats_line(name: str, t: torch.Tensor) -> str:
    """min / max / mean / std / isfinite_ratio on flattened tensor (float32 for display)."""
    if getattr(t, "is_meta", False):
        return f"{name}: (meta tensor, stats skipped)"
    x = t.detach().reshape(-1).to(torch.float32)
    fin = torch.isfinite(x)
    ratio = float(fin.float().mean().item()) if x.numel() > 0 else 0.0
    xf = x[fin]
    if xf.numel() == 0:
        return f"{name}: (no finite values) isfinite_ratio={ratio:.6f}"
    return (
        f"{name}: min={xf.min().item():.6e} max={xf.max().item():.6e} "
        f"mean={xf.mean().item():.6e} std={xf.std(unbiased=False).item():.6e} isfinite_ratio={ratio:.6f}"
    )


def _require_finite(t: torch.Tensor, step: str) -> None:
    if torch.isfinite(t).all():
        return
    bad = (~torch.isfinite(t)).sum().item()
    raise ValueError(f"SegQueryFeatureBridge: non-finite values at step '{step}' (count={bad}, shape={tuple(t.shape)})")


class SegQueryFeatureBridge(nn.Module):
    """
    Pooling-style attention over flattened spatial features.

    - q = Linear(SEG_embedding)
    - k, v = 1x1 conv on feature map, then flatten HW
    - attn = softmax(stable_scaled_logits)
    - delta = attn @ v, then Linear (small init on output projection)
    - out = SEG_embedding + alpha * delta, alpha learnable scalar
    """

    def __init__(self, hidden_dim: int, alpha_init: float = 1e-3):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.v_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.delta_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.normal_(self.delta_proj.weight, std=0.01)
        nn.init.zeros_(self.delta_proj.bias)
        # shape (1,) — 0-dim scalar Parameter breaks HF from_pretrained weight dtype casting (torch.empty()).
        self.alpha = nn.Parameter(torch.tensor([float(alpha_init)], dtype=torch.float32))

    def log_parameter_stats(self, prefix: str = "[SegQueryFeatureBridge.init] ") -> None:
        """Print alpha and weight tensor stats after checkpoint load (diagnostic)."""
        lines = [
            _tensor_stats_line("alpha", self.alpha),
            _tensor_stats_line("q_proj.weight", self.q_proj.weight),
            _tensor_stats_line("k_conv.weight", self.k_conv.weight),
            _tensor_stats_line("v_conv.weight", self.v_conv.weight),
            _tensor_stats_line("delta_proj.weight", self.delta_proj.weight),
        ]
        for ln in lines:
            print(prefix + ln, flush=True)

    def repair_weights_after_pretrained_load(self) -> None:
        """
        HF missing-key init can leave extreme / non-finite values in conv/linear weights.
        Re-init only modules whose parameters are not all finite (minimal fix, idempotent if already ok).
        """
        w0 = self.q_proj.weight
        if getattr(w0, "is_meta", False) or getattr(w0.device, "type", "") == "meta":
            return
        with torch.no_grad():

            def _ok(p: torch.Tensor) -> bool:
                if getattr(p, "is_meta", False) or p.numel() == 0:
                    return True
                return bool(torch.isfinite(p.detach()).all().item())

            if not _ok(self.q_proj.weight) or not _ok(self.q_proj.bias):
                nn.init.xavier_uniform_(self.q_proj.weight)
                nn.init.zeros_(self.q_proj.bias)
            if not _ok(self.k_conv.weight) or not _ok(self.k_conv.bias):
                nn.init.kaiming_uniform_(self.k_conv.weight, a=math.sqrt(5))
                nn.init.zeros_(self.k_conv.bias)
            if not _ok(self.v_conv.weight) or not _ok(self.v_conv.bias):
                nn.init.kaiming_uniform_(self.v_conv.weight, a=math.sqrt(5))
                nn.init.zeros_(self.v_conv.bias)
            if not _ok(self.delta_proj.weight) or not _ok(self.delta_proj.bias):
                nn.init.normal_(self.delta_proj.weight, std=0.01)
                nn.init.zeros_(self.delta_proj.bias)

    def forward(
        self,
        seg_embedding: torch.Tensor,
        feature: torch.Tensor,
        *,
        debug: bool = False,
        bridge_level: Optional[int] = None,
    ) -> torch.Tensor:
        if seg_embedding.ndim != 3:
            raise ValueError(
                f"SegQueryFeatureBridge: expected SEG_embedding.ndim == 3, got {seg_embedding.ndim} "
                f"(shape={tuple(seg_embedding.shape)})"
            )
        if seg_embedding.shape[1] != 1:
            raise ValueError(
                f"SegQueryFeatureBridge: expected SEG_embedding.shape[1] == 1, got shape[1]={seg_embedding.shape[1]} "
                f"(full shape={tuple(seg_embedding.shape)})"
            )
        if feature.ndim != 4:
            raise ValueError(
                f"SegQueryFeatureBridge: expected feature.ndim == 4, got {feature.ndim} "
                f"(shape={tuple(feature.shape)})"
            )
        n_seg = seg_embedding.shape[0]
        if feature.shape[0] != n_seg:
            raise ValueError(
                f"SegQueryFeatureBridge: feature.shape[0] ({feature.shape[0]}) must equal "
                f"SEG_embedding.shape[0] ({n_seg}) (batch-expanded N_seg_total mismatch)."
            )
        c_emb = seg_embedding.shape[2]
        if feature.shape[1] != c_emb:
            raise ValueError(
                f"SegQueryFeatureBridge: feature.shape[1] ({feature.shape[1]}) must equal "
                f"SEG_embedding.shape[2] ({c_emb}) (channel mismatch)."
            )

        n, _, c = seg_embedding.shape
        if c != self.hidden_dim:
            raise ValueError(
                f"SegQueryFeatureBridge: SEG_embedding channel {c} != module hidden_dim {self.hidden_dim}"
            )

        if not torch.isfinite(seg_embedding).all():
            raise ValueError(
                "SegQueryFeatureBridge: SEG_embedding already contains NaN or Inf before bridge "
                f"(shape={tuple(seg_embedding.shape)})"
            )
        if not torch.isfinite(feature).all():
            raise ValueError(
                "SegQueryFeatureBridge: feature (multi_scale_features[level]) already contains NaN or Inf "
                f"(shape={tuple(feature.shape)})"
            )

        in_dtype = seg_embedding.dtype
        in_device = seg_embedding.device

        # Residual / alpha 在 float32 里算；Linear/Conv 权重可能是 bf16/fp16，输入需与权重 dtype 一致后再 cast 到 float32 做 softmax。
        seg_f = seg_embedding.float()
        wd_q = self.q_proj.weight.dtype
        wd_kv = self.k_conv.weight.dtype
        q = self.q_proj(seg_embedding.squeeze(1).to(dtype=wd_q)).to(torch.float32).unsqueeze(1)  # [N, 1, C]
        feat_kv = feature.to(dtype=wd_kv)
        k_map = self.k_conv(feat_kv).to(torch.float32)
        v_map = self.v_conv(feat_kv).to(torch.float32)
        h, w = k_map.shape[-2:]
        hw = h * w
        k_flat = k_map.reshape(n, c, hw).transpose(1, 2).to(torch.float32)  # [N, HW, C]
        v_flat = v_map.reshape(n, c, hw).transpose(1, 2).to(torch.float32)

        _require_finite(q, "q after q_proj")
        _require_finite(k_flat, "k after k_conv/flatten")
        _require_finite(v_flat, "v after v_conv/flatten")

        raw_logits = torch.bmm(q, k_flat.transpose(1, 2))  # [N, 1, HW], before 1/sqrt(C)
        _require_finite(raw_logits, "logits before scale (raw q @ k^T)")
        scale = 1.0 / math.sqrt(float(c))
        scaled_logits = raw_logits * scale
        _require_finite(scaled_logits, "logits after scale ( * 1/sqrt(C) )")

        shifted = scaled_logits - scaled_logits.max(dim=-1, keepdim=True).values
        _require_finite(shifted, "logits after max-subtract (pre-softmax)")
        clamped = shifted.clamp(-50.0, 50.0)
        _require_finite(clamped, "logits after clamp [-50,50] (pre-softmax)")
        attn = F.softmax(clamped, dim=-1)
        if not torch.isfinite(attn).all():
            raise ValueError(
                "SegQueryFeatureBridge: attn after softmax contains NaN or Inf "
                "(check scaled logits / clamp range)"
            )

        delta_pre = torch.bmm(attn, v_flat)  # [N, 1, C]
        _require_finite(delta_pre, "delta before delta_proj (attn @ v)")

        wd_d = self.delta_proj.weight.dtype
        delta = self.delta_proj(delta_pre.squeeze(1).to(dtype=wd_d)).to(torch.float32).unsqueeze(1)
        _require_finite(delta, "delta after delta_proj")

        alpha_f = self.alpha.to(device=in_device, dtype=torch.float32)
        refined_f = seg_f + alpha_f * delta
        _require_finite(refined_f, "refined (SEG + alpha*delta) before cast to input dtype")
        refined = refined_f.to(dtype=in_dtype)

        if debug:
            print("[SegQueryFeatureBridge] --- debug stats (float32) ---", flush=True)
            print(_tensor_stats_line("SEG_embedding", seg_embedding), flush=True)
            print(_tensor_stats_line("feature", feature), flush=True)
            print(_tensor_stats_line("q", q), flush=True)
            print(_tensor_stats_line("k (k_flat)", k_flat), flush=True)
            print(_tensor_stats_line("v (v_flat)", v_flat), flush=True)
            print(_tensor_stats_line("logits before scale (raw_logits)", raw_logits), flush=True)
            print(_tensor_stats_line("logits after scale (scaled_logits)", scaled_logits), flush=True)
            print(_tensor_stats_line("logits after max-subtract", shifted), flush=True)
            print(_tensor_stats_line("logits after clamp [-50,50]", clamped), flush=True)
            print(_tensor_stats_line("attn", attn), flush=True)
            print(_tensor_stats_line("delta before delta_proj", delta_pre), flush=True)
            print(_tensor_stats_line("delta after delta_proj", delta), flush=True)
            print(_tensor_stats_line("refined (float32)", refined_f), flush=True)
            with torch.no_grad():
                print(
                    "[SegQueryFeatureBridge] summary:",
                    f"shape_seg={tuple(seg_embedding.shape)} shape_feat={tuple(feature.shape)} level={bridge_level}",
                    f"alpha={float(self.alpha.detach().cpu().reshape(-1)[0])}",
                    f"delta_norm={float(delta.detach().float().norm().cpu())}",
                    f"SEG_embedding_norm={float(seg_embedding.detach().float().norm().cpu())}",
                    f"refined_norm={float(refined.detach().float().norm().cpu())}",
                    flush=True,
                )

        return refined
