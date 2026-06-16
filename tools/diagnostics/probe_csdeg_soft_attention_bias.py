#!/usr/bin/env python3
"""V2: soft additive attention bias feasibility — standalone, no full model import."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class MiniCrossAttentionLayer(nn.Module):
    """Minimal copy of baseline CrossAttentionLayer (mask2former_transformer_decoder.py:70-130)."""

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.0)

    def forward(self, tgt, memory, memory_mask=None, pos=None, query_pos=None):
        q = tgt if query_pos is None else tgt + query_pos
        k = memory if pos is None else memory + pos
        tgt2 = self.multihead_attn(q, k, memory, attn_mask=memory_mask)[0]
        return self.norm(tgt + self.dropout(tgt2))


def bool_mask_to_additive(bool_mask: torch.Tensor, block_value: float = -1e4) -> torch.Tensor:
    return torch.where(bool_mask, torch.full_like(bool_mask, block_value), torch.zeros_like(bool_mask))


def run_probe(device: str = "cpu") -> dict:
    torch.manual_seed(0)
    Q, B, C, HW, nheads = 1, 2, 256, 64, 8

    tgt = torch.randn(Q, B, C, device=device, requires_grad=True)
    memory = torch.randn(HW, B, C, device=device)
    pos = torch.randn(HW, B, C, device=device)
    query_pos = torch.randn(Q, B, C, device=device)

    bool_mask = torch.zeros(B * nheads, Q, HW, dtype=torch.bool, device=device)
    bool_mask[:, :, HW // 2 :] = True

    evidence_bias = torch.zeros(B * nheads, Q, HW, device=device, requires_grad=True)
    with torch.no_grad():
        evidence_bias[:, :, : HW // 4] = -3.0
    evidence_bias.requires_grad_(True)

    layer = MiniCrossAttentionLayer(C, nheads).to(device)
    mha = layer.multihead_attn

    # Path 1: bool only
    out1, attn1 = mha(
        tgt + query_pos, memory + pos, memory,
        attn_mask=bool_mask,
        need_weights=True,
        average_attn_weights=True,
    )
    loss1 = out1.sum()
    loss1.backward()
    g1 = float(tgt.grad.norm()) if tgt.grad is not None else 0.0
    tgt.grad = None

    # Path 2: bool + soft evidence bias
    combined = bool_mask_to_additive(bool_mask) + evidence_bias
    out2, attn2 = mha(
        tgt + query_pos, memory + pos, memory,
        attn_mask=combined,
        need_weights=True,
        average_attn_weights=True,
    )
    loss2 = out2.sum()
    loss2.backward()
    bg = float(evidence_bias.grad.norm()) if evidence_bias.grad is not None else 0.0

    sup = float(attn2[:, :, : HW // 4].mean())
    mid = float(attn2[:, :, HW // 4 : HW // 2].mean())
    blk = float(attn2[:, :, HW // 2 :].mean())

    finite = bool(torch.isfinite(out1).all() and torch.isfinite(out2).all())
    if not finite:
        verdict = "FAIL"
    elif bg < 1e-9:
        verdict = "PARTIAL"
    elif sup >= mid * 0.9:
        verdict = "PARTIAL"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "device": device,
        "baseline_bool": {"finite": finite, "tgt_grad_norm": g1, "attn_blocked_half_mean": float(attn1[:, :, HW // 2 :].mean())},
        "soft_additive": {
            "finite": finite,
            "evidence_bias_grad_norm": bg,
            "attn_suppressed_quarter": sup,
            "attn_mid_quarter": mid,
            "attn_blocked_half": blk,
            "mid_over_suppressed_ratio": mid / (sup + 1e-8),
        },
        "api_note": "Baseline CrossAttentionLayer passes memory_mask to nn.MultiheadAttention; float additive attn_mask supported by PyTorch MHA.",
        "implementation_note": "Extend CrossAttentionLayer.forward with optional evidence_bias: float [B*nheads,Q,HW] added to bool-derived mask.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    report = run_probe(device)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    md = [
        "# CS-DEG V2 Soft Bias Probe",
        "",
        f"**Verdict: {report['verdict']}**",
        "",
        f"- device: `{device}`",
        f"- evidence_bias grad: `{report['soft_additive']['evidence_bias_grad_norm']}`",
        f"- attn suppressed quarter: `{report['soft_additive']['attn_suppressed_quarter']:.6f}`",
        f"- attn mid quarter: `{report['soft_additive']['attn_mid_quarter']:.6f}`",
        "",
        report["implementation_note"],
    ]
    Path(args.out_md).write_text("\n".join(md) + "\n")
    print(json.dumps({"verdict": report["verdict"], "soft_additive": report["soft_additive"]}, indent=2))


if __name__ == "__main__":
    main()
