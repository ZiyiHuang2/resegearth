#!/usr/bin/env python3
"""CS-DEG++ engineering smoke tests (standalone modules)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from addict import Dict

ROOT = Path(__file__).resolve().parents[2]
CS_DEG = ROOT / "segearth_r2/model/mask_decoder/cs_deg"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ret = _load_module("cs_deg_ret", CS_DEG / "referential_evidence_tokenizer.py")
_depg = _load_module("cs_deg_depg", CS_DEG / "dense_evidence_prompt_generator.py")
_bhef = _load_module("cs_deg_bhef", CS_DEG / "bidirectional_hierarchical_evidence_fusion.py")
_attn = _load_module("cs_deg_attn", CS_DEG / "evidence_guided_attention.py")
_mask = _load_module("cs_deg_mask", CS_DEG / "evidence_mask_head.py")
_hcml = _load_module("cs_deg_hcml", CS_DEG / "hcml_losses.py")

ReferentialEvidenceTokenizer = _ret.ReferentialEvidenceTokenizer
DenseEvidencePromptGenerator = _depg.DenseEvidencePromptGenerator
BidirectionalHierarchicalEvidenceFusion = _bhef.BidirectionalHierarchicalEvidenceFusion
fuse_evidence_attention_mask = _attn.fuse_evidence_attention_mask
normalize_per_sample = _attn.normalize_per_sample
apply_evidence_mask_head = _mask.apply_evidence_mask_head
compute_cs_deg_losses = _hcml.compute_cs_deg_losses


def _make_cfg() -> Dict:
    return Dict({
        "ENABLED": True,
        "EVIDENCE_CONSISTENCY": True,
        "BOUNDARY_LOSS": False,
        "HCML_WARMUP_STEPS": 0,
        "HCML_RAMP_STEPS": 500,
        "LOSS_CONTEXT_WEIGHT": 2.0,
        "LOSS_EVIDENCE_WEIGHT": 1.0,
        "LOSS_RANK_WEIGHT": 0.5,
        "LOSS_SIBLING_AUX_WEIGHT": 0.1,
        "LOSS_CONSISTENCY_WEIGHT": 0.5,
        "HARD_TOPK_RATIO": 0.01,
        "HARD_TOPK_MIN": 10,
        "RANK_MARGIN": 0.2,
    })


def test_ret_zero_init():
    ret = ReferentialEvidenceTokenizer(hidden_dim=256, prompt_token_num=4, num_layers=9)
    output = torch.randn(1, 2, 256)
    tokens = ret(output, layer_index=0)
    assert tokens.shape == (4, 2, 256)
    assert tokens.abs().max() < 1e-6
    print("PASS RET zero-init")


def test_depg_shapes_and_zero_init():
    depg = DenseEvidencePromptGenerator(hidden_dim=256, mask_dim=256, num_levels=3)
    q, b, c = 1, 2, 256
    h = w = 8
    output = torch.randn(q, b, c)
    src = torch.randn(h * w, b, c)
    prev_mask = torch.randn(b, q, 64, 64)
    mf = torch.randn(b, 256, 64, 64)
    ev_tokens = torch.randn(4, b, c)
    sparse, tgt, ctx, unc, tgt_h, ctx_h = depg(
        output, src, (h, w), prev_mask, mf, ev_tokens, 0, compute_high_res=True
    )
    assert sparse.shape == (4, b, c)
    assert tgt.shape == (b, q, h, w)
    assert tgt_h.shape == (b, q, 64, 64)
    assert tgt.abs().max() < 1e-6
    print("PASS DEPG shapes + zero-init")


def test_bhef_identity_gate():
    bhef = BidirectionalHierarchicalEvidenceFusion(hidden_dim=256, gate_init=0.0)
    output = torch.randn(1, 2, 256)
    src = torch.randn(64, 2, 256)
    tokens = torch.randn(4, 2, 256)
    out_ref, tok_ref = bhef(output, src, tokens)
    assert torch.allclose(out_ref, output)
    print("PASS BHEF identity gate")


def test_evidence_mask_head_identity():
    mask = torch.randn(2, 1, 32, 32)
    tgt = torch.randn_like(mask)
    ctx = torch.randn_like(mask)
    gamma = torch.zeros(1)
    eta = torch.zeros(1)
    out = apply_evidence_mask_head(mask, tgt, ctx, gamma, eta)
    assert torch.allclose(out, mask)
    print("PASS evidence mask head identity")


def test_evidence_bias_fusion():
    b, q, hw, heads = 2, 1, 64, 8
    bool_mask = torch.zeros(b * heads, q, hw, dtype=torch.bool)
    bool_mask[:, :, hw // 2 :] = True
    target = torch.randn(b, q, 8, 8)
    context = torch.randn(b, q, 8, 8)
    alpha = torch.nn.Parameter(torch.zeros(1))
    fused, scale = fuse_evidence_attention_mask(
        bool_mask, target, context, alpha, global_step=1000,
        num_heads=heads, bias_max=5.0, bias_warmup_steps=1000,
    )
    assert fused.shape == bool_mask.shape
    assert scale.item() == 0.0
    print("PASS evidence bias fusion")


def test_cs_deg_losses_finite():
    cfg = _make_cfg()
    b, h, w = 2, 32, 32
    outputs = {
        "pred_masks": torch.randn(b, 1, h, w),
        "pred_evidence_logits": torch.randn(b, 1, h, w),
        "pred_context_logits": torch.randn(b, 1, h, w),
    }
    targets = []
    for _ in range(b):
        tgt = torch.zeros(1, h, w)
        tgt[:, 5:15, 5:15] = 1
        targets.append({"masks": tgt, "sibling_masks": torch.zeros_like(tgt)})
    losses = compute_cs_deg_losses(outputs, targets, cfg, global_step=100)
    for k, v in losses.items():
        assert torch.isfinite(v), k
    print("PASS cs-deg losses:", {k: float(v) for k, v in losses.items()})


if __name__ == "__main__":
    test_ret_zero_init()
    test_depg_shapes_and_zero_init()
    test_bhef_identity_gate()
    test_evidence_mask_head_identity()
    test_evidence_bias_fusion()
    test_cs_deg_losses_finite()
    print("ALL STANDALONE SMOKE TESTS PASSED")
