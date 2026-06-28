#!/usr/bin/env python3
"""TG-Swin v1.6 smoke: HTER (stage phrase + hybrid ρ) + PWER (evidence state)."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from segearth_r2.model.mask_encoder.swin_trans import build_swin_b
from segearth_r2.model.mask_encoder.tg_swin import (
    StageWTIHeadAware,
    TextConditionFactory,
    TGSwimController,
)

IDENTITY_THRESH = 1e-5


def _make_tcf(**kwargs):
    defaults = dict(
        text_dim=64, cond_dim=32, reliability_init=0.0,
        version="v1.6", num_stages=4, stage_router=True,
    )
    defaults.update(kwargs)
    return TextConditionFactory(**defaults)


def _make_controller(**kwargs):
    defaults = dict(cond_dim=32, wti_rank=8, window_size=12, head_aware=True)
    defaults.update(kwargs)
    return TGSwimController(**defaults)


def check_v15_equiv_flags_off():
    """All v1.6 flags off + VERSION v1.5 → compact 3-stage TCF (matches v1.5 yaml)."""
    tcf = _make_tcf(
        version="v1.5",
        num_stages=3,
        use_stage_phrase=False,
        use_hybrid_reliability=False,
    )
    seg = torch.randn(3, 64)
    phr = torch.randn(3, 5, 64)
    mask = torch.ones(3, 5, dtype=torch.bool)
    tc, rel = tcf(seg, phrase_hidden=phr, phrase_mask=mask)
    assert tc.shape == (3, 3, 32), tc.shape
    assert rel.shape == (3, 3, 1), rel.shape
    assert rel.mean().item() > 0.4
    print("[PASS] 1/8 v1.5-equiv: flags off → [N,S,C] + [N,S,1]")


def check_full_v16_shapes():
    """All flags on: stage_text_cond [N,S,C], reliability [N,S,1]."""
    tcf = _make_tcf(
        use_stage_phrase=True,
        use_hybrid_reliability=True,
    )
    seg = torch.randn(2, 64)
    phr = torch.randn(2, 6, 64)
    mask = torch.ones(2, 6, dtype=torch.bool)
    tc, rel = tcf(seg, phrase_hidden=phr, phrase_mask=mask)
    assert tc.shape == (2, 4, 32)
    assert rel.shape == (2, 4, 1)
    print("[PASS] 2/8 full v1.6 TCF shapes [N,S,C] + [N,S,1]")


def check_hybrid_rho_global_sum():
    """Hybrid ρ: global branch sums to ~1 per sample."""
    tcf = _make_tcf(use_hybrid_reliability=True, use_stage_phrase=False)
    seg = torch.randn(4, 64)
    phr = torch.randn(4, 5, 64)
    mask = torch.ones(4, 5, dtype=torch.bool)
    _, rel = tcf(seg, phrase_hidden=phr, phrase_mask=mask)
    router = tcf.router
    mixed = router.out_norm(
        router.stage_embed.unsqueeze(0).expand(4, -1, -1)
        + router.seg_proj(router.seg_norm(seg)).unsqueeze(1).expand(-1, 4, -1)
    )
    local = torch.sigmoid(router.reliability_head(mixed))
    global_logits = router.hybrid_global_proj(mixed).squeeze(-1)
    global_rel = F.softmax(global_logits, dim=1)
    global_sum = global_rel.sum(dim=1)
    assert (global_sum - 1.0).abs().max().item() < 1e-4, f"global sum={global_sum}"
    print("[PASS] 3/8 hybrid ρ: sum(global)≈1")


def check_stage_phrase_distinguishable():
    """Stage phrase attn weights differ across stages (synthetic data)."""
    tcf = _make_tcf(use_stage_phrase=True, use_hybrid_reliability=False, stage_phrase_temp=0.05)
    with torch.no_grad():
        for s in range(4):
            tcf.router.stage_phrase_embed[s].zero_()
            tcf.router.stage_phrase_embed[s, s * 8 : (s + 1) * 8] = 8.0

    seg = torch.zeros(2, 64)
    phr = torch.zeros(2, 8, 64)
    for t in range(8):
        phr[:, t, t * 8 : (t + 1) * 8] = 1.0
    mask = torch.ones(2, 8, dtype=torch.bool)
    tcf(seg, phrase_hidden=phr, phrase_mask=mask)
    attn = tcf.router._last_phrase_attn
    assert attn is not None and attn.shape == (2, 4, 8)
    cos_vals = []
    for i in range(4):
        for j in range(i + 1, 4):
            a = attn[0, i]
            b = attn[0, j]
            cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            cos_vals.append(cos)
    mean_cos = sum(cos_vals) / len(cos_vals)
    assert mean_cos < 0.95, f"stage phrase attn too similar: mean_cos={mean_cos:.4f}"
    print(f"[PASS] 4/8 stage phrase attn cross-stage distinguishable (mean_cos={mean_cos:.4f})")


def check_evidence_state_carry():
    """Evidence state non-None and evolves across Swin stages."""
    ctrl = _make_controller(
        cond_dim=32, use_evidence_state=True, evidence_state_dim=16,
        wti_stages=[0, 1, 2, 3],
    )
    tcf = _make_tcf(cond_dim=32, use_stage_phrase=True, use_hybrid_reliability=True)
    seg = torch.randn(2, 64)
    phr = torch.randn(2, 5, 64)
    mask = torch.ones(2, 5, dtype=torch.bool)
    tc, rel = tcf(seg, phrase_hidden=phr, phrase_mask=mask)

    swin = build_swin_b(None)
    imgs = torch.randn(2, 3, 384, 384)
    outs = swin(imgs, text_cond=tc, reliability=rel, tg_swin_controller=ctrl)
    assert len(outs) == 4
    for feat in outs:
        assert feat.shape[0] == 2

    state = None
    norms = []
    stage_dims = [128, 256, 512, 1024]
    for s in range(4):
        x = torch.randn(2, 144, stage_dims[s])
        state = ctrl.update_evidence_state(s, x, state)
        assert state is not None
        norms.append(state.norm(dim=-1).mean().item())
    print(f"[PASS] 5/8 evidence state carry (stage norms={norms})")


def check_identity_alpha_zero():
    """alpha=0 + state zero-init → attn_bias ≈ 0."""
    ctrl = _make_controller(
        cond_dim=32, use_evidence_state=True, evidence_state_dim=16,
        alpha_init=0.0, wti_stages=[1],
    )
    wti = ctrl.wti_blocks["1"]
    x = torch.randn(4, 144, 256)
    tc = torch.randn(2, 32)
    rel = torch.full((2, 1, 1), 0.5)
    state = torch.zeros(2, 16)
    bias, raw, _ = wti(x, tc, rel, evidence_state=state)
    assert bias.abs().max().item() < IDENTITY_THRESH, f"bias max={bias.abs().max()}"
    assert raw.abs().max().item() > 1e-8
    print("[PASS] 6/8 identity: alpha=0 + zero state → bias≈0")


def check_gradients():
    """stage-phrase and evidence-state params have non-zero grad at alpha=0.05."""
    tcf = _make_tcf(use_stage_phrase=True, use_hybrid_reliability=True)
    ctrl = _make_controller(
        cond_dim=32, use_evidence_state=True, evidence_state_dim=16,
        alpha_init=0.05, wti_stages=[1],
    )
    seg = torch.randn(2, 64, requires_grad=False)
    phr = torch.randn(2, 5, 64)
    mask = torch.ones(2, 5, dtype=torch.bool)

    tc, rel = tcf(seg, phrase_hidden=phr, phrase_mask=mask)
    wti = ctrl.wti_blocks["1"]
    x = torch.randn(4, 144, 256)
    state = torch.randn(2, 16, requires_grad=True)
    tc_s = tc[:, 1, :]
    rel_s = rel[:, 1, :]
    bias, _, _ = wti(x, tc_s, rel_s, evidence_state=state)
    loss = bias.sum()
    loss.backward()

    phrase_grad = tcf.router.stage_phrase_query[0].weight.grad
    assert phrase_grad is not None and phrase_grad.abs().max().item() > 0, "no stage-phrase grad"
    state_grad = ctrl.state_proj.weight.grad
    assert state_grad is not None and state_grad.abs().max().item() > 0, "no state_proj grad"
    print("[PASS] 7/8 gradients: stage-phrase + state_proj non-zero at alpha=0.05")


def check_v15_local_only_reliability():
    """USE_HYBRID_RELIABILITY=false → sigmoid-only (v1.5 path)."""
    tcf_off = _make_tcf(version="v1.5", use_hybrid_reliability=False)
    tcf_on = _make_tcf(version="v1.6", use_hybrid_reliability=True, use_stage_phrase=False)
    seg = torch.randn(3, 64)
    phr = torch.randn(3, 5, 64)
    mask = torch.ones(3, 5, dtype=torch.bool)
    torch.manual_seed(42)
    _, rel_off = tcf_off(seg, phrase_hidden=phr, phrase_mask=mask)
    torch.manual_seed(42)
    _, rel_on = tcf_on(seg, phrase_hidden=phr, phrase_mask=mask)
    diff = (rel_off - rel_on).abs().max().item()
    assert diff > 1e-6, "hybrid ρ should change reliability vs v1.5-local"
    print(f"[PASS] 8/8 hybrid ρ ablation: local-only vs hybrid differ (max Δ={diff:.6f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1.6")
    args = parser.parse_args()
    os.chdir(REPO)

    check_v15_equiv_flags_off()
    check_full_v16_shapes()
    check_hybrid_rho_global_sum()
    check_stage_phrase_distinguishable()
    check_evidence_state_carry()
    check_identity_alpha_zero()
    check_gradients()
    check_v15_local_only_reliability()

    print("[PASS] smoke_tgswin_v16 — all checks passed")


if __name__ == "__main__":
    main()
