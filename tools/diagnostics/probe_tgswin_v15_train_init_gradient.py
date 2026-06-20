#!/usr/bin/env python3
"""TG-Swin-WTI v1.5 train-init gradient probe (no manual param perturbation)."""

from __future__ import annotations

import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.mask_encoder.swin_trans import build_swin_b
from segearth_r2.model.mask_encoder.tg_swin import StageWTIHeadAware, TextConditionFactory, TGSwimController

MASK_CONFIG = "segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin.yaml"
IDENTITY_THRESH = 1e-5


def _abs_sum(grad) -> float:
    if grad is None:
        return 0.0
    return float(grad.abs().sum().item())


def _build_from_cfg():
    cfg = get_mask_config(MASK_CONFIG)
    tg = cfg.TG_SWIN
    text_dim = 2560
    tcf = TextConditionFactory(
        text_dim=text_dim,
        cond_dim=int(tg.COND_DIM),
        reliability_init=float(tg.RELIABILITY_INIT),
        version=str(tg.VERSION),
        num_stages=int(tg.NUM_STAGES),
        stage_router=bool(tg.STAGE_ROUTER),
        router_hidden_dim=int(tg.ROUTER_HIDDEN_DIM),
        use_relation_pool=bool(tg.USE_RELATION_AWARE_POOL),
    )
    controller = TGSwimController(
        cond_dim=int(tg.COND_DIM),
        wti_rank=int(tg.WTI_RANK),
        wti_stages=list(tg.WTI_STAGES),
        alpha_init=float(tg.ALPHA_INIT),
        window_size=int(cfg.MODEL.SWIN.WINDOW_SIZE),
        head_aware=bool(tg.HEAD_AWARE),
        num_text_stages=int(tg.NUM_STAGES),
    )
    return tcf, controller, text_dim, tg


def main():
    os.chdir(REPO)
    tcf, controller, text_dim, tg = _build_from_cfg()
    cfg = get_mask_config(MASK_CONFIG)
    assert str(tg.VERSION) == "v1.5", f"expected v1.5 config, got {tg.VERSION}"

    n, s, c = 2, int(tg.NUM_STAGES), int(tg.COND_DIM)
    seg_hidden = torch.randn(n, text_dim)
    phrase_hidden = torch.randn(n, 5, text_dim)
    phrase_mask = torch.ones(n, 5, dtype=torch.bool)

    stage_text_cond, reliability = tcf(
        seg_hidden, phrase_hidden=phrase_hidden, phrase_mask=phrase_mask
    )
    assert stage_text_cond.shape == (n, s, c), stage_text_cond.shape
    assert reliability.shape == (n, s, 1), reliability.shape
    print(f"[PASS] stage_text_cond {tuple(stage_text_cond.shape)} reliability {tuple(reliability.shape)}")
    print(f"       stage_text_cond_abs_mean={stage_text_cond.abs().mean().item():.6e}")
    print(f"       reliability_mean={reliability.mean().item():.4f}")

    wti = controller.wti_blocks["1"]
    assert isinstance(wti, StageWTIHeadAware)
    H, N = wti.num_heads, wti.N
    x_windows = torch.randn(4, N, wti.dim)
    tc = stage_text_cond[:, 1, :]
    rel = reliability[:, 1, :]

    with torch.no_grad():
        raw_bias = wti.compute_raw_bias(x_windows, tc, rel)
        gated, _, _ = wti(x_windows, tc, rel)
    raw_max = float(raw_bias.abs().max().item())
    gated_max = float(gated.abs().max().item())
    assert raw_max > 1e-6, f"raw_bias should be non-zero, max={raw_max}"
    assert gated_max <= IDENTITY_THRESH, f"gated bias max={gated_max}"
    print(f"[PASS] identity: raw_bias_abs_max={raw_max:.6e} gated_bias_abs_max={gated_max:.2e}")

    wti.zero_grad(set_to_none=True)
    gated, raw_bias, _ = wti(x_windows, tc.detach(), rel)
    gated.sum().backward()
    alpha_grad = _abs_sum(wti.alpha.grad)
    assert alpha_grad > 0, f"alpha.grad={alpha_grad}"
    print(f"[PASS] alpha.grad abs sum={alpha_grad:.6e}")

    # B: small alpha -> router / qk projections receive grad
    tcf_b = TextConditionFactory(
        text_dim=text_dim,
        cond_dim=int(tg.COND_DIM),
        reliability_init=float(tg.RELIABILITY_INIT),
        version=str(tg.VERSION),
        num_stages=int(tg.NUM_STAGES),
        stage_router=bool(tg.STAGE_ROUTER),
        router_hidden_dim=int(tg.ROUTER_HIDDEN_DIM),
        use_relation_pool=bool(tg.USE_RELATION_AWARE_POOL),
    )
    wti_b = StageWTIHeadAware(
        dim=wti.dim,
        cond_dim=int(tg.COND_DIM),
        num_heads=wti.num_heads,
        window_size=int(cfg.MODEL.SWIN.WINDOW_SIZE),
        rank=int(tg.WTI_RANK),
        alpha_init=float(tg.ALPHA_INIT),
    )
    with torch.no_grad():
        wti_b.alpha.fill_(0.05)
    stage_text_cond2, rel2 = tcf_b(
        seg_hidden, phrase_hidden=phrase_hidden, phrase_mask=phrase_mask
    )
    tc2 = stage_text_cond2[:, 1, :]
    rel2s = rel2[:, 1, :]
    gated2, _, _ = wti_b(x_windows, tc2, rel2s)
    gated2.sum().backward()
    proj_grads = {
        "visual_q": _abs_sum(wti_b.visual_q.weight.grad),
        "visual_k": _abs_sum(wti_b.visual_k.weight.grad),
        "text_q": _abs_sum(wti_b.text_q.weight.grad),
        "text_k": _abs_sum(wti_b.text_k.weight.grad),
        "router_last": _abs_sum(tcf_b.router.stage_router[-1].weight.grad),
    }
    assert all(v > 0 for v in proj_grads.values()), proj_grads
    print(f"[PASS] projection grads (alpha=0.05 no_grad set): {proj_grads}")

    controller_swin = TGSwimController(
        cond_dim=int(tg.COND_DIM),
        wti_rank=int(tg.WTI_RANK),
        wti_stages=list(tg.WTI_STAGES),
        alpha_init=float(tg.ALPHA_INIT),
        window_size=int(cfg.MODEL.SWIN.WINDOW_SIZE),
        head_aware=bool(tg.HEAD_AWARE),
        num_text_stages=int(tg.NUM_STAGES),
    )
    swin = build_swin_b(None)
    for p in swin.parameters():
        p.requires_grad = False
    stage_text_cond_s, reliability_s = tcf(
        seg_hidden, phrase_hidden=phrase_hidden, phrase_mask=phrase_mask
    )
    outs = swin(
        torch.randn(2, 3, 384, 384),
        text_cond=stage_text_cond_s,
        reliability=reliability_s,
        tg_swin_controller=controller_swin,
    )
    sum(o.float().sum() for o in outs).backward()
    swin_grad = sum(_abs_sum(p.grad) for p in swin.parameters())
    assert swin_grad == 0.0
    wti_s = controller_swin.wti_blocks["1"]
    print(f"[PASS] full Swin frozen backbone grad=0, stage-1 alpha.grad={_abs_sum(wti_s.alpha.grad):.6e}")

    print("[PASS] probe_tgswin_v15_train_init_gradient")


if __name__ == "__main__":
    main()
