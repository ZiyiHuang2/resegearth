#!/usr/bin/env python3
"""Verify TG-Swin-WTI train init: identity forward + non-zero grads without manual perturbation."""

from __future__ import annotations

import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.mask_encoder.swin_trans import build_swin_b
from segearth_r2.model.mask_encoder.tg_swin import StageWTI, TextConditionFactory, TGSwimController

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
        use_phrase_pool=bool(tg.USE_PHRASE_POOL),
    )
    controller = TGSwimController(
        cond_dim=int(tg.COND_DIM),
        wti_rank=int(tg.WTI_RANK),
        wti_stages=list(tg.WTI_STAGES),
        wti_start_layer=int(tg.WTI_START_LAYER),
        bias_max=float(tg.BIAS_MAX),
        alpha_init=float(tg.ALPHA_INIT),
        window_size=int(cfg.MODEL.SWIN.WINDOW_SIZE),
        swin_type="base",
        log_stats=False,
    )
    return tcf, controller, text_dim


def check_wti_identity_and_alpha_grad(controller: TGSwimController, text_cond: torch.Tensor, reliability: torch.Tensor):
    wti = controller.wti_blocks["1"]
    x_windows = torch.randn(4, wti.N, wti.dim)
    with torch.no_grad():
        bias = wti(x_windows, text_cond, reliability)
        bias_max = float(bias.abs().max().item())
    assert bias_max <= IDENTITY_THRESH, f"attn_bias abs max={bias_max} (expected <= {IDENTITY_THRESH})"
    print(f"[PASS] WTI identity: attn_bias.abs().max()={bias_max:.2e} (alpha=0 gate)")

    wti.zero_grad(set_to_none=True)
    bias = wti(x_windows, text_cond.detach(), reliability)
    loss = bias.sum()
    loss.backward()
    alpha_grad = _abs_sum(wti.alpha.grad)
    assert alpha_grad > 0, f"alpha.grad abs sum={alpha_grad}"
    print(f"[PASS] WTI alpha.grad abs sum={alpha_grad:.6e} (real init, loss=bias.sum())")


def check_tcf_grad(tcf: TextConditionFactory, text_dim: int):
    seg_hidden = torch.randn(2, text_dim)
    phrase_hidden = torch.randn(2, 4, text_dim)
    phrase_mask = torch.ones(2, 4, dtype=torch.bool)

    with torch.no_grad():
        text_cond, reliability = tcf(
            seg_hidden, phrase_hidden=phrase_hidden, phrase_mask=phrase_mask
        )
        assert text_cond.abs().max().item() > 0, "text_cond should be non-zero with small out_proj init"
        assert abs(reliability.mean().item() - 0.5) < 0.01, f"reliability init should be ~0.5, got {reliability.mean().item()}"

    tcf.zero_grad(set_to_none=True)
    text_cond, _ = tcf(seg_hidden, phrase_hidden=phrase_hidden, phrase_mask=phrase_mask)
    text_cond.pow(2).sum().backward()
    out_grad = _abs_sum(tcf.out_proj.weight.grad)
    assert tcf.out_proj.weight.grad is not None and out_grad > 0, f"out_proj grad={out_grad}"
    print(f"[PASS] TCF out_proj.weight.grad abs sum={out_grad:.6e}")


def check_full_swin_frozen_backbone(controller: TGSwimController, tcf: TextConditionFactory, text_dim: int):
    swin = build_swin_b(None)
    for p in swin.parameters():
        p.requires_grad = False

    seg_hidden = torch.randn(2, text_dim)
    phrase_hidden = torch.randn(2, 4, text_dim)
    phrase_mask = torch.ones(2, 4, dtype=torch.bool)
    text_cond, reliability = tcf(
        seg_hidden, phrase_hidden=phrase_hidden, phrase_mask=phrase_mask
    )

    x = torch.randn(2, 3, 384, 384)
    controller.zero_grad(set_to_none=True)
    tcf.zero_grad(set_to_none=True)
    outs = swin(
        x,
        text_cond=text_cond,
        reliability=reliability,
        tg_swin_controller=controller,
    )
    loss = sum(o.float().sum() for o in outs)
    loss.backward()

    wti = controller.wti_blocks["1"]
    alpha_grad = _abs_sum(wti.alpha.grad)
    score_grad = _abs_sum(wti.score_proj.weight.grad)
    swin_grad = sum(_abs_sum(p.grad) for p in swin.parameters())

    assert alpha_grad > 0, f"full Swin path alpha.grad={alpha_grad}"
    assert swin_grad == 0.0, f"frozen Swin should have 0 grad, got {swin_grad}"
    print(f"[PASS] full Swin path alpha.grad abs sum={alpha_grad:.6e}, frozen Swin grad=0")

    if wti.score_proj.weight.grad is None or score_grad == 0.0:
        print(
            "[INFO] score_proj grad is 0 at alpha=0 (expected step-0): "
            "score path trains once |alpha|>0 after first optimizer step"
        )
    else:
        print(f"[PASS] score_proj.weight.grad abs sum={score_grad:.6e}")


def check_score_path_connectivity_secondary(controller: TGSwimController):
    """Secondary: score_proj wiring (not train-init gate at alpha=0)."""
    wti = controller.wti_blocks["1"]
    wti.zero_grad(set_to_none=True)
    low_rank = torch.randn(2, 16)
    token_score = wti.score_proj(low_rank)
    token_score.pow(2).sum().backward()
    g = _abs_sum(wti.score_proj.weight.grad)
    assert wti.score_proj.weight.grad is not None and g > 0
    print(f"[SECONDARY PASS] score_proj wiring grad abs sum={g:.6e}")


def main():
    os.chdir(REPO)
    tcf, controller, text_dim = _build_from_cfg()

    seg_hidden = torch.randn(2, text_dim)
    phrase_hidden = torch.randn(2, 4, text_dim)
    phrase_mask = torch.ones(2, 4, dtype=torch.bool)
    text_cond, reliability = tcf(
        seg_hidden, phrase_hidden=phrase_hidden, phrase_mask=phrase_mask
    )

    check_wti_identity_and_alpha_grad(controller, text_cond, reliability)
    check_tcf_grad(tcf, text_dim)
    check_full_swin_frozen_backbone(controller, tcf, text_dim)
    check_score_path_connectivity_secondary(controller)

    print("[PASS] probe_tgswin_train_init_gradient")


if __name__ == "__main__":
    main()
