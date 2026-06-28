#!/usr/bin/env python3
"""DR-EWTI diagnostics: identity, alignment, gradients, checkpoint, state_dict."""

from __future__ import annotations

import io
import os
import sys

import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

# Avoid segearth_r2.model.__init__ (pulls Mipha / transformers CLIP).
import types

_model_pkg = os.path.join(REPO, "segearth_r2", "model")
if "segearth_r2.model" not in sys.modules:
    _stub = types.ModuleType("segearth_r2.model")
    _stub.__path__ = [_model_pkg]
    sys.modules["segearth_r2.model"] = _stub

from segearth_r2.model.mask_decoder.mask_config.config import Config
from segearth_r2.model.mask_encoder.swin_trans import (
    align_coarse_evidence_to_windows,
    build_swin_b,
    window_partition,
)
from segearth_r2.model.mask_encoder.tg_swin import (
    StageDynamicRelationalWTI,
    StageWTIHeadAware,
    TGSwimController,
)

IDENTITY_ATOL = 1e-5
IDENTITY_RTOL = 1e-5


def _make_v15_wti(**kwargs):
    defaults = dict(dim=256, cond_dim=64, num_heads=8, window_size=12, rank=16, alpha_init=0.0)
    defaults.update(kwargs)
    return StageWTIHeadAware(**defaults)


def _make_dr_ext(**kwargs):
    defaults = dict(
        dim=256, cond_dim=64, num_heads=8, rank=16, bias_max=4.0,
        evidence_relation_gate_init=0.0,
    )
    defaults.update(kwargs)
    return StageDynamicRelationalWTI(**defaults)


def test_strict_identity_gate_zero():
    torch.manual_seed(0)
    base = _make_v15_wti()
    dr = _make_dr_ext()
    x = torch.randn(4, 144, 256)
    tc = torch.randn(2, 64)
    rel = torch.full((2, 1), 0.5)
    ev = torch.rand(4, 144)

    raw_v15 = base.compute_raw_bias(x, tc, rel)
    raw_dr = dr.compute_raw_bias(base, x, tc, ev)
    torch.testing.assert_close(raw_v15, raw_dr, atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL)

    bias_v15, _, _ = base(x, tc, rel)
    bias_dr, _, _, _ = dr(base, x, tc, rel, evidence_windows=ev, log_stats=False)
    torch.testing.assert_close(bias_v15, bias_dr, atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL)
    print("[PASS] 11.1 strict identity gate=0")


def test_gate_zero_gradient():
    torch.manual_seed(10)
    base = _make_v15_wti()
    dr = _make_dr_ext()
    with torch.no_grad():
        base.alpha.fill_(0.05)
    x = torch.randn(4, 144, 256)
    tc = torch.randn(2, 64)
    rel = torch.full((2, 1), 0.5)
    ev = torch.rand(4, 144)

    dr.zero_grad(set_to_none=True)
    base.zero_grad(set_to_none=True)
    out, _, _, _ = dr(base, x, tc, rel, evidence_windows=ev, log_stats=False)
    out.sum().backward()

    assert dr.evidence_relation_gate.grad is not None
    gate_grad = float(dr.evidence_relation_gate.grad.abs().sum().detach().cpu())
    assert gate_grad > 0, f"gate grad must be non-zero at init, got {gate_grad}"
    eq_grad = dr.evidence_q.weight.grad
    assert eq_grad is None or float(eq_grad.abs().sum()) == 0.0
    print(f"[PASS] gate=0 gradient non-zero (gate.grad abs sum={gate_grad:.6e})")


def test_evidence_missing_fallback():
    torch.manual_seed(1)
    base = _make_v15_wti()
    dr = _make_dr_ext()
    x = torch.randn(2, 144, 256)
    tc = torch.randn(1, 64)
    rel = torch.full((1, 1), 0.5)

    raw_v15 = base.compute_raw_bias(x, tc, rel)
    raw_dr = dr.compute_raw_bias(base, x, tc, evidence_windows=None)
    torch.testing.assert_close(raw_v15, raw_dr, atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL)
    print("[PASS] 11.2 evidence missing fallback")


def test_evidence_spatial_sensitivity():
    torch.manual_seed(2)
    base = _make_v15_wti()
    dr = _make_dr_ext()
    dr.evidence_relation_gate.data.fill_(5.0)
    nn.init.normal_(dr.evidence_q.weight, std=0.2)
    nn.init.normal_(dr.evidence_k.weight, std=0.2)
    x = torch.randn(4, 144, 256)
    tc = torch.randn(2, 64)

    ev_a = torch.full((4, 144), 0.5)
    ev_a[:, :72] = 0.95
    ev_a[:, 72:] = 0.05
    ev_b = torch.full((4, 144), 0.5)
    ev_b[:, :72] = 0.05
    ev_b[:, 72:] = 0.95

    raw_a = dr.compute_raw_bias(base, x, tc, ev_a)
    raw_b = dr.compute_raw_bias(base, x, tc, ev_b)
    assert (raw_a - raw_b).abs().max().item() > 1e-6

    delta_a = raw_a - base.compute_raw_bias(x, tc, None)
    delta_b = raw_b - base.compute_raw_bias(x, tc, None)
    for head in (0, 1):
        row0 = delta_a[0, head, 0, :]
        row1 = delta_a[0, head, 1, :]
        assert not torch.allclose(row0, row1, atol=1e-7), (
            f"head={head}: evidence delta must vary across query rows (key-only would match)"
        )
    query_variation = delta_a.var(dim=-2)
    assert float(query_variation.max()) > 1e-10

    dr.evidence_relation_gate.data.zero_()
    raw_a0 = dr.compute_raw_bias(base, x, tc, ev_a)
    raw_b0 = dr.compute_raw_bias(base, x, tc, ev_b)
    torch.testing.assert_close(raw_a0, raw_b0, atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL)
    print("[PASS] 11.3 evidence spatial sensitivity + non key-only delta")


def test_shape_and_numeric_alignment():
    torch.manual_seed(3)
    B, H, W = 2, 97, 103
    ws, shift = 12, 6
    coarse = torch.zeros(B, 1, H, W)
    for b in range(B):
        coarse[b, 0, b * 10, b * 10] = 0.9
        coarse[b, 0, H - 1 - b, W - 1 - b] = 0.1

    pad_r = (ws - W % ws) % ws
    pad_b = (ws - H % ws) % ws
    ev_windows = align_coarse_evidence_to_windows(
        coarse, H, W, ws, shift, 0, pad_r, 0, pad_b
    )

    ev_map = coarse.clone()
    ev_map = torch.nn.functional.interpolate(ev_map.float(), size=(H, W), mode="bilinear", align_corners=False)
    ev_map = torch.nn.functional.pad(ev_map, (0, pad_r, 0, pad_b))
    if shift > 0:
        ev_map = torch.roll(ev_map, shifts=(-shift, -shift), dims=(-2, -1))
    manual_windows = window_partition(ev_map.permute(0, 2, 3, 1), ws).view(-1, ws * ws).squeeze(-1)
    torch.testing.assert_close(ev_windows.float(), manual_windows, atol=1e-5, rtol=1e-5)
    print("[PASS] 11.4 numeric alignment (padding + shift + partition)")


def test_numerical_stability_dtypes():
    torch.manual_seed(4)
    tested = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        base = _make_v15_wti().to(dtype)
        dr = _make_dr_ext().to(dtype)
        dr.evidence_relation_gate.data.fill_(1.0)
        x = torch.randn(2, 144, 256, dtype=dtype)
        tc = torch.randn(1, 64, dtype=dtype)
        ev = torch.full((2, 144), 0.5, dtype=dtype)
        try:
            z = dr.evidence_fusion(x, ev)
            raw = dr.compute_raw_bias(base, x, tc, ev)
        except RuntimeError as exc:
            if dtype is torch.float32:
                raise
            print(f"[SKIP] 11.5 {dtype} on CPU ({exc})")
            continue
        assert torch.isfinite(z).all(), f"NaN/Inf in fusion dtype={dtype}"
        assert torch.isfinite(raw).all(), f"NaN/Inf in raw bias dtype={dtype}"
        tested.append(dtype)
    assert torch.float32 in tested
    print(f"[PASS] 11.5 fp stability ({', '.join(str(d).split('.')[-1] for d in tested)})")


def test_gradients_full():
    torch.manual_seed(5)
    base = _make_v15_wti()
    dr = _make_dr_ext()
    x = torch.randn(2, 144, 256)
    tc = torch.randn(1, 64)
    rel = torch.full((1, 1), 0.5)
    ev = torch.rand(2, 144)

    with torch.no_grad():
        base.alpha.fill_(0.05)
    dr.zero_grad(set_to_none=True)
    out, _, _, _ = dr(base, x, tc, rel, evidence_windows=ev, log_stats=False)
    out.sum().backward()
    assert dr.evidence_relation_gate.grad is not None
    assert float(dr.evidence_relation_gate.grad.abs().sum()) > 0

    dr.evidence_relation_gate.data.fill_(0.3)
    dr.zero_grad(set_to_none=True)
    out, _, _, _ = dr(base, x, tc, rel, evidence_windows=ev, log_stats=False)
    out.sum().backward()
    for name in (
        "evidence_q", "evidence_k",
        "evidence_fusion.evidence_feature_proj",
        "evidence_fusion.evidence_visual_proj",
        "evidence_fusion.evidence_add_proj",
        "evidence_fusion.evidence_gate_visual",
        "evidence_fusion.evidence_gate_mask",
    ):
        mod = dr
        for part in name.split("."):
            mod = getattr(mod, part)
        assert mod.weight.grad is not None
        assert float(mod.weight.grad.abs().sum()) > 0
    print("[PASS] 11.6 gradients (gate@0 + full evidence path)")


def test_legacy_stage_cond_index():
    ctrl = TGSwimController(
        cond_dim=64, wti_stages=[1, 2, 3], use_dr_ewti=True, head_aware=True,
        window_size=12, num_text_stages=4,
    )
    tc = torch.randn(2, 4, 64)
    rel = torch.full((2, 4, 1), 0.5)
    for swin_stage, cond_idx in ((1, 1), (2, 2), (3, 3)):
        t_sel, _ = ctrl._select_stage_cond(tc, rel, swin_stage)
        torch.testing.assert_close(t_sel, tc[:, cond_idx, :])
    print("[PASS] legacy stage index: swin 1/2/3 -> cond 1/2/3 (NUM_STAGES=4)")


def test_controller_batch_target_order():
    ctrl = TGSwimController(
        cond_dim=64, wti_stages=[1], head_aware=True, use_dr_ewti=True,
        window_size=12, num_text_stages=4,
    )
    tc = torch.randn(3, 4, 64)
    rel = torch.full((3, 4, 1), 0.5)
    t0, _ = ctrl._select_stage_cond(tc, rel, 1)
    assert t0.shape[0] == 3
    x = torch.randn(6, 144, 256)
    ev = torch.arange(6 * 144, dtype=torch.float32).view(6, 144) / (6 * 144)
    bias = ctrl.compute_bias(1, 0, x, tc, rel, evidence_windows=ev)
    assert bias is not None
    print("[PASS] B>1 target batch compute_bias")


def test_state_dict_no_duplicate_base():
    ctrl = TGSwimController(
        cond_dim=64, wti_stages=[1, 2, 3], use_dr_ewti=True, head_aware=True, window_size=12
    )
    keys = list(ctrl.state_dict().keys())
    assert not any("base_wti" in k for k in keys)
    vq_keys = [k for k in keys if k.endswith("visual_q.weight")]
    assert len(vq_keys) == 3
    assert all(k.startswith("wti_blocks.") for k in vq_keys)
    dr_keys = [k for k in keys if k.startswith("dr_wti_blocks.")]
    assert len(dr_keys) > 0
    assert all("visual_q" not in k for k in dr_keys)
    print(f"[PASS] state_dict: {len(vq_keys)} base WTI paths, {len(dr_keys)} DR-only keys")


def test_v15_controller_state_dict_compat():
    """Controller-level state-dict compatibility (synthetic v1.5 controller, not a trained checkpoint)."""
    ctrl_v15 = TGSwimController(
        cond_dim=64, wti_rank=16, wti_stages=[1, 2, 3], head_aware=True, window_size=12, use_dr_ewti=False
    )
    ctrl_dr = TGSwimController(
        cond_dim=64, wti_rank=16, wti_stages=[1, 2, 3], head_aware=True, window_size=12, use_dr_ewti=True
    )
    v15_sd = ctrl_v15.state_dict()
    missing, unexpected = ctrl_dr.load_state_dict(v15_sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert not any("base_wti" in m for m in missing)
    assert all(m.startswith("dr_wti_blocks.") for m in missing)
    print(f"[PASS] controller state-dict: v1.5→DR missing={len(missing)} (all dr_wti_blocks.*)")
    for m in sorted(missing)[:8]:
        print(f"       missing: {m}")
    if len(missing) > 8:
        print(f"       ... and {len(missing) - 8} more dr_wti_blocks keys")


def test_controller_state_dict_roundtrip():
    ctrl = TGSwimController(
        cond_dim=64, wti_stages=[1, 2], use_dr_ewti=True, head_aware=True, window_size=12
    )
    buf = io.BytesIO()
    torch.save(ctrl.state_dict(), buf)
    buf.seek(0)
    loaded = torch.load(buf, map_location="cpu")
    ctrl2 = TGSwimController(
        cond_dim=64, wti_stages=[1, 2], use_dr_ewti=True, head_aware=True, window_size=12
    )
    ctrl2.load_state_dict(loaded, strict=True)
    for k, v in ctrl.state_dict().items():
        torch.testing.assert_close(v, ctrl2.state_dict()[k])
    print("[PASS] controller state-dict save/reload round-trip")


def test_controller_evidence_merge_roundtrip():
    """Controller-level: v1.5 base state dict + evidence keys (not a trained checkpoint merge)."""
    ctrl_v15 = TGSwimController(
        cond_dim=64, wti_stages=[1, 2], use_dr_ewti=False, head_aware=True, window_size=12
    )
    ctrl_dr = TGSwimController(
        cond_dim=64, wti_stages=[1, 2], use_dr_ewti=True, head_aware=True, window_size=12
    )
    dr_sd = {k: v for k, v in ctrl_dr.state_dict().items() if k.startswith("dr_wti_blocks.")}
    merged = TGSwimController(
        cond_dim=64, wti_stages=[1, 2], use_dr_ewti=True, head_aware=True, window_size=12
    )
    missing, _ = merged.load_state_dict(ctrl_v15.state_dict(), strict=False)
    assert all(m.startswith("dr_wti_blocks.") for m in missing)
    merged.load_state_dict(dr_sd, strict=False)
    for k, v in dr_sd.items():
        torch.testing.assert_close(v, merged.state_dict()[k])
    print(f"[PASS] controller evidence merge: v1.5 base + {len(dr_sd)} evidence keys")


def test_log_stats_off_no_extra():
    base = _make_v15_wti()
    dr = _make_dr_ext()
    x = torch.randn(2, 144, 256)
    tc = torch.randn(1, 64)
    rel = torch.full((1, 1), 0.5)
    ev = torch.rand(2, 144)
    out, raw, gate, diag = dr(base, x, tc, rel, evidence_windows=ev, log_stats=False)
    assert diag is None
    assert out.shape == raw.shape
    print("[PASS] LOG_STATS=false returns no diag dict")


def test_swin_frozen_grad():
    torch.manual_seed(6)
    swin = build_swin_b(None)
    for p in swin.parameters():
        p.requires_grad = False
    ctrl = TGSwimController(
        cond_dim=64, wti_stages=[1], head_aware=True, use_dr_ewti=True, window_size=12, log_stats=False
    )
    with torch.no_grad():
        ctrl.wti_blocks["1"].alpha.fill_(0.05)
    imgs = torch.randn(1, 3, 384, 384)
    tc = torch.randn(1, 3, 64)
    rel = torch.full((1, 3, 1), 0.5)
    coarse = torch.rand(1, 1, 256, 256)
    outs = swin(
        imgs, text_cond=tc, reliability=rel, tg_swin_controller=ctrl,
        coarse_evidence=coarse, enable_tg_swin=True,
    )
    sum(o.float().sum() for o in outs).backward()
    swin_grad = sum(float(p.grad.abs().sum()) for p in swin.parameters() if p.grad is not None)
    assert swin_grad == 0.0
    gate_grad = ctrl.dr_wti_blocks["1"].evidence_relation_gate.grad
    assert gate_grad is not None
    gate_grad_val = float(gate_grad.abs().sum())
    assert gate_grad_val > 0
    print(f"[PASS] frozen Swin; gate.grad={gate_grad_val:.6e}")


def main():
    os.chdir(REPO)

    test_strict_identity_gate_zero()
    test_gate_zero_gradient()
    test_evidence_missing_fallback()
    test_evidence_spatial_sensitivity()
    test_shape_and_numeric_alignment()
    test_numerical_stability_dtypes()
    test_gradients_full()
    test_controller_batch_target_order()
    test_legacy_stage_cond_index()
    test_state_dict_no_duplicate_base()
    test_v15_controller_state_dict_compat()
    test_controller_state_dict_roundtrip()
    test_controller_evidence_merge_roundtrip()
    test_log_stats_off_no_extra()
    test_swin_frozen_grad()

    cfg = Config.fromfile("segearth_r2/model/mask_decoder/mask_config/maskformer2_tgswin_dr_ewti.yaml")
    assert str(cfg.TG_SWIN.VERSION) == "dr-ewti"
    assert int(cfg.TG_SWIN.NUM_STAGES) == 4
    assert list(cfg.TG_SWIN.WTI_STAGES) == [1, 2, 3]
    print("[PASS] configs ok")
    print("[PASS] smoke_tgswin_dr_ewti")


if __name__ == "__main__":
    main()
