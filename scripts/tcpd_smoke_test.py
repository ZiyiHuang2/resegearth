#!/usr/bin/env python3
"""Smoke test for SEG-conditioned TCPD-v2 (forward, identity, gradients, spatial adaptivity)."""

import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from addict import Dict

from segearth_r2.model.mask_decoder.Mask2Former_Simplify.modeling.pixel_decoder.msdeformattn import (
    MSDeformAttnPixelDecoder,
)


def make_input_shape():
    return {
        "res2": Dict({"channel": 128, "stride": 4}),
        "res3": Dict({"channel": 256, "stride": 8}),
        "res4": Dict({"channel": 512, "stride": 16}),
        "res5": Dict({"channel": 1024, "stride": 32}),
    }


def make_dummy_features(bs, device, dtype=torch.float32):
    return {
        "res2": torch.randn(bs, 128, 64, 64, device=device, dtype=dtype),
        "res3": torch.randn(bs, 256, 32, 32, device=device, dtype=dtype),
        "res4": torch.randn(bs, 512, 16, 16, device=device, dtype=dtype),
        "res5": torch.randn(bs, 1024, 8, 8, device=device, dtype=dtype),
    }


def iter_tcpd_params(model):
    for name, param in model.named_parameters():
        if "tcpd" in name:
            yield name, param


def is_gate_param(name):
    return any(
        token in name
        for token in (
            "tcpd_gate_offset", "tcpd_gate_attn", "gate_scale", "tcpd_gate_mask",
            "gate_td", "gate_lat",
        )
    )


def grad_summary(model):
    gate_nonzero = branch_nonzero = 0
    gate_total = branch_total = 0
    rows = []
    for name, param in iter_tcpd_params(model):
        grad = param.grad
        has_grad = grad is not None and grad.abs().sum().item() > 0
        kind = "gate" if is_gate_param(name) else "branch"
        if kind == "gate":
            gate_total += 1
            gate_nonzero += int(has_grad)
        else:
            branch_total += 1
            branch_nonzero += int(has_grad)
        rows.append((name, kind, has_grad))
    return rows, gate_nonzero, gate_total, branch_nonzero, branch_total


def set_tcpd_gates(model, value):
    with torch.no_grad():
        for name, param in iter_tcpd_params(model):
            if is_gate_param(name):
                param.fill_(value)


def build_pd(device, dtype=torch.float32):
    return MSDeformAttnPixelDecoder(
        make_input_shape(),
        transformer_dropout=0.0,
        transformer_nheads=8,
        transformer_dim_feedforward=1024,
        transformer_enc_layers=2,
        conv_dim=256,
        mask_dim=256,
        transformer_in_features=["res3", "res4", "res5"],
        common_stride=4,
    ).to(device=device, dtype=dtype)


def forward_kwargs(mode=None):
    kw = {}
    if mode is not None:
        kw["tcpd_spatial_mode"] = mode
    return kw


def run_backward(pd, features, seg_cond, **kwargs):
    pd.zero_grad(set_to_none=True)
    mask_features, _, multi_scale = pd.forward_features(features, tcpd_condition=seg_cond, **kwargs)
    loss = mask_features.sum()
    for feat in multi_scale:
        loss = loss + feat.sum()
    loss.backward()
    return loss.item()


def check_spatial_adaptivity(pd, device, dtype):
    pd.eval()
    attn = pd.transformer.encoder.layers[0].self_attn
    bs = 1
    len_q = 8
    query = torch.randn(bs, len_q, 256, device=device, dtype=dtype)
    z = torch.randn(bs, 256, device=device, dtype=dtype)
    with torch.no_grad():
        attn.tcpd_gate_offset.fill_(1.0)
        attn.tcpd_gate_attn.fill_(1.0)
        delta_offset, _ = attn.compute_tcpd_deltas(query, z, tcpd_spatial_mode="spatial")
    diff_01 = (delta_offset[:, 0] - delta_offset[:, 1]).abs().max().item()
    print(f"  max |delta_offset[:,0] - delta_offset[:,1]| = {diff_01:.3e}")
    assert diff_01 > 1e-6, "spatial mode should produce per-token varying offsets"
    with torch.no_grad():
        attn.tcpd_gate_offset.zero_()
        attn.tcpd_gate_attn.zero_()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    pd = build_pd(device, dtype)

    bs = 2
    features = make_dummy_features(bs, device, dtype)
    seg_cond = torch.randn(bs, 1, 256, device=device, dtype=dtype, requires_grad=False)

    print("=== 1. Forward shape check (off / global / spatial) ===")
    pd.eval()
    with torch.no_grad():
        mask_off, _, ms_off = pd.forward_features(features)
        mask_g, _, ms_g = pd.forward_features(
            features, tcpd_condition=seg_cond, tcpd_spatial_mode="global"
        )
        mask_s, _, ms_s = pd.forward_features(
            features, tcpd_condition=seg_cond, tcpd_spatial_mode="spatial"
        )
    print(f"  off: mask={tuple(mask_off.shape)}, ms={[tuple(t.shape) for t in ms_off]}")
    print(f"  global: mask={tuple(mask_g.shape)}, ms={[tuple(t.shape) for t in ms_g]}")
    print(f"  spatial: mask={tuple(mask_s.shape)}, ms={[tuple(t.shape) for t in ms_s]}")
    assert mask_off.shape == mask_g.shape == mask_s.shape
    assert all(a.shape == b.shape == c.shape for a, b, c in zip(ms_off, ms_g, ms_s))
    print("  PASS")

    print("\n=== 2. Identity init check (gates=0) ===")
    with torch.no_grad():
        mask_a, _, ms_a = pd.forward_features(features)
        mask_g, _, ms_g = pd.forward_features(
            features, tcpd_condition=seg_cond, tcpd_spatial_mode="global"
        )
        mask_s, _, ms_s = pd.forward_features(
            features, tcpd_condition=seg_cond, tcpd_spatial_mode="spatial"
        )
    mask_diff_g = (mask_a - mask_g).abs().max().item()
    mask_diff_s = (mask_a - mask_s).abs().max().item()
    ms_diff_g = max((a - b).abs().max().item() for a, b in zip(ms_a, ms_g))
    ms_diff_s = max((a - b).abs().max().item() for a, b in zip(ms_a, ms_s))
    print(f"  max |mask_off - mask_global| = {mask_diff_g:.3e}")
    print(f"  max |mask_off - mask_spatial| = {mask_diff_s:.3e}")
    print(f"  max |ms_off - ms_global|     = {ms_diff_g:.3e}")
    print(f"  max |ms_off - ms_spatial|     = {ms_diff_s:.3e}")
    assert mask_diff_g <= 1e-5 and mask_diff_s <= 1e-5
    assert ms_diff_g <= 1e-5 and ms_diff_s <= 1e-5
    print("  PASS")

    print("\n=== 3. Per-target repeat check ===")
    mask_num = [2, 1]
    repeats = torch.tensor(mask_num, device=device, dtype=torch.long)
    pd_features = {k: torch.repeat_interleave(v, repeats=repeats, dim=0) for k, v in features.items()}
    seg_multi = torch.randn(sum(mask_num), 1, 256, device=device, dtype=dtype)
    with torch.no_grad():
        mask_m, _, _ = pd.forward_features(pd_features, tcpd_condition=seg_multi)
    assert mask_m.shape[0] == sum(mask_num)
    print(f"  mask_num={mask_num}, sum_K={sum(mask_num)}, mask batch={mask_m.shape[0]}")
    print("  PASS")

    print("\n=== 4. Backward gradient check (gates=0) ===")
    pd.train()
    for _, param in iter_tcpd_params(pd):
        param.requires_grad_(True)
    loss_val = run_backward(pd, features, seg_cond)
    rows, gate_nz, gate_total, branch_nz, branch_total = grad_summary(pd)
    print(f"  loss={loss_val:.3f}")
    print(f"  gate params with nonzero grad: {gate_nz}/{gate_total}")
    print(f"  branch params with nonzero grad: {branch_nz}/{branch_total} (expected 0 at gate=0)")
    assert gate_nz > 0, "gate params must receive gradients at gate=0"
    assert branch_nz == 0, "branch params should have zero grad when gate=0"
    print("  PASS")

    print("\n=== 5. Backward gradient check (gates=1e-3) ===")
    set_tcpd_gates(pd, 1e-3)
    loss_val = run_backward(pd, features, seg_cond)
    rows, gate_nz, gate_total, branch_nz, branch_total = grad_summary(pd)
    print(f"  loss={loss_val:.3f}")
    print(f"  gate params with nonzero grad: {gate_nz}/{gate_total}")
    print(f"  branch params with nonzero grad: {branch_nz}/{branch_total}")
    msdeform_grad = any(
        has_grad for name, kind, has_grad in rows
        if ("tcpd_delta" in name or "tcpd_spatial_delta" in name) and kind == "branch"
    )
    fpn_grad = any(
        has_grad for name, kind, has_grad in rows
        if "tcpd_fpn_fusion" in name and kind == "branch"
    )
    scale_grad = any(
        has_grad for name, kind, has_grad in rows
        if "tcpd_scale_fusion" in name and kind == "branch"
    )
    assert branch_nz > 0, "branch params must receive gradients when gates are nonzero"
    assert msdeform_grad, "MSDeformAttn delta branches must receive gradients"
    assert fpn_grad, "FPN conditioning branches must receive gradients"
    assert scale_grad, "output scale fusion branches must receive gradients"
    print("  PASS")

    print("\n=== 6. Spatial adaptivity check ===")
    check_spatial_adaptivity(pd, device, dtype)
    print("  PASS")

    print("\n=== 7. get_tcpd_level_weights helper ===")
    z = torch.randn(bs, 256, device=device, dtype=dtype)
    w = pd.tcpd_scale_fusion.get_tcpd_level_weights(z)
    assert w.shape == (bs, 3)
    print(f"  level_weights shape={tuple(w.shape)}, sample={w[0].tolist()}")
    print("  PASS")

    print("\n=== Smoke test PASSED ===")
    return 0


def run_pure_train_sanity():
    print("\n=== 8. Pure train launch sanity (MAX_STEPS=1) ===")
    env = os.environ.copy()
    env["MAX_STEPS"] = "1"
    env["WANDB_MODE"] = "disabled"
    script = os.path.join(REPO, "scripts", "tcpd_train_pure.sh")
    proc = subprocess.run(
        ["bash", script],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    print(proc.stdout[-4000:] if len(proc.stdout) > 4000 else proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        raise RuntimeError(f"tcpd_train_pure.sh failed with code {proc.returncode}")
    for needle in (
        "pure_tcpd_mode: True",
        "ASSERT OK: pure TCPD",
        "ASSERT OK: only TCPD pixel_decoder params trainable",
        "predictor trainable: 0",
        "SEG_token_projector trainable: 0",
        "lm_head trainable: 0",
        "unfrozen LLM layers: none",
    ):
        assert needle in combined, f"missing log evidence: {needle!r}"
    print("  PASS")


if __name__ == "__main__":
    rc = main()
    if os.environ.get("TCPD_SMOKE_SKIP_TRAIN", "") not in ("1", "true", "True"):
        run_pure_train_sanity()
    raise SystemExit(rc)
