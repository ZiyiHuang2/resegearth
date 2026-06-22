#!/usr/bin/env python3
"""Probe TG-Swin v1.6 mechanism stats for Short-B gate."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from segearth_r2.model.mask_encoder.swin_trans import build_swin_b
from segearth_r2.model.mask_encoder.tg_swin import TextConditionFactory, TGSwimController


def _build_modules(cond_dim=32, evidence_dim=16):
    tcf = TextConditionFactory(
        text_dim=64,
        cond_dim=cond_dim,
        version="v1.6",
        num_stages=4,
        stage_router=True,
        use_stage_phrase=True,
        use_hybrid_reliability=True,
    )
    ctrl = TGSwimController(
        cond_dim=cond_dim,
        wti_rank=8,
        window_size=12,
        head_aware=True,
        use_evidence_state=True,
        evidence_state_dim=evidence_dim,
        wti_stages=[0, 1, 2, 3],
        alpha_init=0.05,
    )
    return tcf, ctrl


def probe_mechanism_stats(n_samples: int = 8, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    tcf, ctrl = _build_modules()

    seg = torch.randn(n_samples, 64)
    phr = torch.randn(n_samples, 10, 64)
    for i in range(n_samples):
        for t in range(10):
            phr[i, t, :] = float(t) + i * 0.1
    mask = torch.ones(n_samples, 10, dtype=torch.bool)

    tc, rel = tcf(seg, phrase_hidden=phr, phrase_mask=mask)

    reliability_per_stage_mean = rel.mean(dim=0).squeeze(-1).tolist()

    phrase_attn = tcf.router._last_phrase_attn
    cos_pairs = []
    if phrase_attn is not None:
        n_stages = phrase_attn.shape[1]
        for i in range(n_stages):
            for j in range(i + 1, n_stages):
                a = phrase_attn[:, i].mean(dim=0)
                b = phrase_attn[:, j].mean(dim=0)
                cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
                cos_pairs.append(cos)
    phrase_attn_cross_stage_cosine = (
        sum(cos_pairs) / len(cos_pairs) if cos_pairs else 1.0
    )

    swin = build_swin_b(None)
    imgs = torch.randn(n_samples, 3, 384, 384)
    swin(imgs, text_cond=tc, reliability=rel, tg_swin_controller=ctrl)

    state_norms = []
    state = None
    stage_dims = [128, 256, 512, 1024]
    for s in range(4):
        x = torch.randn(n_samples, 144, stage_dims[s])
        state = ctrl.update_evidence_state(s, x, state)
        state_norms.append(float(state.norm(dim=-1).mean().item()))

    return {
        "reliability_per_stage_mean": reliability_per_stage_mean,
        "phrase_attn_cross_stage_cosine": phrase_attn_cross_stage_cosine,
        "state_norm_per_stage": state_norms,
        "n_samples": n_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--n-samples", type=int, default=8)
    args = parser.parse_args()
    os.chdir(REPO)

    stats = probe_mechanism_stats(n_samples=args.n_samples)
    out = json.dumps(stats, indent=2)
    print(out)

    if args.output:
        with open(args.output, "w") as f:
            f.write(out)
        print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
