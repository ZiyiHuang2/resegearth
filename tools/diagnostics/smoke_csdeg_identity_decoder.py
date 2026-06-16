#!/usr/bin/env python3
"""Save or load decoder state/output for baseline vs csdeg identity check."""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import torch
from addict import Dict


def _build_decoder(repo: Path, cs_deg_enabled: bool, device: str):
    sys.path.insert(0, str(repo))
    for key in list(sys.modules):
        if key.startswith("segearth_r2"):
            del sys.modules[key]

    from segearth_r2.model.mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.mask2former_transformer_decoder import (
        MultiScaleMaskedTransformerDecoderForOPTPreTrain,
    )

    cs_cfg = None
    if cs_deg_enabled:
        cs_cfg = Dict({"ENABLED": True})

    init_kwargs = dict(
        in_channels=256, hidden_dim=256, num_queries=1, nheads=8,
        dim_feedforward=512, dec_layers=2, mask_dim=256,
    )
    if "cs_deg_cfg" in inspect.signature(MultiScaleMaskedTransformerDecoderForOPTPreTrain.__init__).parameters:
        init_kwargs["cs_deg_cfg"] = cs_cfg

    return MultiScaleMaskedTransformerDecoderForOPTPreTrain(**init_kwargs).to(device).eval()


def _forward(dec, device):
    torch.manual_seed(0)
    b = 2
    ms = [
        torch.randn(b, 256, 32, 32, device=device),
        torch.randn(b, 256, 16, 16, device=device),
        torch.randn(b, 256, 8, 8, device=device),
    ]
    mf = torch.randn(b, 256, 64, 64, device=device)
    seg = torch.randn(b, 1, 256, device=device)
    fwd_kwargs = dict(SEG_embedding=seg)
    if "global_step" in inspect.signature(dec.forward).parameters:
        fwd_kwargs["global_step"] = 0
    with torch.no_grad():
        return dec(ms, mf, **fwd_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state-in")
    args = parser.parse_args()

    dec = _build_decoder(Path(args.repo), cs_deg_enabled=False, device=args.device)
    if args.state_in:
        blob = torch.load(args.state_in, map_location=args.device)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        dec.load_state_dict(state, strict=True)

    out = _forward(dec, args.device)
    payload = {
        "state_dict": dec.state_dict(),
        "pred_masks": out["pred_masks"].cpu(),
        "keys": sorted(out.keys()),
        "has_evidence": "pred_evidence_logits" in out,
    }
    torch.save(payload, args.out)


if __name__ == "__main__":
    main()
