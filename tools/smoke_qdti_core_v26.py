#!/usr/bin/env python3
"""Stage 2.6 v3: QDTI-Core shape/numeric smoke (no full dataset)."""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_dec = ROOT / "segearth_r2/model/mask_decoder/Mask2Former_Simplify/modeling/transformer_decoder"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_qdti_mod = _load_module("qdti_core", _dec / "qdti_core.py")
QDTICore = _qdti_mod.QDTICore


def main():
    torch.manual_seed(0)
    B, S, Q, L, nh, n_layers = 2, 64, 5, 12, 8, 9
    Cm = Cq = Dt = 256
    mod = QDTICore(
        Dt, Cm, Cq, 128, max_abs=0.02, num_decoder_layers=n_layers, alpha_init=0.0
    )
    memory = torch.randn(S, B, Cm)
    query = torch.randn(Q, B, Cq)
    text_tokens = torch.randn(B, L, Dt)
    text_mask = torch.ones(B, L, dtype=torch.bool)
    text_mask[:, -2:] = False

    attn_bias, st = mod(
        memory, query, text_tokens, text_mask, nh, layer_idx=7, global_step=100, warmup_steps=500
    )
    P = st["P_bias_qs"]
    assert P.shape == (B, Q, S), f"P_bias_qs {tuple(P.shape)}"
    assert attn_bias.shape == (B * nh, Q, S), f"attn_bias {tuple(attn_bias.shape)}"
    assert float(P.abs().max()) <= 0.02 + 1e-5
    assert float(st["qdti_alpha_l"].item()) == 0.0
    assert abs(float(st["qdti_warmup_factor"].item()) - 0.2) < 1e-5
    print(
        f"[OK] P_bias_qs={tuple(P.shape)} attn_bias={tuple(attn_bias.shape)} "
        f"max={float(P.abs().max()):.4f} alpha_l={float(st['qdti_alpha_l']):.4f} "
        f"warmup={float(st['qdti_warmup_factor']):.4f}"
    )

    attn_bias0, _ = mod(memory, query, text_tokens, text_mask, nh, eval_mode="bypass")
    assert float(attn_bias0.abs().max()) == 0.0
    print("[OK] eval_mode=bypass yields zero bias")

    mod.alpha_l.data[7] = 1.0
    _, st2 = mod(memory, query, text_tokens, text_mask, nh, layer_idx=7, global_step=500, warmup_steps=500)
    assert float(st2["P_bias_qs"].abs().max()) > 0.0
    print("[OK] alpha_l=1 + warmup complete yields non-zero bias")

    assert mod._z_feat_dim == 7 * 128
    print(f"[OK] Z feature dim={mod._z_feat_dim} (7*bias_dim)")

    # diversity loss shape check (mirrors llava_phi._qdti_diversity_loss)
    import torch.nn.functional as F
    P_map = torch.randn(2, 4, 8, 8)
    neg_qs = [0, 2, 3]
    vecs = P_map[0, neg_qs].float().flatten(1)
    assert vecs.shape == (3, 64), vecs.shape
    vecs = F.normalize(vecs, dim=-1, eps=1e-6)
    sim = vecs @ vecs.T
    assert sim.shape == (3, 3), sim.shape
    off_diag = sim.masked_select(~torch.eye(3, dtype=torch.bool))
    assert off_diag.numel() == 6
    print(f"[OK] div_loss spatial vecs={tuple(vecs.shape)} sim={tuple(sim.shape)} off_diag={off_diag.numel()}")

    print("[PASS] smoke_qdti_core_v26")


if __name__ == "__main__":
    main()
