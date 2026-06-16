#!/usr/bin/env python3
"""Module-level shape check + single-batch forward smoke for DGP v6.1."""
import os
import sys

REPO = "/root/rivermind-data/huangziyi/reseg/segearth+DGP"
sys.path.insert(0, REPO)

import torch
from segearth_r2.datasets.dataset import get_mask_config
from segearth_r2.model.language_model.llava_phi import SegEarthR2
from segearth_r2.model.language_model.prompt_query_fusion import (
    DualGranularityPromptAdapter,
    PromptAwareQueryRefiner,
    pack_seg_hidden_states_bq,
    expand_bq_for_mask_num,
    expand_bp_for_mask_num,
)
from segearth_r2.model.mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.qdti import (
    QuerySpecificTextMemoryBias,
)


def check_modules():
    llm_dim, fuse_dim, mem_dim, n_heads = 2560, 256, 256, 8
    B, L, Q, S = 2, 32, 1, 64
    adapter = DualGranularityPromptAdapter(llm_dim, fuse_dim, pg_tokens=1)
    refiner = PromptAwareQueryRefiner(fuse_dim, 512, gate_g_init=0.01, gate_l_init=0.02)
    assert abs(float(refiner.gate_g.detach().item()) - 0.01) < 1e-6
    assert abs(float(refiner.gate_l.detach().item()) - 0.02) < 1e-6
    qdti0 = QuerySpecificTextMemoryBias(fuse_dim, mem_dim, mem_dim, 128, qdti_scale_init=0.0)
    qdti1 = QuerySpecificTextMemoryBias(fuse_dim, mem_dim, mem_dim, 128, qdti_scale_init=1e-3)

    hidden = torch.randn(B, L, llm_dim)
    attn = torch.ones(B, L, dtype=torch.bool)
    seg_mask = torch.zeros(B, L, dtype=torch.bool)
    seg_mask[:, -1] = True
    image_mask = torch.zeros(B, L, dtype=torch.bool)
    image_mask[:, 5:10] = True
    q_seg = torch.randn(B, Q, fuse_dim)

    p_g, p_l, prompt_tokens, prompt_mask, health = adapter(
        hidden, attn, seg_mask, q_seg=q_seg, image_mask=image_mask
    )
    assert p_g.shape == (B, Q, fuse_dim), p_g.shape
    assert p_l.shape == (B, Q, fuse_dim), p_l.shape
    assert prompt_tokens.shape == (B, 2 * Q, fuse_dim), prompt_tokens.shape
    assert prompt_mask.shape == (B, 2 * Q), prompt_mask.shape
    assert health["detail_prompt_source_name"] == "decoupled_attn"

    seg_hidden, seg_valid = pack_seg_hidden_states_bq(hidden, seg_mask)
    assert seg_hidden.shape == (B, Q, llm_dim), seg_hidden.shape
    q_ref, ref_health = refiner(seg_emb := q_seg, p_g, p_l, seg_query_mask=seg_valid, prompt_mask=prompt_mask)
    assert q_ref.shape == (B, Q, fuse_dim), q_ref.shape
    assert "refiner_delta_over_seg" in ref_health
    assert "query_refiner_gate_g" in ref_health
    assert "delta_g_norm" in ref_health

    mask_num = torch.tensor([1, 2])
    q_exp = expand_bq_for_mask_num(q_ref, mask_num)
    assert q_exp.shape == (3, Q, fuse_dim), q_exp.shape
    p_exp = expand_bp_for_mask_num(prompt_tokens, mask_num)
    assert p_exp.shape == (3, 2 * Q, fuse_dim), p_exp.shape

    memory = torch.randn(S, q_exp.shape[0], mem_dim)
    query = q_exp.permute(1, 0, 2)
    p_mask_exp = torch.repeat_interleave(prompt_mask, mask_num, dim=0)
    bias0, _ = qdti0(memory, query, p_exp, p_mask_exp, n_heads)
    assert bias0 is not None, "qdti_scale=0 must return zero extra_attn_bias tensor"
    assert float(bias0.abs().max().item()) == 0.0, "qdti_scale=0 extra_attn_bias must be all zeros"
    bias1, _ = qdti1(memory, query, p_exp, p_mask_exp, n_heads)
    assert bias1.shape == (q_exp.shape[0] * n_heads, Q, S), bias1.shape
    print("[OK] module shape checks passed")


def check_sigmoid_gate_init():
    refiner = PromptAwareQueryRefiner(256, 512, use_sigmoid_gate=True)
    gate_g = float(torch.sigmoid(refiner.gate_g_logit).item())
    gate_l = float(torch.sigmoid(refiner.gate_l_logit).item())
    assert abs(gate_g - 0.01) < 0.002, gate_g
    assert abs(gate_l - 0.02) < 0.002, gate_l
    print("[OK] sigmoid gate init (not sigmoid(-2))")


def check_instruction_mask():
    llm_dim, fuse_dim, B, L, Q = 2560, 256, 1, 12, 1
    adapter = DualGranularityPromptAdapter(llm_dim, fuse_dim)
    hidden = torch.randn(B, L, llm_dim)
    attn = torch.ones(B, L, dtype=torch.bool)
    attn[:, -2:] = False
    seg_mask = torch.zeros(B, L, dtype=torch.bool)
    seg_mask[:, -3] = True
    instruction = torch.zeros(B, L, dtype=torch.bool)
    instruction[:, :8] = True
    q_seg = torch.randn(B, Q, fuse_dim)
    _, _, _, _, health = adapter(
        hidden, attn, seg_mask, q_seg=q_seg, instruction_mask=instruction
    )
    assert health["detail_prompt_source_name"] == "decoupled_attn"
    print("[OK] instruction-only text mask path")


def check_decoupled_pl():
    llm_dim, fuse_dim, B, L, Q = 2560, 256, 1, 16, 2
    adapter = DualGranularityPromptAdapter(llm_dim, fuse_dim)
    hidden = torch.randn(B, L, llm_dim)
    attn = torch.ones(B, L, dtype=torch.bool)
    seg_mask = torch.zeros(B, L, dtype=torch.bool)
    seg_mask[:, -2:] = True
    q_seg = torch.randn(B, Q, fuse_dim)
    _, p_l, _, _, health = adapter(hidden, attn, seg_mask, q_seg=q_seg)
    assert p_l.shape == (B, Q, fuse_dim)
    assert health["detail_prompt_source_name"] == "decoupled_attn"
    print("[OK] decoupled P_l path (Q_detail always, no SEG fallback)")


def check_refer_span_path():
    llm_dim, fuse_dim, B, L, Q = 2560, 256, 1, 16, 1
    adapter = DualGranularityPromptAdapter(llm_dim, fuse_dim)
    hidden = torch.randn(B, L, llm_dim)
    attn = torch.ones(B, L, dtype=torch.bool)
    seg_mask = torch.zeros(B, L, dtype=torch.bool)
    seg_mask[:, -1] = True
    refer_span = torch.zeros(B, L, dtype=torch.bool)
    refer_span[:, 4:7] = True
    q_seg = torch.randn(B, Q, fuse_dim)
    _, _, _, _, health = adapter(
        hidden, attn, seg_mask, q_seg=q_seg, refer_span_mask=refer_span
    )
    assert health["detail_prompt_source_name"] == "refer_id"
    print("[OK] refer span diagnostic source (P_l query still Q_detail)")


def check_baseline_equivalence():
    from segearth_r2.model.mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder import (
        mask2former_transformer_decoder as m2f,
    )
    layer = m2f.CrossAttentionLayer(d_model=256, nhead=8)
    tgt = torch.randn(1, 2, 256)
    mem = torch.randn(64, 2, 256)
    out_a = layer(tgt, mem, extra_attn_bias=None)
    out_b = layer(tgt, mem)
    assert torch.allclose(out_a, out_b), "extra_attn_bias=None must match baseline"
    print("[OK] extra_attn_bias=None baseline equivalence")


def check_last1_qdti_layer():
    from segearth_r2.model.mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder import (
        mask2former_transformer_decoder as m2f,
    )
    dec = m2f.MultiScaleMaskedTransformerDecoderForOPTPreTrain.__new__(
        m2f.MultiScaleMaskedTransformerDecoderForOPTPreTrain
    )
    dec.num_layers = 6
    dec.qdti_apply_layers = "last1"
    assert dec._qdti_layer_active(5) is True
    assert dec._qdti_layer_active(4) is False
    print("[OK] last1 QDTI layer gating")


def check_fail_fast():
    sd = {"lm_head.weight": torch.zeros(1)}
    try:
        SegEarthR2.validate_dgp_qdti_checkpoint(sd, context="unit-test")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "use_dgp_qdti=True" in str(e)
    print("[OK] fail-fast on missing DGP-QDTI weights")


def check_model_keys():
    mask_cfg = get_mask_config(
        "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
    )
    model_path = "/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B"
    if not os.path.isdir(model_path):
        print(f"[SKIP] model path not found: {model_path}")
        return

    class Args:
        use_dgp_qdti = True
        use_qdti_bias = False
        dgp_fuse_dim = 256
        dgp_refiner_hidden_dim = 512
        dgp_pg_tokens = 1
        qdti_bias_dim = 128
        qdti_init_std = 1e-3
        qdti_max_abs = 0.01
        qdti_apply_layers = "last1"
        qdti_scale_init = 1e-3
        scale_hard_loss_weight = 0.0
        load_mask2former = False
        vision_tower_mask = ""

    model = SegEarthR2.from_pretrained(model_path, mask_decoder_cfg=mask_cfg, torch_dtype=torch.float16)
    model.initial_mask_module(pretrained_path=None, model_args=Args())
    keys = [k for k in model.state_dict().keys() if any(m in k for m in SegEarthR2.DGP_PROMPT_KEY_MARKERS)]
    expected = {"prompt_adapter", "query_refiner"}
    found = {m for k in keys for m in expected if m in k}
    assert found == expected, f"missing DGP modules in state_dict: {expected - found}"
    print(f"[OK] training model DGP keys present ({len(keys)} tensors)")


if __name__ == "__main__":
    check_modules()
    check_sigmoid_gate_init()
    check_instruction_mask()
    check_decoupled_pl()
    check_refer_span_path()
    check_baseline_equivalence()
    check_last1_qdti_layer()
    check_fail_fast()
    check_model_keys()
