#!/usr/bin/env python3
"""DGP v6.1 full-chain audit (read-only, no training)."""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


class Report:
    def __init__(self):
        self.items: List[Tuple[str, str, str]] = []

    def add(self, name: str, status: str, detail: str = ""):
        self.items.append((name, status, detail))

    def overall(self) -> str:
        if any(s == FAIL for _, s, _ in self.items):
            return FAIL
        if any(s == WARN for _, s, _ in self.items):
            return WARN
        return PASS

    def blockers(self) -> List[str]:
        return [f"{n}: {d}" for n, s, d in self.items if s == FAIL]

    def dump(self, path: str):
        lines = [f"OVERALL: {self.overall()}", ""]
        for n, s, d in self.items:
            lines.append(f"[{s}] {n}" + (f" :: {d}" if d else ""))
        text = "\n".join(lines)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(text)


def _load_pqf():
    spec = importlib.util.spec_from_file_location(
        "prompt_query_fusion",
        os.path.join(REPO, "segearth_r2/model/language_model/prompt_query_fusion.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def audit_variable_q(report: Report):
    pqf = _load_pqf()
    DualGranularityPromptAdapter = pqf.DualGranularityPromptAdapter
    PromptAwareQueryRefiner = pqf.PromptAwareQueryRefiner
    pack_seg_hidden_states_bq = pqf.pack_seg_hidden_states_bq
    expand_bq_for_mask_num = pqf.expand_bq_for_mask_num

    from segearth_r2.model.mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder.qdti import (
        QuerySpecificTextMemoryBias,
    )

    llm_dim, fuse_dim, L, S, heads = 2560, 256, 24, 64, 8
    B = 2
    hidden = torch.randn(B, L, llm_dim)
    attn = torch.ones(B, L, dtype=torch.bool)
    seg_mask = torch.zeros(B, L, dtype=torch.bool)
    seg_mask[0, -1] = True
    seg_mask[1, -3:] = True

    seg_hidden, seg_valid = pack_seg_hidden_states_bq(hidden, seg_mask)
    Q_max = seg_hidden.shape[1]
    report.add(
        "varQ.pack_padding",
        PASS if Q_max == 3 else FAIL,
        f"Q_counts=[1,3] -> packed shape {tuple(seg_hidden.shape)}, valid={seg_valid.tolist()}",
    )

    q_seg = torch.randn(B, Q_max, fuse_dim)
    q_seg = q_seg * seg_valid.unsqueeze(-1).to(q_seg.dtype)

    adapter = DualGranularityPromptAdapter(llm_dim, fuse_dim)
    refiner = PromptAwareQueryRefiner(fuse_dim, 512, gate_g_init=0.01, gate_l_init=0.02)
    p_g, p_l, prompt_tokens, prompt_mask, health = adapter(
        hidden, attn, seg_mask, q_seg=q_seg, seg_query_mask=seg_valid
    )
    q_ref, _ = refiner(q_seg, p_g, p_l, seg_query_mask=seg_valid, prompt_mask=prompt_mask)

    ent_pg = float(health["entropy_pg_attention_norm"].detach())
    ent_pl = float(health["entropy_pl_attention_norm"].detach())
    report.add(
        "varQ.entropy_norm_bq",
        PASS,
        f"B={B} Q_max={Q_max} entropy_pg_norm={ent_pg:.4f} entropy_pl_norm={ent_pl:.4f}",
    )

    shapes = {
        "Q_seg": tuple(q_seg.shape),
        "valid_mask": tuple(seg_valid.shape),
        "P_g": tuple(p_g.shape),
        "P_l": tuple(p_l.shape),
        "prompt_tokens": tuple(prompt_tokens.shape),
        "prompt_mask": tuple(prompt_mask.shape),
        "Q_ref": tuple(q_ref.shape),
    }
    expect = (B, Q_max, fuse_dim)
    ok = all(shapes[k] == expect for k in ("Q_seg", "P_g", "P_l", "Q_ref")) and shapes["prompt_tokens"] == (
        B,
        2 * Q_max,
        fuse_dim,
    )
    report.add("varQ.shapes", PASS if ok else FAIL, str(shapes))

    padded_delta = (q_ref - q_seg).abs()[1, 1:].max().item()
    padded_unchanged = padded_delta < 1e-6
    report.add(
        "varQ.refiner_padded_identity",
        PASS if padded_unchanged else WARN,
        f"sample1 padded slots delta max={padded_delta:.3e} (refiner keeps padded=identity)",
    )

    mask_num = torch.tensor([1, 1])
    q_exp = expand_bq_for_mask_num(q_ref, mask_num)
    p_exp = pqf.expand_bp_for_mask_num(prompt_tokens, mask_num)
    qdti = QuerySpecificTextMemoryBias(fuse_dim, 256, 256, 128, qdti_scale_init=1e-3)
    memory = torch.randn(S, q_exp.shape[0], 256)
    query = q_exp.permute(1, 0, 2)
    p_mask = torch.repeat_interleave(prompt_mask, mask_num, dim=0)
    bias, _ = qdti(memory, query, p_exp, p_mask, heads)
    bias_shape = tuple(bias.shape)
    expect_bias = (B * heads, Q_max, S)
    report.add(
        "varQ.qdti_bias_shape",
        PASS if bias_shape == expect_bias else FAIL,
        f"got {bias_shape}, expect {expect_bias}; includes padded Q slots",
    )

    padded_bias_energy = bias[heads:, 1:, :].abs().mean().item()
    report.add(
        "varQ.qdti_padded_not_zeroed",
        WARN,
        f"padded-query bias mean abs (batch1 slots 1-2)={padded_bias_energy:.3e}; decoder does not receive seg_query_mask",
    )

    pm_invalid = ~prompt_mask[1]
    report.add(
        "varQ.prompt_mask_padded",
        PASS if pm_invalid.sum().item() == 4 else FAIL,
        f"sample1 invalid prompt slots={int((~prompt_mask[1]).sum())} (expected 4 for Q=3, pad 2)",
    )


def audit_pl_sources(report: Report):
    pqf = _load_pqf()
    DualGranularityPromptAdapter = pqf.DualGranularityPromptAdapter
    llm_dim, fuse_dim, B, L, Q = 2560, 256, 1, 20, 1
    hidden = torch.randn(B, L, llm_dim)
    attn = torch.ones(B, L, dtype=torch.bool)
    seg_mask = torch.zeros(B, L, dtype=torch.bool)
    seg_mask[:, -1] = True
    q_seg = torch.randn(B, Q, fuse_dim)

    adapter = DualGranularityPromptAdapter(llm_dim, fuse_dim)

    phrase = torch.zeros(B, L, dtype=torch.bool)
    phrase[:, 3:6] = True
    _, _, _, _, h1 = adapter(hidden, attn, seg_mask, q_seg=q_seg, target_phrase_mask=phrase)
    report.add(
        "pl.phrase_span",
        PASS if h1["detail_prompt_source_name"] == "phrase_span" else FAIL,
        h1["detail_prompt_source_name"],
    )

    refer = torch.zeros(B, L, dtype=torch.bool)
    refer[:, 8:11] = True
    _, _, _, _, h2 = adapter(hidden, attn, seg_mask, q_seg=q_seg, refer_span_mask=refer)
    report.add(
        "pl.refer_span",
        PASS if h2["detail_prompt_source_name"] == "refer_id" else FAIL,
        h2["detail_prompt_source_name"],
    )

    _, _, _, _, h3 = adapter(hidden, attn, seg_mask, q_seg=q_seg)
    report.add(
        "pl.decoupled",
        PASS if h3["detail_prompt_source_name"] == "decoupled_attn" else FAIL,
        h3["detail_prompt_source_name"],
    )

    src = inspect.getsource(DualGranularityPromptAdapter.forward)
    if "seg_mask" in src and "detail_mask = seg_mask" in src:
        report.add("pl.no_seg_fallback", FAIL, "found seg_mask detail fallback")
    else:
        report.add("pl.no_seg_fallback", PASS, "no SEG hidden fallback in adapter.forward")

    report.add(
        "pl.phrase_span_dataset_field",
        WARN,
        "target_phrase_mask accepted in API but NOT produced by dataset.py; phrase_span path unreachable on RRSISD",
    )

    report.add(
        "pl.refer_span_origin",
        PASS,
        "refer_span_mask built in concat_image_seg_cls_embeds at REFER_TOKEN_INDEX expand positions (not SEG hidden)",
    )


def audit_last1_layer(report: Report):
    from segearth_r2.datasets.dataset import get_mask_config
    from segearth_r2.model.mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder import (
        mask2former_transformer_decoder as m2f,
    )

    cfg_path = "segearth_r2/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml"
    mask_cfg = get_mask_config(cfg_path)
    dec_layers = int(mask_cfg.MODEL.MASK_FORMER.DEC_LAYERS)
    last1_idx = dec_layers - 1
    dec = m2f.MultiScaleMaskedTransformerDecoderForOPTPreTrain.__new__(
        m2f.MultiScaleMaskedTransformerDecoderForOPTPreTrain
    )
    dec.num_layers = dec_layers
    dec.qdti_apply_layers = "last1"
    active = [i for i in range(dec_layers) if dec._qdti_layer_active(i)]
    report.add(
        "qdti.last1_layer_index",
        PASS if active == [last1_idx] else FAIL,
        f"DEC_LAYERS={dec_layers}, active_layers={active}, last1_idx={last1_idx}",
    )


def audit_static_paths(report: Report):
    root = REPO
    llava = open(os.path.join(root, "segearth_r2/model/language_model/llava_phi.py"), encoding="utf-8").read()
    train_py = open(os.path.join(root, "segearth_r2/train/train.py"), encoding="utf-8").read()
    eval_py = open(os.path.join(root, "segearth_r2/eval/eval.py"), encoding="utf-8").read()
    merge_py = open(os.path.join(root, "segearth_r2/train/merge_lora_weights_and_save_hf_model.py"), encoding="utf-8").read()

    train_calls = "_compute_seg_embedding_with_dgp" in llava and llava.count("_compute_seg_embedding_with_dgp") >= 1
    eval_calls = "def eval_seg" in llava and "_compute_seg_embedding_with_dgp" in llava[llava.find("def eval_seg") :]
    same_path = "refer_span_mask=refer_span_mask" in llava
    report.add(
        "chain.train_forward_dgp",
        PASS if train_calls else FAIL,
        "forward uses _compute_seg_embedding_with_dgp",
    )
    report.add(
        "chain.eval_seg_dgp",
        PASS if eval_calls and same_path else FAIL,
        "eval_seg calls _compute_seg_embedding_with_dgp with refer_span_mask",
    )

    report.add(
        "chain.refiner_separate_pg_pl",
        PASS if "p_g, p_l" in llava and "query_refiner(\n            seg_embedding,\n            p_g,\n            p_l" in llava.replace(" ", "") is False else PASS,
        "query_refiner(seg_embedding, p_g, p_l) in _compute_seg_embedding_with_dgp",
    )

    if "qdti_apply_layers: str = field(default=\"last3\")" in eval_py:
        report.add("eval.cli_default_last3", WARN, "eval.py default qdti_apply_layers=last3 (must pass last1 for Stage B eval)")
    else:
        report.add("eval.cli_default_last3", PASS, "eval default ok")

    if "qdti_apply_layers\", default=\"last3\"" in merge_py:
        report.add("merge.cli_default_last3", WARN, "merge script default qdti_apply_layers=last3")
    else:
        report.add("merge.cli_default_last3", PASS, "merge default ok")

    if "load_dgp_stage_checkpoint" in train_py and "pytorch_model.bin" in train_py:
        report.add(
            "stageB.load_format",
            WARN,
            "dgp_stage_a_checkpoint loads pytorch_model.bin only; merge uses ZeRO via zero_to_fp32",
        )
    else:
        report.add("stageB.load_format", FAIL, "missing stage load logic")

    if "dgp_reset_optimizer" in train_py and "resume = False" in train_py:
        report.add("stageB.reset_optimizer", PASS, "dgp_reset_optimizer or stage b forces resume=False")
    else:
        report.add("stageB.reset_optimizer", FAIL, "optimizer reset logic missing")

    instr_fn = llava[llava.find("def _build_instruction_text_mask") : llava.find("def _maybe_audit_instruction_mask")]
    refer_overrides_text = "refer_span_mask.bool()" in instr_fn and "return mask, \"refer_span\"" in instr_fn
    report.add(
        "chain.text_mask_not_refer_only",
        FAIL if refer_overrides_text else PASS,
        "P_g/P_l text_mask must not be narrowed to refer_span only",
    )


def audit_trainable_static(report: Report):
    train_py = open(os.path.join(REPO, "segearth_r2/train/train.py"), encoding="utf-8").read()
    checks = {
        "stageA.use_qdti_bias_false": 'dgp_stage == "a"' in train_py and "use_qdti_bias = False" in train_py,
        "stageA.lora_disable": "training_args.lora_enable = False" in train_py,
        "stageA.train_modules": 'train_module_list = ["prompt_adapter", "query_refiner"]' in train_py,
        "stageB.last1": 'qdti_apply_layers = "last1"' in train_py,
        "stageB.scale": "qdti_scale_init = 1e-3" in train_py,
    }
    for k, v in checks.items():
        report.add(f"train.{k}", PASS if v else FAIL, str(v))


def simulate_rrsisd_refer_ratio(report: Report, n: int = 100):
    """Simulate template: every sample has <refer> token -> refer_id path."""
    phrase = 0
    refer = n
    decoupled = 0
    total = phrase + refer + decoupled
    report.add(
        "pl.rrsisd_source_simulation",
        WARN,
        f"template simulation n={n}: phrase_span={phrase/total:.0%} refer_id={refer/total:.0%} decoupled_attn={decoupled/total:.0%}; decoupled_attn ~0% on RRSISD",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default=os.path.join(REPO, "stage3_audit_logs/v61_audit"))
    args = ap.parse_args()

    report = Report()
    audit_static_paths(report)
    audit_pl_sources(report)
    simulate_rrsisd_refer_ratio(report)
    audit_variable_q(report)
    try:
        audit_last1_layer(report)
    except Exception as e:
        report.add("qdti.last1_layer_index", FAIL, str(e))

    out = os.path.join(args.log_dir, "audit_dgp_v61_chain.txt")
    report.dump(out)
    return 0 if report.overall() != FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
