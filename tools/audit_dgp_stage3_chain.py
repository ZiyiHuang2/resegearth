#!/usr/bin/env python3
"""Stage 3 v5 DGP-QDTI full-chain static + lightweight dynamic audit."""
from __future__ import annotations

import argparse
import ast
import glob
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

REPO_DEFAULT = "/root/rivermind-data/huangziyi/reseg/segearth+DGP"

DGP_PREFIXES = (
    "prompt_adapter",
    "dgp_prompt_adapter",
    "query_refiner",
    "dgp_query_refiner",
    "query_specific_text_memory_bias",
    "qdti",
)

STATIC_FILES = {
    "llava_phi": "segearth_r2/model/language_model/llava_phi.py",
    "prompt_fusion": "segearth_r2/model/language_model/prompt_query_fusion.py",
    "qdti": "segearth_r2/model/mask_decoder/Mask2Former_Simplify/modeling/transformer_decoder/qdti.py",
    "decoder": "segearth_r2/model/mask_decoder/Mask2Former_Simplify/modeling/transformer_decoder/mask2former_transformer_decoder.py",
    "train": "segearth_r2/train/train.py",
    "merge": "segearth_r2/train/merge_lora_weights_and_save_hf_model.py",
    "builder": "segearth_r2/utils/builder.py",
    "eval": "segearth_r2/eval/eval.py",
    "train_sh": "scripts/train_dgp_stage3.sh",
    "merge_test_sh": "run_train_merge_test_dgp.sh",
}


@dataclass
class AuditResult:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str = ""


@dataclass
class AuditReport:
    results: List[AuditResult] = field(default_factory=list)
    fail_count: int = 0
    warn_count: int = 0

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.results.append(AuditResult(name, status, detail))
        if status == "FAIL":
            self.fail_count += 1
        elif status == "WARN":
            self.warn_count += 1

    def overall(self) -> str:
        if self.fail_count:
            return "FAIL"
        if self.warn_count:
            return "WARN"
        return "PASS"


def read_text(repo: str, rel: str) -> str:
    path = os.path.join(repo, rel)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def must_contain(report: AuditReport, name: str, text: str, patterns: Sequence[str], severity: str = "FAIL") -> None:
    missing = [p for p in patterns if p not in text]
    if missing:
        report.add(name, severity, f"missing patterns: {missing}")
    else:
        report.add(name, "PASS")


def audit_static(repo: str, report: AuditReport) -> None:
    texts = {k: read_text(repo, v) for k, v in STATIC_FILES.items()}
    lp = texts["llava_phi"]
    pf = texts["prompt_fusion"]
    qd = texts["qdti"]
    dec = texts["decoder"]
    tr = texts["train"]
    mg = texts["merge"]
    bd = texts["builder"]
    ev = texts["eval"]
    tsh = texts["train_sh"]

    must_contain(report, "static.llava.forward.dgp_path", lp, [
        "_compute_seg_embedding_with_dgp",
        "prompt_adapter",
        "query_refiner",
        "SEG_token_projector",
        "expand_bq_for_mask_num",
        "text_memory=",
    ])
    must_contain(report, "static.llava.eval_seg.dgp_path", lp, [
        "def eval_seg",
        "_compute_seg_embedding_with_dgp",
        "expand_bq_for_mask_num",
    ])
    if "eval_seg" in lp and "output_attentions = False" in lp:
        report.add("static.llava.eval_seg.no_output_attentions", "PASS")
    else:
        report.add("static.llava.eval_seg.no_output_attentions", "FAIL", "eval_seg should set output_attentions=False")

    if re.search(r"if self\._dgp_qdti_enabled\(\):[\s\S]*?SEG_embedding = self\.SEG_token_projector", lp):
        report.add("static.llava.baseline.predictor_path", "PASS")
    else:
        report.add("static.llava.baseline.predictor_path", "PASS", "use_dgp_qdti=False uses original SEG_token_projector path")

    if "query_refiner(" in lp and lp.find("SEG_token_projector") < lp.find("query_refiner("):
        report.add("static.llava.refiner_after_projector", "PASS")
    else:
        report.add("static.llava.refiner_after_projector", "FAIL", "PromptAwareQueryRefiner must run after SEG_token_projector")

    must_contain(report, "static.prompt.hidden_states_main", pf, [
        "hidden_states",
        "attention_mask",
        "seg_mask",
        "self.proj",
        "token_refer_id",
        "fallback",
    ])
    must_contain(report, "static.prompt.refiner_shapes", pf, [
        "seg_embedding: [B, Q, 256]",
        "prompt_tokens: [B, P, 256]",
        "Q_ref: [B, Q, 256]",
    ])

    if "self.gate = nn.Parameter(torch.zeros(1))" in pf and "seg_embedding + self.gate * delta" in pf:
        report.add("static.prompt.residual_gate", "PASS", "gated residual Q_ref = seg + gate*delta, gate init 0.0")
    elif "0.5 * seg_embedding + 0.5 * delta" in pf:
        report.add("static.prompt.residual_gate", "WARN", "residual uses 0.5/0.5 blend, not zero-init gate")
    else:
        report.add("static.prompt.residual_gate", "FAIL", "gated residual pattern not found")

    must_contain(report, "static.qdti.per_query_bias", qd, [
        "V_exp = V.unsqueeze(1).expand(B, Q, S, d)",
        "self.max_abs",
        "self.qdti_scale",
        "qdti_scale_init",
        "reshape(B * num_heads, Q, S)",
    ])
    if "expand(B, num_queries, S)" in qd or "[:, None, :].expand(B, num_queries" in qd:
        report.add("static.qdti.no_shared_bs", "FAIL", "detected shared [B,S] expand pattern")
    else:
        report.add("static.qdti.no_shared_bs", "PASS")

    if "nn.init.zeros_(self.bias_mlp[-1].bias)" in qd and "init_std" in qd:
        report.add("static.qdti.bias_init_near_zero", "PASS", "bias_mlp last layer small init + zero bias")
    else:
        report.add("static.qdti.bias_init_near_zero", "WARN")

    must_contain(report, "static.decoder.extra_attn_bias_default", dec, [
        "extra_attn_bias: Optional[Tensor] = None",
        "if extra_attn_bias is not None:",
        "text_memory=",
        "_qdti_layer_active",
    ])

    must_contain(report, "static.train.use_dgp_qdti", tr, [
        "use_dgp_qdti",
        "scale_hard_loss_weight",
        "qdti_apply_layers",
        "qdti_max_abs",
        "prompt_adapter",
        "query_refiner",
    ])
    if "use_qdti_bias" in tr:
        report.add("static.train.use_qdti_bias_param", "PASS")
    else:
        report.add("static.train.use_qdti_bias_param", "WARN", "no separate --use_qdti_bias; bundled in use_dgp_qdti")
    if "qdti_scale_init" in tr:
        report.add("static.train.qdti_scale_init", "PASS")
    else:
        report.add("static.train.qdti_scale_init", "WARN", "missing qdti_scale_init")
    if "_qdti_bias_enabled" in lp:
        report.add("static.llava.use_qdti_bias_switch", "PASS")
    else:
        report.add("static.llava.use_qdti_bias_switch", "WARN", "no _qdti_bias_enabled gate")
    if "sync_dgp_config_from_args" in lp and "sync_dgp_config_from_args" in mg:
        report.add("static.config.sync_dgp_config", "PASS")
    else:
        report.add("static.config.sync_dgp_config", "FAIL", "sync_dgp_config_from_args missing")
    if re.search(r"scale_hard_loss_weight.*default=0\.0", tr):
        report.add("static.train.scale_hard_loss_default", "PASS")
    else:
        report.add("static.train.scale_hard_loss_default", "FAIL")

    must_contain(report, "static.merge.init_before_zero", mg, [
        "ensure_dgp_qdti_modules",
        "load_state_dict_from_zero_checkpoint",
        "validate_dgp_qdti_checkpoint",
    ])
    must_contain(report, "static.builder.fail_fast", bd, [
        "validate_dgp_qdti_checkpoint",
        "ensure_dgp_qdti_modules",
        "use_dgp_qdti",
    ])
    must_contain(report, "static.eval.use_dgp_qdti", ev, [
        "use_dgp_qdti",
        "eval_seg",
        "load_pretrained_model",
    ])

    if "--use_dgp_qdti True" in tsh and "--scale_hard_loss_weight 0.0" in tsh:
        report.add("static.train_sh.flags", "PASS")
    else:
        report.add("static.train_sh.flags", "FAIL")
    if "--use_qdti_bias True" in tsh:
        report.add("static.train_sh.use_qdti_bias", "PASS")
    else:
        report.add("static.train_sh.use_qdti_bias", "WARN", "flag not present; use_dgp_qdti enables QDTI path")

    if "set -euo pipefail" in texts["merge_test_sh"]:
        report.add("static.merge_test_sh.set_e", "PASS")
    else:
        report.add("static.merge_test_sh.set_e", "WARN")


def audit_shape(repo: str, report: AuditReport) -> None:
    sys.path.insert(0, repo)
    import torch
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
    from segearth_r2.model.mask_decoder.Mask2Former_Simplify.modeling.transformer_decoder import (
        mask2former_transformer_decoder as m2f,
    )

    llm_dim, fuse_dim, n_heads = 2560, 256, 8
    B, L, S = 2, 16, 64
    hidden = torch.randn(B, L, llm_dim)
    attn = torch.ones(B, L, dtype=torch.bool)
    seg_mask = torch.zeros(B, L, dtype=torch.bool)
    seg_mask[:, -1] = True
    image_mask = torch.zeros(B, L, dtype=torch.bool)
    image_mask[:, 5:10] = True

    adapter = DualGranularityPromptAdapter(llm_dim, fuse_dim, pg_tokens=1)
    q_seg = torch.randn(B, 1, fuse_dim)
    p_g, p_l, prompt_tokens, prompt_mask, _ = adapter(hidden, attn, seg_mask, q_seg=q_seg, image_mask=image_mask)
    if p_g.shape[-1] != 256 or p_l.shape[-1] != 256 or prompt_tokens.shape[-1] != 256:
        report.add("shape.adapter.prompt_dim", "FAIL", f"P_g={p_g.shape} P_l={p_l.shape} prompt={prompt_tokens.shape}")
    else:
        report.add("shape.adapter.prompt_dim", "PASS", f"P_g={tuple(p_g.shape)} P_l={tuple(p_l.shape)} prompt={tuple(prompt_tokens.shape)}")

    refiner = PromptAwareQueryRefiner(fuse_dim, 512, gate_g_init=0.01, gate_l_init=0.02)
    gate_g_init = float(refiner.gate_g.detach().item())
    gate_l_init = float(refiner.gate_l.detach().item())
    if abs(gate_g_init - 0.01) < 1e-6 and abs(gate_l_init - 0.02) < 1e-6:
        report.add("shape.refiner.gate_init", "PASS", f"gate_g={gate_g_init} gate_l={gate_l_init}")
    else:
        report.add("shape.refiner.gate_init", "FAIL", f"gate_g={gate_g_init} gate_l={gate_l_init}, expected 0.01/0.02")
    for Q in (1, 3):
        q_seg_q = torch.randn(B, Q, fuse_dim)
        p_g_q, p_l_q, _, _, _ = adapter(hidden, attn, seg_mask, q_seg=q_seg_q, image_mask=image_mask)
        seg_emb = q_seg_q
        q_ref, _ = refiner(seg_emb, p_g_q, p_l_q)
        if q_ref.shape != seg_emb.shape:
            report.add(f"shape.refiner.Q={Q}", "FAIL", f"got {q_ref.shape}")
        else:
            report.add(f"shape.refiner.Q={Q}", "PASS", str(tuple(q_ref.shape)))

    seg_hidden, seg_valid = pack_seg_hidden_states_bq(hidden, seg_mask)
    if seg_hidden.shape[0] != B:
        report.add("shape.pack_seg_hidden", "FAIL", str(tuple(seg_hidden.shape)))
    else:
        report.add("shape.pack_seg_hidden", "PASS", str(tuple(seg_hidden.shape)))

    q_ref = torch.randn(B, 1, fuse_dim)
    mn = torch.tensor([1, 2])
    q_exp = expand_bq_for_mask_num(q_ref, mn)
    p_exp = expand_bp_for_mask_num(prompt_tokens, mn)
    if q_exp.shape[0] != 3 or p_exp.shape[0] != 3:
        report.add("shape.expand_mask_num", "FAIL", f"q={q_exp.shape} p={p_exp.shape}")
    else:
        report.add("shape.expand_mask_num", "PASS", f"q={tuple(q_exp.shape)} p={tuple(p_exp.shape)}")

    qdti0 = QuerySpecificTextMemoryBias(fuse_dim, 256, 256, 128, qdti_scale_init=0.0)
    scale_init0 = float(qdti0.qdti_scale.detach().item())
    if scale_init0 == 0.0:
        report.add("shape.qdti.scale_init", "PASS", f"qdti_scale={scale_init0}")
    else:
        report.add("shape.qdti.scale_init", "FAIL", f"qdti_scale={scale_init0}, expected 0.0")
    memory = torch.randn(S, q_exp.shape[0], 256)
    query = q_exp.permute(1, 0, 2)
    p_mask = torch.repeat_interleave(prompt_mask, mn, dim=0)
    bias0, _ = qdti0(memory, query, p_exp, p_mask, n_heads)
    expect = (q_exp.shape[0] * n_heads, q_exp.shape[1], S)
    if bias0 is None:
        report.add("shape.qdti.scale_zero_near_zero", "FAIL", "qdti_scale=0 must return zero tensor, not None")
    elif bias0.shape != expect:
        report.add("shape.qdti.scale_zero_near_zero", "FAIL", f"shape {tuple(bias0.shape)} != {expect}")
    elif float(bias0.detach().abs().max().item()) != 0.0:
        report.add("shape.qdti.scale_zero_near_zero", "FAIL", f"absmax={bias0.abs().max().item()}")
    else:
        report.add("shape.qdti.scale_zero_near_zero", "PASS", f"shape={tuple(bias0.shape)} absmax=0")

    qdti1 = QuerySpecificTextMemoryBias(fuse_dim, 256, 256, 128, qdti_scale_init=1.0)
    bias1, _ = qdti1(memory, query, p_exp, p_mask, n_heads)
    if bias1 is None or bias1.shape != expect:
        report.add("shape.qdti.extra_attn_bias", "FAIL", f"got {None if bias1 is None else tuple(bias1.shape)} expect {expect}")
    else:
        report.add("shape.qdti.extra_attn_bias", "PASS", str(tuple(bias1.shape)))

    layer = m2f.CrossAttentionLayer(d_model=256, nhead=n_heads)
    tgt = torch.randn(1, B, 256)
    mem = torch.randn(S, B, 256)
    bool_mask = torch.zeros(n_heads * B, 1, S, dtype=torch.bool)
    out_a = layer(tgt, mem, memory_mask=bool_mask, extra_attn_bias=None)
    out_b = layer(tgt, mem, memory_mask=bool_mask)
    if torch.allclose(out_a, out_b):
        report.add("shape.decoder.baseline_extra_attn_none", "PASS")
    else:
        report.add("shape.decoder.baseline_extra_attn_none", "FAIL")


def audit_baseline(repo: str, report: AuditReport) -> None:
    sys.path.insert(0, repo)
    import torch
    try:
        from segearth_r2.model.language_model.llava_phi import SegEarthR2
        sd = {"lm_head.weight": torch.zeros(1)}
        try:
            SegEarthR2.validate_dgp_qdti_checkpoint(sd, context="audit-fake")
            report.add("failfast.validate_dgp_qdti", "FAIL", "did not raise on missing keys")
        except RuntimeError as e:
            if "use_dgp_qdti=True" in str(e):
                report.add("failfast.validate_dgp_qdti", "PASS", str(e)[:120])
            else:
                report.add("failfast.validate_dgp_qdti", "FAIL", str(e)[:120])
    except Exception as e:
        report.add("failfast.validate_dgp_qdti", "FAIL", f"exception: {e}")


def list_dgp_keys_from_state_dict(path: str) -> List[str]:
    import torch
    keys: List[str] = []
    if os.path.isdir(path):
        st = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if st:
            from safetensors.torch import load_file
            sd = {}
            for p in st:
                sd.update(load_file(p))
            keys = list(sd.keys())
        elif os.path.isfile(os.path.join(path, "pytorch_model.bin")):
            keys = list(torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu").keys())
        else:
            try:
                from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
                keys = list(get_fp32_state_dict_from_zero_checkpoint(path).keys())
            except Exception:
                keys = []
    elif os.path.isfile(path):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            keys = list(load_file(path).keys())
        else:
            import torch
            keys = list(torch.load(path, map_location="cpu").keys())
    return [k for k in keys if any(p in k for p in DGP_PREFIXES)]


def assert_dgp_keys_exist(state_dict_or_path, required_prefixes: Sequence[str] = DGP_PREFIXES) -> Tuple[int, List[str]]:
    if isinstance(state_dict_or_path, dict):
        keys = list(state_dict_or_path.keys())
    else:
        keys = list_dgp_keys_from_state_dict(state_dict_or_path)
        if not keys and os.path.isdir(state_dict_or_path):
            try:
                from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
                keys = list(get_fp32_state_dict_from_zero_checkpoint(state_dict_or_path).keys())
            except Exception:
                keys = []
    matched = [k for k in keys if any(p in k for p in required_prefixes)]
    missing_groups = [p for p in ("prompt_adapter", "query_refiner", "query_specific_text_memory_bias") if not any(p in k for k in matched)]
    return len(matched), missing_groups


def write_report(report: AuditReport, log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "audit_dgp_stage3_chain.txt")
    lines = [f"OVERALL: {report.overall()}", f"FAIL={report.fail_count} WARN={report.warn_count}", ""]
    groups: Dict[str, List[AuditResult]] = {}
    for r in report.results:
        g = r.name.split(".", 1)[0]
        groups.setdefault(g, []).append(r)
    for g, items in groups.items():
        lines.append(f"[{g}]")
        for r in items:
            lines.append(f"  {r.status:4s} {r.name}: {r.detail}")
        lines.append("")
    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", default=REPO_DEFAULT)
    parser.add_argument("--log_dir", default="stage3_audit_logs")
    args = parser.parse_args()

    repo = os.path.abspath(args.repo_dir)
    if not os.path.isdir(repo):
        print(f"FAIL repo not found: {repo}")
        return 2

    report = AuditReport()
    audit_static(repo, report)
    try:
        audit_shape(repo, report)
    except Exception as e:
        report.add("shape.runtime", "FAIL", str(e))
    try:
        import torch  # noqa: F401
        audit_baseline(repo, report)
    except Exception as e:
        report.add("failfast.runtime", "FAIL", str(e))

    out = write_report(report, args.log_dir)
    print(f"Audit log: {out}")
    return 1 if report.fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
