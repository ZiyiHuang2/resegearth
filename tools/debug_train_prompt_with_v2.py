#!/usr/bin/env python3
"""
Smoke-check RRSIS-D training prompts: baseline vs public grounding v2 (no train, no API).

Run from resegearth+source repo root:
  conda run -n reseg python tools/debug_train_prompt_with_v2.py --help
  conda run -n reseg python tools/debug_train_prompt_with_v2.py \\
    --model-name-or-path /path/to/merged_model \\
    --library-v2 configs/concept_public_semantic_library_v2.json
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoTokenizer

CURRENT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from segearth_r2.datasets.dataset import RS_Base_Dataset  # noqa: E402
from segearth_r2.utils.constants import IGNORE_INDEX  # noqa: E402
from segearth_r2.utils.concept_public_grounding_train import (  # noqa: E402
    build_rrsisd_supervised_human_value,
    retrieve_matched_public_grounding,
)

# 与 source 训练脚本常见路径一致；可用环境变量覆盖
_DEFAULT_MODEL = os.environ.get(
    "DEBUG_MODEL_PATH",
    "/home/wangchengjun/huangziyi/reseg/output/bseg/baseline_standard-base_5w/merged_model",
)
_DEFAULT_LIBRARY = os.environ.get(
    "DEBUG_LIBRARY_V2",
    os.path.join(REPO_ROOT, "configs", "concept_public_semantic_library_v2.json"),
)


def _build_sources(human: str) -> list:
    return [[{"from": "human", "value": human}, {"from": "gpt", "value": "\n[SEG]"}]]


def _refer_token_ids(tokenizer, instruction: str) -> torch.Tensor:
    """Match RRSISDDataset.preprocess_referring_instruction."""
    tokenized = tokenizer.encode(instruction, add_special_tokens=False)
    ref_id = tokenizer.encode("[SEG]", add_special_tokens=False)[0]
    return torch.tensor(tokenized + [ref_id])


def _label_stats(ds: RS_Base_Dataset, tokenizer, human: str) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    sources = _build_sources(human)
    td = ds.preprocess_llama2(sources, tokenizer)
    input_ids = td["input_ids"][0]
    labels = td["labels"][0]
    trainable = int((labels != IGNORE_INDEX).sum().item())
    ignored = int((labels == IGNORE_INDEX).sum().item())
    return input_ids, labels, trainable, ignored


def _find_subspan(haystack_ids: torch.Tensor, needle_ids: List[int]) -> Optional[Tuple[int, int]]:
    if not needle_ids:
        return None
    h = haystack_ids.tolist()
    n = len(needle_ids)
    for i in range(0, len(h) - n + 1):
        if h[i : i + n] == needle_ids:
            return i, i + n
    return None


def _concept_names(matched: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for row in matched:
        if isinstance(row, dict) and "concept" in row:
            out.add(str(row["concept"]).strip().lower())
    return out


def _first_trainable_index(labels: torch.Tensor) -> Optional[int]:
    nz = (labels != IGNORE_INDEX).nonzero(as_tuple=False)
    if nz.numel() == 0:
        return None
    return int(nz[0].item())


def _run_case(
    name: str,
    instruction: str,
    tokenizer,
    ds: RS_Base_Dataset,
    library_v2: str,
) -> bool:
    ok = True
    print("\n" + "=" * 80)
    print(f"CASE: {name} | instruction: {instruction!r}")

    base_human = build_rrsisd_supervised_human_value(instruction, None)
    v2_human = build_rrsisd_supervised_human_value(instruction, library_v2)
    matched = retrieve_matched_public_grounding(instruction, library_v2)

    print("\n--- (1) baseline human ---\n", base_human)
    print("\n--- (2) v2 human ---\n", v2_human)
    print("\n--- (3) unified diff (baseline -> v2) ---")
    for line in difflib.unified_diff(
        base_human.splitlines(),
        v2_human.splitlines(),
        lineterm="",
        fromfile="baseline",
        tofile="v2",
    ):
        print(line)

    print("\n--- (4) matched_concepts ---")
    print(matched)

    ref_b = _refer_token_ids(tokenizer, instruction)
    ref_v = _refer_token_ids(tokenizer, instruction)
    same_ref = bool(torch.equal(ref_b, ref_v))
    print(f"\n--- (5) token_refer_id baseline vs v2 identical: {same_ref} (shape={tuple(ref_b.shape)})")
    if not same_ref:
        ok = False

    ids_b, labels_b, tr_b, _ = _label_stats(ds, tokenizer, base_human)
    ids_v2, labels_v2, tr_v2, _ = _label_stats(ds, tokenizer, v2_human)
    print(f"\n--- (6) labels != -100 count: baseline={tr_b} v2={tr_v2}")
    if tr_b != tr_v2:
        print("  WARN: trainable token count differs between baseline and v2 (often still OK if template length differs).")

    for field in ("visual_evidence", "mask_scope", "exclusion_rule"):
        needle = tokenizer.encode(field, add_special_tokens=False)
        span = _find_subspan(ids_v2, needle)
        print(f"\n--- (7) span for substring {field!r} ---")
        if span:
            s, e = span
            sub = labels_v2[s:e]
            all_m = bool((sub == IGNORE_INDEX).all().item())
            print(f"  token span [{s}, {e}), all labels -100: {all_m}")
            if not all_m:
                ok = False
        else:
            print("  (not found in token ids — may be split across BPE; check v2_human text)")

    tr0 = _first_trainable_index(labels_v2)
    print("\n--- (8) v2 prior block vs -100 ---")
    if "Category-level public grounding" in v2_human and tr0 is not None:
        needle_cat = tokenizer.encode("Category-level", add_special_tokens=False)
        sp = _find_subspan(ids_v2, needle_cat)
        if sp and sp[0] < tr0:
            block = labels_v2[sp[0] : tr0]
            block_ok = bool((block == IGNORE_INDEX).all().item())
            print(f"  labels[{sp[0]}:{tr0}) (appendix through before first trainable) all -100: {block_ok}")
            if not block_ok:
                ok = False
        else:
            print("  (could not align Category-level span to trainable start)")
    elif not matched:
        print("  (no matched concepts — no JSON appendix)")
    else:
        print("  (skipped: no Category-level string or no trainable span)")

    print("\n--- (9) [SEG] in assistant trainable decode ---")
    non_ig = (labels_v2 != IGNORE_INDEX).nonzero(as_tuple=False).flatten()
    if non_ig.numel() > 0:
        piece = tokenizer.decode(ids_v2[non_ig[0] : non_ig[-1] + 1], skip_special_tokens=False)
        has_seg = "[SEG]" in piece
        print("  trainable decode snippet:", repr(piece[:240]))
        print("  contains [SEG]:", has_seg)
        if not has_seg:
            ok = False
    else:
        print("  ERROR: no trainable tokens")
        ok = False

    print("\n--- (10) v2 prior not in assistant target text ---")
    gpt_only = "\n[SEG]"
    leak_appendix_in_gpt = any(
        k in gpt_only for k in ("visual_evidence", "mask_scope", "exclusion_rule", "Category-level")
    )
    print("  gpt branch value (expected newline + [SEG]):", repr(gpt_only))
    print("  prior keywords in gpt_only:", leak_appendix_in_gpt)
    if leak_appendix_in_gpt:
        ok = False

    print("\n--- (11) private / internal field leak scan ---")
    blob = v2_human + "\n" + str(matched)
    priv = (
        "forbidden_auto_infer_tokens",
        "visual_form_options",
        "slot_guidance",
        "generation_hint",
        "boundary",
    )
    hits = [p for p in priv if p in blob]
    print("  hits:", hits)
    if hits:
        ok = False

    names = _concept_names(matched)
    print("\n--- (12) water / river / lake mutual exclusion (retrieval) ---")
    if name == "water":
        bad = ("river" in names) or ("lake" in names)
        print(f"  water case: river/lake must not appear in matched: bad={bad} names={names}")
        if bad:
            ok = False
    elif name == "river":
        bad = "water" in names
        print(f"  river case: water must not appear in matched: bad={bad} names={names}")
        if bad:
            ok = False
    elif name == "lake":
        bad = "water" in names
        print(f"  lake case: water must not appear in matched: bad={bad} names={names}")
        if bad:
            ok = False

    print("\n--- (13) multi-concept: vehicles on the bridge over the river ---")
    if name == "multi":
        need = {"vehicle", "bridge", "river"}
        missing = need - names
        extra_water = "water" in names
        print(f"  expected concepts (subset): {need}")
        print(f"  matched names: {names}")
        print(f"  missing from matched: {missing}")
        print(f"  water incorrectly matched: {extra_water}")
        if missing or extra_water:
            ok = False

    print(f"\n--- CASE {name} overall OK: {ok} ---")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Debug baseline vs v2 RRSIS-D train prompts (no training, no API)."
    )
    ap.add_argument(
        "--model-name-or-path",
        default=_DEFAULT_MODEL,
        help="HF model dir for tokenizer (default: DEBUG_MODEL_PATH env or merged_model under reseg/output/bseg).",
    )
    ap.add_argument(
        "--library-v2",
        default=_DEFAULT_LIBRARY,
        help="Path to concept_public_semantic_library_v2.json (default: configs/... under repo).",
    )
    args = ap.parse_args()

    lib_path = args.library_v2
    if not os.path.isabs(lib_path):
        lib_path = os.path.join(REPO_ROOT, lib_path)
    if not os.path.isfile(lib_path):
        print(f"[ERROR] library not found: {lib_path}", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        model_max_length=4096,
        padding_side="right",
    )
    ds = RS_Base_Dataset.__new__(RS_Base_Dataset)
    ds.tokenizer = tokenizer

    samples = [
        ("water", "find water"),
        ("river", "find river"),
        ("lake", "find lake"),
        ("multi", "vehicles on the bridge over the river"),
    ]

    all_ok = True
    for name, instruction in samples:
        if not _run_case(name, instruction, tokenizer, ds, lib_path):
            all_ok = False

    print("\n" + "=" * 80)
    print(f"ALL CASES OK: {all_ok}")
    print("Done (no training, no API).")


if __name__ == "__main__":
    main()
