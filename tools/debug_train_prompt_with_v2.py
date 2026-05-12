#!/usr/bin/env python3
"""
Smoke-check RRSIS-D training prompts: baseline vs public grounding v2 (no train, no API).

Usage (from resegearth+tgi repo root):
  python tools/debug_train_prompt_with_v2.py \
    --model-name-or-path /path/to/Mipha-3B \
    --library-v2 /path/to/resegearth+source/configs/concept_public_semantic_library_v2.json
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
from typing import List, Optional, Tuple

from transformers import AutoTokenizer

# repo root = parent of tools/
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


def _build_sources(human: str) -> list:
    return [[{"from": "human", "value": human}, {"from": "gpt", "value": "\n[SEG]"}]]


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name-or-path", required=True, help="HF model dir for tokenizer")
    ap.add_argument(
        "--library-v2",
        required=True,
        help="Path to concept_public_semantic_library_v2.json",
    )
    args = ap.parse_args()

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

    for name, instruction in samples:
        print("\n" + "=" * 80)
        print(f"CASE: {name} | instruction: {instruction!r}")
        base_human = build_rrsisd_supervised_human_value(instruction, None)
        v2_human = build_rrsisd_supervised_human_value(instruction, args.library_v2)
        matched = retrieve_matched_public_grounding(instruction, args.library_v2)
        print("\n--- baseline human ---\n", base_human)
        print("\n--- v2 human ---\n", v2_human)
        print("\n--- unified diff (baseline -> v2) ---")
        for line in difflib.unified_diff(
            base_human.splitlines(),
            v2_human.splitlines(),
            lineterm="",
            fromfile="baseline",
            tofile="v2",
        ):
            print(line)
        print("\n--- matched_concepts (v2 retrieval on raw) ---")
        print(matched)

        inj = "after raw expression text block, before <|vision_bos|> <image>"
        print(f"\n--- injection position note ---\n{inj}")

        _, labels_b, tr_b, _ = _label_stats(ds, tokenizer, base_human)
        ids_v2, labels_v2, tr_v2, _ = _label_stats(ds, tokenizer, v2_human)
        print(f"\n--- label stats ---\ntrainable (!=-100): baseline={tr_b} v2={tr_v2}")

        # Appendix substring span (token ids) for v2 JSON keys
        needle = tokenizer.encode("visual_evidence", add_special_tokens=False)
        span = _find_subspan(ids_v2, needle)
        if span:
            s, e = span
            sub = labels_v2[s:e]
            ok = bool((sub == IGNORE_INDEX).all().item())
            print(f"token span for 'visual_evidence' bytes: [{s}, {e}) all labels -100: {ok}")
        else:
            print("token span for 'visual_evidence': not found (tokenizer-dependent)")

        # Full v2 human: all positions with label != -100 should be assistant tail only
        non_ig = torch.where(labels_v2 != IGNORE_INDEX)[0]
        if non_ig.numel() > 0:
            piece = tokenizer.decode(ids_v2[non_ig[0] : non_ig[-1] + 1], skip_special_tokens=False)
            print("first trainable span decodes (should be assistant / [SEG] only):", repr(piece[:200]))

        blob = "\n".join([v2_human, str(matched)])
        priv = (
            "forbidden_auto_infer_tokens",
            "visual_form_options",
            "slot_guidance",
            "generation_hint",
            "boundary",
        )
        hits = [p for p in priv if p in blob]
        print("private-ish substring hits in prompt+json blob:", hits)

    print("\nDone (no training, no API).")


if __name__ == "__main__":
    main()
