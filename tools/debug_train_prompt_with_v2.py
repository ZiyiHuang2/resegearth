#!/usr/bin/env python3
"""
Smoke-check RRSIS-D training prompts: baseline vs public grounding v2 (no train, no API).

token_refer_id policy matches RRSISDDataset: encode(instruction) + [SEG] for BOTH arms.

Usage (from resegearth+tgi repo root, with transformers+torch):
  python tools/debug_train_prompt_with_v2.py \\
    --model-name-or-path /path/to/Mipha-3B \\
    --library-v2 /path/to/resegearth+source/configs/concept_public_semantic_library_v2.json
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
from typing import List, Optional, Tuple

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


def _build_sources(human: str) -> list:
    return [[{"from": "human", "value": human}, {"from": "gpt", "value": "\n[SEG]"}]]


def _refer_token_ids(tokenizer, instruction: str) -> torch.Tensor:
    """Mirror RRSISDDataset.preprocess_referring_instruction."""
    tokenized = tokenizer.encode(instruction, add_special_tokens=False)
    seg_id = tokenizer.encode("[SEG]", add_special_tokens=False)[0]
    return torch.tensor(tokenized + [seg_id], dtype=torch.long)


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


def _seg_in_assistant(ids: torch.Tensor, labels: torch.Tensor, tokenizer) -> Tuple[bool, str]:
    """Trainable (label != -100) span should decode to assistant / [SEG] only."""
    non_ig = torch.where(labels != IGNORE_INDEX)[0]
    if non_ig.numel() == 0:
        return False, ""
    s, e = int(non_ig[0]), int(non_ig[-1]) + 1
    piece = tokenizer.decode(ids[s:e], skip_special_tokens=False)
    seg_id = tokenizer.convert_tokens_to_ids("[SEG]")
    has_seg = seg_id in ids[s:e].tolist()
    return has_seg, piece


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare baseline vs v2 RRSIS-D train human prompts and label masking (no training)."
    )
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

        ref_b = _refer_token_ids(tokenizer, instruction)
        ref_v2 = _refer_token_ids(tokenizer, instruction)
        refer_equal = bool(torch.equal(ref_b, ref_v2))
        print("\n--- token_refer_id (baseline vs v2, must be identical) ---")
        print("shape:", tuple(ref_b.shape), "| equal:", refer_equal)
        print("ids:", ref_b.tolist())

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
        print("\n--- matched_concepts (v2 retrieval on raw instruction) ---")
        print(matched)

        print("\n--- injection position ---")
        print("v2 JSON block (if any) sits after plain instruction text and before <|vision_bos|> <image>.")

        _, labels_b, tr_b, ig_b = _label_stats(ds, tokenizer, base_human)
        ids_v2, labels_v2, tr_v2, ig_v2 = _label_stats(ds, tokenizer, v2_human)
        print(f"\n--- labels != -100 (trainable token count) ---\nbaseline={tr_b}  v2={tr_v2}")
        print(f"--- labels == -100 count ---\nbaseline={ig_b}  v2={ig_v2}")

        needle = tokenizer.encode("visual_evidence", add_special_tokens=False)
        span = _find_subspan(ids_v2, needle)
        if span:
            s, e = span
            sub = labels_v2[s:e]
            ok = bool((sub == IGNORE_INDEX).all().item())
            print(f"\n--- v2 prior marker span ['visual_evidence'] ---\nindices [{s}, {e})  all labels -100: {ok}")
        else:
            print("\n--- v2 prior marker span ---\n'subword not found as contiguous token run' (skip)")

        seg_ok_b, piece_b = _seg_in_assistant(ids_v2, labels_v2, tokenizer)
        print(f"\n--- [SEG] in assistant trainable span (v2 prompt) ---\ncontains [SEG] id in trainable region: {seg_ok_b}\ndecode preview: {piece_b!r}")

        blob = "\n".join([v2_human, str(matched)])
        priv = (
            "forbidden_auto_infer_tokens",
            "visual_form_options",
            "slot_guidance",
            "generation_hint",
        )
        hits = [p for p in priv if p in blob]
        print("private field name substring hits (excl. English word boundary):", hits)

    print("\nDone (no training, no API).")


if __name__ == "__main__":
    main()
