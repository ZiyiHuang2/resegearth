"""
Training-side helpers for optional public grounding prior (v2 JSON) in RRSIS-D prompts.

Loads retrieval from tools/rrsisd_explicitization_pipeline.py in this repo (no API calls).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PIPELINE_MOD = None


def _load_pipeline():
    global _PIPELINE_MOD
    if _PIPELINE_MOD is not None:
        return _PIPELINE_MOD
    here = Path(__file__).resolve()
    # segearth_r2/utils -> repo root (resegearth+source)
    repo_root = here.parents[2]
    pipe_path = repo_root / "tools" / "rrsisd_explicitization_pipeline.py"
    if not pipe_path.is_file():
        raise FileNotFoundError(f"Missing pipeline module: {pipe_path}")
    spec = importlib.util.spec_from_file_location("rrsisd_explicitization_pipeline_train_bridge", pipe_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _PIPELINE_MOD = mod
    return _PIPELINE_MOD


def retrieve_matched_public_grounding(raw_expression: str, library_path: str) -> List[Dict[str, Any]]:
    mod = _load_pipeline()
    lib = mod.load_concept_public_semantic_library(library_path)
    ctx = mod.retrieve_concept_semantics(raw_expression, lib, private_library_for_audit=None)
    rows = ctx.get("matched_concepts") or []
    return [r for r in rows if isinstance(r, dict)]


def normalize_concept_name(s: Optional[str]) -> str:
    """
    Canonical string for comparing RRSISD category_name to library / matched concept labels.

    Rules: lower, strip, treat underscore and hyphen as spaces, collapse whitespace, then drop
    all non-alphanumeric characters so e.g. "parking lot", "parking-lot", "parking_lot" match.
    """
    if s is None:
        return ""
    t = str(s).strip().lower()
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t


def _concept_label_from_matched_row(row: Any) -> str:
    if isinstance(row, str):
        return row.strip()
    if isinstance(row, dict):
        return str(row.get("concept") or "").strip()
    return ""


def split_matched_concepts_target_and_reference(
    matched_concepts: List[Any],
    category_name: Optional[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split retrieval rows by normalized GT category_name (same equality as filter_matched_concepts_by_category).

    If category_name is empty / unknown, no row counts as target; all matched rows are references
    (exclusion-only in ref-aware prior mode).
    """
    want = normalize_concept_name(category_name)
    target_concepts: List[Dict[str, Any]] = []
    reference_concepts: List[Dict[str, Any]] = []
    for row in matched_concepts or []:
        if not isinstance(row, dict):
            continue
        if want and normalize_concept_name(_concept_label_from_matched_row(row)) == want:
            target_concepts.append(row)
        else:
            reference_concepts.append(row)
    return target_concepts, reference_concepts


def filter_matched_concepts_by_category(
    matched_concepts: List[Any],
    category_name: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Strict target-only filter: keep rows whose public `concept` normalizes equal to category_name.

    If category_name is empty / unknown, returns [] (caller should treat as no prior injection).
    """
    want = normalize_concept_name(category_name)
    if not want:
        return []
    out: List[Dict[str, Any]] = []
    for row in matched_concepts or []:
        if not isinstance(row, dict):
            continue
        if normalize_concept_name(_concept_label_from_matched_row(row)) == want:
            out.append(row)
    return out


def format_grounding_appendix(matched: List[Dict[str, Any]]) -> str:
    if not matched:
        return ""
    header = (
        "Category-level public grounding priors for remote sensing (not observed facts about this image; "
        "do not recite or restate as model output targets):\n"
    )
    return header + json.dumps(matched, ensure_ascii=False, indent=2)


def format_reference_exclusion_prior_section(reference_concepts: List[Dict[str, Any]]) -> str:
    """
    Minimal exclusion block for reference concepts only (no visual_evidence / mask_scope).
    """
    if not reference_concepts:
        return ""
    lines: List[str] = ["[Exclusion Prior]"]
    for row in reference_concepts:
        if not isinstance(row, dict):
            continue
        name = str(row.get("concept") or "").strip()
        if not name:
            continue
        rule = str(row.get("exclusion_rule") or "").strip()
        if rule:
            lines.append(f"- {name}: {rule}")
        else:
            lines.append(f"- {name}: Exclude {name}.")
    if len(lines) <= 1:
        return ""
    return "\n".join(lines)


def build_rrsisd_refaware_exclusion_only_human_value(
    instruction: str,
    concept_public_library_path: str,
    category_name: Optional[str],
    *,
    matched_precalc: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    public_semantic_v2 + ref-aware exclusion-only: target rows get full v2 JSON prior; other matched
    rows contribute only exclusion_rule (or ``Exclude {{concept}}.`` when the library field is empty).
    """
    if matched_precalc is not None:
        matched = list(matched_precalc)
    else:
        matched = retrieve_matched_public_grounding(instruction, concept_public_library_path)
    target_concepts, reference_concepts = split_matched_concepts_target_and_reference(
        matched, category_name
    )
    blocks: List[str] = []
    if target_concepts:
        tgt = format_grounding_appendix(target_concepts)
        if tgt:
            blocks.append(tgt.rstrip())
    excl = format_reference_exclusion_prior_section(reference_concepts)
    if excl:
        blocks.append(excl)
    appendix = "\n\n".join(blocks) if blocks else ""
    parts: List[str] = [
        "This is an image <|sep|> <|user|>\n",
        "Please do Reasoning Segmentation according to the following expression.\n",
        (instruction or "").strip() + "\n",
    ]
    if appendix:
        parts.append(appendix + "\n")
    parts.append("<|vision_bos|> <image> <|vision_eos|>\n<refer> <|assistant|>")
    return "".join(parts)


def build_rrsisd_supervised_human_value(
    instruction: str,
    concept_public_library_path: Optional[str] = None,
    *,
    matched_precalc: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Full human-side string up to (and including) the <|assistant|> handoff token, before gpt adds [SEG].

    - No library: identical layout to legacy RRSIS-D in this repo (image first, then <refer>).
    - With v2 library: raw expression, optional grounding JSON, then image tokens, then <refer>; same
      token_refer_id as baseline (encode(instruction)+[SEG]) is applied in Dataset separately.

    If ``matched_precalc`` is provided (non-None), it must be the already-retrieved (and optionally
    strict-filtered) public rows; this path skips a second retrieval call and must only be used when
    ``concept_public_library_path`` is set.
    """
    if not concept_public_library_path:
        return (
            "This is an image <|vision_bos|> <image> <|vision_eos|> <|sep|> <|user|>, please doing Reasoning Segmentation according to the following instruction:\n"
            "<refer> <|assistant|>"
        )
    if matched_precalc is not None:
        matched = list(matched_precalc)
    else:
        matched = retrieve_matched_public_grounding(instruction, concept_public_library_path)
    appendix = format_grounding_appendix(matched)
    parts: List[str] = [
        "This is an image <|sep|> <|user|>\n",
        "Please do Reasoning Segmentation according to the following expression.\n",
        (instruction or "").strip() + "\n",
    ]
    if appendix:
        parts.append(appendix + "\n")
    parts.append("<|vision_bos|> <image> <|vision_eos|>\n<refer> <|assistant|>")
    return "".join(parts)
