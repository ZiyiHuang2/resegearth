"""
Training-side helpers for optional public grounding prior (v2 JSON) in RRSIS-D supervised prompts.

Loads retrieval from resegearth+source/tools/rrsisd_explicitization_pipeline.py (no API calls).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PIPELINE_MOD = None


def _load_pipeline():
    global _PIPELINE_MOD
    if _PIPELINE_MOD is not None:
        return _PIPELINE_MOD
    here = Path(__file__).resolve()
    # segearth_r2/utils -> segearth_r2 -> resegearth+tgi -> reseg
    reseg_root = here.parents[3]
    pipe_path = reseg_root / "resegearth+source" / "tools" / "rrsisd_explicitization_pipeline.py"
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


def format_grounding_appendix(matched: List[Dict[str, Any]]) -> str:
    if not matched:
        return ""
    header = (
        "Category-level public grounding priors for remote sensing (not observed facts about this image; "
        "do not recite or restate as model output targets):\n"
    )
    return header + json.dumps(matched, ensure_ascii=False, indent=2)


def build_rrsisd_supervised_human_value(instruction: str, concept_public_library_path: Optional[str]) -> str:
    """
    Human-side prompt for RRSIS-D supervised sample.

    Baseline reproduces the legacy template (instruction carried only via <refer> embeddings).

    With v2 library path: plain-text raw expression, optional grounding JSON, then vision/image tokens,
    then <refer> which should carry [SEG] only (see dataset) to avoid duplicating the expression.
    """
    if not concept_public_library_path:
        return (
            "This is an image <|vision_bos|> <image> <|vision_eos|> <|sep|> <|user|>, please doing Reasoning Segmentation according to the following instruction:\n"
            "<refer> <|assistant|>"
        )
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
