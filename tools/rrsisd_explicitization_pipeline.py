#!/usr/bin/env python3
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import pickle
import random
import re
import shutil
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import requests


ALLOWED_OP_TYPES = {
    "article_completion",
    "preposition_completion",
    "relation_verb_insertion",
    "phrase_normalization",
}
ALLOWED_SPATIAL_RELATIONS = {
    "near",
    "adjacent_to",
    "beside",
    "left_of",
    "right_of",
    "above",
    "below",
    "inside",
    "around",
    "surrounded_by",
    "between",
    "overlapping",
    "on",
    "along",
    "unknown",
    "smaller_than",
    "larger_than",
}
ALLOWED_ABS_POSITIONS = {
    "left",
    "right",
    "top",
    "bottom",
    "center",
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
    "unknown",
}
ALLOWED_QUANTITY = {"single", "multiple", "unknown"}
ALLOWED_LLM_STATUS = {
    "success",
    "unchanged",
    "failed",
}
ALLOWED_PIPELINE_STATUS = {
    "failed_postcheck",
    "failed_api_error",
    "skipped_empty_expression",
    "success",
    "unchanged",
}
ALLOWED_JUSTIFICATION_REASON = {"grammar_only", "copied_from_raw"}

HIGH_RISK_WORDS = {
    "white",
    "black",
    "red",
    "blue",
    "green",
    "yellow",
    "gray",
    "grey",
    "brown",
    "dark",
    "bright",
    "small",
    "large",
    "tiny",
    "huge",
    "long",
    "short",
    "wide",
    "narrow",
    "round",
    "circular",
    "rectangular",
    "square",
    "elongated",
    "linear",
    "two",
    "three",
    "four",
    "several",
    "many",
    "multiple",
    "group",
    "cluster",
    "left",
    "right",
    "top",
    "bottom",
    "upper",
    "lower",
    "center",
    "middle",
    "parked",
    "moving",
    "docked",
    "connected",
    "isolated",
    "damaged",
    "road",
    "runway",
    "building",
    "river",
    "bridge",
    "harbor",
    "port",
    "sea",
    "water",
    "field",
    "airport",
    "parking",
    "vehicle",
    "ship",
    "airplane",
}
# Used only in concept_library_audit: visual_form_options.option token coverage vs forbidden_auto_infer_tokens,
# and safe_rewrite token conflict checks. Not used in raw/enhanced leakage validators.
BASIC_STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "with",
    "without",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "by",
    "near",
    "over",
    "under",
    "between",
    "into",
    "through",
    "across",
    "as",
    "is",
    "are",
    "be",
    "body",
    "bodies",
}
FORBIDDEN_EXPLANATION_PHRASES = {
    "this expression refers to",
    "the target is",
    "it describes",
    "based on the image",
    "appears to be",
    "likely",
    "probably",
    "maybe",
}
SLOT_LABEL_TOKENS = {
    "left_of",
    "right_of",
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
    "smaller_than",
    "larger_than",
}
VISUAL_PRESERVE_WORDS = {
    "gray",
    "grey",
    "green",
    "blue",
    "orange",
    "white",
    "black",
    "brown",
    "yellow",
    "red",
    "large",
    "small",
    "tiny",
    "huge",
    "slender",
    "long",
    "short",
    "little",
    "oval",
    "round",
    "circular",
    "rectangular",
    "square",
    "frustum",
    "cone",
}
VISUAL_PRESERVE_PHRASES = {
    "frustum of a cone",
    "ground track field",
    "expressway toll station",
    "expressway service area",
}
DEFAULT_EMPTY_SLOTS = {
    "target_category": None,
    "attributes": [],
    "spatial_relations": ["unknown"],
    "reference_objects": [],
    "absolute_positions": ["unknown"],
    "quantity": "unknown",
    "clean_label_valid": True,
}
DEFAULT_EMPTY_EXPLICITIZATION = {
    "operations": [],
    "added_tokens": [],
    "added_token_justification": [],
}


def utcnow() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def tokenize_lower_words(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def canonical_token_set(text: str) -> Set[str]:
    return set(tokenize_lower_words(text))


def normalize_category_text(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    t = s.lower().strip()
    t = t.replace(" ", "")
    t = t.replace("_", "").replace("-", "")
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t


def normalize_phrase_text(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    t = s.lower().strip()
    t = t.replace(" ", "")
    t = t.replace("_", "").replace("-", "")
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t


def _category_variants(s: Any) -> Set[str]:
    base = normalize_category_text(s)
    if not base:
        return set()
    out = {base}
    if base.endswith("s") and not base.endswith("ss") and len(base) > 4:
        out.add(base[:-1])
    return out


def category_text_match(a: Any, b: Any) -> bool:
    va = _category_variants(a)
    vb = _category_variants(b)
    return len(va.intersection(vb)) > 0


def raw_contains_category_text(raw_expression: str, category_name: Any) -> bool:
    raw_norm = normalize_category_text(raw_expression)
    if not raw_norm:
        return False
    for v in _category_variants(category_name):
        if v and v in raw_norm:
            return True
    return False


def _normalize_label_text(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip().lower()
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"\s+", " ", t)
    return t


def _drop_leading_function_words(text: str) -> str:
    words = text.split()
    while words and words[0] in {"on", "in", "at", "to", "the"}:
        words = words[1:]
    return " ".join(words)


def normalize_absolute_position(label: Any) -> Optional[str]:
    t = _normalize_label_text(label)
    if not t:
        return None
    t = _drop_leading_function_words(t)
    if not t:
        return None
    mapping = {
        "top": "top",
        "upper": "top",
        "bottom": "bottom",
        "lower": "bottom",
        "left": "left",
        "right": "right",
        "center": "center",
        "centre": "center",
        "middle": "center",
        "upper left": "upper_left",
        "upper right": "upper_right",
        "lower left": "lower_left",
        "lower right": "lower_right",
        "unknown": "unknown",
    }
    return mapping.get(t)


def normalize_spatial_relation(label: Any) -> Tuple[Optional[str], Optional[str], bool]:
    # returns: (normalized_spatial_relation, normalized_absolute_position, invalid_flag)
    t = _normalize_label_text(label)
    if not t:
        return None, None, False
    # Absolute-position phrases should be moved out of spatial_relations.
    direct_abs = normalize_absolute_position(t)
    if direct_abs is not None:
        return None, direct_abs, False
    trimmed = _drop_leading_function_words(t)
    abs_pos = normalize_absolute_position(trimmed)
    if abs_pos is not None:
        if " of" in t:
            # e.g. "on the left of" is a relative relation, not absolute.
            pass
        else:
            return None, abs_pos, False
    mapping = {
        "near": "near",
        "near to": "near",
        "close to": "near",
        "adjacent to": "adjacent_to",
        "adjacent_to": "adjacent_to",
        "beside": "beside",
        "next to": "beside",
        "left of": "left_of",
        "on the left of": "left_of",
        "to the left of": "left_of",
        "left_of": "left_of",
        "right of": "right_of",
        "on the right of": "right_of",
        "to the right of": "right_of",
        "right_of": "right_of",
        "above": "above",
        "on top of": "above",
        "below": "below",
        "under": "below",
        "beneath": "below",
        "inside": "inside",
        "inside of": "inside",
        "within": "inside",
        "around": "around",
        "surrounded by": "surrounded_by",
        "surrounded_by": "surrounded_by",
        "between": "between",
        "overlapping": "overlapping",
        "on": "on",
        "along": "along",
        "along with": "along",
        "smaller than": "smaller_than",
        "smaller_than": "smaller_than",
        "larger than": "larger_than",
        "larger_than": "larger_than",
        "unknown": "unknown",
    }
    rel = mapping.get(t)
    if rel is None:
        return None, None, True
    return rel, None, False


def normalize_added_token_reason(reason: Any) -> Optional[str]:
    if not isinstance(reason, str):
        return None
    t = _normalize_label_text(reason)
    if not t:
        return None
    if any(
        bad in t
        for bad in [
            "inferred",
            "inference",
            "visual inference",
            "assumed",
            "guessed",
            "common sense",
            "hallucinated",
            "based on image",
            "based on the image",
            "likely",
            "probably",
        ]
    ):
        return "__forbidden__"
    if any(k in t for k in ["raw", "original", "copied", "present in original", "explicit in raw", "from expression"]):
        return "copied_from_raw"
    if any(k in t for k in ["grammar", "grammatical", "article", "preposition", "function word", "relation verb", "syntax", "phrase normalization"]):
        return "grammar_only"
    if t in ALLOWED_JUSTIFICATION_REASON:
        return t
    return None


def _dedup_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def detect_absolute_positions_in_raw(raw_expression: str) -> List[str]:
    """Extract normalized absolute positions from raw expression text."""
    if not isinstance(raw_expression, str) or not raw_expression.strip():
        return []
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", raw_expression.lower())
    out: List[str] = []
    for k in (1, 2, 3, 4):
        for i in range(len(words) - k + 1):
            phrase = " ".join(words[i : i + k])
            npos = normalize_absolute_position(phrase)
            if npos is not None:
                out.append(npos)
            trimmed_words = phrase.split()
            while trimmed_words and trimmed_words[0] in {"at", "in", "on", "the"}:
                trimmed_words = trimmed_words[1:]
                if not trimmed_words:
                    break
                npos = normalize_absolute_position(" ".join(trimmed_words))
                if npos is not None:
                    out.append(npos)
    return _dedup_keep_order(out)


def _sanitize_slot_string_list(
    values: Any,
    field_label: str,
    normalization_notes: List[str],
) -> List[str]:
    """Drop null/None, '', whitespace-only strings, empty nested lists, and non-strings (no hard fail)."""
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for item in values:
        if item is None:
            normalization_notes.append(f"dropped null in {field_label}")
            continue
        if isinstance(item, list):
            if len(item) == 0:
                normalization_notes.append(f"dropped empty list token in {field_label}")
            else:
                normalization_notes.append(f"dropped non-flat list element in {field_label}")
            continue
        if not isinstance(item, str):
            normalization_notes.append(f"dropped non-string in {field_label}: {type(item).__name__}")
            continue
        if not item.strip():
            normalization_notes.append(f"dropped blank string in {field_label}")
            continue
        out.append(item)
    return out


def raw_supports_absolute_position(raw_expression: str, pos: str) -> bool:
    """True if raw text supports this normalized absolute position (aliases e.g. middle -> center)."""
    if pos == "unknown":
        return True
    raw_l = raw_expression.lower()
    if not raw_l.strip():
        return False
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", raw_l)
    n = len(words)
    for k in (1, 2, 3):
        for i in range(n - k + 1):
            phrase = " ".join(words[i : i + k])
            if normalize_absolute_position(phrase) == pos:
                return True
    for token in re.findall(r"[a-z]+(?:[-_][a-z]+)+", raw_l):
        if normalize_absolute_position(token.replace("_", " ")) == pos:
            return True
    friendly = pos.replace("_", " ")
    if friendly in raw_l:
        return True
    compact = friendly.replace(" ", "")
    raw_compact = re.sub(r"[^a-z0-9]+", "", raw_l)
    if compact and compact in raw_compact:
        return True
    return False


def raw_contains_phrase(raw_expression: str, phrase: str) -> bool:
    raw_norm = normalize_phrase_text(raw_expression)
    phrase_norm = normalize_phrase_text(phrase)
    if not phrase_norm:
        return False
    return phrase_norm in raw_norm


def raw_visual_preservation_check(raw: str, enhanced: str, compressed: str) -> Tuple[bool, str]:
    raw_tokens = canonical_token_set(raw)
    enh_tokens = canonical_token_set(enhanced)
    cmp_tokens = canonical_token_set(compressed)
    raw_l = raw.lower()
    # "little" in comparative phrases is a degree adverb, not a required size attribute token.
    little_as_degree = bool(re.search(r"\b(?:a\s+)?little\s+(?:smaller|larger)\s+than\b", raw_l))
    for w in VISUAL_PRESERVE_WORDS:
        if w == "little" and little_as_degree:
            continue
        if w in raw_tokens and (w not in enh_tokens or w not in cmp_tokens):
            return False, f"raw visual word removed: {w}"
    raw_norm = normalize_phrase_text(raw)
    enh_norm = normalize_phrase_text(enhanced)
    cmp_norm = normalize_phrase_text(compressed)
    for p in VISUAL_PRESERVE_PHRASES:
        pn = normalize_phrase_text(p)
        if pn in raw_norm and (pn not in enh_norm or pn not in cmp_norm):
            return False, f"raw visual phrase removed: {p}"
    return True, "ok"


def check_slot_label_leakage(raw: str, enhanced: str, compressed: str) -> Tuple[bool, str]:
    raw_l = raw.lower()
    enh_l = enhanced.lower()
    cmp_l = compressed.lower()
    for t in SLOT_LABEL_TOKENS:
        if t in enh_l and t not in raw_l:
            return False, f"slot label leaked to natural-language output: {t}"
        if t in cmp_l and t not in raw_l:
            return False, f"slot label leaked to natural-language output: {t}"
    if "middle" in raw_l and "center" not in raw_l:
        if "center" in enh_l or "center" in cmp_l:
            return False, "slot label leaked to natural-language output: center"
    return True, "ok"


def article_definiteness_check(raw: str, enhanced: str, compressed: str) -> Tuple[bool, str]:
    def _first_article(text: str) -> Optional[str]:
        m = re.match(r"^\s*(a|an|the)\b", text.lower())
        return m.group(1) if m else None

    raw_a = _first_article(raw)
    for label, txt in [("enhanced_expression", enhanced), ("compressed_expression", compressed)]:
        out_a = _first_article(txt)
        if raw_a is None or out_a is None:
            continue
        if raw_a == "a" and out_a == "the":
            return False, f"article definiteness changed (A->The) in {label}"
        if raw_a == "the" and out_a in {"a", "an"}:
            return False, f"article definiteness changed (The->A/An) in {label}"
        if raw_a == "a" and out_a == "an":
            rest = txt.lower().strip().split()[1:2]
            if rest and rest[0][0] not in "aeiou":
                return False, f"invalid article correction (A->An) in {label}"
        if raw_a == "an" and out_a == "a":
            rest = txt.lower().strip().split()[1:2]
            if rest and rest[0][0] in "aeiou":
                return False, f"invalid article correction (An->A) in {label}"
    return True, "ok"


def concept_semantic_injection_check(
    raw: str,
    enhanced: str,
    compressed: str,
    concept_semantics_context: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    if not concept_semantics_context:
        return True, "ok"
    matched_concepts = _matched_concepts_full_rows(concept_semantics_context)
    if not matched_concepts:
        return True, "ok"
    def _safe_rewrite_allowed_for_text(text: str) -> bool:
        raw_norm = _normalize_label_text(raw)
        txt_norm = _normalize_label_text(text)
        if not raw_norm or not txt_norm:
            return False
        for c in matched_concepts:
            if not isinstance(c, dict):
                continue
            rewrites = c.get("safe_rewrite_guidance", [])
            if not isinstance(rewrites, list):
                continue
            aliases = c.get("aliases", [])
            alias_list: List[str] = []
            if isinstance(aliases, list):
                alias_list.extend([a for a in aliases if isinstance(a, str)])
            canonical_name = normalize_to_str(c.get("canonical_name"))
            if canonical_name:
                alias_list.append(canonical_name)
            alias_norms = sorted({_normalize_label_text(a) for a in alias_list if _normalize_label_text(a)})
            for r in rewrites:
                if not isinstance(r, dict):
                    continue
                allowed_rewrite = normalize_to_str(r.get("allowed_rewrite"))
                if not allowed_rewrite:
                    continue
                allowed_norm = _normalize_label_text(allowed_rewrite)
                if not allowed_norm or txt_norm != allowed_norm:
                    continue
                if raw_norm == allowed_norm:
                    return True
                for alias_norm in alias_norms:
                    if raw_norm == f"find {alias_norm}":
                        return True
        return False

    safe_rewrite_allow_enh = _safe_rewrite_allowed_for_text(enhanced)
    safe_rewrite_allow_cmp = _safe_rewrite_allowed_for_text(compressed)
    forbidden_tokens: Set[str] = set()
    for c in matched_concepts:
        if not isinstance(c, dict):
            continue
        toks = c.get("forbidden_auto_infer_tokens", [])
        if isinstance(toks, list):
            for t in toks:
                if isinstance(t, str) and t.strip():
                    forbidden_tokens.add(t.strip().lower())
    for t in sorted(forbidden_tokens):
        raw_has = len(_find_phrase_matches(raw, t)) > 0
        enh_has = len(_find_phrase_matches(enhanced, t)) > 0
        cmp_has = len(_find_phrase_matches(compressed, t)) > 0
        enh_injected = (not raw_has) and enh_has and (not safe_rewrite_allow_enh)
        cmp_injected = (not raw_has) and cmp_has and (not safe_rewrite_allow_cmp)
        if enh_injected or cmp_injected:
            return False, f"concept semantic injected new raw fact: {t}"
    return True, "ok"


def visual_form_option_leakage_check(
    raw: str,
    enhanced: str,
    compressed: str,
    concept_semantics_context: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    if not concept_semantics_context:
        return True, "ok"
    matched_concepts = _matched_concepts_full_rows(concept_semantics_context)
    if not matched_concepts:
        return True, "ok"
    def _safe_rewrite_allowed_for_text(text: str) -> bool:
        raw_norm = _normalize_label_text(raw)
        txt_norm = _normalize_label_text(text)
        if not raw_norm or not txt_norm:
            return False
        for c in matched_concepts:
            if not isinstance(c, dict):
                continue
            rewrites = c.get("safe_rewrite_guidance", [])
            if not isinstance(rewrites, list):
                continue
            aliases = c.get("aliases", [])
            alias_list: List[str] = []
            if isinstance(aliases, list):
                alias_list.extend([a for a in aliases if isinstance(a, str)])
            canonical_name = normalize_to_str(c.get("canonical_name"))
            if canonical_name:
                alias_list.append(canonical_name)
            alias_norms = sorted({_normalize_label_text(a) for a in alias_list if _normalize_label_text(a)})
            for r in rewrites:
                if not isinstance(r, dict):
                    continue
                allowed_rewrite = normalize_to_str(r.get("allowed_rewrite"))
                if not allowed_rewrite:
                    continue
                allowed_norm = _normalize_label_text(allowed_rewrite)
                if not allowed_norm or txt_norm != allowed_norm:
                    continue
                if raw_norm == allowed_norm:
                    return True
                for alias_norm in alias_norms:
                    if raw_norm == f"find {alias_norm}":
                        return True
        return False

    safe_rewrite_allow_enh = _safe_rewrite_allowed_for_text(enhanced)
    safe_rewrite_allow_cmp = _safe_rewrite_allowed_for_text(compressed)
    leakage_keywords: Set[str] = set()
    for c in matched_concepts:
        if not isinstance(c, dict):
            continue
        options = c.get("visual_form_options", [])
        if not isinstance(options, list):
            continue
        for opt in options:
            if not isinstance(opt, dict):
                continue
            phrase = normalize_to_str(opt.get("option"))
            if phrase and phrase.strip() and phrase.strip().lower() != "none":
                for kw in _normalize_label_text(phrase).split(" "):
                    if kw and kw not in {"a", "an", "the", "of", "and", "or"}:
                        leakage_keywords.add(kw)
    for kw in sorted(leakage_keywords):
        raw_has = len(_find_phrase_matches(raw, kw)) > 0
        if raw_has:
            continue
        enh_has = len(_find_phrase_matches(enhanced, kw)) > 0
        cmp_has = len(_find_phrase_matches(compressed, kw)) > 0
        enh_leak = enh_has and (not safe_rewrite_allow_enh)
        cmp_leak = cmp_has and (not safe_rewrite_allow_cmp)
        if enh_leak or cmp_leak:
            return False, f"visual_form_option keyword leaked into output without raw evidence: {kw}"
    return True, "ok"


def concept_tag_output_leakage_check(
    raw_expression: str,
    enhanced_expression: str,
    compressed_expression: str,
    concept_context: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    """Block internal tags / do_not_emit phrases in NL outputs unless raw has the same wording (phrase-level, word-boundary)."""
    if not concept_context:
        return True, "ok"
    matched_concepts = _matched_concepts_full_rows(concept_context)
    if not matched_concepts:
        return True, "ok"
    blocked_tokens: Set[str] = set()
    for c in matched_concepts:
        if not isinstance(c, dict):
            continue
        slot_guidance = c.get("slot_guidance", {})
        if not isinstance(slot_guidance, dict):
            continue
        for key in ("allowed_internal_tags", "allowed_slot_hints", "do_not_emit_as_text"):
            vals = slot_guidance.get(key, [])
            if not isinstance(vals, list):
                continue
            for v in vals:
                if isinstance(v, str) and v.strip():
                    blocked_tokens.add(v.strip())
    raw_norm = _normalize_label_text(raw_expression)
    enh_norm = _normalize_label_text(enhanced_expression)
    cmp_norm = _normalize_label_text(compressed_expression)
    for token in sorted(blocked_tokens, key=lambda s: len(s), reverse=True):
        if not _normalize_label_text(token):
            continue
        raw_has = len(_find_phrase_matches(raw_norm, token)) > 0
        if raw_has:
            continue
        if len(_find_phrase_matches(enh_norm, token)) > 0 or len(_find_phrase_matches(cmp_norm, token)) > 0:
            return False, f"concept internal tag leaked into natural language: {token}"
    return True, "ok"


def normalize_to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return None


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_semantic_kb(path: str) -> Dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("semantic kb must be a JSON object.")
    return payload


def load_concept_semantic_library(path: str) -> Dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("concept semantic library must be a JSON object.")
    return payload


def load_concept_public_semantic_library(path: str) -> Dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("concept public semantic library must be a JSON object.")
    return payload


def _deep_copy_semantic_field(val: Any) -> Any:
    if isinstance(val, (dict, list)):
        return copy.deepcopy(val)
    return val


def _full_concept_retrieval_row(concept_key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Shape used for matched_concepts_full (validators / injection checks)."""
    aliases = cfg.get("aliases", [])
    alias_list = [concept_key] + ([a for a in aliases if isinstance(a, str)] if isinstance(aliases, list) else [])
    return {
        "concept_key": concept_key,
        "concept_type": cfg.get("concept_type", ""),
        "canonical_name": cfg.get("canonical_name", concept_key),
        "aliases": _deep_copy_semantic_field(alias_list),
        "matched_alias": [],
        "match_span": [],
        "definition": _deep_copy_semantic_field(cfg.get("definition", "")),
        "remote_sensing_understanding": _deep_copy_semantic_field(cfg.get("remote_sensing_understanding", [])),
        "visual_form_options": _deep_copy_semantic_field(cfg.get("visual_form_options", [])),
        "segmentation_relevance": _deep_copy_semantic_field(cfg.get("segmentation_relevance", [])),
        "safe_rewrite_guidance": _deep_copy_semantic_field(cfg.get("safe_rewrite_guidance", [])),
        "slot_guidance": _deep_copy_semantic_field(cfg.get("slot_guidance", {})),
        "forbidden_auto_infer_tokens": _deep_copy_semantic_field(cfg.get("forbidden_auto_infer_tokens", [])),
        "anti_specialization_rules": _deep_copy_semantic_field(cfg.get("anti_specialization_rules", [])),
        "notes": _deep_copy_semantic_field(cfg.get("notes", "")),
        "concept_semantics": _deep_copy_semantic_field(cfg.get("concept_semantics", [])),
        "safe_negative_boundaries": _deep_copy_semantic_field(cfg.get("safe_negative_boundaries", "")),
    }


STUFF_LIKE_CONCEPT_TYPES = frozenset({"stuff", "region", "stuff_region"})
# Hard caps for minimal public JSON built from a **full** semantic library (no silent truncation; oversize raises ValueError).
PUBLIC_CONTEXT_JSON_MAX_CHARS = 280
PUBLIC_BOUNDARY_MAX_CHARS = 200
_DEFAULT_STUFF_BOUNDARY_PUBLIC = (
    "Do not specialize as river, lake, pond, reservoir, canal, harbor, or wetland unless supported by raw text or image evidence."
)
_STUFF_BOUNDARY_SCRUB_WORDS = ("irregular", "elongated", "clustered", "small", "patches", "narrow")
_STUFF_PUBLIC_SCRUB_CANONICALS_NORMALIZED = frozenset(
    {
        "water",
        "vegetation",
        "farmland",
        "grassland",
        "forest",
        "bare land",
        "river",
        "lake",
    }
)


def skip_visual_form_forbidden_coverage_for_concept_type(concept_type: Any) -> bool:
    if not isinstance(concept_type, str):
        return False
    return concept_type.strip().lower() in STUFF_LIKE_CONCEPT_TYPES


def _use_stuff_like_public_boundary_scrub(concept_entry: Dict[str, Any], type_out: str, canonical: str) -> bool:
    """Stuff/region entries and frozen rows missing concept_type for known stuff-like canonicals."""
    if skip_visual_form_forbidden_coverage_for_concept_type(type_out):
        return True
    if type_out == "unknown" and _normalize_label_text(canonical) in _STUFF_PUBLIC_SCRUB_CANONICALS_NORMALIZED:
        return True
    return False


def is_public_minimal_semantic_library(library: Optional[Dict[str, Any]]) -> bool:
    """Hand-written public library: only concept/type/boundary per entry (see concept_public_semantic_library_v0.json)."""
    if not isinstance(library, dict):
        return False
    if str(library.get("library_kind", "")).strip() == "public_minimal":
        return True
    concepts = library.get("concepts")
    if not isinstance(concepts, dict) or not concepts:
        return False
    for _ck, cfg in concepts.items():
        if not isinstance(cfg, dict):
            return False
        if set(cfg.keys()) != {"concept", "type", "boundary"}:
            return False
    return True


def is_public_grounding_prior_library(library: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(library, dict):
        return False
    return str(library.get("library_kind", "")).strip() == "public_grounding_prior"


GROUNDING_PRIOR_MATCH_POLICY_EXPECTED: Dict[str, str] = {
    "matching": "word_boundary",
    "multi_word": "continuous_phrase",
    "selection": "longest_first_non_overlapping",
    "hierarchy": "child_suppresses_parent",
    "multi_concept": "inject_all_non_overlapping_matches",
    "fallback": "empty",
}
GROUNDING_PRIOR_ENTRY_KEYS = frozenset(
    {"concept", "type", "parent", "visual_evidence", "mask_scope", "exclusion_rule"}
)
GROUNDING_PRIOR_FIELD_CHAR_MAX = 180
GROUNDING_PRIOR_ROW_JSON_MAX_CHARS = 900
GROUNDING_PRIOR_MATCHED_BLOCK_MAX_CHARS = 1200


def _grounding_prior_alias_list(concept_key: str, cfg: Dict[str, Any]) -> List[str]:
    """Retrieval-only alias expansion (no extra JSON fields). Adds simple English plural for single-token labels."""
    vals: List[str] = []
    for a in (concept_key, normalize_to_str(cfg.get("concept")) or ""):
        if isinstance(a, str) and a.strip():
            vals.append(a.strip())
    out: List[str] = []
    seen_norm: Set[str] = set()
    for a in vals:
        n = _normalize_label_text(a)
        if not n or n in seen_norm:
            continue
        seen_norm.add(n)
        out.append(a)
        if " " not in a:
            if not a.lower().endswith("s"):
                pl = a + "s"
                n2 = _normalize_label_text(pl)
                if n2 and n2 not in seen_norm:
                    seen_norm.add(n2)
                    out.append(pl)
    return out


def _suppress_parent_grounding_matches(
    picked: List[Dict[str, Any]],
    concepts: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """When a child (river/lake) matches, drop matched parent water for the same raw text."""
    keys_matched = {c["concept_key"] for c in picked}
    to_drop: Set[str] = set()
    for k in keys_matched:
        cfg = concepts.get(k)
        if not isinstance(cfg, dict):
            continue
        par = cfg.get("parent")
        if par is None or (isinstance(par, str) and not par.strip()):
            continue
        pk = str(par).strip()
        if pk in concepts and pk in keys_matched:
            to_drop.add(pk)
    return [c for c in picked if c["concept_key"] not in to_drop]


def build_grounding_prior_public_row(concept_key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Six-field public payload for MLLM prompts (category-level grounding prior)."""
    par_raw = cfg.get("parent")
    parent_out: Any
    if par_raw is None:
        parent_out = None
    elif isinstance(par_raw, str) and not par_raw.strip():
        parent_out = None
    else:
        parent_out = str(par_raw).strip()
    out: Dict[str, Any] = {
        "concept": str(cfg.get("concept", concept_key)).strip(),
        "type": str(cfg.get("type", "")).strip(),
        "parent": parent_out,
        "visual_evidence": str(cfg.get("visual_evidence", "")).strip(),
        "mask_scope": str(cfg.get("mask_scope", "")).strip(),
        "exclusion_rule": str(cfg.get("exclusion_rule", "")).strip(),
    }
    blob = json.dumps(out, ensure_ascii=False)
    if len(blob) > GROUNDING_PRIOR_ROW_JSON_MAX_CHARS:
        raise ValueError(
            f"grounding prior public row JSON exceeds GROUNDING_PRIOR_ROW_JSON_MAX_CHARS="
            f"{GROUNDING_PRIOR_ROW_JSON_MAX_CHARS}: json_len={len(blob)} concept={out['concept']!r}"
        )
    return out


def build_minimal_public_semantic_context(concept_entry: Dict[str, Any]) -> Dict[str, str]:
    """Public-only view for MLLM prompts: canonical label, coarse type, one boundary sentence."""
    ck = normalize_to_str(concept_entry.get("concept_key")) or ""
    canonical = normalize_to_str(concept_entry.get("canonical_name")) or ck
    ct_raw = concept_entry.get("concept_type", "")
    type_out = ct_raw.strip() if isinstance(ct_raw, str) and ct_raw.strip() else "unknown"
    boundary = ""
    snb = concept_entry.get("safe_negative_boundaries")
    if isinstance(snb, str) and snb.strip():
        boundary = snb.strip()
    elif isinstance(snb, list) and snb:
        boundary = " ".join(x.strip() for x in snb if isinstance(x, str) and x.strip())
    if not boundary:
        asr = concept_entry.get("anti_specialization_rules", [])
        if isinstance(asr, list) and asr:
            first_rule = next((s.strip() for s in asr if isinstance(s, str) and s.strip()), "")
            boundary = first_rule if first_rule else ""
    if not boundary:
        if _use_stuff_like_public_boundary_scrub(concept_entry, type_out, canonical):
            boundary = _DEFAULT_STUFF_BOUNDARY_PUBLIC
        else:
            boundary = "Do not introduce attributes or subclasses not present in the raw expression."
    if _use_stuff_like_public_boundary_scrub(concept_entry, type_out, canonical):
        low = boundary.lower()
        if any(w in low for w in _STUFF_BOUNDARY_SCRUB_WORDS):
            boundary = _DEFAULT_STUFF_BOUNDARY_PUBLIC
    out = {"concept": canonical, "type": type_out, "boundary": boundary}
    if len(out["boundary"]) > PUBLIC_BOUNDARY_MAX_CHARS:
        raise ValueError(
            f"minimal public boundary exceeds PUBLIC_BOUNDARY_MAX_CHARS={PUBLIC_BOUNDARY_MAX_CHARS}: "
            f"len={len(out['boundary'])} concept={canonical!r} — shorten source safe_negative_boundaries or anti_specialization_rules."
        )
    blob = json.dumps(out, ensure_ascii=False)
    if len(blob) > PUBLIC_CONTEXT_JSON_MAX_CHARS:
        raise ValueError(
            f"minimal public context JSON exceeds PUBLIC_CONTEXT_JSON_MAX_CHARS={PUBLIC_CONTEXT_JSON_MAX_CHARS}: "
            f"json_len={len(blob)} concept={canonical!r} boundary_len={len(out['boundary'])}."
        )
    return out


def _matched_concepts_full_rows(concept_semantics_context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(concept_semantics_context, dict):
        return []
    full_rows = concept_semantics_context.get("matched_concepts_full")
    if isinstance(full_rows, list) and full_rows:
        return [x for x in full_rows if isinstance(x, dict)]
    legacy = concept_semantics_context.get("matched_concepts", [])
    if isinstance(legacy, list):
        return [x for x in legacy if isinstance(x, dict)]
    return []


def semantic_context_summary_from_context(semantic_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if semantic_context is None:
        return None
    return {
        "category_context_used": semantic_context.get("category_context", {}),
        "matched_spatial_phrases": semantic_context.get("matched_spatial_phrases", []),
        "forbidden_replacements_used": semantic_context.get("high_risk_replacements", {}),
    }


def concept_semantic_summary_from_context(concept_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if concept_context is None:
        return None
    return {
        "matched_concepts": concept_context.get("matched_concepts", []),
        "global_constraints": concept_context.get("global_constraints", []),
    }


def _phrase_pattern(phrase: str) -> Optional[re.Pattern]:
    label = _normalize_label_text(phrase)
    if not label:
        return None
    parts = [re.escape(p) for p in label.split()]
    body = r"\s+".join(parts)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", flags=re.IGNORECASE)


def _find_phrase_matches(text: str, phrase: str) -> List[Tuple[int, int]]:
    p = _phrase_pattern(phrase)
    if p is None:
        return []
    return [(m.start(), m.end()) for m in p.finditer(text)]


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sha1_8(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def parse_path(path: str) -> List[str]:
    if path in ("$", "root"):
        return []
    return [seg for seg in path.split(".") if seg]


def _extract_values(obj: Any, segments: List[str]) -> List[Any]:
    if not segments:
        return [obj]
    seg = segments[0]
    rest = segments[1:]
    out: List[Any] = []
    if seg == "[]":
        if isinstance(obj, list):
            for item in obj:
                out.extend(_extract_values(item, rest))
        return out
    if seg.endswith("[]"):
        key = seg[:-2]
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], list):
            for item in obj[key]:
                out.extend(_extract_values(item, rest))
        return out
    if isinstance(obj, dict) and seg in obj:
        out.extend(_extract_values(obj[seg], rest))
    return out


def extract_values(obj: Any, path: str) -> List[Any]:
    return _extract_values(obj, parse_path(path))


def _traverse_paths(obj: Any, prefix: str, out: List[Tuple[str, str, Any]]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.append((p, k.lower(), v))
            _traverse_paths(v, p, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5]):
            p = f"{prefix}[]" if prefix else "[]"
            out.append((p, "[]", v))
            _traverse_paths(v, p, out)
            if i >= 4:
                break


def _add_candidate(cands: Dict[str, List[str]], key: str, path: str) -> None:
    if path not in cands[key]:
        cands[key].append(path)


def probe_schema(input_json_path: str, preview_limit: int = 20) -> Dict[str, Any]:
    data = read_json(input_json_path)
    top_level_type = "list" if isinstance(data, list) else "dict"
    candidate_paths: Dict[str, List[str]] = {
        "sample_list": [],
        "sample_id": [],
        "image": [],
        "mask": [],
        "category": [],
        "instance_id": [],
        "expression": [],
    }

    samples: List[Any]
    if isinstance(data, list):
        samples = data
        _add_candidate(candidate_paths, "sample_list", "$")
    elif isinstance(data, dict):
        samples = []
        for k, v in data.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                samples = v
                _add_candidate(candidate_paths, "sample_list", f"{k}[]")
                break
    else:
        samples = []

    for sample in samples[:preview_limit]:
        flat: List[Tuple[str, str, Any]] = []
        _traverse_paths(sample, "", flat)
        for path, low_key, value in flat:
            if low_key in {"id", "sample_id", "image_id"}:
                _add_candidate(candidate_paths, "sample_id", path)
            if any(k in low_key for k in ["image", "img"]) and isinstance(value, (str, int)):
                _add_candidate(candidate_paths, "image", path)
            if "mask" in low_key and isinstance(value, (str, int)):
                _add_candidate(candidate_paths, "mask", path)
            if any(k in low_key for k in ["category", "class", "label"]) and isinstance(value, (str, int)):
                _add_candidate(candidate_paths, "category", path)
            if any(k in low_key for k in ["instance_id", "ins_id", "inst_id", "instance"]) and isinstance(
                value, (str, int)
            ):
                _add_candidate(candidate_paths, "instance_id", path)
            if any(k in low_key for k in ["expression", "expr", "text", "ref"]):
                if isinstance(value, str):
                    _add_candidate(candidate_paths, "expression", path)
                if isinstance(value, list):
                    _add_candidate(candidate_paths, "expression", f"{path}[]")

    detected_layout = "unknown"
    expr_paths = candidate_paths["expression"]
    if any("[]" in p for p in expr_paths):
        if any("[][]" in p for p in expr_paths):
            detected_layout = "nested"
        else:
            detected_layout = "multiple"
    elif expr_paths:
        detected_layout = "single"

    return {
        "top_level_type": top_level_type,
        "num_samples_previewed": min(len(samples), preview_limit),
        "candidate_paths": candidate_paths,
        "detected_expression_layout": detected_layout,
        "requires_user_confirmation": True,
    }


@dataclass
class ProcessingUnit:
    expr_id: str
    ref_id: Optional[str]
    ann_id: Optional[str]
    image_id: Optional[str]
    file_name: Optional[str]
    split: Optional[str]
    category_id: Optional[str]
    category_name: Optional[str]
    bbox: Any
    segmentation_ref: Dict[str, Any]
    raw_expression: str
    expression_index: int
    sent_id: Optional[str]


def build_expr_id(
    ref_id: Any,
    ann_id: Any,
    sent_id: Any,
    expression_index: int,
    raw_expression: str,
) -> str:
    ref_s = normalize_to_str(ref_id) or "null"
    ann_s = normalize_to_str(ann_id) or "null"
    sent_s = normalize_to_str(sent_id)
    if sent_s:
        return f"ref_{ref_s}__ann_{ann_s}__sent_{sent_s}"
    h = sha1_8((raw_expression or "").strip().lower())
    return f"ref_{ref_s}__ann_{ann_s}__idx_{expression_index}__h_{h}"


def choose_sentence_text(sent_obj: Dict[str, Any]) -> str:
    for key in ["sent", "raw", "sentence", "text", "caption"]:
        value = sent_obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def make_structured_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "raw_expression",
            "slots",
            "explicitization",
            "enhanced_expression",
            "compressed_expression",
            "status",
            "notes",
        ],
        "properties": {
            "raw_expression": {"type": "string"},
            "slots": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "target_category",
                    "attributes",
                    "spatial_relations",
                    "reference_objects",
                    "absolute_positions",
                    "quantity",
                    "clean_label_valid",
                ],
                "properties": {
                    "target_category": {"type": ["string", "null"]},
                    "attributes": {"type": "array", "items": {"type": "string"}},
                    "spatial_relations": {"type": "array", "items": {"type": "string"}},
                    "reference_objects": {"type": "array", "items": {"type": "string"}},
                    "absolute_positions": {"type": "array", "items": {"type": "string"}},
                    "quantity": {"type": "string"},
                    "clean_label_valid": {"type": "boolean"},
                },
            },
            "explicitization": {
                "type": "object",
                "additionalProperties": False,
                "required": ["operations", "added_tokens", "added_token_justification"],
                "properties": {
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["type", "before", "after"],
                            "properties": {
                                "type": {"type": "string"},
                                "before": {"type": "string"},
                                "after": {"type": "string"},
                            },
                        },
                    },
                    "added_tokens": {"type": "array", "items": {"type": "string"}},
                    "added_token_justification": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["token", "reason"],
                            "properties": {
                                "token": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "enhanced_expression": {"type": "string"},
            "compressed_expression": {"type": "string"},
            "status": {"type": "string"},
            "notes": {"type": "string"},
        },
    }


def retrieve_semantic_context(
    raw_expression: str,
    category_name: Optional[str],
    semantic_kb: Dict[str, Any],
) -> Dict[str, Any]:
    categories = semantic_kb.get("categories", {}) if isinstance(semantic_kb, dict) else {}
    spatial_vocab = semantic_kb.get("spatial_vocabulary", {}) if isinstance(semantic_kb, dict) else {}
    explicitization_policy = semantic_kb.get("explicitization_policy", {}) if isinstance(semantic_kb, dict) else {}
    high_risk_replacements = semantic_kb.get("high_risk_replacements", {}) if isinstance(semantic_kb, dict) else {}

    category_str = category_name or ""
    category_context: Dict[str, Any] = {}
    matched = False
    category_key_norm = normalize_phrase_text(category_str)
    if isinstance(categories, dict):
        for ds_cat, cfg in categories.items():
            if not isinstance(cfg, dict):
                continue
            aliases = cfg.get("aliases", [])
            alias_norms = {normalize_phrase_text(ds_cat)}
            if isinstance(aliases, list):
                alias_norms.update({normalize_phrase_text(a) for a in aliases if isinstance(a, str)})
            if category_key_norm and category_key_norm in alias_norms:
                category_context = {
                    "dataset_category_name": ds_cat,
                    "canonical_name": cfg.get("canonical_name", ds_cat),
                    "aliases": aliases if isinstance(aliases, list) else [ds_cat],
                    "forbidden_specific_replacements": cfg.get("forbidden_specific_replacements", []),
                }
                matched = True
                break
    if not matched:
        category_context = {
            "dataset_category_name": category_str,
            "canonical_name": category_str,
            "aliases": [category_str] if category_str else [],
            "forbidden_specific_replacements": [],
        }

    matched_spatial_phrases: List[Dict[str, Any]] = []
    raw_label = _normalize_label_text(raw_expression)
    if isinstance(spatial_vocab, dict):
        for raw_phrase, cfg in spatial_vocab.items():
            if not isinstance(raw_phrase, str) or not isinstance(cfg, dict):
                continue
            raw_phrase_label = _normalize_label_text(raw_phrase)
            if not raw_phrase_label:
                continue
            if raw_phrase_label in raw_label:
                matched_spatial_phrases.append(
                    {
                        "raw_phrase": raw_phrase,
                        "type": cfg.get("type", ""),
                        "normalized": cfg.get("normalized", ""),
                    }
                )
    matched_spatial_phrases = sorted(matched_spatial_phrases, key=lambda x: len(str(x.get("raw_phrase", ""))), reverse=True)

    return {
        "category_context": category_context,
        "matched_spatial_phrases": matched_spatial_phrases,
        "explicitization_policy": {
            "allowed_operations": explicitization_policy.get("allowed_operations", list(ALLOWED_OP_TYPES)),
            "forbidden_inference": explicitization_policy.get("forbidden_inference", []),
        },
        "high_risk_replacements": high_risk_replacements if isinstance(high_risk_replacements, dict) else {},
    }


def retrieve_concept_semantics(
    raw_expression: str,
    library: Dict[str, Any],
    *,
    private_library_for_audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    concepts = library.get("concepts", {}) if isinstance(library, dict) else {}
    global_constraints = library.get("global_constraints", []) if isinstance(library, dict) else []
    matched_concepts: List[Dict[str, Any]] = []
    if not isinstance(concepts, dict):
        return {
            "matched_concepts": matched_concepts,
            "matched_concepts_full": matched_concepts,
            "global_constraints": global_constraints if isinstance(global_constraints, list) else [],
        }

    candidates: List[Dict[str, Any]] = []
    for concept_key, cfg in concepts.items():
        if not isinstance(cfg, dict):
            continue
        if is_public_grounding_prior_library(library):
            alias_list = _grounding_prior_alias_list(concept_key, cfg)
        elif is_public_minimal_semantic_library(library):
            alias_list = [concept_key]
            cn = normalize_to_str(cfg.get("concept"))
            if cn and _normalize_label_text(cn) != _normalize_label_text(concept_key):
                alias_list.append(cn)
        else:
            aliases = cfg.get("aliases", [])
            alias_list = [concept_key]
            if isinstance(aliases, list):
                alias_list += [a for a in aliases if isinstance(a, str)]
        dedup_aliases: List[str] = []
        seen: Set[str] = set()
        for a in alias_list:
            n = _normalize_label_text(a)
            if n and n not in seen:
                seen.add(n)
                dedup_aliases.append(a)
        for alias in dedup_aliases:
            for start, end in _find_phrase_matches(raw_expression, alias):
                candidates.append(
                    {
                        "concept_key": concept_key,
                        "cfg": cfg,
                        "alias": alias,
                        "start": start,
                        "end": end,
                        "length": end - start,
                    }
                )

    candidates.sort(key=lambda x: (-x["length"], x["start"]))
    occupied = [False] * max(len(raw_expression), 1)
    picked: List[Dict[str, Any]] = []
    for c in candidates:
        s, e = c["start"], c["end"]
        if any(occupied[i] for i in range(s, e)):
            continue
        picked.append(c)
        for i in range(s, e):
            occupied[i] = True

    if is_public_grounding_prior_library(library):
        picked = _suppress_parent_grounding_matches(picked, concepts)

    private_concepts: Dict[str, Any] = {}
    if isinstance(private_library_for_audit, dict):
        pc = private_library_for_audit.get("concepts", {})
        if isinstance(pc, dict):
            private_concepts = pc

    if is_public_minimal_semantic_library(library):
        by_pub: Dict[str, Dict[str, Any]] = {}
        for c in sorted(picked, key=lambda x: x["start"]):
            concept_key = c["concept_key"]
            cfg = c["cfg"]
            if concept_key not in by_pub:
                by_pub[concept_key] = {
                    "concept": str(cfg.get("concept", concept_key)).strip(),
                    "type": str(cfg.get("type", "")).strip(),
                    "boundary": str(cfg.get("boundary", "")).strip(),
                    "_matched_alias": [],
                    "_match_span": [],
                }
            by_pub[concept_key]["_matched_alias"].append(c["alias"])
            by_pub[concept_key]["_match_span"].append([c["start"], c["end"]])
        matched_public: List[Dict[str, str]] = []
        matched_full: List[Dict[str, Any]] = []
        for concept_key, row in sorted(
            by_pub.items(),
            key=lambda kv: min([sp[0] for sp in kv[1]["_match_span"] if isinstance(sp, list) and sp], default=0),
        ):
            matched_public.append(
                {"concept": row["concept"], "type": row["type"], "boundary": row["boundary"]}
            )
            pcfg = private_concepts.get(concept_key)
            if isinstance(pcfg, dict):
                full_row = _full_concept_retrieval_row(concept_key, pcfg)
                full_row["matched_alias"] = list(row["_matched_alias"])
                full_row["match_span"] = list(row["_match_span"])
                matched_full.append(full_row)
        return {
            "matched_concepts": matched_public,
            "matched_concepts_full": matched_full,
            "global_constraints": global_constraints if isinstance(global_constraints, list) else [],
        }

    if is_public_grounding_prior_library(library):
        by_gp: Dict[str, Dict[str, Any]] = {}
        for c in sorted(picked, key=lambda x: x["start"]):
            concept_key = c["concept_key"]
            cfg = c["cfg"]
            if concept_key not in by_gp:
                by_gp[concept_key] = {
                    "cfg": cfg,
                    "_matched_alias": [],
                    "_match_span": [],
                }
            by_gp[concept_key]["_matched_alias"].append(c["alias"])
            by_gp[concept_key]["_match_span"].append([c["start"], c["end"]])
        matched_public_gp: List[Dict[str, Any]] = []
        matched_full_gp: List[Dict[str, Any]] = []
        for concept_key, row in sorted(
            by_gp.items(),
            key=lambda kv: min([sp[0] for sp in kv[1]["_match_span"] if isinstance(sp, list) and len(sp) == 2], default=0),
        ):
            cfg = row["cfg"]
            pub_row = build_grounding_prior_public_row(concept_key, cfg)
            matched_public_gp.append(pub_row)
            pcfg = private_concepts.get(concept_key)
            if isinstance(pcfg, dict):
                full_row = _full_concept_retrieval_row(concept_key, pcfg)
                full_row["matched_alias"] = list(row["_matched_alias"])
                full_row["match_span"] = list(row["_match_span"])
                matched_full_gp.append(full_row)
            else:
                matched_full_gp.append(dict(pub_row))
        return {
            "matched_concepts": matched_public_gp,
            "matched_concepts_full": matched_full_gp,
            "global_constraints": global_constraints if isinstance(global_constraints, list) else [],
        }

    by_concept: Dict[str, Dict[str, Any]] = {}
    for c in sorted(picked, key=lambda x: x["start"]):
        concept_key = c["concept_key"]
        cfg = c["cfg"]
        if concept_key not in by_concept:
            by_concept[concept_key] = _full_concept_retrieval_row(concept_key, cfg)
        by_concept[concept_key]["matched_alias"].append(c["alias"])
        by_concept[concept_key]["match_span"].append([c["start"], c["end"]])
    matched_concepts_full = list(by_concept.values())
    matched_concepts_public = [build_minimal_public_semantic_context(row) for row in matched_concepts_full]
    return {
        "matched_concepts": matched_concepts_public,
        "matched_concepts_full": matched_concepts_full,
        "global_constraints": global_constraints if isinstance(global_constraints, list) else [],
    }


def _concept_context_intro_block(concept_semantics_context: Dict[str, Any]) -> str:
    matched = concept_semantics_context.get("matched_concepts", [])
    if not isinstance(matched, list) or not matched:
        return "No category-level public concept priors were retrieved for this raw expression.\n"
    r0 = matched[0] if isinstance(matched[0], dict) else {}
    if isinstance(r0, dict) and "visual_evidence" in r0 and "boundary" not in r0:
        return (
            "You are also given category-level public grounding priors for remote sensing "
            "(concept, type, parent, visual_evidence, mask_scope, exclusion_rule).\n"
            "They support target localization and mask-boundary reasoning only; they are not observed facts about this sample.\n"
            "Do not treat priors as permission to add colors, materials, objects, or scene details absent from the raw expression.\n"
            "When multiple priors match, each is listed separately; exclusion texts are not merged and require no conflict resolution.\n"
        )
    if isinstance(r0, dict) and "boundary" in r0:
        return (
            "You are also given minimal public concept context (concept name, coarse type, and one boundary sentence only).\n"
            "It is for understanding concept meaning only, not evidence of observed facts in this sample.\n"
            "Do not treat concept context as if it were already present in the raw expression.\n"
            "The minimal concept JSON below omits private-library fields such as token blocklists, per-form option tables, and structured slot tables.\n"
        )
    return (
        "You are also given retrieved concept semantics context for coarse alignment and validation.\n"
        "It is not a source of new visual facts for this sample.\n"
    )


def _concept_matched_json_header(matched: Any) -> str:
    if isinstance(matched, list) and matched and isinstance(matched[0], dict):
        r0 = matched[0]
        if isinstance(r0, dict) and "visual_evidence" in r0 and "boundary" not in r0:
            return "Retrieved public grounding priors"
    return "Retrieved concept semantics"


def build_llm_messages(
    raw_expression: str,
    allow_synonym_normalization: bool,
    category_name: Optional[str] = None,
    semantic_context: Optional[Dict[str, Any]] = None,
    concept_semantics_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    allow_syn_text = (
        "Synonym normalization is disabled. Keep lexical category terms exactly as in raw expression unless grammar requires function words."
        if not allow_synonym_normalization
        else "Synonym normalization may be used only when semantics are unchanged and no new visual fact is introduced."
    )
    system_prompt = (
        "You are enhancing a remote sensing referring expression for training data.\n"
        "Your task is explicitization-only rewriting.\n"
        "raw expression is the only source of observed facts.\n"
        "Concept tags and internal tags may only appear in slots or concept_tags.\n"
        "They must never appear in enhanced_expression or compressed_expression unless the exact same words already appear in the raw expression.\n"
        "allowed_internal_tags are internal annotations only.\n"
        "do_not_emit_as_text items are forbidden in natural-language outputs unless raw explicitly contains them.\n"
        "enhanced_expression and compressed_expression must remain natural-language referring expressions, not tag strings.\n"
        "Do not infer or invent visual facts.\n"
        "Do not add color, material, size, geometry, number, orientation, absolute location, object state, scene type, "
        "target category, or reference object unless it explicitly appears in the original expression.\n"
        "Only four operation types are allowed:\n"
        "1. article_completion\n"
        "2. preposition_completion\n"
        "3. relation_verb_insertion\n"
        "4. phrase_normalization\n"
        "If the raw expression lacks attribute or spatial information, keep the enhanced expression close to the raw expression and only improve grammar.\n"
        "Forbidden example:\n"
        "Raw: airplane near runway\n"
        "Forbidden: The small white airplane parked near the runway.\n"
        "Reason: small, white, and parked are not present in the raw expression.\n"
        "Return only valid JSON matching the required schema."
    )
    if semantic_context is None:
        semantic_context = {
            "category_context": {
                "dataset_category_name": category_name or "",
                "canonical_name": category_name or "",
                "aliases": [category_name] if category_name else [],
                "forbidden_specific_replacements": [],
            },
            "matched_spatial_phrases": [],
            "explicitization_policy": {
                "allowed_operations": list(ALLOWED_OP_TYPES),
                "forbidden_inference": [],
            },
            "high_risk_replacements": {},
        }
    if concept_semantics_context is None:
        concept_semantics_context = {"matched_concepts": [], "global_constraints": []}
    _matched_list = concept_semantics_context.get("matched_concepts", [])
    _intro_concept = _concept_context_intro_block(concept_semantics_context)
    _matched_header = _concept_matched_json_header(_matched_list)
    user_prompt = (
        f"{allow_syn_text}\n"
        "raw expression is the only source of observed facts.\n"
        "You are given a raw referring expression and a semantic context retrieved from a dataset-specific knowledge base.\n"
        "The semantic context is a constraint, not a source for adding new visual facts.\n"
        f"{_intro_concept}"
        "When the raw expression is vague, keep enhanced and compressed expressions conservative and close to raw wording.\n"
        "Concept tags and internal tags may only appear in slots or concept_tags.\n"
        "They must never appear in enhanced_expression or compressed_expression unless the exact same words already appear in the raw expression.\n"
        "allowed_internal_tags are internal annotations only.\n"
        "do_not_emit_as_text items are forbidden in natural-language outputs unless raw explicitly contains them.\n"
        "enhanced_expression and compressed_expression must remain natural-language referring expressions, not tag strings.\n"
        "semantic_context is for slot normalization and constraint checking only.\n"
        "Do not use semantic_context to delete, replace, simplify, or add visual words in enhanced_expression or compressed_expression.\n"
        "enhanced_expression and compressed_expression must preserve all visual tokens from raw_expression, including colors, sizes, geometry words, attributes, category words, reference objects, and natural-language spatial phrases.\n"
        "Slot labels such as left_of, right_of, upper_left, lower_left, center, smaller_than are allowed only in slots.\n"
        "Slot labels must never appear in enhanced_expression or compressed_expression unless the exact same token already appears in raw_expression.\n"
        "If raw says 'in the middle', enhanced/compressed must keep 'in the middle'; slots may use center.\n"
        "If raw says 'on the left of', enhanced/compressed must keep 'on the left of'; slots may use left_of.\n"
        "If raw says 'on the lower left', enhanced/compressed must keep 'on the lower left'; slots may use lower_left.\n"
        "Do not remove visual attributes that already appear in raw.\n"
        "Do not change article definiteness (A/The/An), except A/An grammar correction.\n"
        "If no safe explicitization is needed, return unchanged.\n"
        "You may use category aliases only to normalize category naming.\n"
        "You may use spatial vocabulary only to normalize spatial phrases already present in the raw expression.\n"
        "You must not introduce colors, sizes, shapes, states, reference objects, or specific subclasses that are not present in the raw expression.\n\n"
        f"Current category:\n{json.dumps(semantic_context.get('category_context', {}), ensure_ascii=False)}\n\n"
        f"Matched spatial phrases from raw expression:\n{json.dumps(semantic_context.get('matched_spatial_phrases', []), ensure_ascii=False)}\n\n"
        f"Allowed explicitization operations:\n{json.dumps(semantic_context.get('explicitization_policy', {}).get('allowed_operations', []), ensure_ascii=False)}\n\n"
        f"Forbidden inference rules:\n{json.dumps(semantic_context.get('explicitization_policy', {}).get('forbidden_inference', []), ensure_ascii=False)}\n\n"
        f"High-risk replacements:\n{json.dumps(semantic_context.get('high_risk_replacements', {}), ensure_ascii=False)}\n\n"
        f"{_matched_header}:\n{json.dumps(_matched_list, ensure_ascii=False)}\n\n"
        f"Concept global constraints:\n{json.dumps(concept_semantics_context.get('global_constraints', []), ensure_ascii=False)}\n\n"
        f"Raw expression:\n{raw_expression}\n"
        "Output strict JSON only."
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


PROMPT_LEAKAGE_FORBIDDEN_FIELD_NAMES = (
    "forbidden_auto_infer_tokens",
    "visual_form_options",
    "mandatory_high_risk_terms",
    "must_put_in_forbidden",
    "do_not_put_in_forbidden",
    "generation_hint",
    "slot_guidance",
)
PROMPT_LEAKAGE_POLLUTION_TERMS = (
    "irregular",
    "elongated",
    "clustered",
    "patches",
    "narrow",
    "dense",
    "sparse",
    "shape",
    "texture",
)


def prompt_leakage_self_check(
    *,
    raw_expression: str = "find water near bridge",
    public_library_path: Optional[str] = None,
    private_library_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Ensure final MLLM user prompt has no private-field names or morphology pollution (no API)."""
    repo_configs = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    pub_path = public_library_path or os.path.join(repo_configs, "concept_public_semantic_library_v0.json")
    priv_path = private_library_path or os.path.join(repo_configs, "concept_semantic_library_verified_7.json")
    public_lib = load_concept_public_semantic_library(pub_path)
    private_lib = load_concept_semantic_library(priv_path)
    ctx = retrieve_concept_semantics(raw_expression, public_lib, private_library_for_audit=private_lib)
    messages = build_llm_messages(
        raw_expression,
        False,
        category_name=None,
        semantic_context=None,
        concept_semantics_context=ctx,
    )
    user_prompt = ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            user_prompt = str(m.get("content", ""))
            break
    failures: List[Dict[str, Any]] = []
    low = user_prompt.lower()
    matched = ctx.get("matched_concepts", []) if isinstance(ctx.get("matched_concepts"), list) else []
    matched_json = json.dumps(matched, ensure_ascii=False)
    low_matched = matched_json.lower()
    for name in PROMPT_LEAKAGE_FORBIDDEN_FIELD_NAMES:
        if name in low:
            failures.append({"kind": "forbidden_field_name_in_user_prompt", "token": name})
    first_m = matched[0] if matched and isinstance(matched[0], dict) else {}
    grounding_matched = isinstance(first_m, dict) and "visual_evidence" in first_m and "boundary" not in first_m
    if not grounding_matched:
        for term in PROMPT_LEAKAGE_POLLUTION_TERMS:
            if term in low_matched:
                failures.append({"kind": "pollution_term_in_matched_concepts_json", "token": term})
    if grounding_matched:
        required = ("water", "bridge", "type", "visual_evidence", "mask_scope", "exclusion_rule")
    else:
        required = ("water", "bridge", "type", "boundary")
    for req in required:
        if req not in low:
            failures.append({"kind": "missing_required_substring", "token": req})
    return {
        "prompt_leakage_self_check_passed": len(failures) == 0,
        "failures": failures,
        "raw_expression": raw_expression,
        "user_prompt_char_count": len(user_prompt),
        "public_library_path": pub_path,
        "private_library_path": priv_path,
    }


def concept_public_context_length_check(
    *,
    raw_expression: str = "find water near bridge",
    public_library_path: Optional[str] = None,
    private_library_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Read-only stats: full user prompt length vs public matched_concepts payload (no API)."""
    repo_configs = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    pub_path = public_library_path or os.path.join(repo_configs, "concept_public_semantic_library_v0.json")
    priv_path = private_library_path or os.path.join(repo_configs, "concept_semantic_library_verified_7.json")
    public_lib = load_concept_public_semantic_library(pub_path)
    private_lib = load_concept_semantic_library(priv_path)
    ctx = retrieve_concept_semantics(raw_expression, public_lib, private_library_for_audit=private_lib)
    matched = ctx.get("matched_concepts", []) if isinstance(ctx.get("matched_concepts"), list) else []
    matched_json = json.dumps(matched, ensure_ascii=False)
    low_matched = matched_json.lower()
    per_entry: List[Dict[str, Any]] = []
    boundary_lens: List[int] = []
    grounding_max_field_lens: List[int] = []
    first_row = matched[0] if matched and isinstance(matched[0], dict) else {}
    grounding_mode = isinstance(first_row, dict) and "visual_evidence" in first_row and "boundary" not in first_row
    for i, row in enumerate(matched):
        if not isinstance(row, dict):
            continue
        one_json = json.dumps(row, ensure_ascii=False)
        if grounding_mode:
            ve = str(row.get("visual_evidence", ""))
            ms = str(row.get("mask_scope", ""))
            ex = str(row.get("exclusion_rule", ""))
            mx = max(len(ve), len(ms), len(ex), len(str(row.get("concept", ""))), len(str(row.get("type", ""))))
            grounding_max_field_lens.append(mx)
            per_entry.append(
                {
                    "index": i,
                    "concept": row.get("concept"),
                    "json_char_count": len(one_json),
                    "max_public_field_char_count": mx,
                }
            )
        else:
            b = str(row.get("boundary", ""))
            boundary_lens.append(len(b))
            per_entry.append(
                {
                    "index": i,
                    "concept": row.get("concept"),
                    "json_char_count": len(one_json),
                    "boundary_char_count": len(b),
                }
            )
    messages = build_llm_messages(
        raw_expression,
        False,
        category_name=None,
        semantic_context=None,
        concept_semantics_context=ctx,
    )
    user_prompt = ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            user_prompt = str(m.get("content", ""))
            break
    low_user = user_prompt.lower()
    private_hits_user = [n for n in PROMPT_LEAKAGE_FORBIDDEN_FIELD_NAMES if n in low_user]
    private_hits_matched = [n for n in PROMPT_LEAKAGE_FORBIDDEN_FIELD_NAMES if n in low_matched]
    pollution_hits_matched_only: List[str] = []
    if not grounding_mode:
        pollution_hits_matched_only = [t for t in PROMPT_LEAKAGE_POLLUTION_TERMS if t in low_matched]
    concept_labels = [str(x.get("concept", "")).strip().lower() for x in matched if isinstance(x, dict)]
    retrieved_concept_count = len(matched)
    matched_json_char_count = len(matched_json)
    expected_pair = {"water", "bridge"}
    concepts_match = set(concept_labels) == expected_pair and retrieved_concept_count == 2
    if grounding_mode:
        concept_block_ok = matched_json_char_count < GROUNDING_PRIOR_MATCHED_BLOCK_MAX_CHARS
        each_public_field_le_budget = True
        for row in matched:
            if not isinstance(row, dict):
                continue
            for k in ("concept", "type", "visual_evidence", "mask_scope", "exclusion_rule"):
                v = row.get(k)
                if isinstance(v, str) and len(v) > GROUNDING_PRIOR_FIELD_CHAR_MAX:
                    each_public_field_le_budget = False
            parv = row.get("parent")
            if isinstance(parv, str) and len(parv) > GROUNDING_PRIOR_FIELD_CHAR_MAX:
                each_public_field_le_budget = False
        passed = bool(
            concepts_match
            and concept_block_ok
            and each_public_field_le_budget
            and not private_hits_user
            and not private_hits_matched
        )
        return {
            "user_prompt_char_count": len(user_prompt),
            "retrieved_concept_count": retrieved_concept_count,
            "matched_concepts_json_char_count": matched_json_char_count,
            "per_matched_concept": per_entry,
            "grounding_prior_mode": True,
            "max_public_field_char_counts": grounding_max_field_lens,
            "private_field_names_in_user_prompt": private_hits_user,
            "private_field_names_in_matched_concepts_json": private_hits_matched,
            "pollution_terms_in_matched_concepts_json": pollution_hits_matched_only,
            "matched_concepts_within_grounding_char_budget": concept_block_ok,
            "each_public_text_field_le_180": each_public_field_le_budget,
            "concepts_expected_water_and_bridge": concepts_match,
            "concept_public_context_length_check_passed": passed,
            "raw_expression": raw_expression,
            "public_library_path": pub_path,
            "private_library_path": priv_path,
        }
    concept_block_under_500 = matched_json_char_count < 500
    each_boundary_le_150 = all((bl <= 150 for bl in boundary_lens)) if boundary_lens else False
    return {
        "user_prompt_char_count": len(user_prompt),
        "retrieved_concept_count": retrieved_concept_count,
        "matched_concepts_json_char_count": matched_json_char_count,
        "per_matched_concept": per_entry,
        "grounding_prior_mode": False,
        "boundary_char_counts": boundary_lens,
        "private_field_names_in_user_prompt": private_hits_user,
        "private_field_names_in_matched_concepts_json": private_hits_matched,
        "pollution_terms_in_matched_concepts_json": pollution_hits_matched_only,
        "concept_block_under_500_chars": concept_block_under_500,
        "each_boundary_char_count_le_150": each_boundary_le_150,
        "concepts_expected_water_and_bridge": concepts_match,
        "concept_public_context_length_check_passed": bool(
            concepts_match
            and concept_block_under_500
            and each_boundary_le_150
            and not private_hits_user
            and not private_hits_matched
            and not pollution_hits_matched_only
        ),
        "raw_expression": raw_expression,
        "public_library_path": pub_path,
        "private_library_path": priv_path,
    }


def call_llm_structured_once(
    *,
    api_url: str,
    api_key: str,
    model: str,
    timeout_sec: int,
    strict_json_schema: bool,
    raw_expression: str,
    allow_synonym_normalization: bool,
    category_name: Optional[str] = None,
    semantic_context: Optional[Dict[str, Any]] = None,
    concept_semantics_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "messages": build_llm_messages(
            raw_expression,
            allow_synonym_normalization,
            category_name=category_name,
            semantic_context=semantic_context,
            concept_semantics_context=concept_semantics_context,
        ),
    }
    if strict_json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "rrsisd_explicitization",
                "strict": True,
                "schema": make_structured_response_schema(),
            },
        }
    else:
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout_sec)
    if resp.status_code >= 400:
        raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:2000]}", response=resp)
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("LLM response content is not JSON string.")
    return json.loads(content)


def _is_retryable_http(status_code: int) -> bool:
    return status_code in {429, 500, 502, 503, 504}


def call_llm_with_retry(
    *,
    api_url: str,
    api_key: str,
    model: str,
    timeout_sec: int,
    strict_json_schema: bool,
    raw_expression: str,
    allow_synonym_normalization: bool,
    category_name: Optional[str] = None,
    semantic_context: Optional[Dict[str, Any]] = None,
    concept_semantics_context: Optional[Dict[str, Any]] = None,
    max_attempts: int,
    initial_wait: float,
    max_wait: float,
    log_path: str,
    expr_id: str,
) -> Dict[str, Any]:
    attempt = 0
    wait = initial_wait
    last_err: Optional[Exception] = None
    while attempt < max_attempts:
        attempt += 1
        try:
            return call_llm_structured_once(
                api_url=api_url,
                api_key=api_key,
                model=model,
                timeout_sec=timeout_sec,
                strict_json_schema=strict_json_schema,
                raw_expression=raw_expression,
                allow_synonym_normalization=allow_synonym_normalization,
                category_name=category_name,
                semantic_context=semantic_context,
                concept_semantics_context=concept_semantics_context,
            )
        except Exception as e:
            last_err = e
            retryable = False
            if isinstance(e, requests.Timeout) or isinstance(e, requests.ConnectionError):
                retryable = True
            elif isinstance(e, requests.HTTPError) and e.response is not None:
                retryable = _is_retryable_http(e.response.status_code)
            if not retryable or attempt >= max_attempts:
                break
            sleep_s = min(max_wait, wait * (2 ** (attempt - 1))) * (0.5 + random.random())
            append_jsonl(
                log_path,
                {
                    "timestamp": utcnow(),
                    "event": "retry",
                    "expr_id": expr_id,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "error": str(e)[:1200],
                    "next_wait_seconds": round(sleep_s, 3),
                },
            )
            time.sleep(sleep_s)
    if last_err is None:
        raise RuntimeError("Unknown LLM failure.")
    raise last_err


def validate_and_normalize_output(
    raw_expression: str,
    llm_output: Dict[str, Any],
    category_name: Optional[str] = None,
    concept_semantics_context: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    out = copy.deepcopy(llm_output)
    required_fields = {
        "raw_expression",
        "slots",
        "explicitization",
        "enhanced_expression",
        "compressed_expression",
        "notes",
    }
    missing = [k for k in required_fields if k not in out]
    if missing:
        return False, f"missing fields: {missing}", out
    normalization_notes: List[str] = []
    raw_llm_status = out.get("status")
    out["raw_llm_status"] = raw_llm_status
    if isinstance(raw_llm_status, str) and raw_llm_status in ALLOWED_LLM_STATUS:
        out["_normalized_llm_status"] = raw_llm_status
        out["raw_llm_status_invalid"] = False
    else:
        out["_normalized_llm_status"] = None
        out["raw_llm_status_invalid"] = True
        normalization_notes.append(f"ignored invalid raw_llm_status `{raw_llm_status}` for final pipeline status")
    if not isinstance(out["raw_expression"], str):
        return False, "raw_expression type invalid", out
    if not isinstance(out["enhanced_expression"], str) or not isinstance(out["compressed_expression"], str):
        return False, "enhanced/compressed type invalid", out
    if not isinstance(out["notes"], str):
        return False, "notes type invalid", out

    slots = out["slots"]
    explicitization = out["explicitization"]
    if not isinstance(slots, dict) or not isinstance(explicitization, dict):
        return False, "slots/explicitization type invalid", out
    for f in ["attributes", "reference_objects"]:
        if f not in slots or not isinstance(slots[f], list) or not all(isinstance(x, str) for x in slots[f]):
            return False, f"slots.{f} invalid", out
    if "spatial_relations" not in slots or not isinstance(slots["spatial_relations"], list):
        return False, "slots.spatial_relations invalid", out
    if "absolute_positions" not in slots or not isinstance(slots["absolute_positions"], list):
        return False, "slots.absolute_positions invalid", out
    slots["spatial_relations"] = _sanitize_slot_string_list(
        slots["spatial_relations"], "slots.spatial_relations", normalization_notes
    )
    slots["absolute_positions"] = _sanitize_slot_string_list(
        slots["absolute_positions"], "slots.absolute_positions", normalization_notes
    )
    if slots.get("target_category") is not None and not isinstance(slots.get("target_category"), str):
        return False, "slots.target_category invalid", out
    quantity = slots.get("quantity")
    quantity_norm = None
    if isinstance(quantity, str):
        q = quantity.strip().lower()
        if q in ALLOWED_QUANTITY:
            quantity_norm = q
    if quantity_norm is None:
        quantity_norm = "unknown"
        normalization_notes.append(f"normalized invalid quantity `{quantity}` to `unknown`")
    slots["quantity"] = quantity_norm

    normalized_abs_positions: List[str] = []
    for pos in slots.get("absolute_positions", []):
        npos = normalize_absolute_position(pos)
        if npos is not None:
            normalized_abs_positions.append(npos)
        elif isinstance(pos, str) and pos.strip():
            normalization_notes.append(f"dropped invalid absolute_position `{pos}`")
    raw_abs_positions = detect_absolute_positions_in_raw(raw_expression)
    slots["absolute_positions"] = _dedup_keep_order(normalized_abs_positions + raw_abs_positions)

    normalized_spatial_relations: List[str] = []
    moved_abs_from_spatial: List[str] = []
    invalid_spatial_labels: List[str] = []
    for rel in slots.get("spatial_relations", []):
        nrel, abs_pos, invalid = normalize_spatial_relation(rel)
        if invalid:
            rel_text = _normalize_label_text(rel)
            if rel_text in {"at", "in"}:
                if slots["absolute_positions"] or raw_abs_positions:
                    normalization_notes.append(
                        f"dropped residual spatial preposition `{rel_text}` due to absolute position evidence"
                    )
                else:
                    normalization_notes.append(
                        f"dropped non-informative spatial preposition `{rel_text}` without absolute evidence"
                    )
                continue
            invalid_spatial_labels.append(rel)
            continue
        if abs_pos is not None:
            moved_abs_from_spatial.append(abs_pos)
            continue
        if nrel is not None:
            normalized_spatial_relations.append(nrel)
    if invalid_spatial_labels:
        return False, f"slots.spatial_relations enum invalid: {invalid_spatial_labels}", out
    if moved_abs_from_spatial:
        normalization_notes.append(f"moved absolute position labels from spatial_relations: {moved_abs_from_spatial}")
    slots["spatial_relations"] = _dedup_keep_order(normalized_spatial_relations)
    slots["absolute_positions"] = _dedup_keep_order(slots["absolute_positions"] + moved_abs_from_spatial)
    raw_l = raw_expression.lower()
    for rel in slots["spatial_relations"]:
        if rel == "smaller_than" and "smaller than" not in raw_l:
            return False, "spatial comparative not in raw: smaller_than", out
        if rel == "larger_than" and "larger than" not in raw_l:
            return False, "spatial comparative not in raw: larger_than", out
    if not isinstance(slots.get("clean_label_valid"), bool):
        return False, "slots.clean_label_valid invalid", out
    if any(rel not in ALLOWED_SPATIAL_RELATIONS for rel in slots["spatial_relations"]):
        return False, "slots.spatial_relations enum invalid", out
    if any(pos not in ALLOWED_ABS_POSITIONS for pos in slots["absolute_positions"]):
        return False, "slots.absolute_positions enum invalid", out

    if "operations" not in explicitization or not isinstance(explicitization["operations"], list):
        return False, "explicitization.operations invalid", out
    for op in explicitization["operations"]:
        if not isinstance(op, dict):
            return False, "operation invalid", out
        if op.get("type") not in ALLOWED_OP_TYPES:
            return False, "operation type invalid", out
        if not isinstance(op.get("before"), str) or not isinstance(op.get("after"), str):
            return False, "operation before/after invalid", out
    if "added_tokens" not in explicitization or not isinstance(explicitization["added_tokens"], list):
        return False, "added_tokens invalid", out
    if not all(isinstance(t, str) for t in explicitization["added_tokens"]):
        return False, "added_tokens item invalid", out
    if "added_token_justification" not in explicitization or not isinstance(
        explicitization["added_token_justification"], list
    ):
        return False, "added_token_justification invalid", out
    added_tokens_list = explicitization.get("added_tokens", [])
    just_list = explicitization.get("added_token_justification", [])
    if not added_tokens_list and not just_list:
        pass
    else:
        for it in just_list:
            if not isinstance(it, dict):
                return False, "added_token_justification item invalid", out
            if not isinstance(it.get("token"), str):
                return False, "added_token_justification.token invalid", out
            reason_raw = it.get("reason")
            reason_norm = normalize_added_token_reason(reason_raw)
            if reason_norm == "__forbidden__":
                return False, f"added_token_justification.reason forbidden: {reason_raw}", out
            if reason_norm is None:
                return False, "added_token_justification.reason invalid", out
            if reason_norm != reason_raw:
                normalization_notes.append(
                    f"normalized added_token_justification.reason `{reason_raw}` -> `{reason_norm}`"
                )
            it["reason"] = reason_norm

    raw_tokens = canonical_token_set(raw_expression)
    enh_text = out["enhanced_expression"]
    cmp_text = out["compressed_expression"]
    enh_tokens = canonical_token_set(enh_text)
    cmp_tokens = canonical_token_set(cmp_text)

    ok_vis, reason_vis = raw_visual_preservation_check(raw_expression, enh_text, cmp_text)
    if not ok_vis:
        return False, reason_vis, out
    ok_concept_inj, reason_concept_inj = concept_semantic_injection_check(
        raw_expression, enh_text, cmp_text, concept_semantics_context
    )
    if not ok_concept_inj:
        return False, reason_concept_inj, out
    ok_vfo_leak, reason_vfo_leak = visual_form_option_leakage_check(
        raw_expression, enh_text, cmp_text, concept_semantics_context
    )
    if not ok_vfo_leak:
        return False, reason_vfo_leak, out
    ok_concept_tag_leak, reason_concept_tag_leak = concept_tag_output_leakage_check(
        raw_expression, enh_text, cmp_text, concept_semantics_context
    )
    if not ok_concept_tag_leak:
        return False, reason_concept_tag_leak, out
    ok_leak, reason_leak = check_slot_label_leakage(raw_expression, enh_text, cmp_text)
    if not ok_leak:
        return False, reason_leak, out
    ok_art, reason_art = article_definiteness_check(raw_expression, enh_text, cmp_text)
    if not ok_art:
        return False, reason_art, out

    new_risky = [w for w in HIGH_RISK_WORDS if w in enh_tokens and w not in raw_tokens]
    new_risky_cmp = [w for w in HIGH_RISK_WORDS if w in cmp_tokens and w not in raw_tokens]
    if new_risky or new_risky_cmp:
        merged = sorted(set(new_risky + new_risky_cmp))
        return False, f"high-risk hallucinated words: {merged}", out

    dataset_category = category_name or ""
    dataset_category_norm = normalize_category_text(dataset_category)
    target_category = slots.get("target_category")
    target_norm = normalize_category_text(target_category) if isinstance(target_category, str) else ""
    raw_contains_dataset_category = raw_contains_category_text(raw_expression, dataset_category)

    if isinstance(target_category, str) and target_category.strip():
        if dataset_category_norm:
            if category_text_match(target_category, dataset_category):
                if dataset_category and target_category != dataset_category:
                    normalization_notes.append(
                        f"normalized target_category `{target_category}` to dataset category_name `{dataset_category}`"
                    )
                slots["target_category"] = dataset_category
            else:
                # Explicit guard: do not allow semantic substitutions unless explicitly present in raw.
                if (
                    dataset_category_norm == "airplane"
                    and target_norm == "aircraft"
                    and "aircraft" in normalize_category_text(raw_expression)
                ):
                    slots["target_category"] = target_category
                else:
                    return False, "target category changed/not in raw", out
        elif target_norm and target_norm not in normalize_category_text(raw_expression):
            return False, "target category changed/not in raw", out
    else:
        if dataset_category_norm and raw_contains_dataset_category:
            slots["target_category"] = dataset_category
            normalization_notes.append("fallback target_category from dataset category_name")
        else:
            slots["target_category"] = None
    for attr in slots.get("attributes", []):
        attr_norm = normalize_phrase_text(attr)
        if not attr_norm:
            continue
        # Multi-word attributes must use phrase-level containment in normalized text space.
        if " " in attr.strip() or "-" in attr or "_" in attr:
            if not raw_contains_phrase(raw_expression, attr):
                return False, f"attribute not in raw: {attr}", out
            continue
        if attr.lower() not in raw_tokens and not raw_contains_phrase(raw_expression, attr):
            return False, f"attribute not in raw: {attr}", out
    raw_phrase_norm = normalize_phrase_text(raw_expression)
    dataset_cat_norm = normalize_phrase_text(dataset_category)
    target_cat_norm = normalize_phrase_text(slots.get("target_category"))
    for ref_obj in slots.get("reference_objects", []):
        ref_norm = normalize_phrase_text(ref_obj)
        if not ref_norm:
            continue
        # Keep strict guards against semantic substitutions.
        if ref_norm == "car" and "vehicle" in raw_phrase_norm and "car" not in raw_phrase_norm:
            return False, f"new reference object introduced: {ref_obj}", out
        if ref_norm == "boat" and "ship" in raw_phrase_norm and "boat" not in raw_phrase_norm:
            return False, f"new reference object introduced: {ref_obj}", out
        if ref_norm == "aircraft" and "airplane" in raw_phrase_norm and "aircraft" not in raw_phrase_norm:
            return False, f"new reference object introduced: {ref_obj}", out
        if ref_norm in raw_phrase_norm:
            continue
        # If reference object equals target/category form, treat as potential self-reference.
        if ref_norm and (ref_norm == dataset_cat_norm or ref_norm == target_cat_norm):
            continue
        return False, f"new reference object introduced: {ref_obj}", out
    for pos in slots.get("absolute_positions", []):
        if pos != "unknown" and not raw_supports_absolute_position(raw_expression, pos):
            return False, f"absolute position not in raw: {pos}", out

    raw_wc = max(word_count(raw_expression), 1)
    if word_count(enh_text) > raw_wc + 8:
        return False, "enhanced expression too long", out
    if word_count(cmp_text) > raw_wc + 5:
        return False, "compressed expression too long", out
    for phrase in FORBIDDEN_EXPLANATION_PHRASES:
        if phrase in enh_text.lower() or phrase in cmp_text.lower():
            return False, f"forbidden explanation phrase: {phrase}", out
    for token in [t.lower() for t in explicitization.get("added_tokens", [])]:
        if token in HIGH_RISK_WORDS and token not in raw_tokens:
            return False, f"added risky token not in raw: {token}", out

    out["raw_expression"] = raw_expression
    out["_normalization_notes"] = normalization_notes
    return True, "ok", out


def fallback_failed_record(
    unit: Dict[str, Any],
    status: str,
    note: str,
    semantic_context_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if status not in ALLOWED_PIPELINE_STATUS:
        raise ValueError(f"invalid pipeline status: {status}")
    rec = {
        "expr_id": unit.get("expr_id"),
        "ref_id": unit.get("ref_id"),
        "ann_id": unit.get("ann_id"),
        "image_id": unit.get("image_id"),
        "file_name": unit.get("file_name"),
        "split": unit.get("split"),
        "category_id": unit.get("category_id"),
        "category_name": unit.get("category_name"),
        "bbox": unit.get("bbox"),
        "segmentation_ref": unit.get("segmentation_ref"),
        "raw": unit.get("raw_expression", ""),
        "enhanced": unit.get("raw_expression", ""),
        "compressed": unit.get("raw_expression", ""),
        "status": status,
        "slots": copy.deepcopy(DEFAULT_EMPTY_SLOTS),
        "explicitization": copy.deepcopy(DEFAULT_EMPTY_EXPLICITIZATION),
        "notes": note,
        "expression_index": unit.get("expression_index"),
        "sent_id": unit.get("sent_id"),
        "source": unit.get("source"),
    }
    if semantic_context_summary is not None:
        rec["semantic_context_summary"] = semantic_context_summary
    return rec


def normalize_success_record(unit: Dict[str, Any], out: Dict[str, Any], pipeline_status: str) -> Dict[str, Any]:
    if pipeline_status not in ALLOWED_PIPELINE_STATUS:
        raise ValueError(f"invalid pipeline status: {pipeline_status}")
    notes = out.get("notes", "")
    norm_notes = out.get("_normalization_notes", [])
    if norm_notes:
        notes = (notes + " | " if notes else "") + "; ".join(norm_notes)
    return {
        "expr_id": unit.get("expr_id"),
        "ref_id": unit.get("ref_id"),
        "ann_id": unit.get("ann_id"),
        "image_id": unit.get("image_id"),
        "file_name": unit.get("file_name"),
        "split": unit.get("split"),
        "category_id": unit.get("category_id"),
        "category_name": unit.get("category_name"),
        "bbox": unit.get("bbox"),
        "segmentation_ref": unit.get("segmentation_ref"),
        "raw": unit.get("raw_expression", ""),
        "enhanced": out["enhanced_expression"].strip(),
        "compressed": out["compressed_expression"].strip(),
        "status": pipeline_status,
        "slots": out["slots"],
        "explicitization": out["explicitization"],
        "notes": notes,
        "expression_index": unit.get("expression_index"),
        "sent_id": unit.get("sent_id"),
        "source": unit.get("source"),
        "raw_llm_status": out.get("raw_llm_status"),
        "raw_llm_status_invalid": bool(out.get("raw_llm_status_invalid", False)),
    }


def decide_pipeline_status(raw_expr: str, normalized: Dict[str, Any]) -> str:
    enh = str(normalized.get("enhanced_expression", "")).strip()
    cmp_text = str(normalized.get("compressed_expression", "")).strip()
    raw = str(raw_expr or "").strip()
    ops = normalized.get("explicitization", {}).get("operations", [])
    added_tokens = normalized.get("explicitization", {}).get("added_tokens", [])
    if enh == raw and (cmp_text == raw or cmp_text == "") and len(ops) == 0 and len(added_tokens) == 0:
        return "unchanged"
    return "success"


def semantic_kb_self_check(semantic_kb: Dict[str, Any]) -> bool:
    ctx = retrieve_semantic_context("x", "Expressway-toll-station", semantic_kb)
    assert ctx["category_context"]["canonical_name"] == "expressway toll station"

    ctx = retrieve_semantic_context("x", "baseballfield", semantic_kb)
    assert "baseball field" in ctx["category_context"]["aliases"]

    ctx = retrieve_semantic_context(
        "The basketball court is on the right of the tennis court in the middle",
        "basketballcourt",
        semantic_kb,
    )
    norms = {m.get("normalized") for m in ctx.get("matched_spatial_phrases", [])}
    assert "right_of" in norms and "center" in norms

    ctx = retrieve_semantic_context(
        "A vehicle is on the right of the tiny expressway toll station at the bottom",
        "vehicle",
        semantic_kb,
    )
    norms = {m.get("normalized") for m in ctx.get("matched_spatial_phrases", [])}
    assert "right_of" in norms and "bottom" in norms

    high_risk = semantic_kb.get("high_risk_replacements", {})
    assert set(["car", "truck", "bus"]).issubset(set(high_risk.get("vehicle", [])))
    assert "aircraft" in high_risk.get("airplane", [])
    usage_rules = semantic_kb.get("explicitization_policy", {}).get("semantic_context_usage", [])
    assert any("normalize slots only" in str(x) for x in usage_rules)

    ctx = retrieve_semantic_context("unknown object", "unknown-category", semantic_kb)
    assert ctx["category_context"]["dataset_category_name"] == "unknown-category"
    assert ctx["category_context"]["canonical_name"] == "unknown-category"
    return True


def concept_library_audit(concept_library: Dict[str, Any]) -> Dict[str, Any]:
    required_fields = {"canonical_name", "aliases", "concept_semantics", "forbidden_auto_infer_tokens", "notes"}
    concepts = concept_library.get("concepts", {}) if isinstance(concept_library, dict) else {}
    concept_count = len(concepts) if isinstance(concepts, dict) else 0
    alias_map: Dict[str, List[Tuple[str, str]]] = {}
    missing_fields: List[Dict[str, Any]] = []
    alias_conflicts: List[Dict[str, Any]] = []
    alias_empty: List[Dict[str, Any]] = []
    strong_hits: List[Dict[str, Any]] = []
    longest_match_risks: List[Dict[str, Any]] = []
    high_risk_missing: List[Dict[str, Any]] = []
    visual_form_options_schema_errors: List[Dict[str, Any]] = []
    visual_form_option_emit_policy_missing: List[Dict[str, Any]] = []
    visual_form_option_leakage_risk: List[Dict[str, Any]] = []
    safe_rewrite_conflicts: List[Dict[str, Any]] = []
    default_none_option_missing: List[Dict[str, Any]] = []
    missing_definition_in_retrieval: List[Dict[str, Any]] = []
    missing_remote_sensing_understanding_in_retrieval: List[Dict[str, Any]] = []
    missing_segmentation_relevance_in_retrieval: List[Dict[str, Any]] = []
    missing_slot_guidance_in_retrieval: List[Dict[str, Any]] = []
    missing_anti_specialization_rules_in_retrieval: List[Dict[str, Any]] = []
    missing_safe_rewrite_guidance_in_retrieval: List[Dict[str, Any]] = []
    missing_visual_form_options_in_retrieval: List[Dict[str, Any]] = []
    missing_forbidden_auto_infer_in_retrieval: List[Dict[str, Any]] = []
    retrieval_field_completeness_errors: List[Dict[str, Any]] = []
    concept_tag_leakage_check_missing: List[Dict[str, Any]] = []
    strong_terms = [
        "always",
        "must",
        "definitely",
        "certainly",
        "usually blue",
        "typically near",
        "always connected to",
        "commonly with",
        "generally accompanied by",
        "often near",
    ]
    risk_dict = {
        "river": ["blue", "winding", "bridge", "bridges", "boat", "boats", "road", "roads", "lake", "lakes", "canal", "canals"],
        "water": ["river", "rivers", "pond", "ponds", "lake", "lakes", "reservoir", "canal", "blue", "green"],
        "bridge": ["river", "rivers", "road", "roads", "water", "vehicle", "vehicles"],
        "airport": ["runway", "runways", "airplane", "airplanes", "terminal", "terminals", "hangar", "hangars"],
        "vehicle": ["car", "cars", "truck", "trucks", "bus", "buses", "road", "roads", "parking lot", "parking lots"],
        "ship": ["boat", "boats", "harbor", "harbors", "dock", "docks", "water"],
    }
    if isinstance(concepts, dict):
        for concept_key, cfg in concepts.items():
            if not isinstance(cfg, dict):
                continue
            for f in sorted(required_fields):
                if f not in cfg:
                    missing_fields.append({"concept": concept_key, "field": f, "path": f"concepts.{concept_key}.{f}"})
            aliases = cfg.get("aliases", [])
            if (not isinstance(aliases, list)) or len(aliases) == 0:
                alias_empty.append({"concept": concept_key, "path": f"concepts.{concept_key}.aliases"})
                aliases = []
            for i, alias in enumerate(aliases):
                if not isinstance(alias, str) or not alias.strip():
                    continue
                a_norm = _normalize_label_text(alias)
                if not a_norm:
                    continue
                alias_map.setdefault(a_norm, []).append((concept_key, f"concepts.{concept_key}.aliases[{i}]"))
            sem_list = cfg.get("concept_semantics", [])
            sem_text = " ".join([s for s in sem_list if isinstance(s, str)]).lower() if isinstance(sem_list, list) else ""
            for st in strong_terms:
                if st in sem_text:
                    strong_hits.append({"concept": concept_key, "term": st, "path": f"concepts.{concept_key}.concept_semantics"})
            forbidden = set([str(t).lower() for t in cfg.get("forbidden_auto_infer_tokens", []) if isinstance(t, str)])
            for risky in risk_dict.get(concept_key, []):
                if risky in sem_text and risky not in forbidden:
                    high_risk_missing.append(
                        {
                            "concept": concept_key,
                            "concept_key": concept_key,
                            "term": risky,
                            "path": f"concepts.{concept_key}",
                            "option": None,
                            "tokens_not_forbidden": [risky],
                        }
                    )
            if concept_key.lower() in forbidden:
                high_risk_missing.append(
                    {
                        "concept": concept_key,
                        "concept_key": concept_key,
                        "term": concept_key,
                        "path": f"concepts.{concept_key}.forbidden_auto_infer_tokens",
                        "option": None,
                        "tokens_not_forbidden": [concept_key.lower()],
                    }
                )
            def _rf_append(field: str, path_rel: str, error: str) -> None:
                retrieval_field_completeness_errors.append(
                    {"concept": concept_key, "field": field, "path": f"concepts.{concept_key}.{path_rel}", "error": error}
                )

            if "definition" not in cfg or not isinstance(cfg.get("definition"), str) or not str(cfg.get("definition", "")).strip():
                missing_definition_in_retrieval.append({"concept": concept_key, "path": f"concepts.{concept_key}.definition"})
                _rf_append("definition", "definition", "missing or empty")
            rsu = cfg.get("remote_sensing_understanding")
            if not isinstance(rsu, list) or not rsu or not all(isinstance(x, str) and x.strip() for x in rsu):
                missing_remote_sensing_understanding_in_retrieval.append(
                    {"concept": concept_key, "path": f"concepts.{concept_key}.remote_sensing_understanding"}
                )
                _rf_append("remote_sensing_understanding", "remote_sensing_understanding", "missing or empty list")
            seg_rel = cfg.get("segmentation_relevance")
            if not isinstance(seg_rel, list) or not seg_rel or not all(isinstance(x, str) and x.strip() for x in seg_rel):
                missing_segmentation_relevance_in_retrieval.append(
                    {"concept": concept_key, "path": f"concepts.{concept_key}.segmentation_relevance"}
                )
                _rf_append("segmentation_relevance", "segmentation_relevance", "missing or empty list")
            srwg = cfg.get("safe_rewrite_guidance")
            if not isinstance(srwg, list) or not srwg or not any(
                isinstance(r, dict) and normalize_to_str(r.get("allowed_rewrite")) for r in srwg
            ):
                missing_safe_rewrite_guidance_in_retrieval.append(
                    {"concept": concept_key, "path": f"concepts.{concept_key}.safe_rewrite_guidance"}
                )
                _rf_append("safe_rewrite_guidance", "safe_rewrite_guidance", "missing or no allowed_rewrite entry")
            if "slot_guidance" not in cfg or not isinstance(cfg.get("slot_guidance"), dict) or not cfg.get("slot_guidance"):
                missing_slot_guidance_in_retrieval.append({"concept": concept_key, "path": f"concepts.{concept_key}.slot_guidance"})
                _rf_append("slot_guidance", "slot_guidance", "missing or not an object")
            asr = cfg.get("anti_specialization_rules")
            if not isinstance(asr, list) or not asr or not all(isinstance(x, str) and x.strip() for x in asr):
                missing_anti_specialization_rules_in_retrieval.append(
                    {"concept": concept_key, "path": f"concepts.{concept_key}.anti_specialization_rules"}
                )
                _rf_append("anti_specialization_rules", "anti_specialization_rules", "missing or empty list")
            fat = cfg.get("forbidden_auto_infer_tokens")
            if not isinstance(fat, list) or not fat or not all(isinstance(x, str) and x.strip() for x in fat):
                missing_forbidden_auto_infer_in_retrieval.append(
                    {"concept": concept_key, "path": f"concepts.{concept_key}.forbidden_auto_infer_tokens"}
                )
                _rf_append("forbidden_auto_infer_tokens", "forbidden_auto_infer_tokens", "missing or empty list")
            vfo_pre = cfg.get("visual_form_options")
            if not isinstance(vfo_pre, list) or not vfo_pre:
                missing_visual_form_options_in_retrieval.append(
                    {"concept": concept_key, "path": f"concepts.{concept_key}.visual_form_options"}
                )
                _rf_append("visual_form_options", "visual_form_options", "missing or empty list")
            vfo = cfg.get("visual_form_options", [])
            if vfo is not None and not isinstance(vfo, list):
                visual_form_options_schema_errors.append(
                    {
                        "concept": concept_key,
                        "path": f"concepts.{concept_key}.visual_form_options",
                        "error": "visual_form_options must be a list when present",
                    }
                )
                vfo = []
            if isinstance(vfo, list):
                if len(vfo) > 0:
                    has_default_none = any(
                        isinstance(item, dict)
                        and _normalize_label_text(normalize_to_str(item.get("option")) or "") == "none"
                        and _normalize_label_text(normalize_to_str(item.get("emit_policy")) or "") == "default for vague raw"
                        for item in vfo
                    )
                    if not has_default_none:
                        default_none_option_missing.append(
                            {
                                "concept": concept_key,
                                "path": f"concepts.{concept_key}.visual_form_options",
                                "error": "missing none/default_for_vague_raw option",
                            }
                        )
                for i, item in enumerate(vfo):
                    item_path = f"concepts.{concept_key}.visual_form_options[{i}]"
                    if not isinstance(item, dict):
                        visual_form_options_schema_errors.append(
                            {"concept": concept_key, "path": item_path, "error": "item must be object"}
                        )
                        continue
                    option = normalize_to_str(item.get("option"))
                    meaning = normalize_to_str(item.get("meaning"))
                    emit_policy = normalize_to_str(item.get("emit_policy"))
                    if not option or not option.strip():
                        visual_form_options_schema_errors.append(
                            {"concept": concept_key, "path": item_path + ".option", "error": "missing/empty"}
                        )
                    if not meaning or not meaning.strip():
                        visual_form_options_schema_errors.append(
                            {"concept": concept_key, "path": item_path + ".meaning", "error": "missing/empty"}
                        )
                    if not emit_policy or not emit_policy.strip():
                        visual_form_option_emit_policy_missing.append(
                            {"concept": concept_key, "path": item_path + ".emit_policy", "error": "missing/empty"}
                        )
                    if option and option.strip():
                        opt_norm = _normalize_label_text(option)
                        if opt_norm and opt_norm != "none" and not skip_visual_form_forbidden_coverage_for_concept_type(
                            cfg.get("concept_type")
                        ):
                            alias_tokens: Set[str] = set()
                            aliases = cfg.get("aliases", [])
                            alias_list = [normalize_to_str(cfg.get("canonical_name")) or concept_key]
                            if isinstance(aliases, list):
                                alias_list.extend([a for a in aliases if isinstance(a, str)])
                            for alias in alias_list:
                                for tok in _normalize_label_text(alias).split(" "):
                                    if tok:
                                        alias_tokens.add(tok)
                            option_tokens = [p for p in opt_norm.split(" ") if p]
                            missing_tokens = [
                                t
                                for t in option_tokens
                                if t not in forbidden and t not in alias_tokens and t not in BASIC_STOP_WORDS
                            ]
                            if missing_tokens:
                                visual_form_option_leakage_risk.append(
                                    {
                                        "concept": concept_key,
                                        "concept_key": concept_key,
                                        "path": item_path,
                                        "option": option,
                                        "tokens_not_forbidden": missing_tokens,
                                    }
                                )
                                high_risk_missing.append(
                                    {
                                        "concept": concept_key,
                                        "concept_key": concept_key,
                                        "path": item_path,
                                        "option": option,
                                        "tokens_not_forbidden": missing_tokens,
                                        "term": ", ".join(missing_tokens),
                                        "source": "visual_form_options",
                                    }
                                )
            rewrites = cfg.get("safe_rewrite_guidance", [])
            if isinstance(rewrites, list):
                aliases_local = cfg.get("aliases", [])
                alias_tokens: Set[str] = set()
                canonical = normalize_to_str(cfg.get("canonical_name")) or concept_key
                for alias in [canonical] + ([a for a in aliases_local if isinstance(a, str)] if isinstance(aliases_local, list) else []):
                    for tok in _normalize_label_text(alias).split(" "):
                        if tok:
                            alias_tokens.add(tok)
                for i, rewrite in enumerate(rewrites):
                    if not isinstance(rewrite, dict):
                        continue
                    allowed_rewrite = normalize_to_str(rewrite.get("allowed_rewrite"))
                    if not allowed_rewrite:
                        continue
                    rewrite_tokens = [t for t in _normalize_label_text(allowed_rewrite).split(" ") if t]
                    conflict_tokens = [
                        t
                        for t in rewrite_tokens
                        if t in forbidden and t not in alias_tokens and t not in BASIC_STOP_WORDS
                    ]
                    if conflict_tokens:
                        safe_rewrite_conflicts.append(
                            {
                                "concept": concept_key,
                                "path": f"concepts.{concept_key}.safe_rewrite_guidance[{i}]",
                                "allowed_rewrite": allowed_rewrite,
                                "conflict_tokens": conflict_tokens,
                                "non_fatal": True,
                            }
                        )
            sg = cfg.get("slot_guidance")
            if isinstance(sg, dict) and (sg.get("allowed_internal_tags") or sg.get("allowed_slot_hints")):
                do_not = sg.get("do_not_emit_as_text", [])
                if not isinstance(do_not, list):
                    do_not = []
                for key_src in ("allowed_internal_tags", "allowed_slot_hints"):
                    tags = sg.get(key_src, [])
                    if not isinstance(tags, list):
                        continue
                    for t in tags:
                        if not isinstance(t, str) or not t.strip():
                            continue
                        matched_dn = any(
                            _normalize_label_text(t) == _normalize_label_text(d) for d in do_not if isinstance(d, str)
                        )
                        if not matched_dn:
                            concept_tag_leakage_check_missing.append(
                                {
                                    "concept": concept_key,
                                    "path": f"concepts.{concept_key}.slot_guidance.{key_src}",
                                    "tag": t,
                                    "error": "allowed tag not listed in do_not_emit_as_text (concept_tag_output_leakage_check coverage)",
                                }
                            )
    for alias, owners in alias_map.items():
        owner_concepts = sorted(set([o[0] for o in owners]))
        if len(owner_concepts) > 1:
            alias_conflicts.append(
                {
                    "alias": alias,
                    "concept_a": owner_concepts[0],
                    "concept_b": owner_concepts[1],
                    "locations": [o[1] for o in owners],
                }
            )
    alias_keys = sorted(alias_map.keys(), key=len)
    for i, a in enumerate(alias_keys):
        for b in alias_keys[i + 1 :]:
            if a in b:
                owners_a = sorted(set([x[0] for x in alias_map[a]]))
                owners_b = sorted(set([x[0] for x in alias_map[b]]))
                if owners_a != owners_b:
                    longest_match_risks.append({"short_alias": a, "long_alias": b, "owners_short": owners_a, "owners_long": owners_b})
    return {
        "longest_match_risk_severity": "warning",
        "longest_match_risk_non_blocking": True,
        "concept_count": concept_count,
        "alias_conflict_count": len(alias_conflicts),
        "missing_field_count": len(missing_fields),
        "high_risk_term_not_forbidden_count": len(high_risk_missing),
        "high_risk_term_not_forbidden_count_definition": "number of detail rows in details.high_risk_terms_not_forbidden",
        "overly_strong_statement_count": len(strong_hits),
        "longest_match_risk_count": len(longest_match_risks),
        "visual_form_options_schema_error_count": len(visual_form_options_schema_errors),
        "visual_form_option_emit_policy_missing_count": len(visual_form_option_emit_policy_missing),
        "visual_form_option_leakage_risk_count": len(visual_form_option_leakage_risk),
        "safe_rewrite_conflict_count": len(safe_rewrite_conflicts),
        "default_none_option_missing_count": len(default_none_option_missing),
        "missing_definition_in_retrieval_count": len(missing_definition_in_retrieval),
        "missing_remote_sensing_understanding_in_retrieval_count": len(missing_remote_sensing_understanding_in_retrieval),
        "missing_segmentation_relevance_in_retrieval_count": len(missing_segmentation_relevance_in_retrieval),
        "missing_slot_guidance_in_retrieval_count": len(missing_slot_guidance_in_retrieval),
        "missing_anti_specialization_rules_in_retrieval_count": len(missing_anti_specialization_rules_in_retrieval),
        "missing_safe_rewrite_guidance_in_retrieval_count": len(missing_safe_rewrite_guidance_in_retrieval),
        "missing_visual_form_options_in_retrieval_count": len(missing_visual_form_options_in_retrieval),
        "missing_forbidden_auto_infer_in_retrieval_count": len(missing_forbidden_auto_infer_in_retrieval),
        "retrieval_field_completeness_error_count": len(retrieval_field_completeness_errors),
        "concept_tag_leakage_check_missing_count": len(concept_tag_leakage_check_missing),
        "golden_concept_errors": [
            e
            for e in retrieval_field_completeness_errors
            if str(e.get("concept", "")) in ("water", "bridge")
        ]
        + [
            e
            for e in concept_tag_leakage_check_missing
            if str(e.get("concept", "")) in ("water", "bridge")
        ],
        "details": {
            "missing_fields": missing_fields,
            "alias_conflicts": alias_conflicts,
            "empty_or_invalid_aliases": alias_empty,
            "high_risk_term_not_forbidden": high_risk_missing,
            "high_risk_terms_not_forbidden": high_risk_missing,
            "overly_strong_statements": strong_hits,
            "longest_match_risks": longest_match_risks,
            "visual_form_options_schema_errors": visual_form_options_schema_errors,
            "visual_form_option_emit_policy_missing": visual_form_option_emit_policy_missing,
            "visual_form_option_leakage_risk": visual_form_option_leakage_risk,
            "visual_form_options_leakage_risk": visual_form_option_leakage_risk,
            "safe_rewrite_conflicts": safe_rewrite_conflicts,
            "default_none_option_missing": default_none_option_missing,
            "missing_definition_in_retrieval": missing_definition_in_retrieval,
            "missing_remote_sensing_understanding_in_retrieval": missing_remote_sensing_understanding_in_retrieval,
            "missing_segmentation_relevance_in_retrieval": missing_segmentation_relevance_in_retrieval,
            "missing_slot_guidance_in_retrieval": missing_slot_guidance_in_retrieval,
            "missing_anti_specialization_rules_in_retrieval": missing_anti_specialization_rules_in_retrieval,
            "missing_safe_rewrite_guidance_in_retrieval": missing_safe_rewrite_guidance_in_retrieval,
            "missing_visual_form_options_in_retrieval": missing_visual_form_options_in_retrieval,
            "missing_forbidden_auto_infer_in_retrieval": missing_forbidden_auto_infer_in_retrieval,
            "retrieval_field_completeness_errors": retrieval_field_completeness_errors,
            "concept_tag_leakage_check_missing": concept_tag_leakage_check_missing,
        },
    }


PUBLIC_SEMANTIC_LIBRARY_V17_KEYS = (
    "water",
    "vegetation",
    "farmland",
    "grassland",
    "forest",
    "bare land",
    "river",
    "lake",
    "bridge",
    "road",
    "building",
    "vehicle",
    "ship",
    "airport",
    "parking lot",
    "playground",
    "harbor",
)
PUBLIC_LIBRARY_POLLUTION_TERMS = PROMPT_LEAKAGE_POLLUTION_TERMS
PUBLIC_LIBRARY_BOUNDARY_MAX_CHARS = 200
_PUBLIC_BOUNDARY_REWRITE_PATTERN = re.compile(
    r"^Do not rewrite '[^']+' as .+ unless the raw text explicitly names it\.$"
)

GROUNDING_PRIOR_FORBIDDEN_VALUE_SUBSTRINGS = (
    "boundary",
    "confuser_guard",
    "visual_form_options",
    "forbidden_auto_infer_tokens",
    "mandatory_high_risk_terms",
    "must_put_in_forbidden",
    "do_not_put_in_forbidden",
    "generation_hint",
    "slot_guidance",
    "remote_sensing_understanding",
    "segmentation_relevance",
)
GROUNDING_PRIOR_BANNED_PHRASES = ("appears as", "is always", "must be")
GROUNDING_PRIOR_BANNED_ABS_WORDS = ("always", "must", "definitely")


def concept_public_grounding_prior_structure_check(library: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-only validation for configs/concept_public_semantic_library_v2.json (no retrieval, no API)."""
    failures: List[Dict[str, Any]] = []
    if not isinstance(library, dict):
        failures.append({"case": "library_not_object"})
        return {"concept_public_grounding_prior_structure_check_passed": False, "failures": failures}
    allowed_top = {"library_kind", "version", "match_policy", "concepts"}
    extra_top = set(library.keys()) - allowed_top
    if extra_top:
        failures.append({"case": "unexpected_top_level_keys", "keys": sorted(extra_top)})
    if str(library.get("library_kind", "")).strip() != "public_grounding_prior":
        failures.append({"case": "library_kind", "value": library.get("library_kind")})
    if str(library.get("version", "")).strip() != "public_semantic_v2":
        failures.append({"case": "version", "value": library.get("version")})
    mp = library.get("match_policy")
    if not isinstance(mp, dict):
        failures.append({"case": "match_policy_not_object"})
    else:
        for k, v in GROUNDING_PRIOR_MATCH_POLICY_EXPECTED.items():
            if mp.get(k) != v:
                failures.append({"case": "match_policy_value", "key": k, "expected": v, "actual": mp.get(k)})
        if set(mp.keys()) != set(GROUNDING_PRIOR_MATCH_POLICY_EXPECTED.keys()):
            failures.append(
                {
                    "case": "match_policy_key_set",
                    "expected": sorted(GROUNDING_PRIOR_MATCH_POLICY_EXPECTED.keys()),
                    "actual": sorted(mp.keys()),
                }
            )
    concepts = library.get("concepts")
    if not isinstance(concepts, dict):
        failures.append({"case": "concepts_not_object"})
        return {"concept_public_grounding_prior_structure_check_passed": False, "failures": failures}
    seen = set(concepts.keys())
    expected = set(PUBLIC_SEMANTIC_LIBRARY_V17_KEYS)
    if seen != expected:
        failures.append(
            {
                "case": "concept_key_set_mismatch",
                "missing": sorted(expected - seen),
                "extra": sorted(seen - expected),
            }
        )
    for ck, cfg in concepts.items():
        if not isinstance(cfg, dict):
            failures.append({"case": "entry_not_object", "concept_key": ck})
            continue
        if set(cfg.keys()) != GROUNDING_PRIOR_ENTRY_KEYS:
            failures.append({"case": "entry_key_set", "concept_key": ck, "keys": sorted(cfg.keys())})
            continue
        ctype = str(cfg.get("type", "")).strip()
        if ctype not in {"stuff", "thing_or_facility"}:
            failures.append({"case": "invalid_type", "concept_key": ck, "type": ctype})
        par = cfg.get("parent")
        if ck in ("river", "lake"):
            if not isinstance(par, str) or _normalize_label_text(par) != _normalize_label_text("water"):
                failures.append({"case": "invalid_parent_for_hydro", "concept_key": ck, "parent": par})
        else:
            if par is not None:
                failures.append({"case": "unexpected_parent", "concept_key": ck, "parent": par})
        for fname in ("concept", "type", "visual_evidence", "mask_scope", "exclusion_rule"):
            val = cfg.get(fname)
            if not isinstance(val, str) or not val.strip():
                failures.append({"case": "empty_or_non_string_field", "concept_key": ck, "field": fname})
                continue
            s = val.strip()
            if len(s) > GROUNDING_PRIOR_FIELD_CHAR_MAX:
                failures.append({"case": "field_too_long", "concept_key": ck, "field": fname, "length": len(s)})
            low = s.lower()
            for w in GROUNDING_PRIOR_BANNED_ABS_WORDS:
                if re.search(rf"\b{re.escape(w)}\b", low):
                    failures.append({"case": "banned_absolute_word", "concept_key": ck, "field": fname, "word": w})
            for phrase in GROUNDING_PRIOR_BANNED_PHRASES:
                if phrase in low:
                    failures.append({"case": "banned_morphology_template_phrase", "concept_key": ck, "field": fname, "phrase": phrase})
            for sub in GROUNDING_PRIOR_FORBIDDEN_VALUE_SUBSTRINGS:
                if sub in low:
                    failures.append({"case": "forbidden_internal_name_in_value", "concept_key": ck, "field": fname, "substring": sub})
        blob_cfg = json.dumps(cfg, ensure_ascii=False).lower()
        for fname in PROMPT_LEAKAGE_FORBIDDEN_FIELD_NAMES:
            if fname in blob_cfg:
                failures.append({"case": "forbidden_field_name_in_entry_json", "concept_key": ck, "name": fname})
    return {"concept_public_grounding_prior_structure_check_passed": len(failures) == 0, "failures": failures}


def concept_public_grounding_prior_retrieval_check(
    *,
    public_library_path: Optional[str] = None,
    private_library_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Dry-run retrieval + prompt checks for v2 public grounding prior (no API)."""
    repo_configs = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    v2_path = public_library_path or os.path.join(repo_configs, "concept_public_semantic_library_v2.json")
    priv_path = private_library_path or os.path.join(repo_configs, "concept_semantic_library_verified_7.json")
    lib = load_concept_public_semantic_library(v2_path)
    private_lib = load_concept_semantic_library(priv_path)
    failures: List[Dict[str, Any]] = []
    sr = concept_public_grounding_prior_structure_check(lib)
    if not sr.get("concept_public_grounding_prior_structure_check_passed"):
        failures.append({"case": "structure_prerequisite_failed", "structure": sr})

    def _norm_concepts(raw: str) -> FrozenSet[str]:
        ctx = retrieve_concept_semantics(raw, lib, private_library_for_audit=private_lib)
        out: Set[str] = set()
        for r in ctx.get("matched_concepts") or []:
            if isinstance(r, dict) and isinstance(r.get("concept"), str):
                out.add(_normalize_label_text(r["concept"]))
        return frozenset(out)

    cases = [
        ("find river", frozenset({_normalize_label_text("river")})),
        ("find lake", frozenset({_normalize_label_text("lake")})),
        ("find water", frozenset({_normalize_label_text("water")})),
        ("riverbank", frozenset()),
        (
            "vehicles on the bridge over the river",
            frozenset(
                {
                    _normalize_label_text("vehicle"),
                    _normalize_label_text("bridge"),
                    _normalize_label_text("river"),
                }
            ),
        ),
        (
            "parking lot near airport",
            frozenset({_normalize_label_text("parking lot"), _normalize_label_text("airport")}),
        ),
        ("unknown target", frozenset()),
    ]
    for raw, expected in cases:
        got = _norm_concepts(raw)
        if got != expected:
            failures.append({"case": "retrieval_mismatch", "raw": raw, "expected": sorted(expected), "got": sorted(got)})

    pl = prompt_leakage_self_check(
        raw_expression="find water near bridge",
        public_library_path=v2_path,
        private_library_path=priv_path,
    )
    if not pl.get("prompt_leakage_self_check_passed"):
        failures.append({"case": "prompt_leakage_self_check", "detail": pl})
    cl = concept_public_context_length_check(
        raw_expression="find water near bridge",
        public_library_path=v2_path,
        private_library_path=priv_path,
    )
    if not cl.get("concept_public_context_length_check_passed"):
        failures.append({"case": "concept_public_context_length_check", "detail": cl})

    return {
        "concept_public_grounding_prior_retrieval_check_passed": len(failures) == 0,
        "failures": failures,
        "public_library_path": v2_path,
        "private_library_path": priv_path,
        "structure_check": sr,
        "prompt_leakage_self_check": pl,
        "concept_public_context_length_check": cl,
    }


def concept_public_semantic_library_self_check(library: Dict[str, Any]) -> Dict[str, Any]:
    """Validate hand-written configs/concept_public_semantic_library_v0.json (no API)."""
    failures: List[Dict[str, Any]] = []
    concepts = library.get("concepts", {}) if isinstance(library, dict) else {}
    if not isinstance(concepts, dict):
        failures.append({"case": "concepts_not_object", "failure_reason": "concepts must be an object"})
        return {"concept_public_semantic_library_self_check_passed": False, "failures": failures}
    seen = set(concepts.keys())
    expected = set(PUBLIC_SEMANTIC_LIBRARY_V17_KEYS)
    if seen != expected:
        failures.append(
            {
                "case": "concept_key_set_mismatch",
                "missing": sorted(expected - seen),
                "extra": sorted(seen - expected),
            }
        )
    for ck in PUBLIC_SEMANTIC_LIBRARY_V17_KEYS:
        cfg = concepts.get(ck)
        if not isinstance(cfg, dict):
            failures.append({"case": "entry_not_object", "concept_key": ck})
            continue
        if set(cfg.keys()) != {"concept", "type", "boundary"}:
            failures.append({"case": "entry_key_set", "concept_key": ck, "keys": sorted(cfg.keys())})
            continue
        ctype = str(cfg.get("type", "")).strip()
        if ctype not in {"stuff", "thing_or_facility"}:
            failures.append({"case": "invalid_type", "concept_key": ck, "type": ctype})
        b = str(cfg.get("boundary", "")).strip()
        if len(b) > PUBLIC_LIBRARY_BOUNDARY_MAX_CHARS:
            failures.append({"case": "boundary_too_long", "concept_key": ck, "length": len(b)})
        if not _PUBLIC_BOUNDARY_REWRITE_PATTERN.match(b):
            failures.append({"case": "boundary_not_do_not_rewrite_template", "concept_key": ck, "boundary_preview": b[:120]})
        mq = re.match(r"^Do not rewrite '([^']+)' as ", b)
        if mq and _normalize_label_text(mq.group(1)) != _normalize_label_text(str(cfg.get("concept", ""))):
            failures.append(
                {
                    "case": "boundary_quoted_concept_mismatch",
                    "concept_key": ck,
                    "quoted": mq.group(1),
                    "expected_concept": cfg.get("concept"),
                }
            )
        b_low = b.lower()
        for t in PUBLIC_LIBRARY_POLLUTION_TERMS:
            if t in b_low:
                failures.append({"case": "boundary_pollution_term", "concept_key": ck, "term": t})
        for fname in PROMPT_LEAKAGE_FORBIDDEN_FIELD_NAMES:
            if fname in b_low:
                failures.append({"case": "boundary_contains_private_field_name", "concept_key": ck, "name": fname})
    return {"concept_public_semantic_library_self_check_passed": len(failures) == 0, "failures": failures}


def public_semantic_context_self_check(
    concept_library: Dict[str, Any],
    *,
    vegetation_seed_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Non-API checks for minimal public context built from a **full** semantic library (legacy path)."""
    failures: List[Dict[str, Any]] = []
    concepts = concept_library.get("concepts", {}) if isinstance(concept_library, dict) else {}
    wcfg = concepts.get("water")
    if isinstance(wcfg, dict):
        try:
            pub_w = build_minimal_public_semantic_context({**wcfg, "concept_key": "water"})
        except ValueError as e:
            failures.append({"case": "water_build_minimal_error", "failure_reason": str(e)})
            pub_w = {}
        if pub_w:
            blob_w = json.dumps(pub_w, ensure_ascii=False)
            for w in _STUFF_BOUNDARY_SCRUB_WORDS:
                if w in blob_w.lower():
                    failures.append({"case": "water_morphology_in_public_json", "token": w, "preview": blob_w[:400]})
                    break
            if len(blob_w) > PUBLIC_CONTEXT_JSON_MAX_CHARS:
                failures.append({"case": "water_public_json_len", "concept_key": "water", "length": len(blob_w)})
    else:
        failures.append({"case": "water_concept_missing", "failure_reason": "no water in library"})
    vpath = vegetation_seed_path
    if not vpath:
        vpath = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "configs", "rrsisd_concepts_seed_next.json"))
    if os.path.isfile(vpath):
        doc = read_json(vpath)
        veg_row = next(
            (c for c in (doc.get("concepts") or []) if isinstance(c, dict) and c.get("concept_key") == "vegetation"),
            None,
        )
        if isinstance(veg_row, dict):
            try:
                pub_v = build_minimal_public_semantic_context(
                    {
                        "concept_key": "vegetation",
                        "canonical_name": veg_row.get("canonical_name", "vegetation"),
                        "concept_type": veg_row.get("concept_type", "stuff"),
                        "anti_specialization_rules": ["Do not add tree or crop subclasses unless stated in raw text."],
                    }
                )
            except ValueError as e:
                failures.append({"case": "vegetation_build_minimal_error", "failure_reason": str(e)})
                pub_v = {}
            if pub_v:
                if set(pub_v.keys()) != {"concept", "type", "boundary"}:
                    failures.append({"case": "vegetation_public_keyset", "keys": sorted(pub_v.keys())})
                blob_v = json.dumps(pub_v, ensure_ascii=False).lower()
                if "visual_form_options" in blob_v or "forbidden_auto_infer_tokens" in blob_v:
                    failures.append({"case": "vegetation_private_name_in_public_blob", "preview": blob_v[:240]})
        else:
            failures.append({"case": "vegetation_seed_row_missing", "path": vpath})
    else:
        failures.append({"case": "vegetation_seed_path_missing", "path": vpath})
    bcfg = concepts.get("bridge")
    if isinstance(bcfg, dict):
        try:
            pub_b = build_minimal_public_semantic_context({**bcfg, "concept_key": "bridge"})
        except ValueError as e:
            failures.append({"case": "bridge_build_minimal_error", "failure_reason": str(e)})
            pub_b = {}
        if pub_b:
            if set(pub_b.keys()) != {"concept", "type", "boundary"}:
                failures.append({"case": "bridge_public_keyset", "keys": sorted(pub_b.keys())})
            blob_b = json.dumps(pub_b, ensure_ascii=False)
            if len(blob_b) > PUBLIC_CONTEXT_JSON_MAX_CHARS:
                failures.append({"case": "bridge_public_json_len", "length": len(blob_b)})
    else:
        failures.append({"case": "bridge_concept_missing", "failure_reason": "no bridge in library"})
    return {"public_semantic_context_self_check_passed": len(failures) == 0, "failures": failures}


def concept_semantic_self_check(concept_library: Dict[str, Any]) -> Dict[str, Any]:
    failed_cases: List[Dict[str, Any]] = []

    def _record_case(
        case_name: str,
        raw: str,
        enhanced: str,
        expected: str,
        actual: str,
        failure_reason: str,
        *,
        exception_type: Optional[str] = None,
        exception_message: Optional[str] = None,
        traceback_tail: Optional[str] = None,
    ) -> None:
        row: Dict[str, Any] = {
            "case_name": case_name,
            "raw": raw,
            "enhanced": enhanced,
            "expected": expected,
            "actual": actual,
            "failure_reason": failure_reason,
        }
        if exception_type:
            row["exception_type"] = exception_type
        if exception_message:
            row["exception_message"] = exception_message
        if traceback_tail:
            row["traceback_tail"] = traceback_tail
        failed_cases.append(row)

    def _run_case(case_name: str, raw: str, enhanced: str, expected: str) -> None:
        try:
            ctx = retrieve_concept_semantics(raw, concept_library)
            ok_inj, reason_inj = concept_semantic_injection_check(raw, enhanced, enhanced, ctx)
            ok_vf, reason_vf = visual_form_option_leakage_check(raw, enhanced, enhanced, ctx)
            ok_tag, reason_tag = concept_tag_output_leakage_check(raw, enhanced, enhanced, ctx)
            passed = bool(ok_inj and ok_vf and ok_tag)
            actual = "pass" if passed else "failed_postcheck"
            expect_pass = expected == "pass"
            if passed != expect_pass:
                reasons = []
                if not ok_inj:
                    reasons.append(reason_inj)
                if not ok_vf:
                    reasons.append(reason_vf)
                if not ok_tag:
                    reasons.append(reason_tag)
                _record_case(
                    case_name,
                    raw,
                    enhanced,
                    expected,
                    actual,
                    "; ".join([r for r in reasons if r]) or "unexpected check result",
                )
        except Exception as e:
            tb_tail = "\n".join(traceback.format_exc().splitlines()[-8:])
            _record_case(
                case_name,
                raw,
                enhanced,
                expected,
                "exception",
                f"{type(e).__name__}: {str(e)}",
                exception_type=type(e).__name__,
                exception_message=str(e),
                traceback_tail=tb_tail,
            )

    concepts = concept_library.get("concepts", {}) if isinstance(concept_library, dict) else {}
    if not isinstance(concepts, dict):
        _record_case(
            "library_shape",
            "",
            "",
            "pass",
            "failed_postcheck",
            "concept_library.concepts is not an object",
        )
    else:
        for ck in ("water", "bridge"):
            if ck not in concepts:
                _record_case(
                    f"required_concept_{ck}",
                    f"find {ck}",
                    "",
                    "pass",
                    "failed_postcheck",
                    f"required concept missing: {ck}",
                )
        retrieval_core_fields = (
            "definition",
            "remote_sensing_understanding",
            "visual_form_options",
            "segmentation_relevance",
            "safe_rewrite_guidance",
            "slot_guidance",
            "forbidden_auto_infer_tokens",
            "anti_specialization_rules",
        )
        for ck in ("water", "bridge"):
            ctx_rb = retrieve_concept_semantics(f"find {ck}", concept_library)
            full_rows = ctx_rb.get("matched_concepts_full") or ctx_rb.get("matched_concepts", [])
            row = {c["concept_key"]: c for c in full_rows if isinstance(c, dict)}.get(ck)
            if row is None:
                _record_case(
                    f"retrieval_presence_{ck}",
                    f"find {ck}",
                    "",
                    "pass",
                    "failed_postcheck",
                    f"retrieve_concept_semantics did not return {ck}",
                )
                continue
            for f in retrieval_core_fields:
                if f not in row or row.get(f) in (None, "", [], {}):
                    _record_case(
                        f"retrieval_field_{ck}_{f}",
                        f"find {ck}",
                        "",
                        "pass",
                        "failed_postcheck",
                        f"missing or empty retrieval field: {f}",
                    )

    # Case-level checks (replaces deprecated assertion logic).
    _run_case("water_safe", "find water", "the water body", "pass")
    _run_case("water_visual_form_leak", "find water", "the winding small river", "failed_postcheck")
    water_ctx_for_tag = retrieve_concept_semantics("find water", concept_library)
    wfull_rows = water_ctx_for_tag.get("matched_concepts_full") or water_ctx_for_tag.get("matched_concepts", [])
    water_row_for_tag = {c.get("concept_key"): c for c in wfull_rows if isinstance(c, dict)}.get("water", {})
    slot_guidance = water_row_for_tag.get("slot_guidance", {}) if isinstance(water_row_for_tag, dict) else {}
    candidate_tags = []
    if isinstance(slot_guidance, dict):
        for src in ("allowed_internal_tags", "allowed_slot_hints"):
            vals = slot_guidance.get(src, [])
            if isinstance(vals, list):
                candidate_tags.extend([v for v in vals if isinstance(v, str) and v.strip()])
    if candidate_tags:
        t = candidate_tags[0]
        _run_case("water_internal_tag_leak", "find water", f"the {t}", "failed_postcheck")
    _run_case("bridge_safe", "find bridge", "the bridge", "pass")
    _run_case("bridge_reference_leak", "find bridge", "the bridge over water", "failed_postcheck")
    _run_case("bridge_over_water_allowed", "find bridge over water", "the bridge over water", "pass")

    return {
        "concept_semantic_self_check_passed": len(failed_cases) == 0,
        "failed_cases": failed_cases,
    }


def validator_self_check() -> bool:
    # Category normalization (dataset id vs natural phrase)
    assert normalize_category_text("baseball field") == normalize_category_text("baseballfield")
    assert category_text_match("baseballfield", "baseball field")
    assert category_text_match("groundtrackfield", "ground track field")
    assert category_text_match("tenniscourt", "tennis court")
    assert category_text_match("basketballcourt", "basketball court")
    assert category_text_match("trainstation", "train station")
    assert category_text_match("storagetank", "storage tank")
    assert category_text_match("windmill", "windmill")
    assert category_text_match("vehicle", "vehicle")
    assert category_text_match("airplane", "airplane")
    assert not category_text_match("vehicle", "car")
    assert not category_text_match("ship", "boat")
    assert not category_text_match("airplane", "aircraft")

    # Absolute positions
    assert normalize_absolute_position("lower left") == "lower_left"
    assert normalize_absolute_position("middle") == "center"
    assert normalize_absolute_position("upper-left") == "upper_left"

    # Spatial: position tokens move to absolute; relation synonyms map
    rel, abs_pos, invalid = normalize_spatial_relation("lower left")
    assert rel is None and abs_pos == "lower_left" and not invalid
    rel, abs_pos, invalid = normalize_spatial_relation("next to")
    assert rel == "beside" and abs_pos is None and not invalid

    # Added-token reasons
    assert normalize_added_token_reason("grammar completion") == "grammar_only"
    assert normalize_added_token_reason("copied from raw") == "copied_from_raw"
    assert normalize_added_token_reason("visual inference") == "__forbidden__"

    def _minimal_llm_out(**kwargs: Any) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "raw_expression": "",
            "slots": {
                "target_category": None,
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            "explicitization": {"operations": [], "added_tokens": [], "added_token_justification": []},
            "enhanced_expression": "",
            "compressed_expression": "",
            "status": "success",
            "notes": "",
        }
        base.update(kwargs)
        return base

    # 1 baseballfield + raw phrase
    ok, _, n = validate_and_normalize_output(
        "The baseball field on the top",
        _minimal_llm_out(
            raw_expression="The baseball field on the top",
            slots={
                "target_category": "baseball field",
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": ["top"],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The baseball field on the top",
            compressed_expression="The baseball field on the top",
        ),
        category_name="baseballfield",
    )
    assert ok and n["slots"]["target_category"] == "baseballfield"

    # 2 tenniscourt + lower left
    ok, _, n = validate_and_normalize_output(
        "A tennis court on the lower left",
        _minimal_llm_out(
            raw_expression="A tennis court on the lower left",
            slots={
                "target_category": "tennis court",
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": ["lower_left"],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="A tennis court on the lower left",
            compressed_expression="A tennis court on the lower left",
        ),
        category_name="tenniscourt",
    )
    assert ok and n["slots"]["target_category"] == "tenniscourt"

    # 3 groundtrackfield + color words present in raw
    ok, _, n = validate_and_normalize_output(
        "The yellow and orange ground track field on the left",
        _minimal_llm_out(
            raw_expression="The yellow and orange ground track field on the left",
            slots={
                "target_category": "ground track field",
                "attributes": ["yellow", "orange"],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": ["left"],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The yellow and orange ground track field on the left",
            compressed_expression="The yellow and orange ground track field on the left",
        ),
        category_name="groundtrackfield",
    )
    assert ok and n["slots"]["target_category"] == "groundtrackfield"

    # 4 absolute_positions list normalization
    ok, _, n = validate_and_normalize_output(
        "x on the lower left",
        _minimal_llm_out(
            raw_expression="x on the lower left",
            slots={
                "target_category": None,
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": ["lower left"],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="x on the lower left",
            compressed_expression="x on the lower left",
        ),
        category_name=None,
    )
    assert ok and "lower_left" in n["slots"]["absolute_positions"]

    # 5 middle -> center + raw has middle
    ok, _, n = validate_and_normalize_output(
        "object in the middle",
        _minimal_llm_out(
            raw_expression="object in the middle",
            slots={
                "target_category": None,
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": ["middle"],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="object in the middle",
            compressed_expression="object in the middle",
        ),
        category_name=None,
    )
    assert ok and "center" in n["slots"]["absolute_positions"]

    # 6 spatial lower left -> absolute_positions, spatial empty
    ok, _, n = validate_and_normalize_output(
        "obj lower left area",
        _minimal_llm_out(
            raw_expression="obj lower left area",
            slots={
                "target_category": None,
                "attributes": [],
                "spatial_relations": ["lower left"],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="obj lower left area",
            compressed_expression="obj lower left area",
        ),
        category_name=None,
    )
    assert ok and "lower_left" in n["slots"]["absolute_positions"] and n["slots"]["spatial_relations"] == []

    # 7 next to -> beside
    ok, _, n = validate_and_normalize_output(
        "a near b next to c",
        _minimal_llm_out(
            raw_expression="a near b next to c",
            slots={
                "target_category": None,
                "attributes": [],
                "spatial_relations": ["next to"],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="a near b next to c",
            compressed_expression="a near b next to c",
        ),
        category_name=None,
    )
    assert ok and n["slots"]["spatial_relations"] == ["beside"]

    # 8-9 reasons via full validate path
    ok, _, n = validate_and_normalize_output(
        "raw text",
        _minimal_llm_out(
            raw_expression="raw text",
            slots={
                "target_category": None,
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            explicitization={
                "operations": [],
                "added_tokens": ["the"],
                "added_token_justification": [{"token": "the", "reason": "grammar completion"}],
            },
            enhanced_expression="raw text",
            compressed_expression="raw text",
        ),
        category_name=None,
    )
    assert ok and n["explicitization"]["added_token_justification"][0]["reason"] == "grammar_only"

    ok, _, n = validate_and_normalize_output(
        "raw text",
        _minimal_llm_out(
            raw_expression="raw text",
            slots={
                "target_category": None,
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            explicitization={
                "operations": [],
                "added_tokens": ["raw"],
                "added_token_justification": [{"token": "raw", "reason": "copied from raw"}],
            },
            enhanced_expression="raw text",
            compressed_expression="raw text",
        ),
        category_name=None,
    )
    assert ok and n["explicitization"]["added_token_justification"][0]["reason"] == "copied_from_raw"

    # 10 visual inference must fail
    ok, reason, _ = validate_and_normalize_output(
        "raw text",
        _minimal_llm_out(
            raw_expression="raw text",
            slots={
                "target_category": None,
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            explicitization={
                "operations": [],
                "added_tokens": ["x"],
                "added_token_justification": [{"token": "x", "reason": "visual inference"}],
            },
            enhanced_expression="raw text",
            compressed_expression="raw text",
        ),
        category_name=None,
    )
    assert not ok and "forbidden" in reason

    # 11 quantity "" -> unknown
    ok, _, n = validate_and_normalize_output(
        "word",
        _minimal_llm_out(
            raw_expression="word",
            slots={
                "target_category": None,
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "",
                "clean_label_valid": True,
            },
            enhanced_expression="word",
            compressed_expression="word",
        ),
        category_name=None,
    )
    assert ok and n["slots"]["quantity"] == "unknown"

    # 12 target_category null + raw contains category_name -> fallback + notes
    ok, _, n = validate_and_normalize_output(
        "The baseball field on the top",
        _minimal_llm_out(
            raw_expression="The baseball field on the top",
            slots={
                "target_category": None,
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": ["top"],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The baseball field on the top",
            compressed_expression="The baseball field on the top",
        ),
        category_name="baseballfield",
    )
    assert ok and n["slots"]["target_category"] == "baseballfield"
    notes_joined = "; ".join(n.get("_normalization_notes", []))
    assert "fallback target_category from dataset category_name" in notes_joined

    # Empty added_tokens + empty justification must pass
    ok, _, _ = validate_and_normalize_output(
        "plain",
        _minimal_llm_out(
            raw_expression="plain",
            enhanced_expression="plain",
            compressed_expression="plain",
        ),
        category_name=None,
    )
    assert ok

    # Null / blank entries in spatial_relations are dropped, not hard-fail
    ok, _, n = validate_and_normalize_output(
        "a near b",
        _minimal_llm_out(
            raw_expression="a near b",
            slots={
                "target_category": None,
                "attributes": [],
                "spatial_relations": [None, "", "  ", "near"],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="a near b",
            compressed_expression="a near b",
        ),
        category_name=None,
    )
    assert ok and "near" in n["slots"]["spatial_relations"]

    # V4.1: "at/in" in spatial_relations are residual prepositions, not enum hard failures.
    ok, _, n = validate_and_normalize_output(
        "A baseball field at the bottom",
        _minimal_llm_out(
            raw_expression="A baseball field at the bottom",
            slots={
                "target_category": "baseball field",
                "attributes": [],
                "spatial_relations": ["at"],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="A baseball field at the bottom",
            compressed_expression="A baseball field at the bottom",
        ),
        category_name="baseballfield",
    )
    assert ok and "bottom" in n["slots"]["absolute_positions"] and "at" not in n["slots"]["spatial_relations"]

    ok, _, n = validate_and_normalize_output(
        "A small slender train station in the middle",
        _minimal_llm_out(
            raw_expression="A small slender train station in the middle",
            slots={
                "target_category": "train station",
                "attributes": ["small"],
                "spatial_relations": ["in"],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="A small slender train station in the middle",
            compressed_expression="A small slender train station in the middle",
        ),
        category_name="trainstation",
    )
    assert ok and "center" in n["slots"]["absolute_positions"] and "in" not in n["slots"]["spatial_relations"]

    ok, _, n = validate_and_normalize_output(
        "The expressway toll station in the middle",
        _minimal_llm_out(
            raw_expression="The expressway toll station in the middle",
            slots={
                "target_category": "station",
                "attributes": [],
                "spatial_relations": ["in"],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The expressway toll station in the middle",
            compressed_expression="The expressway toll station in the middle",
        ),
        category_name=None,
    )
    assert ok and "center" in n["slots"]["absolute_positions"]

    # Invalid raw_llm_status should not force failed_postcheck when content passes.
    ok, _, n = validate_and_normalize_output(
        "The tiny dam at the bottom",
        _minimal_llm_out(
            raw_expression="The tiny dam at the bottom",
            status="valid",
            slots={
                "target_category": None,
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": ["bottom"],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The tiny dam at the bottom",
            compressed_expression="The tiny dam at the bottom",
        ),
        category_name=None,
    )
    assert ok and n["raw_llm_status_invalid"] is True and decide_pipeline_status("The tiny dam at the bottom", n) != "failed_postcheck"

    ok, _, n = validate_and_normalize_output(
        "The airport in the middle",
        _minimal_llm_out(
            raw_expression="The airport in the middle",
            status="no_change",
            slots={
                "target_category": None,
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": ["middle"],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The airport in the middle",
            compressed_expression="The airport in the middle",
        ),
        category_name=None,
    )
    assert ok and n["raw_llm_status_invalid"] is True and decide_pipeline_status("The airport in the middle", n) != "failed_postcheck"

    # High-risk blockers must remain strict.
    ok, _, _ = validate_and_normalize_output(
        "vehicle on road",
        _minimal_llm_out(
            raw_expression="vehicle on road",
            slots={
                "target_category": "car",
                "attributes": [],
                "spatial_relations": ["on"],
                "reference_objects": ["road"],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="vehicle on road",
            compressed_expression="vehicle on road",
        ),
        category_name="vehicle",
    )
    assert not ok

    ok, _, _ = validate_and_normalize_output(
        "airplane near runway",
        _minimal_llm_out(
            raw_expression="airplane near runway",
            slots={
                "target_category": "airplane",
                "attributes": ["white"],
                "spatial_relations": ["near"],
                "reference_objects": ["runway"],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="white airplane near runway",
            compressed_expression="white airplane near runway",
        ),
        category_name="airplane",
    )
    assert not ok

    # V4.2 complex spatial phrases + reference objects
    ok, _, n = validate_and_normalize_output(
        "The baseball field is on the left of the orange and red ground track field in the middle",
        _minimal_llm_out(
            raw_expression="The baseball field is on the left of the orange and red ground track field in the middle",
            slots={
                "target_category": "baseball field",
                "attributes": [],
                "spatial_relations": ["on the left of", "in the middle"],
                "reference_objects": ["orange and red ground track field"],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The baseball field is on the left of the orange and red ground track field in the middle",
            compressed_expression="The baseball field is on the left of the orange and red ground track field in the middle",
        ),
        category_name="baseballfield",
    )
    assert ok and "left_of" in n["slots"]["spatial_relations"] and "center" in n["slots"]["absolute_positions"]

    ok, _, n = validate_and_normalize_output(
        "The basketball court is on the right of the tennis court in the middle",
        _minimal_llm_out(
            raw_expression="The basketball court is on the right of the tennis court in the middle",
            slots={
                "target_category": "basketball court",
                "attributes": [],
                "spatial_relations": ["on the right of", "in the middle"],
                "reference_objects": ["tennis court"],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The basketball court is on the right of the tennis court in the middle",
            compressed_expression="The basketball court is on the right of the tennis court in the middle",
        ),
        category_name="basketballcourt",
    )
    assert ok and "right_of" in n["slots"]["spatial_relations"] and "center" in n["slots"]["absolute_positions"]

    ok, _, n = validate_and_normalize_output(
        "The bridge is on the upper left of the large green golf field",
        _minimal_llm_out(
            raw_expression="The bridge is on the upper left of the large green golf field",
            slots={
                "target_category": "bridge",
                "attributes": [],
                "spatial_relations": ["on the upper left"],
                "reference_objects": ["large green golf field"],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The bridge is on the upper left of the large green golf field",
            compressed_expression="The bridge is on the upper left of the large green golf field",
        ),
        category_name=None,
    )
    assert ok and "upper_left" in n["slots"]["absolute_positions"]

    ok, _, n = validate_and_normalize_output(
        "A vehicle is on the right of the tiny expressway toll station at the bottom",
        _minimal_llm_out(
            raw_expression="A vehicle is on the right of the tiny expressway toll station at the bottom",
            slots={
                "target_category": "vehicle",
                "attributes": [],
                "spatial_relations": ["on the right of"],
                "reference_objects": ["tiny expressway toll station"],
                "absolute_positions": ["bottom"],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="A vehicle is on the right of the tiny expressway toll station at the bottom",
            compressed_expression="A vehicle is on the right of the tiny expressway toll station at the bottom",
        ),
        category_name="vehicle",
    )
    assert ok and "right_of" in n["slots"]["spatial_relations"] and "bottom" in n["slots"]["absolute_positions"]

    ok, _, n = validate_and_normalize_output(
        "The vehicle is a little smaller than the vehicle in the middle",
        _minimal_llm_out(
            raw_expression="The vehicle is a little smaller than the vehicle in the middle",
            slots={
                "target_category": "vehicle",
                "attributes": [],
                "spatial_relations": ["smaller than", "in the middle"],
                "reference_objects": ["vehicle"],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The vehicle is a little smaller than the vehicle in the middle",
            compressed_expression="The vehicle is a little smaller than the vehicle in the middle",
        ),
        category_name="vehicle",
    )
    assert ok and "smaller_than" in n["slots"]["spatial_relations"] and "center" in n["slots"]["absolute_positions"]

    # Category normalization with hyphen/space
    ok, _, _ = validate_and_normalize_output(
        "The large expressway toll station",
        _minimal_llm_out(
            raw_expression="The large expressway toll station",
            slots={
                "target_category": "expressway toll station",
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The large expressway toll station",
            compressed_expression="The large expressway toll station",
        ),
        category_name="Expressway-toll-station",
    )
    assert ok
    ok, _, _ = validate_and_normalize_output(
        "The large frustum of a cone chimney",
        _minimal_llm_out(
            raw_expression="The large frustum of a cone chimney",
            slots={
                "target_category": "chimney",
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The large frustum of a cone chimney",
            compressed_expression="The large frustum of a cone chimney",
        ),
        category_name="chimney",
    )
    assert ok
    ok, _, _ = validate_and_normalize_output(
        "The stadium has the baseball field at the bottom",
        _minimal_llm_out(
            raw_expression="The stadium has the baseball field at the bottom",
            slots={
                "target_category": "stadium",
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": ["baseball field"],
                "absolute_positions": ["bottom"],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The stadium has the baseball field at the bottom",
            compressed_expression="The stadium has the baseball field at the bottom",
        ),
        category_name="stadium",
    )
    assert ok

    # Strict reference-object synonym blockers
    ok, _, _ = validate_and_normalize_output(
        "A vehicle on the road",
        _minimal_llm_out(
            raw_expression="A vehicle on the road",
            slots={
                "target_category": "vehicle",
                "attributes": [],
                "spatial_relations": ["on"],
                "reference_objects": ["car"],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="A vehicle on the road",
            compressed_expression="A vehicle on the road",
        ),
        category_name="vehicle",
    )
    assert not ok
    ok, _, _ = validate_and_normalize_output(
        "A ship near port",
        _minimal_llm_out(
            raw_expression="A ship near port",
            slots={
                "target_category": "ship",
                "attributes": [],
                "spatial_relations": ["near"],
                "reference_objects": ["boat"],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="A ship near port",
            compressed_expression="A ship near port",
        ),
        category_name="ship",
    )
    assert not ok

    # Enhanced adds color absent in raw must still fail
    ok, _, _ = validate_and_normalize_output(
        "airplane near runway",
        _minimal_llm_out(
            raw_expression="airplane near runway",
            slots={
                "target_category": "airplane",
                "attributes": [],
                "spatial_relations": ["near"],
                "reference_objects": ["runway"],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="white airplane near runway",
            compressed_expression="white airplane near runway",
        ),
        category_name="airplane",
    )
    assert not ok

    # Zero-delta candidate should pass as unchanged after normalization.
    ok, _, n = validate_and_normalize_output(
        "The baseball field is on the left of the orange and red ground track field in the middle",
        _minimal_llm_out(
            raw_expression="The baseball field is on the left of the orange and red ground track field in the middle",
            slots={
                "target_category": "baseball field",
                "attributes": [],
                "spatial_relations": ["on the left of", "in the middle"],
                "reference_objects": ["orange and red ground track field"],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            explicitization={"operations": [], "added_tokens": [], "added_token_justification": []},
            enhanced_expression="The baseball field is on the left of the orange and red ground track field in the middle",
            compressed_expression="The baseball field is on the left of the orange and red ground track field in the middle",
        ),
        category_name="baseballfield",
    )
    assert ok and decide_pipeline_status(
        "The baseball field is on the left of the orange and red ground track field in the middle", n
    ) == "unchanged"

    # SKB v1.1 text isolation checks
    ok, reason, _ = validate_and_normalize_output(
        "The large basketball court",
        _minimal_llm_out(
            raw_expression="The large basketball court",
            slots={
                "target_category": "basketball court",
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The basketball court",
            compressed_expression="The basketball court",
        ),
        category_name="basketballcourt",
    )
    assert not ok and "raw visual word removed: large" in reason

    ok, reason, _ = validate_and_normalize_output(
        "The large expressway toll station",
        _minimal_llm_out(
            raw_expression="The large expressway toll station",
            slots={
                "target_category": "expressway toll station",
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The expressway toll station",
            compressed_expression="The expressway toll station",
        ),
        category_name="Expressway-toll-station",
    )
    assert not ok and "raw visual word removed: large" in reason

    ok, reason, _ = validate_and_normalize_output(
        "A windmill on the lower left",
        _minimal_llm_out(
            raw_expression="A windmill on the lower left",
            slots={
                "target_category": "windmill",
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": ["lower_left"],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The windmill is on the lower left",
            compressed_expression="The windmill is on the lower left",
        ),
        category_name="windmill",
    )
    assert not ok and "A->The" in reason

    ok, reason, _ = validate_and_normalize_output(
        "A vehicle is on the left of the small blue and gray chimney",
        _minimal_llm_out(
            raw_expression="A vehicle is on the left of the small blue and gray chimney",
            slots={
                "target_category": "vehicle",
                "attributes": ["small", "blue", "gray"],
                "spatial_relations": ["left_of"],
                "reference_objects": ["chimney"],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="A vehicle is on the left of the small blue and gray chimney",
            compressed_expression="A vehicle is left_of the small blue and gray chimney",
        ),
        category_name="vehicle",
    )
    assert not ok and "slot label leaked" in reason

    ok, _, _ = validate_and_normalize_output(
        "The airport in the middle",
        _minimal_llm_out(
            raw_expression="The airport in the middle",
            slots={
                "target_category": "airport",
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": ["center"],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The airport in the middle",
            compressed_expression="The airport in the middle",
        ),
        category_name="airport",
    )
    assert ok

    ok, _, _ = validate_and_normalize_output(
        "The basketball court is on the right of the tennis court in the middle",
        _minimal_llm_out(
            raw_expression="The basketball court is on the right of the tennis court in the middle",
            slots={
                "target_category": "basketball court",
                "attributes": [],
                "spatial_relations": ["right_of"],
                "reference_objects": ["tennis court"],
                "absolute_positions": ["center"],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The basketball court is on the right of the tennis court in the middle",
            compressed_expression="The basketball court is on the right of the tennis court in the middle",
        ),
        category_name="basketballcourt",
    )
    assert ok

    ok, _, _ = validate_and_normalize_output(
        "A oval ground track field",
        _minimal_llm_out(
            raw_expression="A oval ground track field",
            slots={
                "target_category": "ground track field",
                "attributes": ["oval"],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="An oval ground track field",
            compressed_expression="An oval ground track field",
        ),
        category_name="groundtrackfield",
    )
    assert ok

    ok, _, _ = validate_and_normalize_output(
        "A windmill on the lower left",
        _minimal_llm_out(
            raw_expression="A windmill on the lower left",
            slots={
                "target_category": "windmill",
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": ["lower_left"],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="A windmill is on the lower left",
            compressed_expression="A windmill is on the lower left",
        ),
        category_name="windmill",
    )
    assert ok

    sc_summary = {
        "category_context_used": {"dataset_category_name": "vehicle"},
        "matched_spatial_phrases": [{"raw_phrase": "on the right of", "normalized": "right_of"}],
        "forbidden_replacements_used": {"vehicle": ["car", "truck", "bus"]},
    }
    failed_post = fallback_failed_record(
        {"raw_expression": "x", "expr_id": "e1"},
        "failed_postcheck",
        "Reason for failure: test",
        semantic_context_summary=sc_summary,
    )
    failed_api = fallback_failed_record(
        {"raw_expression": "x", "expr_id": "e2"},
        "failed_api_error",
        "Reason for failure: test",
        semantic_context_summary=sc_summary,
    )
    assert failed_post.get("semantic_context_summary") is not None
    assert failed_api.get("semantic_context_summary") is not None

    # SKB v1.1.1 fixes
    ok, _, _ = validate_and_normalize_output(
        "A large frustum of a cone chimney",
        _minimal_llm_out(
            raw_expression="A large frustum of a cone chimney",
            slots={
                "target_category": "chimney",
                "attributes": ["large", "frustum of a cone"],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="A large frustum of a cone chimney",
            compressed_expression="A large frustum of a cone chimney",
        ),
        category_name="chimney",
    )
    assert ok

    ok, _, _ = validate_and_normalize_output(
        "A small frustum of a cone chimney on the left",
        _minimal_llm_out(
            raw_expression="A small frustum of a cone chimney on the left",
            slots={
                "target_category": "chimney",
                "attributes": ["small", "frustum of a cone"],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": ["left"],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="A small frustum of a cone chimney on the left",
            compressed_expression="A small frustum of a cone chimney on the left",
        ),
        category_name="chimney",
    )
    assert ok

    ok, reason, _ = validate_and_normalize_output(
        "The vehicle is a little smaller than the vehicle in the middle",
        _minimal_llm_out(
            raw_expression="The vehicle is a little smaller than the vehicle in the middle",
            slots={
                "target_category": "vehicle",
                "attributes": [],
                "spatial_relations": ["smaller than"],
                "reference_objects": ["vehicle"],
                "absolute_positions": ["center"],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The vehicle is smaller than the vehicle in the middle",
            compressed_expression="The vehicle is smaller than the vehicle in the middle",
        ),
        category_name="vehicle",
    )
    assert ok, reason

    ok, reason, _ = validate_and_normalize_output(
        "The little vehicle",
        _minimal_llm_out(
            raw_expression="The little vehicle",
            slots={
                "target_category": "vehicle",
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="The vehicle",
            compressed_expression="The vehicle",
        ),
        category_name="vehicle",
    )
    assert not ok and "raw visual word removed: little" in reason

    ok, reason, _ = validate_and_normalize_output(
        "A little storage tank",
        _minimal_llm_out(
            raw_expression="A little storage tank",
            slots={
                "target_category": "storage tank",
                "attributes": [],
                "spatial_relations": [],
                "reference_objects": [],
                "absolute_positions": [],
                "quantity": "unknown",
                "clean_label_valid": True,
            },
            enhanced_expression="A storage tank",
            compressed_expression="A storage tank",
        ),
        category_name="storagetank",
    )
    assert not ok and "raw visual word removed: little" in reason

    return True


def _cleanup_output_targets(output_dir: str, targets: List[str], dirs: List[str]) -> None:
    for path in targets:
        if os.path.exists(path):
            os.remove(path)
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(output_dir, exist_ok=True)


def _prepare_process_output_dir(output_dir: str, resume: bool, overwrite: bool) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume-from-checkpoint are mutually exclusive.")

    result_jsonl_path = os.path.join(output_dir, "intermediate", "enhancement_results.jsonl")
    failed_cases_path = os.path.join(output_dir, "failed_cases.jsonl")
    log_path = os.path.join(output_dir, "enhancement_log.jsonl")
    summary_path = os.path.join(output_dir, "summary_report.json")
    enhanced_training_path = os.path.join(output_dir, "enhanced_training.json")
    adapter_path = os.path.join(output_dir, "data_adapter.py")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    intermediate_dir = os.path.join(output_dir, "intermediate")

    os.makedirs(output_dir, exist_ok=True)
    existing_artifacts = any(
        os.path.exists(p)
        for p in [
            result_jsonl_path,
            failed_cases_path,
            log_path,
            summary_path,
            enhanced_training_path,
            adapter_path,
            checkpoint_dir,
        ]
    )

    if overwrite:
        _cleanup_output_targets(
            output_dir,
            [failed_cases_path, log_path, summary_path, enhanced_training_path, adapter_path],
            [checkpoint_dir, intermediate_dir],
        )
    elif (not resume) and existing_artifacts:
        raise ValueError(
            "Output directory already contains process artifacts. Use --overwrite to reset, or --resume-from-checkpoint."
        )

    os.makedirs(intermediate_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)


def _load_written_expr_ids_from_jsonl(path: str) -> Set[str]:
    out: Set[str] = set()
    if not os.path.exists(path):
        return out
    for row in read_jsonl(path):
        expr_id = row.get("expr_id")
        if isinstance(expr_id, str):
            out.add(expr_id)
    return out


def _unit_expr_id(unit: Dict[str, Any], idx: int) -> str:
    expr_id = normalize_to_str(unit.get("expr_id"))
    if expr_id:
        return expr_id
    return f"unit_{idx}__h_{sha1_8(json.dumps(unit, ensure_ascii=False, sort_keys=True))}"


def save_checkpoint(
    checkpoint_dir: str,
    last_processed_index: int,
    processed_expr_ids: Iterable[str],
    partial_jsonl_path: str,
    failed_cases_path: str,
    config: Dict[str, Any],
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"enhancement_ckpt_{last_processed_index:06d}.json")
    payload = {
        "last_processed_index": last_processed_index,
        "processed_expr_ids": sorted(set(processed_expr_ids)),
        "partial_jsonl_path": partial_jsonl_path,
        "failed_cases_path": failed_cases_path,
        "config": config,
        "timestamp": utcnow(),
    }
    write_json(path, payload)
    return path


def load_resume_state(checkpoint_dir: str, result_jsonl_path: str) -> Set[str]:
    processed: Set[str] = set()
    if os.path.exists(checkpoint_dir):
        ck_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("enhancement_ckpt_") and f.endswith(".json")])
        if ck_files:
            latest = read_json(os.path.join(checkpoint_dir, ck_files[-1]))
            for expr_id in latest.get("processed_expr_ids", []):
                processed.add(str(expr_id))
    if os.path.exists(result_jsonl_path):
        for row in read_jsonl(result_jsonl_path):
            expr_id = row.get("expr_id")
            if isinstance(expr_id, str):
                processed.add(expr_id)
    return processed


def aggregate_to_training_json(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # preserve expression-level metadata as primary payload
    records = []
    for r in rows:
        records.append(
            {
                "expr_id": r.get("expr_id"),
                "ref_id": r.get("ref_id"),
                "ann_id": r.get("ann_id"),
                "image_id": r.get("image_id"),
                "file_name": r.get("file_name"),
                "split": r.get("split"),
                "category_id": r.get("category_id"),
                "category_name": r.get("category_name"),
                "bbox": r.get("bbox"),
                "segmentation_ref": r.get("segmentation_ref"),
                "raw": r.get("raw"),
                "enhanced": r.get("enhanced"),
                "compressed": r.get("compressed"),
                "status": r.get("status"),
                "slots": r.get("slots"),
                "explicitization": r.get("explicitization"),
                "notes": r.get("notes"),
                "expression_index": r.get("expression_index"),
                "sent_id": r.get("sent_id"),
                "source": r.get("source"),
            }
        )
    # optional grouped structure
    by_ref: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        key = str(rec.get("ref_id"))
        if key not in by_ref:
            by_ref[key] = {
                "ref_id": rec.get("ref_id"),
                "ann_id": rec.get("ann_id"),
                "image_id": rec.get("image_id"),
                "file_name": rec.get("file_name"),
                "category_id": rec.get("category_id"),
                "category_name": rec.get("category_name"),
                "split": rec.get("split"),
                "bbox": rec.get("bbox"),
                "segmentation_ref": rec.get("segmentation_ref"),
                "expressions": [],
            }
        by_ref[key]["expressions"].append(rec)
    return {
        "dataset": "RRSIS-D",
        "enhancement_version": "v1.0_explicitization_only",
        "num_records": len(records),
        "records": records,
        "refs": list(by_ref.values()),
    }


def generate_data_adapter_py(output_path: str) -> None:
    code = """#!/usr/bin/env python3
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional


class MaskLoader:
    def __init__(self, instances_json_path: str) -> None:
        self.instances_json_path = instances_json_path
        with open(instances_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        annotations = payload.get("annotations", [])
        self.ann_by_id: Dict[Any, Dict[str, Any]] = {}
        for ann in annotations:
            if isinstance(ann, dict) and "id" in ann:
                self.ann_by_id[ann["id"]] = ann
        self._cache: Dict[Any, Dict[str, Any]] = {}

    def get_annotation(self, annotation_id: Any) -> Optional[Dict[str, Any]]:
        if annotation_id in self._cache:
            return self._cache[annotation_id]
        ann = self.ann_by_id.get(annotation_id)
        if ann is not None:
            self._cache[annotation_id] = ann
        return ann

    def get_bbox(self, annotation_id: Any) -> Any:
        ann = self.get_annotation(annotation_id)
        if ann is None:
            return None
        return ann.get("bbox")

    def get_segmentation(self, annotation_id: Any) -> Any:
        ann = self.get_annotation(annotation_id)
        if ann is None:
            return None
        return ann.get("segmentation")

    def get_mask_ref(self, annotation_id: Any) -> Any:
        ann = self.get_annotation(annotation_id)
        if ann is None:
            return None
        return {"annotation_id": annotation_id, "annotation": ann}


class EnhancedRRSISDAdapter:
    def __init__(
        self,
        enhanced_json_path: str,
        expression_mode: str = "enhanced",
        view_mode: str = "flatten",
        failed_policy: str = "use_raw",
        instances_json_path: Optional[str] = None,
    ) -> None:
        if expression_mode not in {"raw", "enhanced", "compressed"}:
            raise ValueError("expression_mode must be raw|enhanced|compressed")
        if view_mode not in {"flatten", "grouped", "virtual_refs"}:
            raise ValueError("view_mode must be flatten|grouped|virtual_refs")
        if failed_policy not in {"use_raw", "skip", "raise_error"}:
            raise ValueError("failed_policy must be use_raw|skip|raise_error")
        self.expression_mode = expression_mode
        self.view_mode = view_mode
        self.failed_policy = failed_policy
        self.mask_loader = MaskLoader(instances_json_path) if instances_json_path else None
        with open(enhanced_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.records = payload.get("records", [])
        if not self.records:
            refs = payload.get("refs", [])
            for ref in refs:
                for expr in ref.get("expressions", []):
                    self.records.append(expr)
        self._flatten = self._build_flatten_items()
        self._grouped = self._build_grouped_items()

    def _select_text(self, rec: Dict[str, Any]) -> str:
        status = rec.get("status")
        if status not in {"success", "unchanged"}:
            if self.failed_policy == "skip":
                return ""
            if self.failed_policy == "raise_error":
                raise ValueError(f"unusable status={status}, expr_id={rec.get('expr_id')}")
            return rec.get("raw", "")
        return rec.get(self.expression_mode, rec.get("raw", ""))

    def _build_flatten_items(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rec in self.records:
            text = self._select_text(rec)
            if text == "":
                continue
            ann_id = rec.get("ann_id")
            mask = rec.get("segmentation_ref")
            if self.mask_loader and ann_id is not None:
                mask = self.mask_loader.get_mask_ref(ann_id)
            out.append(
                {
                    "image": rec.get("file_name"),
                    "mask": mask,
                    "text_expression": text,
                    "sample_id": f"image_{rec.get('image_id')}",
                    "ref_id": rec.get("ref_id"),
                    "ann_id": ann_id,
                    "image_id": rec.get("image_id"),
                    "expr_id": rec.get("expr_id"),
                    "split": rec.get("split"),
                    "category_id": rec.get("category_id"),
                    "category_name": rec.get("category_name"),
                    "bbox": rec.get("bbox"),
                    "segmentation_ref": rec.get("segmentation_ref"),
                    "slots": rec.get("slots", {}),
                    "meta": rec,
                }
            )
        return out

    def _build_grouped_items(self) -> List[Dict[str, Any]]:
        by_ref: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for item in self._flatten:
            by_ref[item.get("ref_id")].append(item)
        grouped = []
        for ref_id, rows in by_ref.items():
            first = rows[0]
            grouped.append(
                {
                    "ref_id": ref_id,
                    "ann_id": first.get("ann_id"),
                    "image_id": first.get("image_id"),
                    "file_name": first.get("image"),
                    "split": first.get("split"),
                    "category_id": first.get("category_id"),
                    "category_name": first.get("category_name"),
                    "bbox": first.get("bbox"),
                    "segmentation_ref": first.get("segmentation_ref"),
                    "expressions": rows,
                }
            )
        return grouped

    def get_legacy_refs(self) -> List[Dict[str, Any]]:
        refs: Dict[Any, Dict[str, Any]] = {}
        for rec in self.records:
            text = self._select_text(rec)
            if text == "":
                continue
            ref_id = rec.get("ref_id")
            if ref_id not in refs:
                refs[ref_id] = {
                    "ref_id": ref_id,
                    "ann_id": rec.get("ann_id"),
                    "image_id": rec.get("image_id"),
                    "file_name": rec.get("file_name"),
                    "category_id": rec.get("category_id"),
                    "split": rec.get("split"),
                    "sent_ids": [],
                    "sentences": [],
                }
            sent_id = rec.get("sent_id")
            refs[ref_id]["sent_ids"].append(sent_id)
            refs[ref_id]["sentences"].append(
                {
                    "sent": text,
                    "raw": rec.get("raw", ""),
                    "sent_id": sent_id,
                    "tokens": [],
                    "expr_id": rec.get("expr_id"),
                }
            )
        return list(refs.values())

    def __len__(self) -> int:
        if self.view_mode == "flatten":
            return len(self._flatten)
        if self.view_mode == "grouped":
            return len(self._grouped)
        return len(self.get_legacy_refs())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.view_mode == "flatten":
            return self._flatten[idx]
        if self.view_mode == "grouped":
            return self._grouped[idx]
        return self.get_legacy_refs()[idx]
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)


def build_units_from_refs(args: argparse.Namespace) -> None:
    instances = read_json(args.instances_json)
    with open(args.refs_file, "rb") as f:
        refs = pickle.load(f)
    if not isinstance(refs, list):
        raise ValueError("refs file must be a list-like object.")

    annotations = instances.get("annotations", [])
    images = instances.get("images", [])
    categories = instances.get("categories", [])
    ann_by_id = {ann["id"]: ann for ann in annotations if isinstance(ann, dict) and "id" in ann}
    img_by_id = {img["id"]: img for img in images if isinstance(img, dict) and "id" in img}
    cat_name_by_id = {
        cat.get("id"): cat.get("name")
        for cat in categories
        if isinstance(cat, dict) and "id" in cat
    }

    split_filter = (args.split or "all").lower()
    if split_filter not in {"train", "val", "test", "all"}:
        raise ValueError("--split must be train/val/test/all.")
    refs_used = [r for r in refs if split_filter == "all" or str(r.get("split", "")).lower() == split_filter]

    split_ref_counts: Dict[str, int] = {}
    split_unit_counts: Dict[str, int] = {}
    sentences_len_distribution: Dict[str, int] = {}
    for r in refs_used:
        split = str(r.get("split", "unknown"))
        split_ref_counts[split] = split_ref_counts.get(split, 0) + 1
        sent_len = len(r.get("sentences", [])) if isinstance(r.get("sentences"), list) else 0
        sentences_len_distribution[str(sent_len)] = sentences_len_distribution.get(str(sent_len), 0) + 1

    units: List[Dict[str, Any]] = []
    unmatched_rows: List[Dict[str, Any]] = []

    unmatched_ref_ann_ids: List[Any] = []
    image_id_mismatches: List[Dict[str, Any]] = []
    matched_ref_count = 0
    matched_ann_ids: Set[Any] = set()

    for r in refs_used:
        ann_id = r.get("ann_id")
        ref_id = r.get("ref_id")
        ref_image_id = r.get("image_id")
        split = r.get("split")
        file_name = r.get("file_name")

        if ann_id not in ann_by_id:
            unmatched_ref_ann_ids.append(ann_id)
            unmatched_rows.append(
                {
                    "reason": "ann_id_not_found",
                    "ref_id": ref_id,
                    "ann_id": ann_id,
                    "image_id": ref_image_id,
                    "split": split,
                    "file_name": file_name,
                }
            )
            continue

        ann = ann_by_id[ann_id]
        ann_image_id = ann.get("image_id")
        if ref_image_id is not None and ann_image_id is not None and ref_image_id != ann_image_id:
            mismatch_obj = {
                "ref_id": ref_id,
                "ann_id": ann_id,
                "ref_image_id": ref_image_id,
                "annotation_image_id": ann_image_id,
            }
            image_id_mismatches.append(mismatch_obj)
            unmatched_rows.append({"reason": "image_id_mismatch", **mismatch_obj, "split": split, "file_name": file_name})
            continue

        sents = r.get("sentences", [])
        if not isinstance(sents, list) or len(sents) == 0:
            unmatched_rows.append(
                {
                    "reason": "missing_sentences",
                    "ref_id": ref_id,
                    "ann_id": ann_id,
                    "image_id": ref_image_id,
                    "split": split,
                    "file_name": file_name,
                }
            )
            continue

        matched_ref_count += 1
        matched_ann_ids.add(ann_id)
        unit_count_for_ref = 0
        for expression_index, sent_obj in enumerate(sents):
            if not isinstance(sent_obj, dict):
                continue
            raw_expression = choose_sentence_text(sent_obj)
            if not raw_expression:
                continue
            sent_id = sent_obj.get("sent_id")
            expr_id = build_expr_id(
                ref_id=ref_id,
                ann_id=ann_id,
                sent_id=sent_id,
                expression_index=expression_index,
                raw_expression=raw_expression,
            )
            category_id = ann.get("categories_id", ann.get("category_id", r.get("category_id")))
            category_name = cat_name_by_id.get(category_id)
            resolved_image_id = ref_image_id if ref_image_id is not None else ann_image_id
            resolved_file_name = file_name
            if not resolved_file_name:
                image_meta = img_by_id.get(resolved_image_id)
                if isinstance(image_meta, dict):
                    resolved_file_name = image_meta.get("file_name")
            unit = {
                "expr_id": expr_id,
                "ref_id": ref_id,
                "ann_id": ann_id,
                "image_id": resolved_image_id,
                "file_name": resolved_file_name,
                "split": split,
                "category_id": category_id,
                "category_name": category_name,
                "bbox": ann.get("bbox", []),
                "segmentation_ref": {
                    "source_file": "instances.json",
                    "annotation_id": ann_id,
                },
                "raw_expression": raw_expression,
                "expression_index": expression_index,
                "sent_id": sent_id,
                "source": {
                    "ref_file": os.path.abspath(args.refs_file),
                    "instance_file": os.path.abspath(args.instances_json),
                    "expression_path": "sentences[].sent",
                    "join_key": "ann_id -> annotations.id",
                },
            }
            units.append(unit)
            split_unit_counts[str(split)] = split_unit_counts.get(str(split), 0) + 1
            unit_count_for_ref += 1

        if unit_count_for_ref == 0:
            unmatched_rows.append(
                {
                    "reason": "no_valid_sentence_text",
                    "ref_id": ref_id,
                    "ann_id": ann_id,
                    "image_id": ref_image_id,
                    "split": split,
                    "file_name": file_name,
                }
            )

    unmatched_ann_ids = [aid for aid in ann_by_id.keys() if aid not in matched_ann_ids]
    unmatched_ref_count = len(unmatched_rows)
    matched_ratio = (matched_ref_count / len(refs_used)) if refs_used else 0.0
    join_safe_to_use = True
    if matched_ratio < 0.98:
        join_safe_to_use = False
    if unmatched_ref_count > 0:
        join_safe_to_use = False
    if len(image_id_mismatches) > 0:
        join_safe_to_use = False

    join_report = {
        "join_key_checked": "ann_id -> annotations.id",
        "num_refs": len(refs_used),
        "num_annotations": len(ann_by_id),
        "num_units": len(units),
        "matched_ref_count": matched_ref_count,
        "unmatched_ref_count": unmatched_ref_count,
        "unmatched_ann_count": len(unmatched_ann_ids),
        "matched_ratio": round(matched_ratio, 6),
        "image_id_mismatch_count": len(image_id_mismatches),
        "first_10_unmatched_ref_ann_ids": unmatched_ref_ann_ids[:10],
        "first_10_unmatched_annotation_ids": unmatched_ann_ids[:10],
        "first_10_image_id_mismatches": image_id_mismatches[:10],
        "split_ref_counts": split_ref_counts,
        "split_unit_counts": split_unit_counts,
        "sentences_len_distribution": sentences_len_distribution,
        "join_safe_to_use": join_safe_to_use,
    }

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.unmatched_output_jsonl) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.join_report) or ".", exist_ok=True)

    write_json(args.join_report, join_report)

    written = 0
    preview_limit = args.preview_limit if args.preview_limit and args.preview_limit > 0 else None
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for unit in units:
            if preview_limit is not None and written >= preview_limit:
                break
            f.write(json.dumps(unit, ensure_ascii=False) + "\n")
            written += 1
    with open(args.unmatched_output_jsonl, "w", encoding="utf-8") as f:
        for row in unmatched_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_units_jsonl(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input_jsonl)
    all_units = [r for r in rows if isinstance(r, dict)]
    limit = args.limit if args.limit is not None and args.limit > 0 else None
    scoped_units: List[Dict[str, Any]] = []
    scoped_expr_ids: Set[str] = set()
    duplicate_input_expr_ids: Set[str] = set()
    for idx, unit in enumerate(all_units):
        expr_id = _unit_expr_id(unit, idx)
        unit["expr_id"] = expr_id
        if expr_id in scoped_expr_ids:
            duplicate_input_expr_ids.add(expr_id)
            continue
        scoped_expr_ids.add(expr_id)
        scoped_units.append(unit)
        if limit is not None and len(scoped_units) >= limit:
            break
    units = scoped_units
    if not units:
        raise ValueError("input jsonl has no valid rows.")

    output_dir = args.output_dir
    _prepare_process_output_dir(
        output_dir=output_dir,
        resume=bool(args.resume_from_checkpoint),
        overwrite=bool(args.overwrite),
    )
    intermediate_dir = os.path.join(output_dir, "intermediate")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    result_jsonl_path = os.path.join(intermediate_dir, "enhancement_results.jsonl")
    failed_cases_path = os.path.join(output_dir, "failed_cases.jsonl")
    log_path = os.path.join(output_dir, "enhancement_log.jsonl")
    enhanced_training_path = os.path.join(output_dir, "enhanced_training.json")
    summary_path = os.path.join(output_dir, "summary_report.json")
    adapter_path = os.path.join(output_dir, "data_adapter.py")

    processed_expr_ids: Set[str] = set()
    if args.resume_from_checkpoint:
        processed_expr_ids = load_resume_state(checkpoint_dir, result_jsonl_path)
    written_result_expr_ids = _load_written_expr_ids_from_jsonl(result_jsonl_path) if args.resume_from_checkpoint else set()
    written_failed_expr_ids = _load_written_expr_ids_from_jsonl(failed_cases_path) if args.resume_from_checkpoint else set()
    semantic_kb: Optional[Dict[str, Any]] = None
    if getattr(args, "semantic_kb", None):
        semantic_kb = load_semantic_kb(args.semantic_kb)
    concept_semantic_library: Optional[Dict[str, Any]] = None
    concept_public_semantic_library: Optional[Dict[str, Any]] = None
    if getattr(args, "concept_semantic_library", None):
        concept_semantic_library = load_concept_semantic_library(args.concept_semantic_library)
    if getattr(args, "concept_public_semantic_library", None):
        concept_public_semantic_library = load_concept_public_semantic_library(args.concept_public_semantic_library)

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key env: {args.api_key_env}")

    append_jsonl(
        log_path,
        {
            "timestamp": utcnow(),
            "event": "start_processing_units_jsonl",
            "input_jsonl": args.input_jsonl,
            "total_input_units": len(all_units),
            "total_units": len(units),
            "limit": limit,
            "already_processed": len(processed_expr_ids),
            "duplicate_input_expr_id_count": len(duplicate_input_expr_ids),
        },
    )
    for expr_id in sorted(duplicate_input_expr_ids):
        append_jsonl(
            log_path,
            {
                "timestamp": utcnow(),
                "event": "duplicate_expr_id_detected",
                "expr_id": expr_id,
                "action": "skipped_duplicate_input",
            },
        )

    processed_count = 0
    for idx, unit in enumerate(units):
        expr_id = _unit_expr_id(unit, idx)
        unit["expr_id"] = expr_id
        if expr_id in processed_expr_ids:
            continue

        raw_expr = str(unit.get("raw_expression", "")).strip()
        semantic_context = (
            retrieve_semantic_context(
                raw_expression=raw_expr,
                category_name=normalize_to_str(unit.get("category_name")),
                semantic_kb=semantic_kb,
            )
            if semantic_kb is not None
            else None
        )
        primary_concept_lib = concept_public_semantic_library or concept_semantic_library
        private_for_audit = concept_semantic_library if concept_public_semantic_library is not None else None
        concept_semantics_context = (
            retrieve_concept_semantics(
                raw_expr,
                primary_concept_lib,
                private_library_for_audit=private_for_audit,
            )
            if primary_concept_lib is not None
            else None
        )
        semantic_summary = semantic_context_summary_from_context(semantic_context)
        concept_semantic_summary = concept_semantic_summary_from_context(concept_semantics_context)
        combined_semantic_summary: Optional[Dict[str, Any]] = None
        if semantic_summary is not None or concept_semantic_summary is not None:
            combined_semantic_summary = {}
            if semantic_summary is not None:
                combined_semantic_summary.update(semantic_summary)
            if concept_semantic_summary is not None:
                combined_semantic_summary["concept_semantics_used"] = concept_semantic_summary

        if not raw_expr:
            record = fallback_failed_record(
                unit,
                "skipped_empty_expression",
                "Empty expression. Fallback to raw expression.",
                semantic_context_summary=combined_semantic_summary,
            )
            if expr_id not in written_result_expr_ids:
                append_jsonl(result_jsonl_path, record)
                written_result_expr_ids.add(expr_id)
            else:
                append_jsonl(log_path, {"timestamp": utcnow(), "event": "duplicate_expr_id_detected", "expr_id": expr_id, "action": "skip_result_write"})
            if expr_id not in written_failed_expr_ids:
                append_jsonl(failed_cases_path, record)
                written_failed_expr_ids.add(expr_id)
            else:
                append_jsonl(log_path, {"timestamp": utcnow(), "event": "duplicate_expr_id_detected", "expr_id": expr_id, "action": "skip_failed_write"})
            processed_expr_ids.add(expr_id)
            processed_count += 1
            continue

        try:
            llm_output = call_llm_with_retry(
                api_url=args.api_url,
                api_key=api_key,
                model=args.model,
                timeout_sec=args.timeout_sec,
                strict_json_schema=bool(args.strict_json_schema),
                raw_expression=raw_expr,
                allow_synonym_normalization=bool(args.allow_synonym_normalization),
                category_name=normalize_to_str(unit.get("category_name")),
                semantic_context=semantic_context,
                concept_semantics_context=concept_semantics_context,
                max_attempts=args.max_attempts,
                initial_wait=args.initial_wait,
                max_wait=args.max_wait,
                log_path=log_path,
                expr_id=expr_id,
            )
        except Exception as e:
            record = fallback_failed_record(
                unit,
                "failed_api_error",
                f"API failed after retries: {str(e)[:1200]}. Fallback to raw expression.",
                semantic_context_summary=combined_semantic_summary,
            )
            if expr_id not in written_result_expr_ids:
                append_jsonl(result_jsonl_path, record)
                written_result_expr_ids.add(expr_id)
            else:
                append_jsonl(log_path, {"timestamp": utcnow(), "event": "duplicate_expr_id_detected", "expr_id": expr_id, "action": "skip_result_write"})
            if expr_id not in written_failed_expr_ids:
                append_jsonl(failed_cases_path, record)
                written_failed_expr_ids.add(expr_id)
            else:
                append_jsonl(log_path, {"timestamp": utcnow(), "event": "duplicate_expr_id_detected", "expr_id": expr_id, "action": "skip_failed_write"})
            append_jsonl(log_path, {"timestamp": utcnow(), "event": "api_error", "expr_id": expr_id, "error": str(e)[:2000]})
            processed_expr_ids.add(expr_id)
            processed_count += 1
            if processed_count % args.checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_dir,
                    idx + 1,
                    processed_expr_ids,
                    result_jsonl_path,
                    failed_cases_path,
                    {"args": vars(args)},
                )
            continue

        ok, reason, normalized = validate_and_normalize_output(
            raw_expr,
            llm_output,
            category_name=normalize_to_str(unit.get("category_name")),
            concept_semantics_context=concept_semantics_context,
        )
        if not ok:
            record = fallback_failed_record(
                unit,
                "failed_postcheck",
                f"Reason for failure: {reason}. Fallback to raw expression.",
                semantic_context_summary=combined_semantic_summary,
            )
            if expr_id not in written_result_expr_ids:
                append_jsonl(result_jsonl_path, record)
                written_result_expr_ids.add(expr_id)
            else:
                append_jsonl(log_path, {"timestamp": utcnow(), "event": "duplicate_expr_id_detected", "expr_id": expr_id, "action": "skip_result_write"})
            if expr_id not in written_failed_expr_ids:
                append_jsonl(failed_cases_path, record)
                written_failed_expr_ids.add(expr_id)
            else:
                append_jsonl(log_path, {"timestamp": utcnow(), "event": "duplicate_expr_id_detected", "expr_id": expr_id, "action": "skip_failed_write"})
            append_jsonl(log_path, {"timestamp": utcnow(), "event": "failed_postcheck", "expr_id": expr_id, "reason": reason})
        else:
            pipeline_status = decide_pipeline_status(raw_expr, normalized)
            record = normalize_success_record(unit, normalized, pipeline_status=pipeline_status)
            if combined_semantic_summary is not None:
                record["semantic_context_summary"] = combined_semantic_summary
            if expr_id not in written_result_expr_ids:
                append_jsonl(result_jsonl_path, record)
                written_result_expr_ids.add(expr_id)
            else:
                append_jsonl(log_path, {"timestamp": utcnow(), "event": "duplicate_expr_id_detected", "expr_id": expr_id, "action": "skip_result_write"})
            append_jsonl(log_path, {"timestamp": utcnow(), "event": "processed", "expr_id": expr_id, "status": record["status"]})
            if record["status"] not in {"success", "unchanged"}:
                if expr_id not in written_failed_expr_ids:
                    append_jsonl(failed_cases_path, record)
                    written_failed_expr_ids.add(expr_id)
                else:
                    append_jsonl(log_path, {"timestamp": utcnow(), "event": "duplicate_expr_id_detected", "expr_id": expr_id, "action": "skip_failed_write"})
        processed_expr_ids.add(expr_id)
        processed_count += 1
        if processed_count % args.checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_dir,
                idx + 1,
                processed_expr_ids,
                result_jsonl_path,
                failed_cases_path,
                {"args": vars(args)},
            )

    if processed_count % args.checkpoint_interval != 0 and processed_count > 0:
        save_checkpoint(
            checkpoint_dir,
            len(units),
            processed_expr_ids,
            result_jsonl_path,
            failed_cases_path,
            {"args": vars(args)},
        )

    result_rows = read_jsonl(result_jsonl_path)
    by_expr = {}
    for r in result_rows:
        expr_id = r.get("expr_id")
        if isinstance(expr_id, str) and expr_id in scoped_expr_ids:
            by_expr[expr_id] = r
    dedup_rows = list(by_expr.values())
    enhanced_obj = aggregate_to_training_json(dedup_rows)
    write_json(enhanced_training_path, enhanced_obj)
    generate_data_adapter_py(adapter_path)

    status_breakdown: Dict[str, int] = {}
    for r in dedup_rows:
        st = r.get("status", "unknown")
        status_breakdown[st] = status_breakdown.get(st, 0) + 1
    summary = {
        "dataset": "RRSIS-D",
        "enhancement_version": "v1.0_explicitization_only",
        "total_input_units": len(all_units),
        "limit": limit,
        "processed_scope_units": len(units),
        "total_expressions": len(dedup_rows),
        "success_count": status_breakdown.get("success", 0),
        "unchanged_count": status_breakdown.get("unchanged", 0),
        "failed_count": status_breakdown.get("failed", 0),
        "failed_postcheck_count": status_breakdown.get("failed_postcheck", 0),
        "failed_api_error_count": status_breakdown.get("failed_api_error", 0),
        "skipped_count": status_breakdown.get("skipped_empty_expression", 0),
        "status_breakdown": status_breakdown,
        "checkpoint_count": len([f for f in os.listdir(checkpoint_dir) if f.startswith("enhancement_ckpt_")]),
        "duplicate_input_expr_id_count": len(duplicate_input_expr_ids),
        "failed_cases_count": sum(1 for r in dedup_rows if r.get("status") not in {"success", "unchanged"}),
        "jsonl_result_file": os.path.join("intermediate", "enhancement_results.jsonl"),
        "output_file": "enhanced_training.json",
        "adapter_file": "data_adapter.py",
        "failed_cases_file": "failed_cases.jsonl",
        "log_file": "enhancement_log.jsonl",
    }
    summary["build_invalid"] = summary["failed_cases_count"] > summary["total_expressions"]
    write_json(summary_path, summary)
    append_jsonl(log_path, {"timestamp": utcnow(), "event": "finished", "summary_report": summary_path})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RRSIS-D explicitization-only pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_p = subparsers.add_parser("probe_schema", help="Probe dataset schema paths.")
    probe_p.add_argument("--input-json", required=True)
    probe_p.add_argument("--preview-limit", type=int, default=20)
    probe_p.add_argument("--probe-output", default=None)

    build_p = subparsers.add_parser("build_units_from_refs", help="Build canonical raw units from refs + instances.")
    build_p.add_argument("--instances-json", required=True)
    build_p.add_argument("--refs-file", required=True)
    build_p.add_argument("--output-jsonl", required=True)
    build_p.add_argument("--join-report", required=True)
    build_p.add_argument("--unmatched-output-jsonl", required=True)
    build_p.add_argument("--split", default="all", help="train|val|test|all")
    build_p.add_argument("--preview-limit", type=int, default=0)

    process_p = subparsers.add_parser("process_units_jsonl", help="Run enhancement on canonical raw units jsonl.")
    process_p.add_argument("--input-jsonl", required=True)
    process_p.add_argument("--output-dir", required=True)
    process_p.add_argument("--api-url", default="https://api.openai.com/v1/chat/completions")
    process_p.add_argument("--model", required=True)
    process_p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    process_p.add_argument("--timeout-sec", type=int, default=90)
    process_p.add_argument("--strict-json-schema", action="store_true")
    process_p.add_argument("--allow-synonym-normalization", action="store_true")
    process_p.add_argument("--max-attempts", type=int, default=5)
    process_p.add_argument("--initial-wait", type=float, default=1.0)
    process_p.add_argument("--max-wait", type=float, default=60.0)
    process_p.add_argument("--checkpoint-interval", type=int, default=100)
    process_p.add_argument("--resume-from-checkpoint", action="store_true")
    process_p.add_argument("--overwrite", action="store_true", help="Overwrite existing process artifacts in output dir.")
    process_p.add_argument("--limit", type=int, default=0, help="Process only first N units from input jsonl (0=all).")
    process_p.add_argument("--semantic-kb", default=None, help="Optional semantic knowledge base JSON path.")
    process_p.add_argument(
        "--concept-semantic-library",
        default=None,
        help="Optional full concept semantic library JSON (private/audit); used with public library for validators.",
    )
    process_p.add_argument(
        "--concept-public-semantic-library",
        default=None,
        help="Optional public library JSON: v0 public_minimal (concept/type/boundary) or v2 public_grounding_prior (six-field grounding priors).",
    )
    subparsers.add_parser("validator_self_check", help="Run validator normalization self-check.")
    kb_check_p = subparsers.add_parser("semantic_kb_self_check", help="Run semantic knowledge base retrieval self-check.")
    kb_check_p.add_argument("--semantic-kb", required=True)
    concept_audit_p = subparsers.add_parser("concept_library_audit", help="Audit concept semantic library quality.")
    concept_audit_p.add_argument("--concept-semantic-library", required=True)
    concept_audit_p.add_argument("--audit-output", required=True)
    concept_self_check_p = subparsers.add_parser("concept_semantic_self_check", help="Run concept semantic retrieval/injection self-check.")
    concept_self_check_p.add_argument("--concept-semantic-library", required=True)
    public_ctx_p = subparsers.add_parser(
        "public_semantic_context_self_check",
        help="Validate minimal public concept context for MLLM prompts (no API).",
    )
    public_ctx_p.add_argument("--concept-semantic-library", required=True)
    public_ctx_p.add_argument(
        "--vegetation-seed",
        default=None,
        help="Optional path to rrsisd_concepts_seed_next.json (defaults to configs/ next to repo).",
    )
    public_lib_check_p = subparsers.add_parser(
        "concept_public_semantic_library_self_check",
        help="Validate hand-written public minimal concept library JSON (no API).",
    )
    public_lib_check_p.add_argument(
        "--public-library",
        default=None,
        help="Path to public library JSON (default: configs/concept_public_semantic_library_v0.json under repo).",
    )
    gp_struct_p = subparsers.add_parser(
        "concept_public_grounding_prior_structure_check",
        help="Validate configs/concept_public_semantic_library_v2.json structure only (no API).",
    )
    gp_struct_p.add_argument(
        "--public-library",
        default=None,
        help="Path to v2 public grounding prior JSON (default: configs/concept_public_semantic_library_v2.json).",
    )
    gp_ret_p = subparsers.add_parser(
        "concept_public_grounding_prior_retrieval_check",
        help="Dry-run v2 retrieval + prompt length/leakage checks (no API).",
    )
    gp_ret_p.add_argument(
        "--public-library",
        default=None,
        help="Path to v2 public grounding prior JSON (default: configs/concept_public_semantic_library_v2.json).",
    )
    gp_ret_p.add_argument(
        "--private-library",
        default=None,
        help="Full semantic library for audit merge (default: configs/concept_semantic_library_verified_7.json).",
    )
    prompt_leak_p = subparsers.add_parser(
        "prompt_leakage_self_check",
        help="Verify MLLM user prompt has no private-field leaks for public+private retrieval (no API).",
    )
    prompt_leak_p.add_argument(
        "--raw-expression",
        default="find water near bridge",
        help="Raw expression to simulate (default: find water near bridge).",
    )
    prompt_leak_p.add_argument(
        "--public-library",
        default=None,
        help="Public minimal library path (default: configs/concept_public_semantic_library_v0.json).",
    )
    prompt_leak_p.add_argument(
        "--private-library",
        default=None,
        help="Full semantic library for audit merge (default: configs/concept_semantic_library_verified_7.json).",
    )
    len_check_p = subparsers.add_parser(
        "concept_public_context_length_check",
        help="Read-only stats: user prompt length vs public matched_concepts size (no API).",
    )
    len_check_p.add_argument(
        "--raw-expression",
        default="find water near bridge",
        help="Raw expression to simulate (default: find water near bridge).",
    )
    len_check_p.add_argument(
        "--public-library",
        default=None,
        help="Public minimal library path (default: configs/concept_public_semantic_library_v0.json).",
    )
    len_check_p.add_argument(
        "--private-library",
        default=None,
        help="Full semantic library for audit merge (default: configs/concept_semantic_library_verified_7.json).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "probe_schema":
        report = probe_schema(args.input_json, preview_limit=args.preview_limit)
        if args.probe_output:
            write_json(args.probe_output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "build_units_from_refs":
        build_units_from_refs(args)
        return
    if args.command == "process_units_jsonl":
        process_units_jsonl(args)
        return
    if args.command == "validator_self_check":
        passed = validator_self_check()
        print(json.dumps({"validator_self_check_passed": bool(passed)}, ensure_ascii=False, indent=2))
        return
    if args.command == "semantic_kb_self_check":
        semantic_kb = load_semantic_kb(args.semantic_kb)
        passed = semantic_kb_self_check(semantic_kb)
        print(json.dumps({"semantic_kb_self_check_passed": bool(passed)}, ensure_ascii=False, indent=2))
        return
    if args.command == "concept_library_audit":
        concept_library = load_concept_semantic_library(args.concept_semantic_library)
        report = concept_library_audit(concept_library)
        write_json(args.audit_output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "concept_semantic_self_check":
        concept_library = load_concept_semantic_library(args.concept_semantic_library)
        result = concept_semantic_self_check(concept_library)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "public_semantic_context_self_check":
        concept_library = load_concept_semantic_library(args.concept_semantic_library)
        result = public_semantic_context_self_check(
            concept_library,
            vegetation_seed_path=getattr(args, "vegetation_seed", None),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "concept_public_semantic_library_self_check":
        pl_path = getattr(args, "public_library", None) or os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "configs", "concept_public_semantic_library_v0.json")
        )
        lib = load_concept_public_semantic_library(pl_path)
        result = concept_public_semantic_library_self_check(lib)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "concept_public_grounding_prior_structure_check":
        pl_path = getattr(args, "public_library", None) or os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "configs", "concept_public_semantic_library_v2.json")
        )
        lib = load_concept_public_semantic_library(pl_path)
        result = concept_public_grounding_prior_structure_check(lib)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "concept_public_grounding_prior_retrieval_check":
        result = concept_public_grounding_prior_retrieval_check(
            public_library_path=getattr(args, "public_library", None),
            private_library_path=getattr(args, "private_library", None),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "prompt_leakage_self_check":
        result = prompt_leakage_self_check(
            raw_expression=str(getattr(args, "raw_expression", "") or "find water near bridge"),
            public_library_path=getattr(args, "public_library", None),
            private_library_path=getattr(args, "private_library", None),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "concept_public_context_length_check":
        result = concept_public_context_length_check(
            raw_expression=str(getattr(args, "raw_expression", "") or "find water near bridge"),
            public_library_path=getattr(args, "public_library", None),
            private_library_path=getattr(args, "private_library", None),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
