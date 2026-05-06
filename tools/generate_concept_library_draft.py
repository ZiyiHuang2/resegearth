#!/usr/bin/env python3
"""
Offline draft generator for concept_semantic_library (per-concept API calls).
Does not run expression enhancement or process_units_jsonl.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = REPO_ROOT / "tools" / "rrsisd_explicitization_pipeline.py"

_PIPELINE_MOD: Optional[Any] = None


def _load_pipeline_module() -> Any:
    global _PIPELINE_MOD
    if _PIPELINE_MOD is not None:
        return _PIPELINE_MOD
    spec = importlib.util.spec_from_file_location("rrsisd_explicitization_pipeline", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pipeline module from {PIPELINE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _PIPELINE_MOD = mod
    return mod


def _basic_stop_words() -> Set[str]:
    # Keep only true function-word-like stop terms for forbidden-token checks.
    # Domain content words such as "body/bodies" should be allowed as standalone
    # high-risk tokens (e.g., "water body" alias subword "body").
    stops = set(_load_pipeline_module().BASIC_STOP_WORDS)
    stops.discard("body")
    stops.discard("bodies")
    return stops


def _normalize_label_text(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip().lower()
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"\s+", " ", t)
    return t


def normalize_phrase(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip().lower()
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def is_disallowed_forbidden_token(
    token: str, canonical_name: str, aliases: List[str], basic_stop_words: Set[str]
) -> bool:
    norm_token = normalize_phrase(token)
    if not norm_token:
        return False
    if norm_token == "none":
        return True
    norm_stops = {normalize_phrase(x) for x in basic_stop_words}
    if norm_token in norm_stops:
        return True
    if norm_token == normalize_phrase(canonical_name):
        return True
    for a in aliases:
        if norm_token == normalize_phrase(a):
            return True
    return False


def _skip_vfo_forbidden_token_coverage(seed_row: Optional[Dict[str, Any]], obj: Dict[str, Any]) -> bool:
    """stuff / region / stuff_region: do not require visual_form option words in forbidden_auto_infer_tokens."""
    mod = _load_pipeline_module()
    ct = ""
    if isinstance(seed_row, dict):
        v = seed_row.get("concept_type")
        if isinstance(v, str) and v.strip():
            ct = v.strip()
    if not ct and isinstance(obj, dict):
        v2 = obj.get("concept_type")
        if isinstance(v2, str) and v2.strip():
            ct = v2.strip()
    return bool(mod.skip_visual_form_forbidden_coverage_for_concept_type(ct))


def _forbidden_token_policy_self_check() -> bool:
    stops = _basic_stop_words()
    # should pass
    assert not is_disallowed_forbidden_token("exposed", "bare land", ["bare land", "bareland", "exposed soil"], stops)
    assert not is_disallowed_forbidden_token("car", "parking lot", ["parking lot", "parking lots", "car park"], stops)
    assert not is_disallowed_forbidden_token("ground", "playground", ["playground", "playgrounds", "sports ground"], stops)
    assert not is_disallowed_forbidden_token("channel", "river", ["river", "rivers", "river channel"], stops)
    assert not is_disallowed_forbidden_token("body", "lake", ["lake", "lakes", "lake area"], stops)
    assert not is_disallowed_forbidden_token("port", "harbor", ["harbor", "harbors", "port area"], stops)
    # should fail
    assert is_disallowed_forbidden_token("river", "river", ["river", "rivers", "river channel"], stops)
    assert is_disallowed_forbidden_token("river channel", "river", ["river", "rivers", "river channel"], stops)
    assert is_disallowed_forbidden_token("parking lot", "parking lot", ["parking lot", "parking lots", "car park"], stops)
    assert is_disallowed_forbidden_token("car park", "parking lot", ["parking lot", "parking lots", "car park"], stops)
    assert is_disallowed_forbidden_token("in", "river", ["river", "rivers", "river channel"], stops)
    assert is_disallowed_forbidden_token("with", "river", ["river", "rivers", "river channel"], stops)
    assert is_disallowed_forbidden_token("none", "river", ["river", "rivers", "river channel"], stops)
    return True


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _make_concept_json_schema() -> Dict[str, Any]:
    """OpenAI-compatible json_schema (strict) for a single concept entry."""
    string_arr: Dict[str, Any] = {"type": "array", "items": {"type": "string"}}
    vfo_item: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["option", "meaning", "emit_policy"],
        "properties": {
            "option": {"type": "string"},
            "meaning": {"type": "string"},
            "emit_policy": {"type": "string"},
        },
    }
    sr_item: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["condition", "allowed_rewrite", "not_allowed"],
        "properties": {
            "condition": {"type": "string"},
            "allowed_rewrite": {"type": "string"},
            "not_allowed": {"type": "string"},
        },
    }
    slot: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["allowed_internal_tags", "allowed_slot_hints", "do_not_emit_as_text"],
        "properties": {
            "allowed_internal_tags": string_arr,
            "allowed_slot_hints": string_arr,
            "do_not_emit_as_text": string_arr,
        },
    }
    props = {
        "concept_key": {"type": "string"},
        "canonical_name": {"type": "string"},
        "aliases": string_arr,
        "definition": {"type": "string"},
        "remote_sensing_understanding": string_arr,
        "visual_form_options": {"type": "array", "items": vfo_item},
        "segmentation_relevance": string_arr,
        "safe_rewrite_guidance": {"type": "array", "items": sr_item},
        "slot_guidance": slot,
        "forbidden_auto_infer_tokens": string_arr,
        "anti_specialization_rules": string_arr,
        "notes": {"type": "string"},
        "concept_semantics": string_arr,
    }
    required = list(props.keys())
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": props,
    }


def _validate_generated_concept(obj: Any, expected_key: str, seed_row: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "root is not an object"
    if obj.get("concept_key") != expected_key:
        return False, f"concept_key mismatch (expected {expected_key!r})"
    str_fields = ("canonical_name", "definition", "notes")
    for f in str_fields:
        v = obj.get(f)
        if not isinstance(v, str) or not v.strip():
            return False, f"missing or empty string field: {f}"
    if not isinstance(obj.get("aliases"), list) or not obj["aliases"]:
        return False, "aliases missing or empty"
    if not all(isinstance(a, str) and a.strip() for a in obj["aliases"]):
        return False, "aliases must be non-empty strings"
    rsu = obj.get("remote_sensing_understanding")
    if not isinstance(rsu, list) or not rsu or not all(isinstance(x, str) and x.strip() for x in rsu):
        return False, "remote_sensing_understanding invalid"
    fat = obj.get("forbidden_auto_infer_tokens")
    if not isinstance(fat, list) or not fat or not all(isinstance(x, str) and x.strip() for x in fat):
        return False, "forbidden_auto_infer_tokens invalid"
    forbidden_set = {_normalize_label_text(x) for x in fat if isinstance(x, str) and str(x).strip()}
    stops = _basic_stop_words()
    cn = obj.get("canonical_name")
    if not isinstance(cn, str) or not cn.strip():
        return False, "canonical_name missing or empty"
    aliases_list = [a for a in (obj.get("aliases", []) or []) if isinstance(a, str)]
    alias_word_set: Set[str] = set()
    if isinstance(cn, str) and cn.strip():
        cn_norm = _normalize_label_text(cn)
        for w in cn_norm.split():
            if w:
                alias_word_set.add(w)
    for a in aliases_list:
        a_norm = _normalize_label_text(a)
        for w in a_norm.split():
            if w:
                alias_word_set.add(w)
    for raw_tok in fat:
        if not isinstance(raw_tok, str) or not raw_tok.strip():
            continue
        if normalize_phrase(raw_tok) == "none":
            return False, "forbidden_auto_infer_tokens contains disallowed self/alias/stop token: none"
        if is_disallowed_forbidden_token(raw_tok, cn, aliases_list, stops):
            return False, f"forbidden_auto_infer_tokens contains disallowed self/alias/stop token: {normalize_phrase(raw_tok)}"
    vfo = obj.get("visual_form_options")
    if not isinstance(vfo, list) or not vfo:
        return False, "visual_form_options invalid"
    has_none_default = False
    for i, item in enumerate(vfo):
        if not isinstance(item, dict):
            return False, f"visual_form_options[{i}] not an object"
        for k in ("option", "meaning", "emit_policy"):
            if k not in item or not isinstance(item[k], str) or not str(item[k]).strip():
                return False, f"visual_form_options[{i}].{k} missing or empty"
        opt_raw = item.get("option", "")
        if isinstance(opt_raw, str) and "_" in opt_raw and opt_raw.strip().lower() != "none":
            return False, "visual_form_options option uses underscore-style token (use natural-language phrase)"
        opt_n = _normalize_label_text(item.get("option", ""))
        pol_n = _normalize_label_text(item.get("emit_policy", ""))
        if opt_n == "none" and pol_n == "default for vague raw":
            has_none_default = True
            continue
        if not _skip_vfo_forbidden_token_coverage(seed_row, obj):
            for tok in opt_n.split():
                if not tok or tok in stops or tok in alias_word_set:
                    continue
                if tok not in forbidden_set:
                    return False, f"visual_form_options[{i}] content word {tok!r} missing from forbidden_auto_infer_tokens"
    mandatory_terms = []
    if isinstance(seed_row, dict):
        mt = seed_row.get("mandatory_high_risk_terms", [])
        if isinstance(mt, list):
            mandatory_terms = [m for m in mt if isinstance(m, str) and _normalize_label_text(m)]
    for term in mandatory_terms:
        t_norm = _normalize_label_text(term)
        if (not t_norm) or normalize_phrase(term) == "none" or is_disallowed_forbidden_token(term, cn, aliases_list, stops):
            continue
        if t_norm not in forbidden_set:
            return False, f"mandatory_high_risk_term missing from forbidden_auto_infer_tokens: {term}"
    must_terms = []
    if isinstance(seed_row, dict):
        gh = seed_row.get("generation_hint", {})
        if isinstance(gh, dict):
            mt2 = gh.get("must_put_in_forbidden", [])
            if isinstance(mt2, list):
                must_terms = [m for m in mt2 if isinstance(m, str) and _normalize_label_text(m)]
    for term in must_terms:
        t_norm = _normalize_label_text(term)
        if (not t_norm) or normalize_phrase(term) == "none" or is_disallowed_forbidden_token(term, cn, aliases_list, stops):
            continue
        if t_norm not in forbidden_set:
            return False, f"generation_hint.must_put_in_forbidden missing from forbidden_auto_infer_tokens: {term}"
    if not has_none_default:
        return False, "missing required none + default_for_vague_raw visual_form_option"
    seg = obj.get("segmentation_relevance")
    if not isinstance(seg, list) or not seg or not all(isinstance(x, str) and x.strip() for x in seg):
        return False, "segmentation_relevance invalid"
    sr = obj.get("safe_rewrite_guidance")
    if not isinstance(sr, list) or not sr:
        return False, "safe_rewrite_guidance invalid"
    for i, r in enumerate(sr):
        if not isinstance(r, dict):
            return False, f"safe_rewrite_guidance[{i}] not an object"
        for k in ("condition", "allowed_rewrite", "not_allowed"):
            if k not in r or not isinstance(r[k], str) or not str(r[k]).strip():
                return False, f"safe_rewrite_guidance[{i}].{k} missing or empty"
    sg = obj.get("slot_guidance")
    if not isinstance(sg, dict):
        return False, "slot_guidance not an object"
    for k in ("allowed_internal_tags", "allowed_slot_hints", "do_not_emit_as_text"):
        if k not in sg or not isinstance(sg[k], list):
            return False, f"slot_guidance.{k} missing or not a list"
        if not all(isinstance(x, str) for x in sg[k]):
            return False, f"slot_guidance.{k} items must be strings"
    do_not = sg.get("do_not_emit_as_text", [])
    for key_src in ("allowed_internal_tags", "allowed_slot_hints"):
        for t in sg.get(key_src, []) or []:
            if not isinstance(t, str) or not t.strip():
                continue
            matched = any(_normalize_label_text(t) == _normalize_label_text(d) for d in do_not if isinstance(d, str))
            if not matched:
                return False, f"slot_guidance.{key_src} entry {t!r} not listed in do_not_emit_as_text"
    asr = obj.get("anti_specialization_rules")
    if not isinstance(asr, list) or not asr or not all(isinstance(x, str) and x.strip() for x in asr):
        return False, "anti_specialization_rules invalid"
    cs = obj.get("concept_semantics")
    if not isinstance(cs, list) or not cs or not all(isinstance(x, str) and x.strip() for x in cs):
        return False, "concept_semantics invalid (required for downstream audit)"
    # Guard against downstream audit high-risk misses before persisting.
    mod = _load_pipeline_module()
    mini_cfg = copy.deepcopy(obj)
    if isinstance(seed_row, dict) and isinstance(seed_row.get("concept_type"), str) and seed_row["concept_type"].strip():
        mini_cfg["concept_type"] = seed_row["concept_type"].strip()
    mini_lib = {"dataset": "RRSIS-D", "version": "draft_single", "global_constraints": [], "concepts": {expected_key: mini_cfg}}
    mini_audit = mod.concept_library_audit(mini_lib)
    if int(mini_audit.get("high_risk_term_not_forbidden_count") or 0) > 0:
        details = mini_audit.get("details", {}).get("high_risk_term_not_forbidden", [])
        return False, f"high_risk_term_not_forbidden_count>0: {details}"
    return True, "ok"


def _build_system_prompt() -> str:
    return (
        "You are generating a concept-level semantic library entry for remote sensing referring expression segmentation.\n"
        "You are not rewriting any current expression.\n"
        "The output is not a per-sample answer.\n"
        "The concept library will be used as a versioned and auditable knowledge asset.\n"
        "For each concept, generate conservative, neutral, remote-sensing-oriented semantic information.\n"
        "visual_form_options are candidate forms in remote sensing imagery, not observed facts.\n"
        "Each visual_form_option must have an emit_policy.\n"
        "Every concept must include a none/default_for_vague_raw option (option exactly \"none\", emit_policy exactly \"default_for_vague_raw\").\n"
        "safe_rewrite_guidance must describe conservative rewriting, e.g. imperative find-X to a generic definite noun phrase.\n"
        "forbidden_auto_infer_tokens must include words likely to be incorrectly injected into natural language outputs.\n"
        "forbidden_auto_infer_tokens must NOT include canonical_name tokens, alias tokens, the literal none option, or basic stop words.\n"
        "Do not include the canonical concept name, aliases, \"none\", or BASIC_STOP_WORDS in forbidden_auto_infer_tokens.\n"
        "forbidden_auto_infer_tokens should only include risky attributes, subtypes, materials, shapes, forms, contexts, or other words that must not be injected into natural-language outputs unless present in raw text.\n"
        "Examples:\n"
        "- For water, do not include water, water body, body, bodies, or none as forbidden tokens.\n"
        "- For bridge, do not include bridge, bridges, with, or none as forbidden tokens.\n"
        "- For water visual form \"linear water\", forbidden should include linear but not water.\n"
        "- For bridge visual form \"multi-span bridge with piers\", forbidden should include multi, span, piers but not bridge or with.\n"
        "anti_specialization_rules must prevent a general class from being rewritten as a specific subclass unless raw text explicitly states it.\n"
        "slot_guidance rules (hard):\n"
        "Every item in slot_guidance.allowed_internal_tags must also appear in slot_guidance.do_not_emit_as_text (same string).\n"
        "Every item in slot_guidance.allowed_slot_hints must also appear in slot_guidance.do_not_emit_as_text (same string).\n"
        "Internal tags and slot hints are for structured fields only and must never be emitted as natural-language expressions.\n"
        "visual_form_options rules (hard):\n"
        "visual_form_options.option must be a natural-language phrase, not an underscore-style tag.\n"
        "Bad: multi_span_with_piers. Good: multi-span bridge with piers (hyphens allowed; underscores not in option text except forbidden none key).\n"
        "Every content word in each visual_form_options.option must appear in forbidden_auto_infer_tokens unless it is a common function word from a small closed class (articles, prepositions, conjunctions, auxiliary verbs), the canonical concept name, a whole-seed-alias token, or the literal none option.\n"
        "Do not use strong claims such as always, must, definitely, certainly.\n"
        "Do not claim that a concept always has a specific color, shape, neighbor object, or substructure.\n"
        "Return only JSON matching the provided schema."
    )


def _build_user_prompt(seed_row: Dict[str, Any], failure_reason: Optional[str] = None) -> str:
    retry = ""
    if failure_reason and failure_reason.strip():
        retry = (
            "\nThe previous JSON failed automated validation. Regenerate a corrected entry.\n"
            "Validation failure (fix all issues; do not add commentary outside JSON):\n"
            f"{failure_reason.strip()}\n"
            "If validation failed because:\n"
            "\"visual_form_options[i] content word '<word>' missing from forbidden_auto_infer_tokens\",\n"
            "then you must either:\n"
            "1) add '<word>' to forbidden_auto_infer_tokens, or\n"
            "2) remove '<word>' from that visual_form_options.option.\n"
            "Do not repeat the same validation error.\n"
            "Example: if option contains \"simple bridge\", then forbidden_auto_infer_tokens must include \"simple\", unless \"simple\" is removed from the option.\n"
        )
    mandatory_hint = ""
    mandatory = seed_row.get("mandatory_high_risk_terms", []) if isinstance(seed_row, dict) else []
    if isinstance(mandatory, list) and mandatory:
        mandatory_txt = ", ".join([m for m in mandatory if isinstance(m, str)])
        mandatory_hint = (
            "The following mandatory_high_risk_terms must appear in forbidden_auto_infer_tokens unless they are the canonical concept name, aliases, "
            "\"none\", or BASIC_STOP_WORDS.\n"
            f"mandatory_high_risk_terms: {mandatory_txt}\n"
        )
    generation_hint = ""
    gh = seed_row.get("generation_hint", {}) if isinstance(seed_row, dict) else {}
    if isinstance(gh, dict):
        rec = gh.get("recommended_visual_form_options", [])
        avoid = gh.get("avoid_visual_form_words", [])
        dont_forbidden = gh.get("do_not_put_in_forbidden", [])
        must_forbidden = gh.get("must_put_in_forbidden", [])
        lines: List[str] = []
        if isinstance(rec, list) and rec:
            lines.append("Recommended visual_form_options:")
            lines.extend([f"- {x}" for x in rec if isinstance(x, str)])
        if isinstance(avoid, list) and avoid:
            lines.append("Avoid these visual form words:")
            lines.extend([f"- {x}" for x in avoid if isinstance(x, str)])
        if isinstance(dont_forbidden, list) and dont_forbidden:
            lines.append("Do not put these words in forbidden_auto_infer_tokens:")
            lines.extend([f"- {x}" for x in dont_forbidden if isinstance(x, str)])
        if isinstance(must_forbidden, list) and must_forbidden:
            lines.append("These terms must appear in forbidden_auto_infer_tokens:")
            lines.extend([f"- {x}" for x in must_forbidden if isinstance(x, str)])
        if lines:
            generation_hint = (
                "\nGeneration hint (seed-provided, not output fields):\n"
                + "\n".join(lines)
                + "\n"
                "If you use any other content word in visual_form_options, it must also appear in forbidden_auto_infer_tokens.\n"
                "Do not use avoid_visual_form_words unless you also include them in forbidden_auto_infer_tokens. Prefer not to use them.\n"
            )
    return (
        "Generate one concept semantic library entry from the seed below.\n"
        "The concept_key, canonical_name, and aliases must stay consistent with the seed (you may reorder aliases but must include every seed alias).\n"
        "Fill all other fields for downstream audit and retrieval.\n"
        "slot_guidance: every allowed_internal_tags and allowed_slot_hints string must be duplicated in do_not_emit_as_text.\n"
        "visual_form_options.option: natural language only; list every non-trivial option word in forbidden_auto_infer_tokens per the rules above.\n"
        f"{mandatory_hint}"
        f"{generation_hint}"
        f"{retry}"
        "Seed (JSON):\n"
        f"{json.dumps(seed_row, ensure_ascii=False, indent=2)}"
    )


def _append_failed(failed_path: Path, row: Dict[str, Any]) -> None:
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(failed_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _call_api_once(
    *,
    api_url: str,
    api_key: str,
    model: str,
    timeout_sec: int,
    strict_json_schema: bool,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload: Dict[str, Any] = {"model": model, "temperature": 0, "messages": messages}
    if strict_json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "concept_semantic_entry",
                "strict": True,
                "schema": _make_concept_json_schema(),
            },
        }
    else:
        payload["response_format"] = {"type": "json_object"}
    resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout_sec)
    if resp.status_code >= 400:
        raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("LLM response content is not a string")
    content = _strip_json_fences(content)
    return json.loads(content)


def _aliases_cover_seed(generated: Dict[str, Any], seed_aliases: List[str]) -> Tuple[bool, str]:
    gen_set = {_normalize_label_text(a) for a in generated.get("aliases", []) if isinstance(a, str)}
    for a in seed_aliases:
        if _normalize_label_text(a) not in gen_set:
            return False, f"generated aliases missing seed alias: {a!r}"
    return True, "ok"


def _human_review_concepts(audit: Dict[str, Any]) -> List[str]:
    seen: Set[str] = set()
    details = audit.get("details", {})
    if not isinstance(details, dict):
        return []
    for _k, rows in details.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("concept"):
                seen.add(str(row["concept"]))
    for row in details.get("longest_match_risks", []) or []:
        if isinstance(row, dict):
            for k in ("owners_short", "owners_long"):
                for c in row.get(k) or []:
                    seen.add(str(c))
    return sorted(seen)


def _run_self_check_with_details(
    draft: Dict[str, Any], seed_key_list: List[str]
) -> Tuple[bool, List[Dict[str, str]]]:
    mod = _load_pipeline_module()
    failed_cases: List[Dict[str, str]] = []

    def _record(
        case_name: str, raw: str, enhanced: str, expected: str, actual: str, failure_reason: str
    ) -> None:
        failed_cases.append(
            {
                "case_name": case_name,
                "raw": raw,
                "enhanced": enhanced,
                "expected": expected,
                "actual": actual,
                "failure_reason": failure_reason,
            }
        )

    def _eval_case(case_name: str, raw: str, enhanced: str, expected_pass: bool) -> None:
        ctx = mod.retrieve_concept_semantics(raw, draft)
        ok_inj, reason_inj = mod.concept_semantic_injection_check(raw, enhanced, enhanced, ctx)
        ok_vfo, reason_vfo = mod.visual_form_option_leakage_check(raw, enhanced, enhanced, ctx)
        ok_tag, reason_tag = mod.concept_tag_output_leakage_check(raw, enhanced, enhanced, ctx)
        ok = bool(ok_inj and ok_vfo and ok_tag)
        if ok != expected_pass:
            actual = "pass" if ok else "failed_postcheck"
            expected = "pass" if expected_pass else "failed_postcheck"
            reasons = []
            if not ok_inj:
                reasons.append(reason_inj)
            if not ok_vfo:
                reasons.append(reason_vfo)
            if not ok_tag:
                reasons.append(reason_tag)
            _record(case_name, raw, enhanced, expected, actual, "; ".join([r for r in reasons if r]) or "unknown")

    for ck in seed_key_list:
        cfg = (draft.get("concepts", {}) or {}).get(ck)
        if not isinstance(cfg, dict):
            _record(f"{ck}_missing_in_draft", f"find {ck}", "", "pass", "failed_postcheck", "concept not present in draft")
            continue
        canonical = str(cfg.get("canonical_name", ck)).strip() or ck
        # Safe rewrite should pass if guidance exists.
        rewrite = None
        for r in cfg.get("safe_rewrite_guidance", []) or []:
            if isinstance(r, dict) and isinstance(r.get("allowed_rewrite"), str) and r["allowed_rewrite"].strip():
                rewrite = r["allowed_rewrite"].strip()
                break
        if rewrite:
            _eval_case(f"{ck}_safe_rewrite", f"find {canonical}", rewrite, True)

        # Internal tags/hints must fail when emitted in natural language.
        sg = cfg.get("slot_guidance", {}) if isinstance(cfg.get("slot_guidance"), dict) else {}
        for src in ("allowed_internal_tags", "allowed_slot_hints"):
            tags = sg.get(src, []) if isinstance(sg.get(src), list) else []
            for t in tags[:2]:
                if isinstance(t, str) and t.strip():
                    _eval_case(f"{ck}_{src}_leak", f"find {canonical}", f"the {t} {canonical}", False)

        # Non-none option emitted under vague raw should fail.
        opts = cfg.get("visual_form_options", []) if isinstance(cfg.get("visual_form_options"), list) else []
        for item in opts:
            if not isinstance(item, dict):
                continue
            option = str(item.get("option", "")).strip()
            if option and _normalize_label_text(option) != "none":
                _eval_case(f"{ck}_vfo_vague_raw_leak", f"find {canonical}", f"the {option}", False)
                break

    return len(failed_cases) == 0, failed_cases


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate concept_semantic_library draft via API (per concept).")
    p.add_argument("--concept-seed", default="configs/rrsisd_concepts_seed.json", help="Path to seed JSON.")
    p.add_argument("--output-json", default="configs/concept_semantic_library_draft.json", help="Output draft library path.")
    p.add_argument(
        "--api-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        help="Chat completions compatible endpoint.",
    )
    p.add_argument("--model", default="qwen-plus", help="Model name.")
    p.add_argument("--api-key-env", default="DASHSCOPE_API_KEY", help="Environment variable name for API key.")
    p.add_argument("--timeout-sec", type=int, default=90)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--initial-wait", type=float, default=1.0)
    p.add_argument("--max-wait", type=float, default=20.0)
    p.add_argument("--strict-json-schema", action="store_true", help="Use json_schema response_format.")
    p.add_argument("--overwrite", action="store_true", help="Ignore existing draft when merging; regenerate all seed concepts.")
    p.add_argument(
        "--audit-output",
        default="outputs/concept_library_draft_audit_report.json",
        help="Path for concept_library_audit JSON report.",
    )
    p.add_argument(
        "--failed-log",
        default="outputs/failed_concepts.jsonl",
        help="Append-only log for failed concept generations.",
    )
    p.add_argument(
        "--generation-report",
        default="outputs/concept_library_generation_report.json",
        help="Write final generation summary JSON here.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _forbidden_token_policy_self_check()
    seed_path = _resolve_path(args.concept_seed)
    out_path = _resolve_path(args.output_json)
    audit_path = _resolve_path(args.audit_output)
    failed_path = _resolve_path(args.failed_log)
    report_path = _resolve_path(args.generation_report)

    with open(seed_path, "r", encoding="utf-8") as f:
        seed_doc = json.load(f)
    if not isinstance(seed_doc, dict):
        raise SystemExit("Seed file must be a JSON object.")
    concepts_seed = seed_doc.get("concepts")
    if not isinstance(concepts_seed, list) or not concepts_seed:
        raise SystemExit("Seed must contain non-empty concepts array.")

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"Missing API key: environment variable {args.api_key_env!r} is not set.")

    draft: Dict[str, Any] = {
        "version": str(seed_doc.get("version", "draft")),
        "dataset": str(seed_doc.get("dataset", "RRSIS-D")),
        "global_constraints": copy.deepcopy(seed_doc.get("global_constraints", [])),
        "concepts": {},
    }
    if not isinstance(draft["global_constraints"], list):
        draft["global_constraints"] = []

    if args.overwrite:
        draft["concepts"] = {}
    elif out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict) and isinstance(existing.get("concepts"), dict):
                for ck, cfg in existing["concepts"].items():
                    if isinstance(cfg, dict) and _validate_generated_concept(cfg, ck, None)[0]:
                        draft["concepts"][ck] = cfg
        except (json.JSONDecodeError, OSError):
            pass

    sys_msg = _build_system_prompt()
    generated_count = 0
    failed_count = 0
    seed_ok_rows: List[Dict[str, Any]] = []

    for row in concepts_seed:
        if not isinstance(row, dict):
            failed_count += 1
            _append_failed(
                failed_path,
                {"timestamp": _utcnow(), "concept_key": None, "failure_reason": "seed row not an object"},
            )
            continue
        ck = row.get("concept_key")
        if not isinstance(ck, str) or not ck.strip():
            failed_count += 1
            _append_failed(
                failed_path,
                {"timestamp": _utcnow(), "concept_key": None, "failure_reason": "missing concept_key in seed"},
            )
            continue
        seed_ok_rows.append(row)
        if ck in draft["concepts"] and not args.overwrite:
            print(f"[skip] {ck} (already present in draft)", flush=True)
            continue

        last_reason = ""
        concept_success = False
        for attempt in range(1, args.max_attempts + 1):
            user_msg = _build_user_prompt(row, last_reason if last_reason else None)
            messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
            try:
                parsed = _call_api_once(
                    api_url=args.api_url,
                    api_key=api_key,
                    model=args.model,
                    timeout_sec=args.timeout_sec,
                    strict_json_schema=bool(args.strict_json_schema),
                    messages=messages,
                )
                ok, reason = _validate_generated_concept(parsed, ck, row)
                if not ok:
                    last_reason = reason
                    if attempt >= args.max_attempts:
                        raise ValueError(reason)
                    time.sleep(min(1.0, float(args.initial_wait)))
                    continue
                seed_aliases = row.get("aliases", [])
                if not isinstance(seed_aliases, list):
                    seed_aliases = []
                ok2, reason2 = _aliases_cover_seed(parsed, [a for a in seed_aliases if isinstance(a, str)])
                if not ok2:
                    last_reason = reason2
                    if attempt >= args.max_attempts:
                        raise ValueError(reason2)
                    time.sleep(min(1.0, float(args.initial_wait)))
                    continue
                draft["concepts"][ck] = copy.deepcopy(parsed)
                generated_count += 1
                concept_success = True
                print(f"[ok] {ck} (saved, attempt {attempt}/{args.max_attempts})", flush=True)
                break
            except Exception as e:
                msg = str(e)
                if len(msg) > 2000:
                    msg = msg[:2000] + "..."
                last_reason = (last_reason or "").strip() or msg
                _append_failed(
                    failed_path,
                    {
                        "timestamp": _utcnow(),
                        "concept_key": ck,
                        "attempt_index": attempt,
                        "max_attempts": args.max_attempts,
                        "failure_reason": last_reason,
                        "error_type": type(e).__name__,
                        "record_type": "attempt_failed",
                    },
                )
                if attempt >= args.max_attempts:
                    failed_count += 1
                    _append_failed(
                        failed_path,
                        {
                            "timestamp": _utcnow(),
                            "concept_key": ck,
                            "attempt_index": attempt,
                            "max_attempts": args.max_attempts,
                            "failure_reason": last_reason,
                            "error_type": type(e).__name__,
                            "attempts": args.max_attempts,
                            "record_type": "exhausted_summary",
                        },
                    )
                    print(f"[fail] {ck}: exhausted {args.max_attempts} attempts ({type(e).__name__})", flush=True)
                else:
                    time.sleep(min(1.0, float(args.initial_wait)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(draft, f, ensure_ascii=False, indent=2)

    audit_report: Dict[str, Any] = {}
    audit_proc: Dict[str, Any] = {}
    self_check_passed: Optional[bool] = None
    self_check_blocked_reason: Optional[str] = None
    self_check_failed_cases: List[Dict[str, str]] = []

    seed_target_count = len(seed_ok_rows)
    seed_key_list = [str(r["concept_key"]) for r in seed_ok_rows if isinstance(r.get("concept_key"), str)]
    concepts_present_in_draft_count = sum(1 for k in seed_key_list if k in draft.get("concepts", {}))

    if draft["concepts"]:
        cmd = [
            sys.executable,
            str(PIPELINE_PATH),
            "concept_library_audit",
            "--concept-semantic-library",
            str(out_path),
            "--audit-output",
            str(audit_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        audit_proc = {
            "returncode": proc.returncode,
            "stdout_chars": len(proc.stdout or ""),
            "stderr_tail": (proc.stderr or "")[-800:],
            "stdout_tail": (proc.stdout or "")[-800:],
        }
        try:
            audit_report = json.loads(proc.stdout) if proc.stdout.strip() else {}
        except json.JSONDecodeError:
            audit_report = {}
        if not audit_report:
            mod = _load_pipeline_module()
            audit_report = mod.concept_library_audit(draft)
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(audit_report, f, ensure_ascii=False, indent=2)
    else:
        audit_proc = {"skipped": True, "reason": "no concepts in draft"}

    if concepts_present_in_draft_count < seed_target_count:
        self_check_passed = False
        self_check_blocked_reason = "generated_concept_count_less_than_seed_count"
    elif draft["concepts"]:
        try:
            detail_passed, self_check_failed_cases = _run_self_check_with_details(draft, seed_key_list)
            mod = _load_pipeline_module()
            baseline_result = mod.concept_semantic_self_check(draft)
            baseline_failed_cases: List[Dict[str, Any]] = []
            if isinstance(baseline_result, dict):
                baseline_passed = bool(baseline_result.get("concept_semantic_self_check_passed", False))
                bfc = baseline_result.get("failed_cases", [])
                if isinstance(bfc, list):
                    baseline_failed_cases = [x for x in bfc if isinstance(x, dict)]
            else:
                baseline_passed = bool(baseline_result)
            self_check_passed = bool(detail_passed and baseline_passed)
            if (not baseline_passed) and detail_passed:
                if baseline_failed_cases:
                    self_check_failed_cases.extend(baseline_failed_cases)
                else:
                    self_check_failed_cases.append(
                        {
                            "case_name": "baseline_concept_semantic_self_check",
                            "raw": "",
                            "enhanced": "",
                            "expected": "pass",
                            "actual": "failed_postcheck",
                            "failure_reason": "baseline concept_semantic_self_check returned false",
                        }
                    )
        except Exception as e:
            self_check_passed = False
            self_check_failed_cases.append(
                {
                    "case_name": "self_check_exception",
                    "raw": "",
                    "enhanced": "",
                    "expected": "pass",
                    "actual": "exception",
                    "failure_reason": f"{type(e).__name__}: {str(e)[:500]}",
                }
            )
            audit_proc["self_check_error"] = str(e)[:500]
    else:
        self_check_passed = False
        self_check_blocked_reason = "no_concepts_in_draft"

    report: Dict[str, Any] = {
        "seed_concept_count": seed_target_count,
        "concepts_present_in_draft_count": concepts_present_in_draft_count,
        "generated_concept_count_this_run": generated_count,
        "failed_concept_count_this_run": failed_count,
        "output_json_path": str(out_path),
        "failed_concepts_path": str(failed_path),
        "audit_report_path": str(audit_path),
        "audit_subprocess": audit_proc,
        "concept_semantic_self_check_passed": self_check_passed,
        "self_check_blocked_reason": self_check_blocked_reason,
        "self_check_failed_cases": self_check_failed_cases,
        "audit_core": {
            "concept_count": audit_report.get("concept_count"),
            "alias_conflict_count": audit_report.get("alias_conflict_count"),
            "missing_field_count": audit_report.get("missing_field_count"),
            "high_risk_term_not_forbidden_count": audit_report.get("high_risk_term_not_forbidden_count"),
            "overly_strong_statement_count": audit_report.get("overly_strong_statement_count"),
            "visual_form_options_schema_error_count": audit_report.get("visual_form_options_schema_error_count"),
            "visual_form_option_emit_policy_missing_count": audit_report.get("visual_form_option_emit_policy_missing_count"),
            "default_none_option_missing_count": audit_report.get("default_none_option_missing_count"),
            "safe_rewrite_conflict_count": audit_report.get("safe_rewrite_conflict_count"),
        },
        "high_risk_term_not_forbidden_details_path": (
            "details.high_risk_term_not_forbidden"
            if int(audit_report.get("high_risk_term_not_forbidden_count") or 0) > 0
            else None
        ),
        "high_risk_term_not_forbidden_preview": (
            (audit_report.get("details", {}) or {}).get("high_risk_term_not_forbidden", [])[:5]
            if int(audit_report.get("high_risk_term_not_forbidden_count") or 0) > 0
            else []
        ),
        "human_review_concepts": _human_review_concepts(audit_report) if audit_report else [],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
