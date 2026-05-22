"""
Test-time controlled prompt ensemble for RRSIS-D (inference only).

Reads segearth_r2/configs/prompts/rrsisd_prompt_v2.yaml and generates K paraphrases
with prompt_0 fixed to the original instruction.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _word_count(text: str) -> int:
    return len(text.strip().split())


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _pick_variant(synonyms: List[str], variant_idx: int) -> str:
    if not synonyms:
        return ""
    return synonyms[variant_idx % len(synonyms)]


def _word_level_synonyms(synonyms: List[str]) -> List[str]:
    """Single-token synonyms safe for in-phrase substitution."""
    words = [s for s in synonyms if s and " " not in s.strip()]
    return words if words else list(synonyms)


def _find_word_boundary_match(text_lower: str, phrase: str) -> Optional[re.Match]:
    phrase = phrase.strip().lower()
    if not phrase:
        return None
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.search(pattern, text_lower)


class RRSISDPromptEnsembleGenerator:
    """Generate K positive prompts from a raw RRSIS-D referring expression."""

    def __init__(self, config_path: str):
        config_path = os.path.abspath(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)

        self.k_prompts = int(self.cfg["k_prompts"])
        self.templates: List[str] = list(self.cfg.get("templates") or [])
        self.rules: Dict[str, Any] = dict(self.cfg.get("rules") or {})

        self.target_synonyms: Dict[str, List[str]] = dict(self.cfg.get("target_synonyms") or {})
        self.spatial_synonyms: Dict[str, List[str]] = dict(self.cfg.get("spatial_synonyms") or {})
        self.relation_synonyms: Dict[str, List[str]] = dict(self.cfg.get("relation_synonyms") or {})
        self.attribute_synonyms: Dict[str, List[str]] = dict(self.cfg.get("attribute_synonyms") or {})

        fallback = self.rules.get("fallback_mode")
        if fallback is None:
            fallback = self.cfg.get("fallback_mode") or []
        self.fallback_templates: List[str] = list(fallback)

        self.min_words = int(self.rules.get("min_words", 0))
        self.max_words = int(self.rules.get("max_words", 10_000))
        self.keep_original_as_prompt_0 = bool(self.rules.get("keep_original_as_prompt_0", True))
        self.deduplicate = bool(self.rules.get("deduplicate", True))
        self.lite_only = bool(self.rules.get("lite_only", False))
        self.original_weight = float(self.rules.get("original_weight", 0.6))
        self.area_ratio_min = float(self.rules.get("area_ratio_min", 0.5))
        self.area_ratio_max = float(self.rules.get("area_ratio_max", 1.8))
        self.enable_keyword_gate = bool(self.rules.get("enable_keyword_gate", False))
        self.trigger_keywords = [str(x).lower() for x in (self.rules.get("trigger_keywords", []) or [])]
        self.confidence_ratio_min = float(self.rules.get("confidence_ratio_min", 0.9))

        self._target_entries = self._build_synonym_entries(self.target_synonyms)
        self._spatial_entries = self._build_synonym_entries(self.spatial_synonyms)
        # Detect spatial words by canonical key only (avoid matching "on the left" before "left").
        self._spatial_detect_entries = [
            (key, key, list(self.spatial_synonyms.get(key, []) or [key]))
            for key in self.spatial_synonyms
        ]
        self._relation_entries = self._build_synonym_entries(self.relation_synonyms)
        self._attribute_entries = self._build_synonym_entries(self.attribute_synonyms)

    @staticmethod
    def _build_synonym_entries(syn_map: Dict[str, List[str]]) -> List[Tuple[str, str, List[str]]]:
        """(canonical_key, matched_phrase, all_synonyms) sorted by phrase length desc."""
        entries: List[Tuple[str, str, List[str]]] = []
        for key, syns in syn_map.items():
            phrases = [key] + list(syns)
            seen = set()
            ordered: List[str] = []
            for p in phrases:
                pl = p.strip().lower()
                if pl and pl not in seen:
                    seen.add(pl)
                    ordered.append(p.strip())
            for phrase in sorted(ordered, key=lambda x: len(x), reverse=True):
                entries.append((key, phrase, ordered))
        entries.sort(key=lambda x: len(x[1]), reverse=True)
        return entries

    def _match_leftmost(
        self, text_lower: str, entries: List[Tuple[str, str, List[str]]]
    ) -> Optional[Tuple[str, str, List[str], re.Match]]:
        """Pick the earliest match in the sentence (main referent is usually leftmost)."""
        best: Optional[Tuple[int, int, str, str, List[str], re.Match]] = None
        for key, phrase, syns in entries:
            m = _find_word_boundary_match(text_lower, phrase)
            if m is None:
                continue
            score = (m.start(), -len(phrase))
            if best is None or score < (best[0], best[1]):
                best = (m.start(), -len(phrase), key, phrase, syns, m)
        if best is None:
            return None
        return best[2], best[3], best[4], best[5]

    def _match_all(
        self, text_lower: str, entries: List[Tuple[str, str, List[str]]]
    ) -> List[Tuple[int, str, str, List[str], re.Match]]:
        hits: List[Tuple[int, str, str, List[str], re.Match]] = []
        for key, phrase, syns in entries:
            m = _find_word_boundary_match(text_lower, phrase)
            if m is not None:
                hits.append((m.start(), key, phrase, syns, m))
        hits.sort(key=lambda x: x[0])
        return hits

    def _match_target(
        self, text_lower: str, attr_match: Optional[re.Match]
    ) -> Optional[Tuple[str, str, List[str], re.Match]]:
        hits = self._match_all(text_lower, self._target_entries)
        if not hits:
            return None
        if len(hits) == 1:
            _, key, phrase, syns, m = hits[0]
            return key, phrase, syns, m

        best = None
        best_score = None
        for start, key, phrase, syns, m in hits:
            score = -start
            before = text_lower[:start].rstrip()
            if before.endswith("of the") or before.endswith("of a"):
                score -= 1000
            if attr_match is not None and attr_match.end() <= start:
                score += 50
            if best_score is None or score > best_score:
                best_score = score
                best = (key, phrase, syns, m)
        return best

    def _match_relation(
        self, text_lower: str, spatial_match: Optional[re.Match]
    ) -> Optional[Tuple[str, str, List[str], re.Match]]:
        hits = self._match_all(text_lower, self._relation_entries)
        if not hits:
            return None
        if spatial_match is not None:
            for start, key, phrase, syns, m in hits:
                if start >= spatial_match.start():
                    return key, phrase, syns, m
        _, key, phrase, syns, m = hits[-1]
        return key, phrase, syns, m

    def _extract_spatial_phrase(self, text: str, text_lower: str, spatial_match: Optional[re.Match]) -> str:
        if spatial_match is None:
            return ""
        start = spatial_match.start()
        # Prefer full prepositional phrase when spatial cue is mid-sentence.
        for prefix in ("at the ", "on the ", "in the ", "to the "):
            idx = text_lower.rfind(prefix, 0, start + 1)
            if idx >= 0:
                return text[idx:].strip()
        return text[start:].strip()

    def _extract_relation_phrase(
        self,
        text_lower: str,
        relation_match: Optional[re.Match],
        spatial_phrase: str = "",
    ) -> str:
        if spatial_phrase:
            return spatial_phrase
        if relation_match is None:
            return ""
        rel = relation_match.group(0).strip().lower()
        idx = relation_match.start()
        tail = text_lower[idx:].strip()
        if rel == "of" and " of " in tail:
            return tail
        return rel

    def _extract_area_phrase(self, spatial_phrase: str, spatial_key: Optional[str]) -> str:
        if not spatial_phrase:
            return ""
        sp = spatial_phrase.strip()
        sp_lower = sp.lower()
        m = re.match(r"^(?:at|on|in)\s+the\s+(.+)$", sp_lower, flags=re.IGNORECASE)
        if m:
            return sp[m.start(1) :].strip()
        for prefix in ("at the ", "on the ", "in the "):
            if sp_lower.startswith(prefix):
                return sp[len(prefix) :].strip()
        return sp

    def _parse_instruction(self, instruction: str) -> Dict[str, Any]:
        text = instruction.strip()
        text_lower = text.lower()

        attr_key = attr_surface = None
        attr_match = None
        a_hit = self._match_leftmost(text_lower, self._attribute_entries)
        if a_hit is not None:
            attr_key, attr_surface, attr_syns, attr_match = a_hit

        target_key = target_surface = None
        target_match = None
        t_hit = self._match_target(text_lower, attr_match)
        if t_hit is not None:
            target_key, target_surface, target_syns, target_match = t_hit

        spatial_key = spatial_surface = None
        spatial_match = None
        s_hit = self._match_leftmost(text_lower, self._spatial_detect_entries)
        if s_hit is not None:
            spatial_key, spatial_surface, spatial_syns, spatial_match = s_hit

        relation_key = relation_surface = None
        relation_match = None
        r_hit = self._match_relation(text_lower, spatial_match)
        if r_hit is not None:
            relation_key, relation_surface, relation_syns, relation_match = r_hit

        spatial_phrase = self._extract_spatial_phrase(text, text_lower, spatial_match)
        relation_phrase = self._extract_relation_phrase(
            text_lower, relation_match, spatial_phrase=spatial_phrase
        )
        area_phrase = self._extract_area_phrase(spatial_phrase, spatial_key)

        return {
            "original": text,
            "target_key": target_key,
            "target_surface": target_surface,
            "target_syns": self.target_synonyms.get(target_key, []) if target_key else [],
            "attr_key": attr_key,
            "attr_surface": attr_surface,
            "attr_syns": self.attribute_synonyms.get(attr_key, []) if attr_key else [],
            "spatial_key": spatial_key,
            "spatial_surface": spatial_surface,
            "spatial_syns": self.spatial_synonyms.get(spatial_key, []) if spatial_key else [],
            "spatial_phrase": spatial_phrase,
            "relation_key": relation_key,
            "relation_surface": relation_surface,
            "relation_syns": self.relation_synonyms.get(relation_key, []) if relation_key else [],
            "relation_phrase": relation_phrase,
            "area_phrase": area_phrase,
        }

    def _is_valid_prompt(self, prompt: str) -> bool:
        wc = _word_count(prompt)
        return self.min_words <= wc <= self.max_words

    def _fill_template(self, template: str, parsed: Dict[str, Any], variant_idx: int) -> Optional[str]:
        attr = ""
        if parsed.get("attr_key"):
            attr = _pick_variant(parsed["attr_syns"], variant_idx)

        target = ""
        if parsed.get("target_key"):
            target = _pick_variant(parsed["target_syns"], variant_idx)
        elif parsed.get("target_surface"):
            target = parsed["target_surface"]

        spatial_phrase = parsed.get("spatial_phrase") or ""
        if parsed.get("spatial_key") and spatial_phrase:
            spatial_word = _pick_variant(_word_level_synonyms(parsed["spatial_syns"]), variant_idx)
            if parsed.get("spatial_surface"):
                spatial_phrase = re.sub(
                    r"\b" + re.escape(parsed["spatial_surface"]) + r"\b",
                    spatial_word,
                    spatial_phrase,
                    count=1,
                    flags=re.IGNORECASE,
                )
        elif parsed.get("spatial_key") and not spatial_phrase:
            spatial_word = _pick_variant(parsed["spatial_syns"], variant_idx)
            spatial_phrase = f"at the {spatial_word}"

        relation_phrase = parsed.get("relation_phrase") or ""
        if parsed.get("relation_key") and relation_phrase:
            rel_word = _pick_variant(parsed["relation_syns"], variant_idx)
            if parsed.get("relation_surface"):
                relation_phrase = re.sub(
                    re.escape(parsed["relation_surface"]),
                    rel_word,
                    relation_phrase,
                    count=1,
                    flags=re.IGNORECASE,
                )

        area_phrase = parsed.get("area_phrase") or ""
        if parsed.get("spatial_key") and area_phrase and parsed.get("spatial_surface"):
            area_word = _pick_variant(_word_level_synonyms(parsed["spatial_syns"]), variant_idx)
            area_phrase = re.sub(
                r"\b" + re.escape(parsed["spatial_surface"]) + r"\b",
                area_word,
                area_phrase,
                count=1,
                flags=re.IGNORECASE,
            )

        if not target and not spatial_phrase and not relation_phrase:
            return None

        try:
            prompt = template.format(
                attr=attr,
                target=target or "object",
                spatial_phrase=spatial_phrase,
                relation_phrase=relation_phrase,
                area_phrase=area_phrase,
            )
        except KeyError:
            return None

        prompt = re.sub(r"\s+", " ", prompt).strip()
        prompt = re.sub(r"\s+,", ",", prompt)
        return prompt

    def _dedupe_preserve_order(self, prompts: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for p in prompts:
            key = _norm_key(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(p.strip())
        return out

    def generate(self, instruction: str) -> List[str]:
        instruction = instruction.strip()
        prompts: List[str] = []

        # Conservative mode: only original + light templates/fallback.
        if self.lite_only:
            if self.keep_original_as_prompt_0 and instruction:
                prompts.append(instruction)
            for fb in self.fallback_templates:
                if len(prompts) >= self.k_prompts:
                    break
                candidate = fb.format(original=instruction)
                candidate = re.sub(r"\s+", " ", candidate).strip()
                if candidate and self._is_valid_prompt(candidate):
                    prompts.append(candidate)
            if self.deduplicate:
                prompts = self._dedupe_preserve_order(prompts)
            if self.keep_original_as_prompt_0 and instruction:
                rest = [p for p in prompts if _norm_key(p) != _norm_key(instruction)]
                prompts = [instruction] + rest
            while len(prompts) < self.k_prompts and instruction:
                prompts.append(instruction)
            return prompts[: self.k_prompts]

        parsed = self._parse_instruction(instruction)

        if self.keep_original_as_prompt_0:
            prompts.append(instruction)

        variant_idx = 0
        for template in self.templates:
            if len(prompts) >= self.k_prompts:
                break
            candidate = self._fill_template(template, parsed, variant_idx)
            variant_idx += 1
            if candidate is None:
                continue
            if not self._is_valid_prompt(candidate):
                continue
            prompts.append(candidate)

        while len(prompts) < self.k_prompts:
            filled_any = False
            for fb in self.fallback_templates:
                if len(prompts) >= self.k_prompts:
                    break
                candidate = fb.format(original=instruction)
                candidate = re.sub(r"\s+", " ", candidate).strip()
                if self._is_valid_prompt(candidate):
                    prompts.append(candidate)
                    filled_any = True
            if not filled_any:
                break

        if self.deduplicate:
            prompts = self._dedupe_preserve_order(prompts)

        if self.keep_original_as_prompt_0 and instruction:
            # Ensure prompt_0 is always the original sentence.
            rest = [p for p in prompts if _norm_key(p) != _norm_key(instruction)]
            prompts = [instruction] + rest

        if len(prompts) < self.k_prompts and instruction:
            while len(prompts) < self.k_prompts:
                prompts.append(instruction)

        return prompts[: self.k_prompts]



def build_token_refer_id(tokenizer, instruction: str, refer_token: str = "[SEG]"):
    """Same tokenization as RRSISDDataset.preprocess_referring_instruction."""
    tokenized = tokenizer.encode(instruction, add_special_tokens=False)
    refer_token_id = [tokenizer.encode(refer_token, add_special_tokens=False)[0]]
    tokenized = tokenized + refer_token_id
    import torch

    return torch.tensor(tokenized)
