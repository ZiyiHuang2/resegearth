import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SemanticKBConfig:
    inject_mode: str = "hard"  # hard | always | never
    hard_query_max_tokens: int = 7
    max_prefix_chars: int = 72
    include_fields: str = "cat,rel,ctx,shape,scale"
    category_priors_path: Optional[str] = None
    relation_priors_path: Optional[str] = None


class SemanticRSKB:
    STOP_WORDS = {
        "the", "a", "an", "this", "that", "these", "those",
        "in", "on", "at", "of", "to", "with", "and", "for",
        "segment", "image",
    }

    def __init__(self, config: Optional[SemanticKBConfig] = None):
        self.cfg = config or SemanticKBConfig()
        valid_modes = {"hard", "always", "never"}
        if self.cfg.inject_mode not in valid_modes:
            raise ValueError(f"Unsupported inject_mode: {self.cfg.inject_mode}")

        base_dir = os.path.dirname(__file__)
        category_path = self.cfg.category_priors_path or os.path.join(base_dir, "category_priors.json")
        relation_path = self.cfg.relation_priors_path or os.path.join(base_dir, "relation_priors.json")

        self.category_priors = self._load_json(category_path)
        self.relation_priors = self._load_json(relation_path)

        self.category_keys = sorted(self.category_priors.keys(), key=lambda x: len(x), reverse=True)
        self.relation_keys = sorted(self.relation_priors.keys(), key=lambda x: len(x), reverse=True)
        self.include_fields = [field.strip() for field in self.cfg.include_fields.split(",") if field.strip()]

    def _load_json(self, path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _normalize(self, query: str) -> str:
        query = query.lower().strip()
        query = query.replace("<image>", "image").replace("<refer>", "refer")
        query = query.replace("[seg]", "segment")
        query = re.sub(r"\s+", " ", query)
        return query

    def _tokenize(self, query: str):
        return re.findall(r"[a-z0-9]+", query)

    def _find_phrase(self, query: str, phrases) -> str:
        padded = f" {query} "
        for phrase in phrases:
            if f" {phrase} " in padded:
                return phrase
        return ""

    def _extract_context(self, query: str, relation: str) -> str:
        if not relation:
            return ""
        match = re.search(rf"{re.escape(relation)}\s+([a-z0-9]+(?:\s+[a-z0-9]+)?)", query)
        if not match:
            return ""
        context = match.group(1).strip()
        context_tokens = [token for token in self._tokenize(context) if token not in self.STOP_WORDS]
        return " ".join(context_tokens[:2]).strip()

    def parse(self, query: str) -> Dict[str, str]:
        query_norm = self._normalize(query)
        tokens = self._tokenize(query_norm)
        category = self._find_phrase(query_norm, self.category_keys)
        relation = self._find_phrase(query_norm, self.relation_keys)
        context = self._extract_context(query_norm, relation)

        if not category:
            for token in tokens:
                if token not in self.STOP_WORDS:
                    category = token
                    break

        category_prior = self.category_priors.get(category, {})

        result = {
            "query": query_norm,
            "cat": category,
            "rel": relation,
            "ctx": context,
            "shape": category_prior.get("shape", ""),
            "scale": category_prior.get("scale", ""),
        }
        return result

    def _is_hard_sample(self, parsed: Dict[str, str]) -> bool:
        tokens = self._tokenize(parsed["query"])
        short_query = len(tokens) <= self.cfg.hard_query_max_tokens
        has_relative_relation = bool(parsed["rel"]) and (
            self.relation_priors.get(parsed["rel"], {}).get("type") == "relative_location"
        )
        has_any_relation = bool(parsed["rel"])
        has_context = bool(parsed["ctx"])
        category_prior = self.category_priors.get(parsed["cat"], {})
        is_small_target = bool(category_prior.get("small_target", False)) or category_prior.get("scale") == "small"
        weak_category = parsed["cat"] == "" or parsed["cat"] in {"target", "object"}
        return short_query and (has_relative_relation or weak_category or (is_small_target and (has_any_relation or has_context)))

    def should_inject(self, parsed: Dict[str, str]) -> bool:
        if self.cfg.inject_mode == "always":
            return True
        if self.cfg.inject_mode == "never":
            return False
        return self._is_hard_sample(parsed)

    def build_prefix(self, parsed: Dict[str, str]) -> str:
        parts = []
        for key in self.include_fields:
            value = parsed.get(key, "")
            if value:
                parts.append(f"{key}={value}")
        prefix = " ".join(parts).strip()
        if len(prefix) <= self.cfg.max_prefix_chars:
            return prefix
        trimmed = prefix[: self.cfg.max_prefix_chars]
        if " " in trimmed:
            trimmed = trimmed.rsplit(" ", 1)[0]
        return trimmed.strip()

    def augment(self, query: str) -> str:
        parsed = self.parse(query)
        if not self.should_inject(parsed):
            return parsed["query"]
        prefix = self.build_prefix(parsed)
        return f"{prefix} || {parsed['query']}" if prefix else parsed["query"]