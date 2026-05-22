import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SSKBConfig:
    min_query_tokens: int = 3
    max_prefix_chars: int = 64
    inject_mode: str = "auto"  # auto | always | never


class StructuredRSKB:
    LOC_WORDS = {
        "left", "right", "top", "bottom", "upper", "lower",
        "near", "next", "behind", "front", "center", "middle",
    }
    STOP_WORDS = {
        "the", "a", "an", "in", "on", "at", "of", "to", "with", "and", "this", "that"
    }

    def __init__(self, config: Optional[SSKBConfig] = None):
        self.cfg = config or SSKBConfig()
        valid_modes = {"auto", "always", "never"}
        if self.cfg.inject_mode not in valid_modes:
            raise ValueError(f"Unsupported inject_mode: {self.cfg.inject_mode}")

    def _normalize(self, query: str) -> str:
        query = query.lower().strip()
        query = re.sub(r"\s+", " ", query)
        return query

    def _tokenize(self, query: str):
        return re.findall(r"[a-z0-9]+", query)

    def parse_slots(self, query: str) -> Dict[str, str]:
        q = self._normalize(query)
        tokens = self._tokenize(q)
        if not tokens:
            return {}

        cat = next(
            (
                token
                for token in tokens
                if token not in self.STOP_WORDS and token not in self.LOC_WORDS
            ),
            "",
        )

        loc = next((token for token in tokens if token in self.LOC_WORDS), "")

        ctx = ""
        if loc and loc in tokens:
            loc_idx = tokens.index(loc)
            for token in tokens[loc_idx + 1:]:
                if token not in self.STOP_WORDS and token not in self.LOC_WORDS:
                    ctx = token
                    break

        slots = {}
        if cat:
            slots["cat"] = cat
        if loc:
            slots["loc"] = loc
        if ctx:
            slots["ctx"] = ctx
        return slots

    def should_inject(self, query: str, slots: Dict[str, str]) -> bool:
        if self.cfg.inject_mode == "always":
            return True
        if self.cfg.inject_mode == "never":
            return False
        tokens = self._tokenize(self._normalize(query))
        return len(tokens) >= self.cfg.min_query_tokens and ("loc" in slots or "ctx" in slots)

    def build_prefix(self, slots: Dict[str, str]) -> str:
        parts = []
        if "cat" in slots:
            parts.append(f"cat={slots['cat']}")
        if "loc" in slots:
            parts.append(f"loc={slots['loc']}")
        if "ctx" in slots:
            parts.append(f"ctx={slots['ctx']}")
        prefix = " ".join(parts).strip()
        return prefix[: self.cfg.max_prefix_chars].strip()

    def augment(self, query: str) -> str:
        query_norm = self._normalize(query)
        slots = self.parse_slots(query_norm)
        if not self.should_inject(query_norm, slots):
            return query_norm
        prefix = self.build_prefix(slots)
        return f"{prefix} || {query_norm}" if prefix else query_norm