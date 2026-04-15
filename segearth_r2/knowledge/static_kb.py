# segearth_r2/knowledge/static_kb.py
class StaticRSKB:
    def __init__(self, mode: str = "rrsisd"):
        self.mode = mode
        self.prior = self._build_prior(mode)

    def _build_prior(self, mode: str) -> str:
        priors = {
            "rrsisd": (
                "Referring segmentation prior: identify the target mainly by category words, "
                "location words, nearby objects, and scene context. "
                "In remote sensing images, small targets and weak boundaries are common."
            ),
            "generic": (
                "Segmentation prior: use target category, spatial location, nearby objects, "
                "and scene context to identify the referred region."
            ),
        }
        return priors.get(mode, priors["generic"])

    def sanitize(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("<image>", "image")
        text = text.replace("<refer>", "refer")
        text = text.replace("[SEG]", "segment")
        return " ".join(text.split())

    def _safe_trim(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        trimmed = text[:max_chars]
        if " " in trimmed:
            trimmed = trimmed.rsplit(" ", 1)[0]
        return trimmed.rstrip(" ,;:.")

    def augment(self, instruction: str, max_chars: int = 220) -> str:
        instruction = self.sanitize(instruction)
        merged = f"{self.prior}\nQuery: {instruction}"          # ← 这里加了 \n
        return self._safe_trim(merged, max_chars)