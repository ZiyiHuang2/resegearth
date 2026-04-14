class StaticRSKB:
    def __init__(self, mode: str = "generic"):
        self.mode = mode
        self.prior = self._build_prior(mode)

    def _build_prior(self, mode: str) -> str:
        if mode == "generic":
            return (
                "Remote sensing image prior: object scales may vary greatly; "
                "small targets and fragmented boundaries are common; "
                "spatial relations and surrounding context such as roads, buildings, "
                "vegetation and water can help identify the target."
            )
        elif mode == "rrsisd":
            return (
                "Remote sensing referring segmentation prior: targets may appear at different scales; "
                "visual boundaries can be weak or fragmented; "
                "target identity often depends on location words, nearby objects, "
                "and scene context rather than category name alone."
            )
        else:
            return (
                "Remote sensing image prior: use spatial location, nearby objects, "
                "and scene context to identify the target."
            )

    def sanitize(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("<image>", "image")
        text = text.replace("<refer>", "refer")
        text = text.replace("[SEG]", "segment")
        return " ".join(text.split())

    def augment(self, instruction: str, max_chars: int = 320) -> str:
        instruction = self.sanitize(instruction)
        merged = f"{self.prior}\nQuery: {instruction}"
        if len(merged) > max_chars:
            merged = merged[:max_chars].rstrip()
        return merged