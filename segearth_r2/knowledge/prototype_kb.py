import json
import os
import re
from typing import Dict, Optional

import torch


class PrototypeRSKB:
    def __init__(
        self,
        kb_path: Optional[str] = None,
        unknown_key: str = "unknown",
        visual_dim: int = 256,
    ):
        self.kb_path = kb_path
        self.unknown_key = unknown_key
        self.visual_dim = visual_dim
        self.entries = self._load_entries(kb_path)
        if self.unknown_key not in self.entries:
            self.entries[self.unknown_key] = {
                "text_proto": "a generic target object in remote sensing imagery",
                "visual_proto": [0.0] * visual_dim,
            }

    def _load_entries(self, kb_path: Optional[str]) -> Dict:
        if kb_path is None or not os.path.exists(kb_path):
            return {}
        with open(kb_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, dict):
            raise ValueError(f"Prototype KB must be a dict, got {type(entries)}")
        return entries

    def available_keys(self):
        return list(self.entries.keys())

    def _default_text(self, key: str) -> str:
        if key == self.unknown_key:
            return "a generic target object in remote sensing imagery"
        return f"a typical {key} target in remote sensing imagery"

    def _load_visual_proto(self, value):
        if value is None:
            return torch.zeros(self.visual_dim, dtype=torch.float32)
        if isinstance(value, list):
            tensor = torch.tensor(value, dtype=torch.float32)
        elif isinstance(value, str):
            if not os.path.exists(value):
                tensor = torch.zeros(self.visual_dim, dtype=torch.float32)
            else:
                tensor = torch.load(value, map_location="cpu")
                if isinstance(tensor, dict):
                    if "visual_proto" in tensor:
                        tensor = tensor["visual_proto"]
                    elif "feature" in tensor:
                        tensor = tensor["feature"]
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.tensor(tensor, dtype=torch.float32)
        else:
            tensor = torch.tensor(value, dtype=torch.float32)
        tensor = tensor.float().flatten()
        if tensor.numel() == 0:
            tensor = torch.zeros(self.visual_dim, dtype=torch.float32)
        return tensor

    def normalize_key(self, key: Optional[str]) -> str:
        if not key:
            return self.unknown_key
        key = str(key).strip().lower().replace("-", " ")
        if key in self.entries:
            return key
        return self.unknown_key

    def resolve_key_from_text(self, text: str) -> str:
        text = (text or "").lower()
        for key in self.entries.keys():
            if key == self.unknown_key:
                continue
            if re.search(rf"\b{re.escape(key)}\b", text):
                return key
        return self.unknown_key

    def get_item(self, key: Optional[str] = None, text: Optional[str] = None) -> Dict:
        if key is None:
            key = self.resolve_key_from_text(text or "")
        key = self.normalize_key(key)
        entry = self.entries.get(key, self.entries[self.unknown_key])
        text_proto = entry.get("text_proto", self._default_text(key))
        visual_proto = self._load_visual_proto(entry.get("visual_proto", entry.get("visual_proto_path")))
        return {
            "kb_key": key,
            "text_proto": text_proto,
            "visual_proto": visual_proto,
        }
