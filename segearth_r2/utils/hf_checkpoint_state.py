# Utilities to load flat HF checkpoints (single bin / safetensors / sharded) for gate restore checks.
import json
import os
from typing import Dict, List, Tuple

import torch


def load_flat_state_dict_from_hf_folder(folder: str) -> Dict[str, torch.Tensor]:
    """Load all weight tensors from a HuggingFace-style model directory (CPU)."""
    folder = os.path.abspath(os.path.expanduser(folder))
    bin_path = os.path.join(folder, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu")

    st_path = os.path.join(folder, "model.safetensors")
    if os.path.isfile(st_path):
        from safetensors.torch import load_file

        return dict(load_file(st_path))

    index_path = os.path.join(folder, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        from safetensors.torch import load_file

        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        shards = sorted(set(weight_map.values()))
        sd: Dict[str, torch.Tensor] = {}
        for shard in shards:
            shard_path = os.path.join(folder, shard)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Missing shard file: {shard_path}")
            sd.update(load_file(shard_path))
        return sd

    raise FileNotFoundError(
        f"No pytorch_model.bin, model.safetensors, or model.safetensors.index.json under: {folder}"
    )


def gate_keys_in_state_dict(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    return sorted(k for k in state_dict if k.startswith("seg_visual_prior_gate."))


def count_gate_keys(state_dict: Dict[str, torch.Tensor]) -> int:
    return len(gate_keys_in_state_dict(state_dict))


def filter_state_dict_by_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in state_dict.items() if k.startswith(prefix)}


def hf_folder_contains_trained_gate(folder: str) -> bool:
    try:
        sd = load_flat_state_dict_from_hf_folder(folder)
    except FileNotFoundError:
        return False
    return count_gate_keys(sd) > 0
