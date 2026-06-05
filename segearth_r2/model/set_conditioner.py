"""Pre-decoder set conditioning for LaSeRS multi-[SEG] samples."""

from __future__ import annotations

import json
import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_CATEGORY_WARNED = False
_EMPTY_CATEGORY_WARNED = False

DEFAULT_LASERS_CATEGORY_VOCAB = [
    "airplane", "airport", "baseball field", "basketball court", "bridge",
    "chimney", "dam", "dry cargo ship", "expressway service area",
    "expressway toll station", "golf course", "ground track field", "harbor",
    "helicopter", "intersection", "motorboat", "overpass", "park",
    "parking lot", "passenger ship", "playground", "railway", "river",
    "roundabout", "ship", "soccer ball field", "stadium", "storage tank",
    "swimming pool", "tennis court", "tugboats", "vehicle", "warship",
    "windmill",
]


def _normalize_phrase(phrase: str) -> str:
    text = " ".join(phrase.strip().lower().split())
    return text.strip(string.punctuation + " ")


class CategoryLabelStats:
    """Accumulates unknown / empty category labels for LaSeRS."""

    def __init__(self):
        self.unknown_phrases: Counter = Counter()
        self.empty_label_samples = 0
        self.empty_with_p_tag_samples = 0
        self.total_samples = 0

    def record_sample(
        self,
        phrases: Sequence[str],
        unknown: Sequence[str],
        multi_hot_sum: float,
    ):
        self.total_samples += 1
        for phrase in unknown:
            self.unknown_phrases[_normalize_phrase(phrase)] += 1
        if multi_hot_sum == 0:
            self.empty_label_samples += 1
            if phrases:
                self.empty_with_p_tag_samples += 1

    def summary(self) -> Dict[str, int]:
        return {
            "total_samples": self.total_samples,
            "empty_label_samples": self.empty_label_samples,
            "empty_with_p_tag_samples": self.empty_with_p_tag_samples,
            "unknown_unique": len(self.unknown_phrases),
            "unknown_total": int(sum(self.unknown_phrases.values())),
        }


def load_lasers_category_vocab(path: Optional[str] = None) -> List[str]:
    if path is not None:
        vocab_path = Path(path)
        if vocab_path.is_file():
            with open(vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "categories" in data:
                return [_normalize_phrase(x) for x in data["categories"]]
            if isinstance(data, list):
                return [_normalize_phrase(x) for x in data]
    return [_normalize_phrase(x) for x in DEFAULT_LASERS_CATEGORY_VOCAB]


def extract_category_phrases_from_answer(answer: str) -> List[str]:
    """Extract unique <p>...</p> phrases from GT answer (order preserved)."""
    if not answer:
        return []
    seen = set()
    phrases: List[str] = []
    for raw in re.findall(r"<p>\s*(.*?)\s*</p>", answer, flags=re.IGNORECASE):
        phrase = _normalize_phrase(raw)
        if phrase and phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)
    return phrases


def build_category_multi_hot(
    phrases: Sequence[str],
    vocab: Sequence[str],
) -> torch.Tensor:
    multi_hot, _, _ = build_category_set_labels(phrases, vocab)
    return multi_hot


def build_category_set_labels(
    phrases: Sequence[str],
    vocab: Sequence[str],
    stats: Optional[CategoryLabelStats] = None,
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """Build [vocab_size] multi-hot; return (labels, matched, unknown)."""
    vocab_index = {_normalize_phrase(v): i for i, v in enumerate(vocab)}
    multi_hot = torch.zeros(len(vocab), dtype=torch.float32)
    matched: List[str] = []
    unknown: List[str] = []
    for phrase in phrases:
        key = _normalize_phrase(phrase)
        if not key:
            continue
        if key in vocab_index:
            multi_hot[vocab_index[key]] = 1.0
            matched.append(key)
        else:
            unknown.append(key)
    if stats is not None:
        stats.record_sample(phrases, unknown, float(multi_hot.sum().item()))
    return multi_hot, matched, unknown


def build_category_set_labels_from_answer(
    answer: str,
    vocab: Sequence[str],
    stats: Optional[CategoryLabelStats] = None,
) -> Tuple[torch.Tensor, List[str], List[str], List[str]]:
    phrases = extract_category_phrases_from_answer(answer)
    labels, matched, unknown = build_category_set_labels(phrases, vocab, stats=stats)
    return labels, phrases, matched, unknown


def warn_empty_category_labels_once(phrases: Sequence[str], unknown: Sequence[str]):
    global _EMPTY_CATEGORY_WARNED
    if _EMPTY_CATEGORY_WARNED:
        return
    _EMPTY_CATEGORY_WARNED = True
    logger.warning(
        "LaSeRS sample has no recognized category in vocab (all-zero category_set_labels). "
        "phrases=%s unknown=%s",
        list(phrases),
        list(unknown),
    )


def regroup_seg_embeddings(
    seg_embedding: torch.Tensor,
    mask_num: Union[List[int], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Flatten [sum(K_i), C] -> padded [B, Kmax, C] with valid_mask."""
    if seg_embedding.dim() == 3 and seg_embedding.shape[1] == 1:
        seg_flat = seg_embedding.squeeze(1)
    else:
        seg_flat = seg_embedding

    if isinstance(mask_num, torch.Tensor):
        counts = [int(x) for x in mask_num.detach().cpu().tolist()]
    else:
        counts = [int(x) for x in mask_num]

    device = seg_flat.device
    dtype = seg_flat.dtype
    bsz = len(counts)
    kmax = max(counts) if counts else 0
    if kmax == 0:
        return (
            torch.zeros(bsz, 0, seg_flat.shape[-1], device=device, dtype=dtype),
            torch.zeros(bsz, 0, dtype=torch.bool, device=device),
            counts,
        )

    seg_group = torch.zeros(bsz, kmax, seg_flat.shape[-1], device=device, dtype=dtype)
    valid_mask = torch.zeros(bsz, kmax, dtype=torch.bool, device=device)
    offset = 0
    for b, k in enumerate(counts):
        if k > 0:
            seg_group[b, :k] = seg_flat[offset : offset + k]
            valid_mask[b, :k] = True
        offset += k
    return seg_group, valid_mask, counts


def flatten_seg_embeddings(
    seg_group: torch.Tensor,
    valid_mask: torch.Tensor,
    counts: Sequence[int],
) -> torch.Tensor:
    """Restore [sum(K_i), C] in the same order as regroup."""
    chunks = []
    for b, k in enumerate(counts):
        if k > 0:
            chunks.append(seg_group[b, :k])
    if not chunks:
        return seg_group.new_zeros(0, seg_group.shape[-1])
    return torch.cat(chunks, dim=0)


class SetConditioner(nn.Module):
    """Set-aware refinement on grouped SEG embeddings; Q_set modulates the main path."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        gate_init: float = 1e-3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_attn = nn.Linear(hidden_dim, 1, bias=False)
        self.set_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.residual_gate = nn.Parameter(torch.tensor(1e-3))
        self._init_refinement_modules(gate_init)

    def _init_refinement_modules(self, gate_init: float):
        gate_init = max(float(gate_init), 1e-3)
        nn.init.zeros_(self.set_proj.weight)
        nn.init.zeros_(self.set_proj.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.zeros_(self.gate_proj.weight)
        # Keep sigmoid(gate) low at init (not ~0.5); residual_gate scales the delta path.
        nn.init.constant_(self.gate_proj.bias, -4.0)
        with torch.no_grad():
            self.residual_gate.fill_(gate_init)

    def _pool_q_set(self, seg_group: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        scores = self.q_attn(seg_group).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        return torch.sum(weights.unsqueeze(-1) * seg_group, dim=1)

    def forward(
        self,
        seg_embedding: torch.Tensor,
        mask_num: Union[List[int], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seg_group, valid_mask, counts = regroup_seg_embeddings(seg_embedding, mask_num)
        if seg_group.numel() == 0:
            empty_q = seg_group.new_zeros(seg_group.shape[0], self.hidden_dim)
            empty_gate = seg_group.new_zeros(0)
            return seg_embedding, empty_q, valid_mask, empty_gate, empty_gate

        seg_group_orig = seg_group
        q_set = self._pool_q_set(seg_group_orig, valid_mask)
        q_set_expand = q_set.unsqueeze(1).expand(-1, seg_group_orig.shape[1], -1)
        seg_with_set = seg_group_orig + self.set_proj(q_set_expand)
        padding_mask = ~valid_mask
        encoded = self.set_encoder(seg_with_set, src_key_padding_mask=padding_mask)
        delta = self.output_proj(encoded)
        gate_input = torch.cat([seg_group_orig, q_set_expand], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        seg_refined = seg_group_orig + self.residual_gate * gate * delta
        seg_flat = flatten_seg_embeddings(seg_refined, valid_mask, counts)
        refined = seg_flat.unsqueeze(1)
        gate_valid = gate.masked_select(valid_mask.unsqueeze(-1))
        gate_mean = gate_valid.mean() if gate_valid.numel() else gate.new_tensor(0.0)
        return refined, q_set, valid_mask, gate_mean, gate


class CountHead(nn.Module):
    def __init__(self, hidden_dim: int, set_max_count: int = 10):
        super().__init__()
        self.set_max_count = set_max_count
        self.num_classes = set_max_count + 1
        self.head = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, q_set: torch.Tensor) -> torch.Tensor:
        return self.head(q_set)


class CategorySetHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, q_set: torch.Tensor) -> torch.Tensor:
        return self.head(q_set)


def compute_set_count_loss(
    count_head: CountHead,
    q_set: torch.Tensor,
    mask_num: Union[List[int], torch.Tensor],
    set_max_count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(mask_num, torch.Tensor):
        targets = mask_num.detach().long().flatten()
    else:
        targets = torch.tensor(mask_num, device=q_set.device, dtype=torch.long)
    targets = targets.clamp(0, set_max_count)
    logits = count_head(q_set)
    loss = F.cross_entropy(logits, targets)
    acc = (logits.argmax(dim=-1) == targets).float().mean()
    return loss, acc


def compute_set_category_loss(
    category_head: CategorySetHead,
    q_set: torch.Tensor,
    category_set_labels: torch.Tensor,
) -> torch.Tensor:
    logits = category_head(q_set)
    return F.binary_cross_entropy_with_logits(logits, category_set_labels.to(logits.dtype))


def warn_category_labels_missing_once():
    global _CATEGORY_WARNED
    if not _CATEGORY_WARNED:
        logger.warning(
            "use_set_category_loss=True but batch has no category_set_labels; "
            "skipping category loss for this step."
        )
        _CATEGORY_WARNED = True
