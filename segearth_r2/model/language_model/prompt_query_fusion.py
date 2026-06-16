"""Dual-granularity prompt fusion and SEG query refinement (DGP v6.1 guardrails)."""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# sigmoid(-4.60) ≈ 0.01, sigmoid(-3.89) ≈ 0.02
SIGMOID_GATE_G_LOGIT = -4.60
SIGMOID_GATE_L_LOGIT = -3.89


def pack_seg_hidden_states_bq(
    hidden_states: torch.Tensor,
    seg_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather [SEG] hidden states into [B, Q, H] with explicit validity mask."""
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be [B,L,H], got {tuple(hidden_states.shape)}")
    if seg_mask.shape[:2] != hidden_states.shape[:2]:
        raise ValueError(
            f"seg_mask {tuple(seg_mask.shape)} incompatible with hidden_states {tuple(hidden_states.shape)}"
        )
    B, _, H = hidden_states.shape
    seg_mask = seg_mask.bool()
    q_counts = seg_mask.sum(dim=1)
    Q_max = max(int(q_counts.max().item()), 1)
    out = hidden_states.new_zeros(B, Q_max, H)
    valid = torch.zeros(B, Q_max, dtype=torch.bool, device=hidden_states.device)
    for b in range(B):
        pos = seg_mask[b]
        n = int(pos.sum().item())
        if n > 0:
            out[b, :n] = hidden_states[b, pos]
            valid[b, :n] = True
    return out, valid


def expand_bq_for_mask_num(
    tensor_bq: torch.Tensor,
    mask_num,
) -> torch.Tensor:
    """Repeat each batch row by mask_num -> [B_exp, Q, D]."""
    if tensor_bq.dim() != 3:
        raise ValueError(f"tensor_bq must be [B,Q,D], got {tuple(tensor_bq.shape)}")
    if mask_num is None:
        return tensor_bq
    mn = torch.as_tensor(mask_num, device=tensor_bq.device, dtype=torch.long).flatten()
    if mn.numel() == 0:
        return tensor_bq
    if mn.numel() != tensor_bq.shape[0]:
        raise ValueError(
            f"mask_num length {mn.numel()} != batch {tensor_bq.shape[0]} for expand_bq_for_mask_num"
        )
    return torch.repeat_interleave(tensor_bq, mn, dim=0)


def expand_bp_for_mask_num(
    tensor_bp: torch.Tensor,
    mask_num,
) -> torch.Tensor:
    """Repeat each batch row by mask_num -> [B_exp, P, D]."""
    if tensor_bp.dim() != 3:
        raise ValueError(f"tensor_bp must be [B,P,D], got {tuple(tensor_bp.shape)}")
    if mask_num is None:
        return tensor_bp
    mn = torch.as_tensor(mask_num, device=tensor_bp.device, dtype=torch.long).flatten()
    if mn.numel() == 0:
        return tensor_bp
    if mn.numel() != tensor_bp.shape[0]:
        raise ValueError(
            f"mask_num length {mn.numel()} != batch {tensor_bp.shape[0]} for expand_bp_for_mask_num"
        )
    return torch.repeat_interleave(tensor_bp, mn, dim=0)


def _masked_mean_cosine(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean cosine similarity over valid query positions -> scalar."""
    cos = F.cosine_similarity(a, b, dim=-1)
    if mask is None:
        return cos.mean()
    valid = mask.bool()
    if not valid.any():
        return cos.new_zeros(())
    return cos.masked_fill(~valid, 0.0).sum() / valid.sum().clamp_min(1).to(cos.dtype)


def _attn_entropy(attn: torch.Tensor, query_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean attention entropy over valid queries -> scalar."""
    eps = 1e-8
    ent = -(attn * (attn + eps).log()).sum(dim=-1)
    if query_mask is None:
        return ent.mean()
    valid = query_mask.bool()
    if not valid.any():
        return ent.new_zeros(())
    return ent.masked_fill(~valid, 0.0).sum() / valid.sum().clamp_min(1).to(ent.dtype)


def _attn_entropy_normalized(
    attn: torch.Tensor,
    kv_mask: torch.Tensor,
    query_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """entropy / log(valid_kv_count) per query, averaged over valid queries."""
    eps = 1e-8
    valid_kv = kv_mask.bool().sum(dim=-1).clamp_min(1).to(attn.dtype)
    max_ent = torch.log(valid_kv + eps).unsqueeze(1)
    ent = -(attn * (attn + eps).log()).sum(dim=-1)
    ent_norm = ent / max_ent.clamp_min(eps)
    if query_mask is None:
        return ent_norm.mean()
    valid = query_mask.bool()
    if not valid.any():
        return ent_norm.new_zeros(())
    return ent_norm.masked_fill(~valid, 0.0).sum() / valid.sum().clamp_min(1).to(ent_norm.dtype)


def _detail_source_code(source: str) -> float:
    return {"phrase_span": 0.0, "refer_id": 1.0, "decoupled_attn": 2.0}.get(source, -1.0)


def _resolve_detail_source_name(
    target_phrase_mask: Optional[torch.Tensor],
    refer_span_mask: Optional[torch.Tensor],
    text_mask: torch.Tensor,
) -> str:
    """Diagnostic label only; P_l query always uses decoupled Q_detail."""
    if target_phrase_mask is not None:
        phrase_mask = target_phrase_mask.bool() & text_mask
        if phrase_mask.any(dim=1).any():
            return "phrase_span"

    if refer_span_mask is not None:
        refer_mask = refer_span_mask.bool() & text_mask
        if refer_mask.any(dim=1).any():
            return "refer_id"

    return "decoupled_attn"


class DualGranularityPromptAdapter(nn.Module):
    """Build P_g / P_l from Q_seg and instruction-only text tokens (v6.1 guardrails)."""

    def __init__(self, llm_dim: int, fuse_dim: int, pg_tokens: int = 1):
        super().__init__()
        self.llm_dim = int(llm_dim)
        self.fuse_dim = int(fuse_dim)
        self.pg_tokens = max(int(pg_tokens), 1)
        self.proj = nn.Linear(self.llm_dim, self.fuse_dim)
        self.ln_q = nn.LayerNorm(self.fuse_dim)
        self.ln_pg = nn.LayerNorm(self.fuse_dim)
        self.detail_proj = nn.Linear(self.fuse_dim, self.fuse_dim)

    @staticmethod
    def build_instruction_text_mask(
        attention_mask: torch.Tensor,
        seg_mask: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Instruction-only KV mask for text_tokens.
        Excludes pad, image, [SEG], and (when provided) non-instruction positions
        such as assistant answer / template tail.
        """
        text_mask = attention_mask.bool() & ~seg_mask.bool()
        if image_mask is not None:
            text_mask = text_mask & ~image_mask.bool()
        if instruction_mask is not None:
            text_mask = text_mask & instruction_mask.bool()
        return text_mask

    def _cross_attn(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        kv_mask: torch.Tensor,
        seg_query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, Q, D]
            key_value: [B, S, D]
            kv_mask: [B, S]
        Returns:
            out: [B, Q, D], attn: [B, Q, S]
        """
        scale = self.fuse_dim ** 0.5
        logits = torch.matmul(query, key_value.transpose(1, 2)) / scale
        logits = logits.masked_fill(~kv_mask.unsqueeze(1), -1e4)
        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, key_value)
        if seg_query_mask is not None:
            valid = seg_query_mask.bool().unsqueeze(-1)
            out = torch.where(valid, out, query)
        return out, attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        seg_mask: torch.Tensor,
        q_seg: torch.Tensor,
        target_phrase_mask: Optional[torch.Tensor] = None,
        refer_span_mask: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
        seg_query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            P_g: [B, Q, fuse_dim]
            P_l: [B, Q, fuse_dim]
            prompt_tokens: [B, 2Q, fuse_dim]
            prompt_mask: [B, 2Q]
            health: scalar tensors for monitoring
        """
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be [B,L,H_lm], got {tuple(hidden_states.shape)}")
        if q_seg.dim() != 3 or q_seg.shape[-1] != self.fuse_dim:
            raise ValueError(f"q_seg must be [B,Q,{self.fuse_dim}], got {tuple(q_seg.shape)}")

        text_mask = self.build_instruction_text_mask(
            attention_mask, seg_mask, image_mask, instruction_mask
        )
        text_tokens = self.proj(hidden_states)

        p_g, pg_attn = self._cross_attn(q_seg, text_tokens, text_mask, seg_query_mask)

        q_detail_raw = self.ln_q(q_seg) - self.ln_pg(p_g.detach())
        q_detail = self.detail_proj(q_detail_raw)
        p_l, pl_attn = self._cross_attn(q_detail, text_tokens, text_mask, seg_query_mask)

        detail_source = _resolve_detail_source_name(target_phrase_mask, refer_span_mask, text_mask)

        B, Q, _ = q_seg.shape
        prompt_tokens = torch.cat([p_g, p_l], dim=1)
        if seg_query_mask is None:
            prompt_mask = torch.ones(B, 2 * Q, dtype=torch.bool, device=q_seg.device)
        else:
            valid = seg_query_mask.bool()
            prompt_mask = torch.cat([valid, valid], dim=1)

        health = {
            "cos_pg_pl": _masked_mean_cosine(p_g, p_l, seg_query_mask),
            "cos_pg_qseg": _masked_mean_cosine(p_g, q_seg, seg_query_mask),
            "cos_pl_qseg": _masked_mean_cosine(p_l, q_seg, seg_query_mask),
            "cos_qdetail_qseg": _masked_mean_cosine(q_detail, q_seg, seg_query_mask),
            "entropy_pg_attention": _attn_entropy(pg_attn, seg_query_mask),
            "entropy_pl_attention": _attn_entropy(pl_attn, seg_query_mask),
            "entropy_pg_attention_norm": _attn_entropy_normalized(pg_attn, text_mask, seg_query_mask),
            "entropy_pl_attention_norm": _attn_entropy_normalized(pl_attn, text_mask, seg_query_mask),
            "detail_prompt_source": q_seg.new_tensor(_detail_source_code(detail_source)),
            "detail_prompt_source_name": detail_source,
            "_q_detail": q_detail,
        }

        if seg_query_mask is not None and seg_query_mask.any():
            pg_top = pg_attn.argmax(dim=-1).float()
            pl_top = pl_attn.argmax(dim=-1).float()
            flip = (pg_top.long() != pl_top.long()).to(q_seg.dtype)
            valid = seg_query_mask.bool()
            health["target_flip_rate"] = flip.masked_fill(~valid, 0.0).sum() / valid.sum().clamp_min(1).to(flip.dtype)
            health["pg_top_token_idx_mean"] = pg_top.masked_fill(~valid, 0.0).sum() / valid.sum().clamp_min(1).to(pg_top.dtype)
            health["pl_top_token_idx_mean"] = pl_top.masked_fill(~valid, 0.0).sum() / valid.sum().clamp_min(1).to(pl_top.dtype)
        else:
            health["target_flip_rate"] = q_seg.new_zeros(())
            health["pg_top_token_idx_mean"] = q_seg.new_zeros(())
            health["pl_top_token_idx_mean"] = q_seg.new_zeros(())

        return p_g, p_l, prompt_tokens, prompt_mask, health


class PromptAwareQueryRefiner(nn.Module):
    """Q_ref = Q_seg + gate_g * CrossAttn(Q_seg, P_g) + gate_l * CrossAttn(Q_seg, P_l)."""

    def __init__(
        self,
        dim: int = 256,
        hidden_dim: int = 512,
        gate_g_init: float = 0.01,
        gate_l_init: float = 0.02,
        use_sigmoid_gate: bool = False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.use_sigmoid_gate = bool(use_sigmoid_gate)
        if self.use_sigmoid_gate:
            self.gate_g_logit = nn.Parameter(torch.tensor([SIGMOID_GATE_G_LOGIT], dtype=torch.float32))
            self.gate_l_logit = nn.Parameter(torch.tensor([SIGMOID_GATE_L_LOGIT], dtype=torch.float32))
        else:
            self.gate_g = nn.Parameter(torch.tensor([float(gate_g_init)], dtype=torch.float32))
            self.gate_l = nn.Parameter(torch.tensor([float(gate_l_init)], dtype=torch.float32))
        self.ln_q = nn.LayerNorm(self.dim)
        self.ln_kv = nn.LayerNorm(self.dim)

    def _gate_values(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_sigmoid_gate:
            return torch.sigmoid(self.gate_g_logit), torch.sigmoid(self.gate_l_logit)
        return self.gate_g, self.gate_l

    def _cross_attn_delta(
        self,
        q_seg: torch.Tensor,
        prompt: torch.Tensor,
        prompt_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """CrossAttn(Q_seg, P): Q_seg queries attend over per-query prompt vectors."""
        q = self.ln_q(q_seg)
        kv = self.ln_kv(prompt)
        scale = self.dim ** 0.5
        logits = torch.matmul(q, kv.transpose(1, 2)) / scale
        if prompt_valid is not None:
            logits = logits.masked_fill(~prompt_valid.bool().unsqueeze(1), -1e4)
        attn = torch.softmax(logits, dim=-1)
        delta = torch.matmul(attn, kv)
        if prompt_valid is not None:
            valid = prompt_valid.bool().unsqueeze(-1)
            delta = torch.where(valid, delta, torch.zeros_like(delta))
        return delta

    def forward(
        self,
        seg_embedding: torch.Tensor,
        p_g: torch.Tensor,
        p_l: torch.Tensor,
        seg_query_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            seg_embedding: [B, Q, D] (Q_seg)
            p_g: [B, Q, D]
            p_l: [B, Q, D]
        Returns:
            Q_ref: [B, Q, D], health dict
        """
        if seg_embedding.dim() != 3 or p_g.dim() != 3 or p_l.dim() != 3:
            raise ValueError(
                f"seg_embedding {tuple(seg_embedding.shape)}, p_g {tuple(p_g.shape)}, "
                f"p_l {tuple(p_l.shape)} must be [B,Q,D]"
            )
        if seg_embedding.shape != p_g.shape or seg_embedding.shape != p_l.shape:
            raise ValueError(
                f"shape mismatch: seg={tuple(seg_embedding.shape)} p_g={tuple(p_g.shape)} p_l={tuple(p_l.shape)}"
            )

        gate_g, gate_l = self._gate_values()
        delta_g = self._cross_attn_delta(seg_embedding, p_g, seg_query_mask)
        delta_l = self._cross_attn_delta(seg_embedding, p_l, seg_query_mask)
        q_ref = seg_embedding + gate_g * delta_g + gate_l * delta_l

        if seg_query_mask is not None:
            valid = seg_query_mask.bool().unsqueeze(-1)
            q_ref = torch.where(valid, q_ref, seg_embedding)

        delta_g_norm = delta_g.detach().float().norm()
        delta_l_norm = delta_l.detach().float().norm()
        seg_norm = seg_embedding.detach().float().norm().clamp_min(1e-8)
        health = {
            "delta_g_norm": delta_g_norm,
            "delta_l_norm": delta_l_norm,
            "refiner_delta_norm": (gate_g.detach() * delta_g + gate_l.detach() * delta_l).detach().float().norm(),
            "seg_query_norm": seg_norm,
            "refiner_delta_over_seg": (gate_g.detach() * delta_g + gate_l.detach() * delta_l).detach().float().norm() / seg_norm,
            "cos_qref_qseg": _masked_mean_cosine(q_ref, seg_embedding, seg_query_mask),
            "query_refiner_gate_g": gate_g.detach().float().reshape(-1)[0],
            "query_refiner_gate_l": gate_l.detach().float().reshape(-1)[0],
            # backward-compatible alias
            "query_refiner_gate": gate_l.detach().float().reshape(-1)[0],
        }
        return q_ref, health

    def forward_with_deltas(
        self,
        seg_embedding: torch.Tensor,
        p_g: torch.Tensor,
        p_l: torch.Tensor,
        seg_query_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Like forward but also returns Delta_g / Delta_l for smoke diagnostics."""
        gate_g, gate_l = self._gate_values()
        delta_g = self._cross_attn_delta(seg_embedding, p_g, seg_query_mask)
        delta_l = self._cross_attn_delta(seg_embedding, p_l, seg_query_mask)
        q_ref = seg_embedding + gate_g * delta_g + gate_l * delta_l
        if seg_query_mask is not None:
            valid = seg_query_mask.bool().unsqueeze(-1)
            q_ref = torch.where(valid, q_ref, seg_embedding)
        delta_g_norm = delta_g.detach().float().norm()
        delta_l_norm = delta_l.detach().float().norm()
        seg_norm = seg_embedding.detach().float().norm().clamp_min(1e-8)
        health = {
            "delta_g_norm": delta_g_norm,
            "delta_l_norm": delta_l_norm,
            "refiner_delta_norm": (gate_g.detach() * delta_g + gate_l.detach() * delta_l).detach().float().norm(),
            "seg_query_norm": seg_norm,
            "refiner_delta_over_seg": (gate_g.detach() * delta_g + gate_l.detach() * delta_l).detach().float().norm() / seg_norm,
            "cos_qref_qseg": _masked_mean_cosine(q_ref, seg_embedding, seg_query_mask),
            "query_refiner_gate_g": gate_g.detach().float().reshape(-1)[0],
            "query_refiner_gate_l": gate_l.detach().float().reshape(-1)[0],
            "query_refiner_gate": gate_l.detach().float().reshape(-1)[0],
        }
        return q_ref, health, delta_g, delta_l
