# TG-Swin-WTI v1.5 — Design Brief

**Fork:** `/root/rivermind-data/huangziyi/reseg/segearth+tgswin`

## Positioning

| Version | Role |
|---------|------|
| **v1** | Clean probe — single flat `text_cond`, per-key token bias `[BW,1,1,N]` |
| **v1.5** | **Short-run adjudication build** — enough Swin-internal text–vision complexity to decide whether WTI is worth v2; **not** full v2 |
| **v2 (future)** | PES / memory / contrastive / decoder loops — **explicitly out of v1.5** |

## v1.5 contributions

1. **Stage-wise Text Router (TCF)** — distinct condition per Swin stage via seg / phrase / relation-attention branches + stage gates
2. **Head-aware Low-rank WTI** — pairwise attention bias `[BW,H,N,N]` from low-rank QK routing modulated by text
3. **Mechanism Stats** — `pop_stats()` for context-leak diagnosis without storing full attention maps

## Identity principle

Forward identity at step 0: **`alpha = 0`** → `tanh(alpha)=0` gates all WTI bias to zero.

- Router / projection weights use **small random init** (not all-zero)
- `raw_bias` can be non-zero at init → **`alpha.grad` is non-zero** on first backward
- Baseline-compatible forward without manual perturbation

## Explicit non-goals (v1.5)

- PES / progressive evidence state
- Stage memory tokens
- Context contrastive loss
- CS-DEG decoder evidence loop
- SET++ / union / group query
- Mask decoder query logic changes
- Pixel decoder / MSDeformAttn changes
- Swin backbone unfreeze

## Insertion point

`WindowAttention.forward` — additive `attn_text_bias` before softmax.

## Warmstart anchor

```
/root/rivermind-data/huangziyi/reseg/output/base/standard-base-lasers-siglip1-8w-gd4/merged_model
```

Baseline multi_cate gIoU anchor: **42.45**
