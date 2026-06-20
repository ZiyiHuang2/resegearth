# TG-Swin-WTI v1.5 — Implementation Spec

**Fork:** `/root/rivermind-data/huangziyi/reseg/segearth+tgswin`  
**Python:** `/root/rivermind-data/miniconda3/envs/reseg/bin/python`

## Config

| File | TG_SWIN |
|------|---------|
| `maskformer2_swin_base_384_bs16_50ep.yaml` | `ENABLED: false` |
| `maskformer2_tgswin.yaml` | v1.5 enabled (alias) |
| `maskformer2_tgswin_v15.yaml` | v1.5 enabled (run script default) |

Key fields: `VERSION: "v1.5"`, `STAGE_ROUTER`, `HEAD_AWARE`, `LOW_RANK_QK`, `WTI_RANK`, `NUM_STAGES`, `WTI_STAGES`.

## TCF — `text_condition_factory.py`

### StageWiseTextRouter (v1.5)

| | |
|---|---|
| Input | `seg_hidden [N,D]`, `phrase_hidden [N,L,D]`, `phrase_mask [N,L]` |
| Output | `stage_text_cond [N,S,COND_DIM]`, `reliability [N,S,1]` |

Branches:
- **seg:** `LN → Linear → cond_dim`
- **phrase:** masked mean-pool → `LN → Linear`
- **relation:** seg-query attention over phrase tokens
- **stage router:** MLP gates + learnable `stage_embed[S,C]`

Init: router last layer small random; reliability bias = `RELIABILITY_INIT` (0.0 → sigmoid≈0.5).

### TextConditionFactory

- `version=="v1.5"` or `stage_router=True` → router path
- else → v1 flat `[N,C]` path (backward compat)

## WTI — `window_text_interaction.py`

### StageWTIHeadAware (v1.5 default)

| | |
|---|---|
| Input | `x_windows [BW,N,C]`, `stage_text_cond [B,C]`, `reliability [B,1]` |
| Output | `attn_bias [BW,H,N,N]` |

Low-rank pairwise:
```
visual_q/k = Linear(C → H*r)(x) → [BW,N,H,r]
text_q/k   = Linear(C → H*r)(tc) → [BW,H,r]
bias = tanh(einsum(q_bias, k_bias) / sqrt(r)) * BIAS_MAX
attn_bias = bias * tanh(alpha) * stage_scale * reliability
```

`compute_raw_bias()` exposed for diagnostics (gate before alpha).

### StageWTIKeyBias (v1 legacy)

Output `[BW,1,1,N]` when `HEAD_AWARE=false`.

### TGSwimController

- `_select_stage_cond`: slices `[B,S,C]` by `swin_stage_idx`
- `pop_stats()` fields: `alpha`, `bias_abs_mean`, `bias_abs_max`, `raw_bias_abs_mean`, `reliability_mean`, `head_bias_entropy`, `stage_idx`, `layer_idx`, `active`

## Swin — `swin_trans.py`

`WindowAttention`: `attn = attn + attn_text_bias` (supports `[BW,H,N,N]` or legacy `[BW,1,1,N]`).

## Forward — `llava_phi.py`

1. LLM → `hidden_states`
2. `build_text_cond(..., refer_span_mask)` → `[N,S,C]`, `[N,S,1]`
3. `_repeat_images_per_target`
4. Swin + controller (pixel decoder unchanged)

## Train / merge

- `train_swin_backbone=False` → frozen Swin
- `train_module_list`: `tg_swin_tcf`, `tg_swin_controller` (no LoRA on TG-Swin linears)
- Merge includes router + head-aware WTI weights

## Gradient path

At `alpha=0`: forward identity; `raw_bias ≠ 0` → `alpha.grad > 0`.  
At small `alpha`: visual/text projections + router receive gradients.

## Diagnostics

| Script | Purpose |
|--------|---------|
| `probe_tgswin_v15_train_init_gradient.py` | v1.5 identity + alpha + projection grads |
| `probe_tgswin_train_init_gradient.py` | v1 legacy |
| `smoke_tgswin_core.py` | Unified smoke (includes v1.5 probe) |
| `probe_tgswin_identity.py` | Disabled path / state_dict |
| `probe_tgswin_shape_alignment.py` | Repeat + stage cond shapes |

## Files modified (v1 → v1.5)

- `tg_swin/text_condition_factory.py` — StageWiseTextRouter
- `tg_swin/window_text_interaction.py` — StageWTIHeadAware
- `tg_swin/__init__.py`
- `swin_trans.py` — bias shape doc
- `llava_phi.py` — v1.5 init kwargs
- `mask_config/maskformer2_tgswin*.yaml`
- `run_train_merge_test_tgswin.sh`
- `tools/diagnostics/*`
