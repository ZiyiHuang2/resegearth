# TCPD Design Notes

## Scope

- TCPD-only.
- Uses only current projected `[SEG]` embedding.
- No `[SET]`.
- No expression token bank.
- Predictor unchanged.
- No eval metric claim.

## TCPD-v1 (global) vs TCPD-v2 (spatial)

### v1 global (ablation mode `tcpd_spatial_mode=global`)

MSDeformAttn conditioning is **target-global** and broadcast over all spatial query tokens:

```text
delta_offset = Linear(z)  -> [N, 1, heads, levels, points, 2]
delta_attn   = Linear(z)  -> [N, 1, heads, levels*points]
```

This can learn per-target sampling bias but not location-varying language-visual interaction.

### v2 spatial (default when `use_tcpd=True`)

Each spatial query token is conditioned jointly with the target embedding:

```text
q = query                         [N, Len_q, C]
z = tcpd_z[:, None, :]            [N, 1, C]
joint = GELU(LayerNorm(q_proj + z_proj))
delta_offset = offset_head(joint) [N, Len_q, heads, levels, points, 2]
delta_attn   = attn_head(joint)   [N, Len_q, heads, levels*points]
```

Why spatial conditioning: referential segmentation needs different sampling/attention patterns at different spatial locations for the same target phrase.

Gated residual (identity init):

```text
sampling_offsets += gate_offset * delta_offset
attention_logits += gate_attn * delta_attn
```

- `gate_offset = 0`, `gate_attn = 0` at init
- branch weights xavier; bias zero

## FPN conditioning injection point

Baseline FPN top-down fusion is text-blind:

```text
cur = lateral_conv(x)
td  = interpolate(out[-1])
y   = cur + td
output_conv(y)
```

TCPD-v2 adds optional conditioning **before** `output_conv` via `TCPDFPNFusion`:

```text
y = cur + td
y += gate_td * sigmoid(MLP(z)) * td_adapter(td)
y += gate_lat * (1 - alpha) * lat_adapter(cur)
```

Controlled by `tcpd_condition_fpn=True` (default). Gates init 0; adapters xavier.

## Output scale fusion

`TCPDScaleFusion` applies per-target level weights from `level_mlp(z)` to gated residual adapters on multi-scale features and mask features. Controlled by `tcpd_condition_output_scale=True`. Helper `get_tcpd_level_weights(z)` exposes softmax weights for probes (not logged every step).

## Injection Points

- `segearth_r2/model/language_model/llava_phi.py`
  - **`get_SEG_embedding` + `SEG_token_projector`:** LLM hidden state at `[SEG]` → `SEG_embedding`.
  - **`_repeat_image_features`:** when `use_tcpd=True`, repeats Swin dict features to `sum_K`.
  - **`_get_tcpd_forward_kwargs()`:** reads v2 flags from `model.config`.
  - **`forward` / `eval_seg`:** `pixel_decoder.forward_features(..., tcpd_condition=SEG_embedding, **kwargs)`.
  - **Predictor:** unchanged.

- `segearth_r2/model/mask_decoder/.../pixel_decoder/msdeformattn.py`
  - **`forward_features`:** encoder + FPN + scale fusion; passes v2 kwargs.
  - **`TCPDFPNFusion`:** text-conditioned top-down fusion.
  - **`TCPDScaleFusion`:** output adapters.

- `segearth_r2/model/mask_decoder/.../ops/modules/ms_deform_attn.py`
  - **`MSDeformAttn`:** global or spatial TCPD deltas; `compute_tcpd_deltas()` for tests.

## Config flags and ablation matrix

| Flag | Default (TCPD on) | Ablation |
|------|-------------------|----------|
| `use_tcpd` | `False` (baseline) | off vs on |
| `tcpd_spatial_mode` | `spatial` | `global` = v1 broadcast |
| `tcpd_condition_msdeform` | `True` | MSDeformAttn-only off |
| `tcpd_condition_fpn` | `True` | FPN-only off |
| `tcpd_condition_output_scale` | `True` | scale fusion off |
| `tcpd_train` | `False` | unfreeze `*tcpd*` inside frozen PD |

Example ablations:

- MSDeformAttn-only: `tcpd_condition_fpn=False`, `tcpd_condition_output_scale=False`
- FPN-only: `tcpd_condition_msdeform=False`, `tcpd_condition_output_scale=False`
- Full TCPD-v2: all three `True`, `tcpd_spatial_mode=spatial`

## Training scripts

| Script | Purpose |
|--------|---------|
| `scripts/tcpd_train_pure.sh` | **Pure TCPD-only:** freeze LLM, predictor, SEG_projector, lm_head, base PD; train only `*tcpd*` |
| `scripts/tcpd_train_test4_plus.sh` | **Test4 + TCPD:** `--test4_mode True` (LLM 30–31, predictor, SEG_projector) plus TCPD |
| `scripts/tcpd_smoke_train.sh` | Alias to pure train with `MAX_STEPS=1` |

Pure mode assertions in `log_test4_train_config()`:

```text
non_tcpd_pixel_decoder_trainable == 0
tcpd_trainable > 0
predictor_trainable == 0
SEG_token_projector_trainable == 0
lm_head_trainable == 0
unfrozen_llm_layers == none
```

## Disable path

- `use_tcpd=False` (default): original text-blind PD path.
- TCPD gates init 0 → identity forward at step 0 when enabled.

## Merge / Eval

Export via `merge_lora_weights_and_save_hf_model.py` preserves:

```text
use_tcpd, tcpd_condition_source, tcpd_spatial_mode,
tcpd_condition_msdeform, tcpd_condition_fpn, tcpd_condition_output_scale
```

`scripts/test4_merge_checkpoint.sh` forwards env vars (`USE_TCPD`, `TCPD_SPATIAL_MODE`, etc.).

Eval/model load prints TCPD config once via `llava_phi._log_tcpd_config_once()`.

## Validation

`scripts/tcpd_smoke_test.py`:

- off / global / spatial forward shapes
- identity at gates=0
- gradients (gates=0 and gates=1e-3) for MSDeformAttn, FPN, scale fusion
- spatial adaptivity: per-token delta_offset differs across query positions
- optional pure train launch (`MAX_STEPS=1 bash scripts/tcpd_train_pure.sh`)

Do not claim gIoU improvement.
