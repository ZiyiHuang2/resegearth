# Full Path Audit

Config: `ours_full_enhanced_tgswin_setpp.yaml`  
Method: Enhanced TG-Swin encoder grounding + SET++ decoder set consistency

## 1. Does SEG enter TG-Swin?

- **Answer:** Yes. SEG enters Swin **only** as `text_cond` / `reliability` via `TextConditionFactory`, not as visual tokens.

- **Code path:**
  1. `build_text_cond()` → `get_SEG_embedding()` + `_gather_refer_phrase_hidden(refer_span_mask)` → `tg_swin_tcf(...)`
  2. `_prepare_full_tg_swin_inputs()` validates shapes
  3. `get_vision_tower_feature(..., text_cond=..., reliability=...)`
  4. `SwinTransformerBlock.forward` → `tg_swin_controller.compute_bias(..., text_cond, reliability)`
  5. `StageEnhancedWTIHeadAware.compute_raw_bias(x_windows, text_cond, ...)` → `attn_text_bias`

- **Tensor shapes (Full, per-target):**
  - `seg_hidden`: `[T, hidden_size]`
  - `text_cond`: `[T, S, C]` (TCF v1.5 stage router, S=4, C=256)
  - `reliability`: `[T, S, 1]`
  - `SEG_embedding` (decoder only): `[T, 1, mask_dim]`

- **Assertions added:**
  - `_prepare_full_tg_swin_inputs`: `text_cond` / `reliability` not None; `text_cond.shape[0] == n_target == sum(mask_num)`
  - `get_vision_tower_feature` (enhanced-wti-v2): same checks + `coarse_evidence is None`

## 2. Does SET enter TG-Swin?

- **Answer:** Yes. SET enters Swin **only** as `set_control` via `SETControlHead`, modulating WTI gate. It does **not** enter visual tokens.

- **Code path:**
  1. `get_SET_embedding()` → `SET_token_projector` → `SET_embedding_b` `[B,1,C]`
  2. `_repeat_set_embedding_per_target` → `SET_embedding_t` `[T,1,C]`
  3. `build_set_control()` → `SETControlHead` → `set_control` `[T,S,1]`
  4. `get_vision_tower_feature(..., set_control=...)`
  5. `TGSwimController.compute_bias`: stage-select + broadcast `[T,S,1]` → `[BW,1,1,1]`
  6. `StageEnhancedWTIHeadAware.forward`: `gate = tanh(alpha) * stage_scale * rel; gate *= set_control`

- **Tensor shapes:**
  - `SET_embedding_b`: `[B, 1, C]`
  - `SET_embedding_t`: `[T, 1, C]`
  - `set_control`: `[T, S, 1]`

- **Assertions added:**
  - If `tg_swin_set_control` is not None: `set_control is not None` and `set_control.shape[0] == n_target`
  - `get_vision_tower_feature` (Full): `set_control` required when SET control head exists

## 3. Does coarse evidence enter Full?

- **Answer:** No.

- **Code path checked:**
  - Config: `USE_COARSE_EVIDENCE=False`, `USE_DR_EWTI=False`
  - `_assert_full_no_coarse_path()` in `_prepare_full_tg_swin_inputs`
  - Full forward passes `coarse_evidence=None` explicitly
  - `get_vision_tower_feature` asserts `coarse_evidence is None` for enhanced-wti-v2
  - `SwinTransformerBlock`: `evidence_windows` only built when `coarse_evidence is not None` (never in Full)

- **Guard added:** `_assert_full_no_coarse_path`, `_resolve_coarse_evidence` (legacy only), explicit `coarse_evidence=None` in Full branch

## 4. Is Swin modified by token concat or attention bias?

- **Answer:** Attention bias only. No SEG/SET concat into `x_windows` or image features.

- **Code path:**
  - Visual: `images_expanded` → Swin patch embed → window partition → `x_windows`
  - Modulation: `attn = attn + relative_bias + attn_text_bias` in `WindowAttention.forward`
  - `SEG_embedding` / `SET_embedding` also go to **Mask2Former predictor** (decoder queries) — separate from Swin; this is expected SET++ behavior, not duplicate Swin injection

## 5. Is forward too crowded?

- **Refactor done:** Minimal extraction only; no behavior change.

- **Functions extracted:**
  - `FullTGSwinInputs` dataclass
  - `_prepare_full_tg_swin_inputs()` — SEG/SET/TG-Swin prep + asserts
  - `_log_full_batch_semantics()` — optional debug (unchanged role, added `reliability` shape)

- **Full training forward now reads as:**
  ```python
  if per_target_swin:
      tg_prep = self._prepare_full_tg_swin_inputs(...)
      images_expanded = self._repeat_images_per_target(...)
      image_features = self.get_vision_tower_feature(..., coarse_evidence=None, enable_tg_swin=True)
      ...
  ```

- **No large behavior changes.**

## 6. Remaining risks

1. **Decoder vs encoder dual use of SEG/SET:** `SEG_embedding` / `SET_embedding` feed both TG-Swin (via TCF/set_control) and Mask2Former predictor. This is by design (encoder grounding + decoder SET++), not a Swin concat bug — but increases compute vs encoder-only grounding ablations.

2. **Per-target repeat cost:** `T = sum(mask_num)` expands Swin batch; memory scales with T, independent of attention-map saving.

3. **Missing refer span:** If `refer_span_mask` is None, TCF falls back to local context pool (warning in router). Full still runs but phrase grounding may weaken.

4. **Legacy TG-Swin configs** without enhanced-wti-v2 still allow silent Swin fallback when `text_cond` is missing; Full config will hard-fail instead (intentional).
