# Full Implementation Audit

**Branch:** `ours` (verified via `git branch --show-current`)  
**Repo path:** `huangziyi/reseg/segearth+dr-ewti-setpp/`  
**Audit date:** 2026-06-29  
**Mode:** read-only (no code changes)

## Overall Verdict
- **PASS**
- One-sentence summary: Current Full training/eval forward implements Enhanced TG-Swin encoder grounding (SEG→`text_cond`, SET→`set_control`→WTI gate) plus SET++ decoder-side set consistency with coarse/DR disabled, per-target batch semantics, and image-level regrouped set losses; no Swin visual-token concat of SEG/SET was found.

## Required Design
- SEG -> text_cond -> WTI bias content
- SET -> set_control -> WTI gate modulation
- image -> Swin visual tokens
- coarse disabled

## Checklist

### 1. Full config
- **Verdict:** PASS
- **Evidence:**
  - Full yaml exists: `segearth_r2/model/mask_decoder/mask_config/ours_full_enhanced_tgswin_setpp.yaml`
  - `scripts/train_full.sh` line 32: `--mask_config 'segearth_r2/model/mask_decoder/mask_config/ours_full_enhanced_tgswin_setpp.yaml'`
  - `run_train_merge_test.sh` default `MASK_CONFIG` points to the same yaml (line 64)
  - Yaml values match required Full flags (see below)
- **Risk:** Low. A different entry script or env override could still point at a legacy yaml; `run_train_merge_test.sh` and `train_full.sh` are correct.
- **Required fix:** None

```
PASS
实际使用的 mask_config 路径：
  segearth_r2/model/mask_decoder/mask_config/ours_full_enhanced_tgswin_setpp.yaml
TG_SWIN.ENABLED 实际值：True
USE_COARSE_EVIDENCE 实际值：False
USE_DR_EWTI 实际值：False
GATE_MODE 实际值："legacy"
```

---

### 2. SEG into TG-Swin
- **Verdict:** PASS
- **Evidence:**
  - Train forward (`llava_phi.py` ~1574–1604): when `per_target_swin`, calls `_prepare_full_tg_swin_inputs` then `get_vision_tower_feature(..., text_cond=..., reliability=...)`.
  - `_prepare_full_tg_swin_inputs` (~448–516):
    - `build_text_cond(hidden_states, SEG_token_embedding_indices, refer_span_mask)` 
    - `get_SEG_embedding(hidden_states, ...)` → LLM last hidden at `[SEG]` positions (~1470–1478)
    - `_gather_refer_phrase_hidden` + `TextConditionFactory` / `StageWiseTextRouter` with phrase span (~688–699, `text_condition_factory.py` ~164–226)
    - Asserts `text_cond is not None`, `text_cond.shape[0] == n_target == sum(mask_num)`
  - `get_vision_tower_feature` (~709–767): for enhanced-wti-v2, **asserts** `text_cond` and `reliability`; raises `RuntimeError` on silent plain-Swin fallback (~748–750).
  - Swin block (~281–287 `swin_trans.py`): `tg_swin_controller.compute_bias(..., text_cond, reliability, set_control=...)`.
  - WTI content: `StageEnhancedWTIHeadAware.compute_raw_bias` uses `text_cond` via `text_proj` (~258–277 `window_text_interaction.py`).
- **Shapes:**
  - `seg_hidden`: `[T, D_llm]` from LLM hidden states
  - `text_cond`: `[T, S, C]` with `S = num_stages` (default 3 = `len(WTI_STAGES)`), `C = COND_DIM` (256)
  - `reliability`: `[T, S, 1]` (stage router path)
  - `T = sum(mask_num)`
- **Risk:** If `refer_span_mask` is missing, phrase pooling falls back to local context or zeros; TCF still runs but phrase grounding may weaken (non-blocking for path correctness).
- **Required fix:** None

```
Does SEG enter TG-Swin? YES
Code path:
  hidden_states -> get_SEG_embedding -> build_text_cond -> tg_swin_tcf -> text_cond/reliability
  -> get_vision_tower_feature -> SwinTransformerBlock -> TGSwimController.compute_bias
  -> StageEnhancedWTIHeadAware.compute_raw_bias
Expected shape: text_cond [T, S, C], T=sum(mask_num)
Actual shape: Asserted in _prepare_full_tg_swin_inputs; TCF router returns [n_target, num_stages, cond_dim]
Risk: Phrase span optional degradation only
```

---

### 3. SET into TG-Swin
- **Verdict:** PASS
- **Evidence:**
  - `_prepare_full_tg_swin_inputs` (~482–504):
    - `get_SET_embedding(hidden_states, SET_token_embedding_indices)` (~1480–1488)
    - `SET_token_projector` → `SET_embedding_b`
    - `_repeat_set_embedding_per_target` → `SET_embedding_t` `[T, 1, C]`
    - `build_set_control` → `SETControlHead` → `set_control` `[T, S, 1]` (`set_control_head.py` ~34–38)
    - Asserts `set_control.shape[0] == n_target`; asserts non-None when `tg_swin_set_control` exists
  - `get_vision_tower_feature` (~726–729): asserts `set_control is not None` when `tg_swin_set_control` is enabled
  - `TGSwimController.compute_bias` (~545–551, ~582): selects stage slice of `set_control`, broadcasts to windows, passes to WTI module
  - `StageEnhancedWTIHeadAware.forward` legacy gate (~325–327): `gate = gate * set_control`; `attn_bias = raw_bias * gate`
- **Shapes:**
  - `SET_embedding_b`: `[B, 1, C]` (asserted)
  - `SET_embedding_t`: `[T, 1, C]` (asserted)
  - `set_control`: `[T, S, 1]` (asserted last dim == 1)
- **Risk:** Low. `SETControlHead.text_dim` uses decoder hidden dim (256) after `SET_token_projector`, not raw LLM dim — intentional and consistent.
- **Required fix:** None

```
Does SET enter TG-Swin as gate modulation? YES
Code path:
  hidden_states -> get_SET_embedding -> SET_token_projector -> _repeat_set_embedding_per_target
  -> build_set_control -> SETControlHead -> set_control
  -> get_vision_tower_feature -> TGSwimController.compute_bias -> StageEnhancedWTIHeadAware.forward (gate *= set_control)
Expected shape: set_control [T, S, 1]
Actual shape: Asserted in _prepare_full_tg_swin_inputs and documented in set_control_head.py
Risk: None blocking
```

---

### 4. No SET/SEG token concat into Swin
- **Verdict:** PASS
- **Evidence:**
  - Grep across `swin_trans.py`: no `torch.cat` involving SEG/SET/text tokens into `x` or `x_windows`. Only `text_cond` / `set_control` passed as kwargs to `compute_bias`.
  - `WindowAttention.forward` (~145–180): `x` remains visual window tokens; modulation via `attn_text_bias` only.
  - `torch.cat([SET_query, SEG_query], ...)` exists only in **Mask2Former decoder** (`mask2former_transformer_decoder.py` ~633, ~671) — decoder query assembly, not Swin encoder input.
  - `get_SEG_embedding` / `get_SET_embedding` `torch.cat` operations concatenate **batch rows of token hiddens**, not visual features.
- **Risk:** None for Swin concat. Decoder dual-use of SEG/SET embeddings is by design (encoder grounding + decoder SET++).
- **Required fix:** None

```
Are SET/SEG concatenated into Swin visual tokens? NO
If YES, location: N/A
Should be fixed? N/A
```

---

### 5. Swin modified only by attention bias
- **Verdict:** PASS
- **Evidence:**
  - `WindowAttention.forward` (~157–174):
    ```python
    attn = (q @ k.transpose(-2, -1))
    attn = attn + relative_position_bias.unsqueeze(0)
    if attn_text_bias is not None:
        attn = attn + attn_text_bias
    attn = self.softmax(attn)
    x = (attn @ v) ...
    ```
  - TG-Swin produces `attn_text_bias` only; does not alter `v`, `x_windows` token sequence, or channel concat.
  - `StageEnhancedWTIHeadAware` modulates bias magnitude via gate (reliability × set_control); still pre-softmax additive path through `attn_bias = raw_bias * gate`.
- **Risk:** Low. DR/coarse branches in `TGSwimController.compute_bias` (~563–577) could modify bias differently, but Full config sets `use_dr_ewti=False` and `coarse_evidence=None`, so DR branch is not taken.
- **Required fix:** None

```
Is Swin modified only by pre-softmax attention bias? YES
Code path: SwinTransformerBlock -> compute_bias -> WindowAttention(attn_text_bias=...)
Risk: Legacy DR path exists in code but inactive under Full yaml
```

---

### 6. Coarse disabled
- **Verdict:** PASS
- **Evidence:**
  - Full yaml: `USE_COARSE_EVIDENCE: False`, `USE_DR_EWTI: False`, `USE_EVIDENCE_STATE: False`
  - `_assert_full_no_coarse_path` (~383–393): raises if coarse/DR/evidence_state enabled under enhanced-wti-v2
  - Train forward (~1590): `coarse_evidence = None` explicitly; never calls `_resolve_coarse_evidence` on Full path
  - `get_vision_tower_feature` (~730): `assert coarse_evidence is None` for enhanced-wti-v2
  - `_init_tg_swin_modules` (~537–538): forces `use_dr_ewti = False` when `enhanced_wti`
  - `TGSwimController.compute_bias` DR branch requires `self.use_dr_ewti and evidence_windows is not None` (~563–567) — not satisfied on Full
  - `predict_coarse_masks` / `get_shared_coarse_evidence` / `EvidenceTokenFusion` / `StageDynamicRelationalWTI` remain in repo for legacy/diagnostic scripts only
- **Risk:** Low accidental activation only if yaml flags flipped or a non-Full script overrides config.
- **Required fix:** None

```
Does coarse evidence enter Full? NO
Any possible accidental path? Only via config override (USE_COARSE_EVIDENCE=True) or calling legacy helpers directly in custom scripts
Guard/assert exists? YES (_assert_full_no_coarse_path, coarse_evidence=None assert, use_dr_ewti forced False)
Risk: Config misuse, not default Full path
```

---

### 7. Per-target batch semantics
- **Verdict:** PASS (minor assert gap)
- **Evidence:**
  - Gated by `per_target_swin` (~1532–1538): `TG_SWIN.ENABLED`, `mask_num`, `seg_info`, `PER_TARGET_SWING_REPEAT=True`
  - `B = len(mask_num)`, `T = sum(mask_num)` asserted via SEG/text_cond/set_control alignment (~470–500)
  - `images_expanded = _repeat_images_per_target(images, mask_num)` (~1591, ~607–621)
  - `target_to_image = _build_target_to_image(mask_num, device)` (~806–808): `[T]` with `repeat_interleave(arange(B), mask_num)`
  - Swin called on `images_expanded` with `text_cond`/`set_control` batch dim `T` (~1597–1604)
  - Optional debug log `_log_full_batch_semantics` prints all shapes when `debug_batch_semantics=True`
- **Shapes (expected vs code):**

| Tensor | Expected | Verified |
|--------|----------|----------|
| B | `len(mask_num)` | YES |
| T | `sum(mask_num)` | YES (asserts) |
| images | `[B, ...]` | YES |
| images_expanded | `[T, ...]` | YES (repeat per mask_num; no explicit assert on dim 0) |
| SEG_embedding | `[T, 1, C]` | YES (projector after n_target check) |
| SET_embedding repeated | `[T, 1, C]` | YES (asserted) |
| text_cond | `[T, S, C]` | YES (asserted batch dim) |
| set_control | `[T, S, 1]` | YES (asserted) |
| target_to_image | `[T]` | YES (built explicitly) |

- **Missing asserts:** `images_expanded.shape[0] == T` not explicitly asserted (inferred from repeat logic).
- **Risk:** Low. Repeat helpers are straightforward; silent mismatch unlikely.
- **Required fix:** None (optional assert is non-blocking)

```
Are per-target batch semantics consistent? YES
B: len(mask_num)
T: sum(mask_num)
images_expanded: [T, C, H, W] via _repeat_images_per_target
text_cond: [T, S, C]
set_control: [T, S, 1]
target_to_image: [T]
Missing asserts: images_expanded.shape[0] == T
```

---

### 8. Regrouped set loss
- **Verdict:** PASS
- **Evidence:**
  - `Mask_Criterion._use_regrouped_set_loss` (~229–236): requires `setpp_enable`, `setpp_regroup_set_loss`, `per_target_mode`, `target_to_image`, `mask_num`
  - Helpers present: `regroup_per_target_union` (~238–250), `build_regrouped_union_target_tensor` (~252–282)
  - `loss_set_union` (~293–316): regroup branch → `[B, 1, H, W]` then BCE + dice
  - `loss_setpp_coverage` (~350–365): regroup branch on `pred_seg`
  - `loss_setpp_consistency` (~394–418): regroup branch on both `pred_set` and `pred_seg`
  - `forward_per_target` (~714–737): uses regrouped path when `_use_regrouped_set_loss` true
  - Training defaults: `run_train_merge_test.sh` / `train_full.sh` set `--setpp_regroup_set_loss True`
  - Live run artifacts (`output/full/full/`): logs show `loss_union`, `loss_setpp_coverage`, `loss_setpp_consistency` — consistent with regroup path active
- **Risk:** If metadata missing (`per_target_mode`, `target_to_image`, `mask_num`), falls back to non-regroup target-level union rows (~718–732) — would be wrong semantically but does not trigger under Full train forward (metadata attached ~1631–1634).
- **Required fix:** None

```
Does per-target set loss regroup to image-level? YES
Functions found: _use_regrouped_set_loss, regroup_per_target_union, build_regrouped_union_target_tensor
loss_set_union regrouped? YES
loss_setpp_coverage regrouped? YES
loss_setpp_consistency regrouped? YES
Risk: Fallback path if metadata stripped
```

---

### 9. Aux outputs metadata
- **Verdict:** PASS
- **Evidence:**
  - `llava_phi.py` forward (~1635–1641): copies `per_target_mode`, `target_to_image`, `mask_num`, `valid_seg_mask` into each aux dict
  - `Mask_Criterion.forward_per_target` (~760–768): explicitly re-injects `target_to_image`, `mask_num`, `per_target_mode`, `valid_seg_mask` before recursive aux loss
  - `Mask_Criterion.forward` (~847–849): inherits metadata from main outputs when missing in aux
  - Loss fns derive masks via `outputs.get("pred_set_union_mask", outputs["pred_masks"][:, 0:1])` and `pred_seg_masks` / `[:, 1:]` fallback (~290, ~351, ~395)
- **Risk:** Low. If aux dict lacks `pred_masks` entirely, loss would fail loudly (KeyError), not silently wrong.
- **Required fix:** None

```
Are aux_outputs compatible with regrouped set loss? YES
Risk: None blocking under current forward
```

---

### 10. Trainable modules
- **Verdict:** PASS
- **Evidence:**
  - `train.py` (~548–555): when `TG_SWIN.ENABLED`, adds `tg_swin_tcf`, `tg_swin_controller`; when `USE_SET_TGSWIN_CONTROL`, adds `tg_swin_set_control`
  - Base list includes `predictor`, `pixel_decoder`, `SEG_token_projector`, `SET_token_projector`, `lm_head` (~548–550)
  - Post-LoRA loop sets `requires_grad=True` for all `train_module_list` names (~576–583)
  - `save_trainable_parameters` writes `trainable_parameters.txt` (~385–415, called ~585)
  - Verified artifact `output/full/full/trainable_parameters.txt`:
    - Contains `tg_swin_tcf`, `tg_swin_controller`, `tg_swin_set_control`, `pixel_decoder`, `predictor`, `SEG_token_projector`, `SET_token_projector`, `lm_head`, LoRA adapters
    - `total_trainable_params=162681836`, `trainable_ratio=0.048801`
  - Swin backbone (`vision_tower_mask`) frozen by default (`train_swin_backbone=False`) — intentional for Full recipe
- **Risk:** None blocking. User expecting full Swin backbone training would need `train_swin_backbone=True` (out of Full spec).
- **Required fix:** None

```
Are Full modules trainable? YES
Missing trainable modules: None (all required Full modules present)
Risk: Swin backbone intentionally frozen
```

---

### 11. Code crowding
- **Verdict:** PARTIAL (readability only; behavior OK)
- **Evidence:**
  - `llava_phi.py` is **1960 lines**; main `forward` (~1490–1785) mixes LLM pass, TG-Swin prep, Swin features, pixel decoder, predictor, loss metadata, debug logging
  - Partial extraction already exists: `_prepare_full_tg_swin_inputs`, `_log_full_batch_semantics`, `_assert_full_no_coarse_path`, `_resolve_coarse_evidence`
  - Legacy coarse/DR helpers still live in same file (~840–920) though guarded off for Full
- **Minimal extraction suggestion (no behavior change):**
  ```python
  _prepare_full_tg_swin_inputs(...)      # already exists
  _run_full_swin_encoder(...)            # images_expanded + get_vision_tower_feature + pixel_decoder
  _attach_per_target_metadata(...)       # mask_outputs dict enrichment
  _debug_full_batch_semantics(...)       # already exists as _log_full_batch_semantics
  ```
  Move legacy coarse helpers to `llava_phi_coarse_legacy.py` (import-only) to shrink main forward file.
- **No behavior change required:** YES

```
Is current forward too crowded? PARTIAL
Suggested minimal extraction: _run_full_swin_encoder, _attach_per_target_metadata; optional legacy module split
No behavior change required: YES
```

---

## Final Blocking Issues
None. Under `ours_full_enhanced_tgswin_setpp.yaml` and the standard Full train entrypoints, the implementation matches the specified Full main path.

## Non-blocking Issues
1. **`llava_phi.py` size (~1960 lines):** forward mixes encoder prep, decoder, loss metadata, and legacy coarse helpers — readable but heavy.
2. **Missing assert:** `images_expanded.shape[0] == sum(mask_num)` not explicit (low risk).
3. **`eval_seg` pre-multimodal Swin call** (~1817–1818): if `per_target_swin` is false while Full yaml is loaded, `get_vision_tower_feature(images)` raises — safe fail, but documents that Full eval requires `mask_num`.
4. **Config typo:** yaml/code use `PER_TARGET_SWING_REPEAT` (SWING not SWIN) — consistent internally, cosmetic only.
5. **Legacy coarse/DR code** remains in tree for diagnostics/other configs — inactive on Full yaml.

## Minimal Patch Plan
Not required for PASS. Optional cleanup only (in order):
1. Add one assert: `images_expanded.shape[0] == n_target` in `_prepare_full_tg_swin_inputs` or immediately after `_repeat_images_per_target`.
2. Extract `_attach_per_target_metadata(mask_outputs, ...)` from `forward` (~1631–1641).
3. Extract `_run_full_swin_encoder(images_expanded, tg_prep)` wrapping `get_vision_tower_feature` + `pixel_decoder.forward_features`.
4. Move `predict_coarse_masks` / `get_shared_coarse_evidence` to a legacy submodule (import unchanged behavior).

---

## Self-check Commands

Executed on branch `ours`:

```bash
cd huangziyi/reseg/segearth+dr-ewti-setpp
python -m py_compile segearth_r2/train/train.py
python -m py_compile segearth_r2/model/language_model/llava_phi.py
python -m py_compile segearth_r2/model/mask_encoder/swin_trans.py
python -m py_compile segearth_r2/model/mask_encoder/tg_swin/window_text_interaction.py
python -m py_compile segearth_r2/model/mask_encoder/tg_swin/text_condition_factory.py
python -m py_compile segearth_r2/model/mask_encoder/tg_swin/set_control_head.py
python -m py_compile segearth_r2/model/mask_decoder/mask_criterion/Mask_Criterion.py
```

**Result:** All exited 0 (success).

---

## Final Answer

**当前代码是否已经等于我要的 Full 主路径？ → YES (PASS)**

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Full config: Enhanced TG-Swin + SET++ | PASS |
| 2 | coarse / DR fully off | PASS |
| 3 | SEG via `text_cond` into Enhanced WTI | PASS |
| 4 | SET via `set_control` into WTI gate | PASS |
| 5 | No SEG/SET concat into Swin visual tokens | PASS |
| 6 | Swin modulated only via pre-softmax attention bias | PASS |
| 7 | Per-target batch semantics consistent | PASS |
| 8 | Set consistency loss regroups to image-level | PASS |
| 9 | Full key modules trainable | PASS |
| 10 | Code maintainability | PARTIAL (non-blocking) |
