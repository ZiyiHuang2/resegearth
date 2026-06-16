# TG-MSA image-route redesign for SegEarth-R2

## Position

The main route must be the image route. Decoder/query modules are allowed only as auxiliary stabilizers. The method should be framed as target-guided visual feature reorganization before mask prediction, not as query binding.

Recommended method name:

**TG-VFR: Target-Guided Visual Feature Reorganization for Referring Remote Sensing Segmentation**

## Current SegEarth-R2 image path

The actual path is:

```text
images
  -> Swin mask tower
  -> res2/res3/res4/res5
  -> MSDeformAttnPixelDecoder.forward_features(features)
       transformer encoder uses res5/res4/res3
       FPN adds res2 back for high-resolution mask_features
  -> mask_features, multi_scale_features
  -> repeat_interleave by mask_num
  -> Mask2Former-style decoder with SEG_embedding
  -> one mask per flattened SEG
```

Important consequences:

1. Before `pixel_decoder.forward_features`, visual features are image-level and target-agnostic.
2. Inside the pixel decoder, MSDeformAttn sees only visual features. Strong text injection here risks repeating TCPD-style instability.
3. After repeat by `mask_num`, every target has its own visual feature copy. This is the safest place for per-target target/background/sibling visual calibration.
4. The decoder is too late to be the main image-route contribution; it should read better visual features rather than repair all target ambiguity alone.

## Real defects to address

### Defect 1: target-blind multiscale visual encoding

Swin and pixel decoder encode all objects jointly. Remote sensing scenes often contain repeated small objects and similar textures. The target phrase arrives too late, so early visual features do not prioritize target-relevant scale/texture/semantic cues.

### Defect 2: no target/background visual separation before mask decoding

The current repeated `mask_features` are identical for all targets from the same image. Only `SEG_embedding` differs at decoder input. For multi-cate or sibling targets, this places too much burden on query-mask decoding.

### Defect 3: no sibling-target suppression on image features

Same-image target A and target B share image features and are decoded independently. There is no image-side mechanism saying: enhance A while suppressing B-like visual evidence.

### Defect 4: no image-side diagnostic losses

Current supervision is mask loss after decoder. If target-conditioned visual features are wrong, the loss only reveals the final mask error, not whether visual features failed to separate target/background/sibling regions.

## Literature mechanisms and where they fit

### VSPNet: shallow texture / deep semantic split

Use for Stage A. Shallow features should preserve texture and boundary cues; deep features should encode semantic target relevance.

Do not copy VSPNet backbone changes. SegEarth-R2 should keep Swin frozen and only add output adapters.

### SBANet: scale-wise alignment and dynamic feature selection

Use for Stage A router. A scale router should decide how much each level contributes for a target, rather than applying a single uniform text gate.

Do not fully update language tokens or add heavy bidirectional transformer blocks in the image backbone.

### STDNet: target-background twin-stream

Use for Stage B. The key idea is not to create a new decoder, but to split visual calibration into target-enhanced and background/sibling-suppressed residuals.

Do not duplicate Mask2Former decoder.

### FarmSeg_VLM / RS2-SAM2: dense visual prompt

Use as a future optional diagnostic. Existing predicted masks or image-side prototypes can act as dense visual prompts, but only after C/A/B evidence shows it is needed.

Do not port SAM/SAM2 or introduce a second foundation-model pipeline.

### LQMFormer

Use only as auxiliary decoder regularization. It supports anti-collapse, but the main image route must not become a query-method paper.

## Proposed architecture

### Stage 0: Target context builder

File: `segearth_r2/model/tgmsa/context.py`

Inputs:

- `SEG_embedding`: `[sumK, 1, C]`
- `mask_num`: list length B
- optional `hidden_states` / token masks later, if phrase-level token context is needed

Outputs:

- `target_embed`: `[sumK, C]`
- `image_embed`: `[B, C]`, pooled from each image's targets
- `peer_embed`: `[sumK, C]`, mean of same-image other targets
- `hard_peer_embed`: `[sumK, C]`, most similar same-image target
- index maps: target-to-image and group slices

Reason: do not duplicate mask_num regrouping logic across Stage A/B/C.

### Stage A: Scale-wise Target Visual Router

File: `segearth_r2/model/tgmsa/visual_router.py`

Insertion point:

```python
image_features = self.get_vision_tower_feature(images)
target_ctx = self.tgmsa_context(SEG_embedding, mask_num)
image_features, visual_ctx = self.tgmsa_visual_router(image_features, target_ctx)
mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(image_features)
```

Design:

- Keep `res2/res3/res4/res5` image-batch shaped, not repeated.
- Use `image_embed` for image-level routing before pixel decoder.
- res2/res3: spatial texture router with depthwise/local conv + text-conditioned residual scale.
- res4/res5: channel/scale semantic router with scale scores.
- Extract lightweight per-target visual prototypes from res4/res5 using target-conditioned attention pooling, but do not create full per-target feature maps here.

Outputs:

- routed `res2/res3/res4/res5`
- `target_visual_proto`: `[sumK, C]`
- `peer_visual_proto`: `[sumK, C]`
- `scale_scores`: `[sumK or B, 4]` for diagnostics

Why here:

- This borrows VSPNet/SBANet correctly: scale-aware visual feature reorganization without altering Swin or MSDeformAttn internals.

### Stage B: Target-Background-Sibling Visual Calibrator

File: `segearth_r2/model/tgmsa/calibration.py`

Insertion point:

```python
mask_features, multi_scale_features = repeat_by_mask_num(...)
mask_features, multi_scale_features, image_losses, image_stats = self.tgmsa_visual_calibrator(
    mask_features, multi_scale_features, target_ctx, visual_ctx, targets_optional
)
```

Design:

- Operates after repeat, so each `[SEG]` gets its own visual copy.
- Generate three visual directions:
  - target residual from `target_visual_proto + target_embed`
  - background residual from image/background pooled features
  - sibling residual from `peer_visual_proto + hard_peer_embed`
- Apply residual calibration:

```text
F' = F + alpha_t * Target(F) - alpha_b * Background(F) - alpha_s * Sibling(F)
```

- Use low-rank channel modulation plus small spatial mask logits, not full attention over all pixels at all scales.
- Identity initialization for all residual alphas.

Why here:

- This is the strongest image-route insertion point. It is per-target, visual, after the pixel decoder, and avoids TCPD-style internal injection.

### Stage B losses: image-side visual contrast

File: `segearth_r2/model/tgmsa/losses.py`

Use only during training when masks are available.

Possible losses:

1. Target compactness:
   mask-region pooled features should align with `target_visual_proto`.
2. Background separation:
   outside-mask pooled features should be away from target prototype.
3. Sibling suppression:
   if two targets are from the same image, target A's mask-region feature should be farther from sibling B prototype.
4. Scale entropy / scale sparsity diagnostic:
   prevent all targets from using identical scale scores.

These losses should be low weight and separately logged. They are better aligned with image-route novelty than decoder query losses.

### Stage C: Decoder auxiliary only

File: `segearth_r2/model/tgmsa/decoder_aux.py`

Keep current peer-contrast dynamic query binding, but rename and position it as auxiliary:

- `DecoderAuxBinding`
- optional flag: `use_tgmsa_decoder_aux`
- small loss weights
- not used in first image-route proof unless A/B are stable

Purpose:

- prevent decoder collapse after image features are improved
- provide diagnostic binding logits

Not the main paper contribution.

## Code organization

Recommended final structure:

```text
segearth_r2/model/tgmsa/
  __init__.py
  context.py
  visual_router.py
  calibration.py
  losses.py
  decoder_aux.py
  registry.py or config.py
```

`llava_phi.py` should only orchestrate:

```python
target_ctx = self.tgmsa_context(SEG_embedding, mask_num)
image_features, visual_ctx = self.tgmsa_visual_router(image_features, target_ctx)
mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(image_features)
mask_features, multi_scale_features = repeat_by_mask_num(...)
mask_features, multi_scale_features, tgmsa_losses = self.tgmsa_visual_calibrator(...)
mask_outputs = self.predictor(...)
```

Avoid putting algorithm logic in `llava_phi.py`.

## Config design

Separate image-route flags from decoder auxiliary flags:

```text
use_tgmsa_visual_router
use_tgmsa_visual_calibrator
use_tgmsa_visual_contrast_loss
use_tgmsa_decoder_aux
```

Keep old flags temporarily as compatibility aliases if needed, but paper experiments should use image-route names.

Loss weights:

```text
tgmsa_target_compact_loss_weight
tgmsa_background_separation_loss_weight
tgmsa_sibling_suppression_loss_weight
tgmsa_scale_diversity_loss_weight
tgmsa_decoder_aux_loss_weight
```

Diagnostics:

```text
tgmsa_scale_scores
tgmsa_target_proto_norm
tgmsa_sibling_proto_similarity
tgmsa_background_similarity
tgmsa_calibration_alpha
```

## Experiment order under image-route mainline

1. B-only smoke and 50-step engineering test
   - safest because it operates after repeat and does not disturb pixel decoder internals.

2. A-only smoke and 50-step engineering test
   - checks whether router affects memory and stability.

3. A+B 1000-step screen
   - main proof of image-route claim.

4. A+B plus image-side contrast losses
   - only if A+B is stable but weak.

5. Decoder auxiliary only after image route is validated
   - not before.

Do not use C-only as the main experiment anymore.

## Hard rejections

- No `[SET]`, no union mask, no query0.
- No full joint decoder rewrite.
- No MSDeformAttn internal modulation in the first image-route version.
- No SAM/SAM2 second pipeline.
- No A+B+C full run before B-only/A-only/A+B attribution.
- No claim that Stage A/B/C are equally important; image route is the method, decoder auxiliary is optional.

## Top-tier claim after redesign

The strongest claim becomes:

> SegEarth-R2's flattened per-target decoding fails to enforce target-specific visual evidence before mask prediction. TG-VFR reorganizes image features through scale-wise target routing and target-background-sibling visual calibration, enabling the decoder to segment from target-conditioned visual representations rather than relying on late query binding alone.

This is image-route first, structurally adapted to SegEarth-R2, and distinct from SET++.
