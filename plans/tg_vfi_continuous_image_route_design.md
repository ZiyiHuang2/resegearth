# TG-VFI continuous image-route design for SegEarth-R2

## 1. Position

The main line should be the image branch, not a decoder-query method.

Proposed method name:

**TG-VFI: Target-Guided Visual Feedback Interaction**

Core claim:

> SegEarth-R2 currently gives the mask decoder target-specific language queries, but the image features reaching the decoder are mostly target-blind. TG-VFI reorganizes image features through recurrent target-guided visual feedback before mask prediction, so the decoder reads target-conditioned visual evidence instead of repairing ambiguity only at the final query stage.

This is deliberately different from SET++:

- no `[SET]` token
- no union mask objective
- no query0 / query1..K semantic layout
- no group-level set decoder
- no SET projector or union supervision

Decoder-side query regularization can remain an auxiliary diagnostic, but it is not the main contribution.

## 2. Current SegEarth-R2 structure

The relevant path is:

```text
images
  -> frozen Swin mask tower
  -> res2/res3/res4/res5
  -> MSDeformAttnPixelDecoder.forward_features(features)
       transformer encoder consumes res3/res4/res5
       FPN path adds res2 back for high-resolution mask_features
  -> mask_features, multi_scale_features
  -> repeat_interleave by mask_num
  -> Mask2Former-style decoder with SEG_embedding
  -> one predicted mask per flattened [SEG]
```

Consequences:

1. Before the pixel decoder, features are image-level and shared by all targets in that image.
2. After `repeat_interleave(mask_num)`, every target has its own visual feature copy.
3. The current decoder can use different `[SEG]` embeddings, but the underlying visual features are still weakly separated by target.
4. Multi-target LaSeRS failures are therefore more likely to come from target-blind visual evidence and sibling confusion than from missing query count alone.

## 3. Why a simple adapter is not enough

The previous TG-MSA draft has three limitations:

1. Stage A only averages all `[SEG]` embeddings per image and applies one-shot gates to Swin outputs. This loses per-target information before the pixel decoder.
2. Stage B only applies linear target/background residuals after the pixel decoder. It does not build visual prototypes from the actual image, so it is closer to text FiLM than visual interaction.
3. Stage C dynamic query binding is useful but makes the paper drift toward a decoder-query method. That does not match the user's current image-route requirement.

For a stronger paper-level design, TG-VFI needs persistent target state and recurrent visual feedback:

```text
target language state
  -> image feature selection
  -> visual prototype extraction
  -> target/background/sibling state update
  -> scale-aware feature recalibration
  -> decoder
```

## 4. Literature mechanisms to borrow, with boundaries

### LAVT

LAVT argues for language-aware visual encoding inside the visual feature extraction stage, rather than relying only on a late cross-modal decoder. TG-VFI should borrow this principle, but not replace SegEarth-R2's frozen Swin with a new language-aware Swin.

Use in TG-VFI:

- continuous target influence during visual feature construction
- visual-branch-first framing

Boundary:

- do not unfreeze or rebuild the Swin backbone in the first implementation

### SBANet

SBANet's useful idea is scale-wise bidirectional alignment and dynamic feature selection. For remote sensing, target scale varies strongly, so a uniform text gate is not enough.

Use in TG-VFI:

- per-target scale state
- dynamic res2/res3/res4/res5 weighting
- encoder-decoder gap bridging

Boundary:

- do not copy a full bidirectional language encoder
- do not let this become another text-token refinement paper

### RMSIN

RMSIN is important because remote sensing referring segmentation depends on intra-scale detail and cross-scale interaction. This supports a visual module that does more than channel gating.

Use in TG-VFI:

- intra-scale local interaction for res2/res3
- cross-scale target state exchange among res3/res4/res5
- optional orientation-aware local kernels later

Boundary:

- do not add rotation-specific convolution in version 1 unless the first image-route experiments show boundary/orientation failure

### STDNet / BAGJP target-background twin stream

These works support explicit target/background separation. The important transferable idea is not the whole decoder, but the target-vs-non-target visual decomposition.

Use in TG-VFI:

- target prototype
- background prototype
- sibling prototype for same-image competing targets

Boundary:

- do not duplicate the Mask2Former decoder

### LQMFormer

LQMFormer supports dynamic query and anti-collapse regularization. This is useful only as auxiliary support after the visual route is stable.

Use in TG-VFI:

- optional decoder auxiliary loss
- query collapse diagnostic

Boundary:

- not a first-round main experiment

## 5. Main design: TG-VFI

TG-VFI consists of four image-route components and one optional decoder auxiliary.

```text
SEG_embedding + mask_num
  -> TargetVisualState
  -> RecurrentScaleVisualInteractor
  -> PixelDecoderBridge
  -> TargetBackgroundSiblingCalibrator
  -> Mask2Former decoder
```

### 5.1 TargetVisualState

File:

```text
segearth_r2/model/tgmsa/context.py
```

Inputs:

- `SEG_embedding`: `[sumK, 1, C]`
- `mask_num`: list or tensor with length `B`
- optional token-level hidden states in a later version

Outputs:

- `target_embed`: `[sumK, C]`
- `image_embed`: `[B, C]`
- `peer_embed`: `[sumK, C]`
- `hard_peer_embed`: `[sumK, C]`
- `target_to_image`: `[sumK]`
- `group_slices`
- recurrent state slots:
  - `target_state`
  - `background_state`
  - `sibling_state`
  - `scale_state`

Purpose:

This module owns all regrouping logic. No other module should manually split by `mask_num`.

Design detail:

- `target_state` starts from projected `[SEG]`.
- `image_embed` is only used for image-level operations before repeat.
- `peer_embed` is the average of other targets in the same image.
- `hard_peer_embed` is the most similar sibling target by cosine similarity.
- State updates use residual GRU-style MLP blocks with zero-initialized output scale, so the module starts close to identity.

### 5.2 RecurrentScaleVisualInteractor

File:

```text
segearth_r2/model/tgmsa/visual_interactor.py
```

Insertion point:

```python
image_features = self.get_vision_tower_feature(images)
target_ctx = self.tgmsa_context(SEG_embedding, mask_num)
image_features, visual_state = self.tgmsa_visual_interactor(image_features, target_ctx)
```

This replaces the old `SwinOutputTargetFilter`.

Design:

1. Keep `res2/res3/res4/res5` image-batch shaped. Do not repeat high-resolution maps per target before the pixel decoder.
2. Maintain target states per `[SEG]`, but apply image-level shared feature updates through grouped target summaries.
3. For each scale:
   - extract target-conditioned visual prototypes by attention pooling
   - update target state from visual evidence
   - update image feature by target-state-conditioned local residual
4. Use shallow/deep split:
   - `res2/res3`: local spatial interaction, depthwise convolution, small target-aware spatial maps
   - `res4/res5`: semantic channel interaction and scale selection
5. Add cross-scale state exchange:
   - high-level target state guides low-level detail selection
   - low-level prototype updates boundary/detail confidence for the target

Pseudo-flow:

```text
state_0 = target_embed

for level in [res5, res4, res3, res2]:
    proto_l = attention_pool(feature_l, state_l, grouped_by_image)
    state_l = state_l + StateUpdate(state_l, proto_l, peer_state)
    scale_weight_l = ScaleRouter(state_l, proto_l)
    feature_l = feature_l + alpha_l * VisualResidual(feature_l, grouped_state_l, scale_weight_l)

return routed_features, visual_state
```

Why this is a sustained interaction module:

- target state changes after reading each visual level
- later feature levels use updated target state
- each target gets its own prototype even though pre-pixel features stay image-batch shaped
- the module can diagnose whether targets collapse to identical scale states

Expected tensor behavior:

- input features:
  - `res2`: `[B, 128, H/4, W/4]`
  - `res3`: `[B, 256, H/8, W/8]`
  - `res4`: `[B, 512, H/16, W/16]`
  - `res5`: `[B, 1024, H/32, W/32]`
- target state:
  - `[sumK, C]`
- grouped image feature update:
  - compute per-image summary from targets
  - apply one routed residual per image/level
- target prototypes:
  - `[sumK, C]`

### 5.3 PixelDecoderBridge

File:

```text
segearth_r2/model/tgmsa/pixel_bridge.py
```

This is the answer to the question: why not directly modify internals?

TG-VFI should modify pixel-decoder-adjacent internal interfaces, but not first attack the MSDeformAttn sampling core.

Safe internal points:

1. After `input_proj` and before the deformable encoder:
   - add low-rank target-aware scale bias through image-level grouped target state
   - this lets the encoder see target-prioritized levels without changing reference points
2. During FPN lateral fusion:
   - apply target-aware residual to the lower-level lateral feature
   - this is where res2 detail re-enters mask feature construction
3. Before the final `mask_features` projection:
   - expose a hook to pass calibrated feature maps to Stage B

Unsafe first-version point:

- directly modulating MSDeformAttn offsets, sampling locations, or attention weights per target

Reason:

- before repeat, the pixel decoder does not have per-target feature maps
- repeating inside the deformable encoder multiplies memory and changes the training surface sharply
- previous TCPD-style internal injection already showed a high-risk failure mode
- reference-point geometry is a fragile core mechanism; changing it first would make attribution poor

Version 1 decision:

- implement `PixelDecoderBridge` as a wrapper or small internal hooks around projection/FPN fusion
- keep the deformable attention operation itself unchanged

Version 2 decision, only if needed:

- add a constrained internal attention-bias term with zero initialization
- run same-checkpoint on/off evaluation before any long training

### 5.4 TargetBackgroundSiblingCalibrator

File:

```text
segearth_r2/model/tgmsa/calibration.py
```

Insertion point:

```python
mask_features, multi_scale_features = repeat_by_mask_num(...)
mask_features, multi_scale_features, tg_vfi_losses, tg_vfi_stats = self.tgmsa_calibrator(
    mask_features,
    multi_scale_features,
    target_ctx,
    visual_state,
    gt_masks_optional,
)
```

This replaces the old `PixelTargetBackgroundCalibrator`.

Design:

- operates after repeat, so each target gets its own visual feature copy
- uses visual prototypes, not only text projections
- decomposes calibration into three directions:
  - target enhancement
  - background suppression
  - sibling suppression

Formula:

```text
F'_k = F_k
     + alpha_t * T(F_k, target_state_k, target_proto_k)
     - alpha_b * B(F_k, background_state_k, background_proto_k)
     - alpha_s * S(F_k, sibling_state_k, sibling_proto_k)
```

Implementation constraints:

- low-rank channel modulation
- small spatial logits at each scale
- no full dense cross-attention over all pixels for all targets in version 1
- all residual alphas initialized to zero

Why this fits LaSeRS:

- multi-cate and K>1 failures require target-vs-similar-object separation
- background alone is not enough; sibling suppression is needed when two valid targets appear in the same image
- this makes visual features target-specific before the decoder predicts masks

### 5.5 Image-side losses

File:

```text
segearth_r2/model/tgmsa/losses.py
```

Losses are optional and low-weight.

1. Target compactness:

```text
pooled_feature(mask_k) should align with target_proto_k
```

2. Background separation:

```text
pooled_feature(non_mask_k) should be separated from target_proto_k
```

3. Sibling suppression:

```text
feature inside mask_k should be less similar to sibling_proto_j than to target_proto_k
```

4. Scale diversity:

```text
same-image targets should not always choose identical scale distributions
```

These losses are better aligned with an image-route paper than decoder-only query diversity.

### 5.6 DecoderAuxBinding

File:

```text
segearth_r2/model/tgmsa/decoder_aux.py
```

Keep the current dynamic query binding idea only as auxiliary:

- optional flag: `use_tgmsa_decoder_aux`
- small loss weight
- disabled in the first image-route proof

Purpose:

- diagnose query collapse
- stabilize final binding after image features become target-aware

Not allowed:

- presenting this as the main method
- running C-only as the first main experiment

## 6. Code structure

Recommended final structure:

```text
segearth_r2/model/tgmsa/
  __init__.py
  context.py
  visual_interactor.py
  pixel_bridge.py
  calibration.py
  losses.py
  decoder_aux.py
  config.py
```

`llava_phi.py` should orchestrate only:

```python
target_ctx = self.tgmsa_context(SEG_embedding, mask_num)

image_features = self.get_vision_tower_feature(images)
image_features, visual_state = self.tgmsa_visual_interactor(image_features, target_ctx)

mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(
    image_features,
    tg_vfi_bridge=self.tgmsa_pixel_bridge if enabled else None,
    tg_vfi_state=visual_state,
)

mask_features = torch.repeat_interleave(mask_features, repeats=mask_num, dim=0)
multi_scale_features = [
    torch.repeat_interleave(x, repeats=mask_num, dim=0)
    for x in multi_scale_features
]

mask_features, multi_scale_features, tg_vfi_losses = self.tgmsa_calibrator(
    mask_features,
    multi_scale_features,
    target_ctx,
    visual_state,
    gt_masks_optional,
)

pred_masks = self.predictor(...)
```

Algorithmic logic should not be embedded directly in `llava_phi.py`.

## 7. Config flags

Use image-route names:

```text
use_tg_vfi
use_tg_vfi_visual_interactor
use_tg_vfi_pixel_bridge
use_tg_vfi_calibrator
use_tg_vfi_image_losses
use_tg_vfi_decoder_aux
```

Main hyperparameters:

```text
tg_vfi_state_dim
tg_vfi_interaction_steps
tg_vfi_proto_topk
tg_vfi_scale_temperature
tg_vfi_alpha_init
tg_vfi_target_loss_weight
tg_vfi_background_loss_weight
tg_vfi_sibling_loss_weight
tg_vfi_scale_loss_weight
tg_vfi_decoder_aux_loss_weight
```

Compatibility:

- old `use_tgmsa_*` flags can remain as aliases during transition
- new experiments should use `tg_vfi_*`

## 8. Diagnostics

Log these values:

```text
tg_vfi_scale_entropy
tg_vfi_scale_scores_res2
tg_vfi_scale_scores_res3
tg_vfi_scale_scores_res4
tg_vfi_scale_scores_res5
tg_vfi_target_proto_norm
tg_vfi_target_peer_cosine
tg_vfi_target_background_cosine
tg_vfi_sibling_margin
tg_vfi_alpha_visual
tg_vfi_alpha_calibrator_target
tg_vfi_alpha_calibrator_background
tg_vfi_alpha_calibrator_sibling
```

Failure signals:

- all targets in one image have near-identical scale scores
- sibling cosine remains high after calibration
- target/background cosine does not separate
- residual alphas explode early
- image losses improve while mask metrics degrade

## 9. Experiment plan

GPU constraint:

- when the A100 is busy, only run `max_steps=1` smoke
- no performance claims from 1-step runs
- no resource抢占

### Phase 0: static and CPU validation

Checks:

- import
- config parse
- module shape tests
- identity initialization tests
- forward/backward non-NaN with tiny tensors

No GPU required.

### Phase 1: 1-step smoke

Run separately:

1. `visual_interactor` only
2. `calibrator` only
3. `pixel_bridge` only
4. `visual_interactor + calibrator`

Pass condition:

- no shape error
- no NaN
- losses finite
- trainable parameter list matches intended modules

### Phase 2: 50-step engineering stability

Run only when GPU is free.

Order:

1. calibrator only
2. visual interactor only
3. visual interactor + calibrator
4. visual interactor + pixel bridge

Goal:

- exclude memory blowup
- exclude unstable residual growth
- inspect diagnostics

### Phase 3: 1000-step direction screen

Run:

1. `VFI-Calib`
2. `VFI-Interact`
3. `VFI-Interact+Calib`
4. `VFI-Interact+Bridge`, only if previous modules are stable

Do not run all components at once in the first screen.

Pass condition:

- LaSeRS multi-cate or K>1 subset not worse than baseline by more than 1 gIoU point
- no TCPD-style global collapse
- diagnostics show target/sibling separation improving

### Phase 4: 5000-step decision

Keep only the best two from Phase 3.

Metrics:

- LaSeRS `test_multi_cate`
- `test_long_query`
- K>1 instance subset
- `test_single_cate`
- `test_explicit`

Pass condition:

- multi-cate or K>1 has stable positive gain
- single/explicit degradation <= 0.5 gIoU point

### Phase 5: full run

Only the 5000-step winner gets a 50000/80000-step full run.

Required audit:

- launch command
- merged config
- trainable parameters
- `trainer_state.json`
- checkpoint provenance
- prediction counts
- grouped metrics JSON
- same-checkpoint module on/off evaluation if internal pixel bridge is active

## 10. Explicit rejection list

Do not do these in the main route:

- no SET++ union-query design
- no C-only decoder-query main experiment
- no direct MSDeformAttn offset/reference-point modulation in version 1
- no full SAM/SAM2/CLIP second pipeline
- no training Swin backbone in the first implementation
- no all-components full run before attribution
- no claim that 1-step or 50-step results prove performance

## 11. Why this is stronger than the previous TG-MSA

Previous TG-MSA:

```text
text gate after Swin
  + text residual after pixel decoder
  + decoder dynamic query
```

TG-VFI:

```text
persistent target visual state
  + recurrent scale-wise visual interaction
  + safe pixel-decoder bridge
  + target/background/sibling calibration
  + image-side contrastive diagnostics
```

The difference is structural:

- target state is updated by visual evidence
- visual features are updated by target state
- same-image sibling targets are modeled explicitly
- pixel decoder is touched at safe feature-fusion interfaces
- decoder auxiliary is not allowed to become the main story

This better matches the user's requirement for a non-simple, continuous interaction module and keeps the contribution aligned with the image branch.

