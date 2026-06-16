# TG-MSA design audit and experiment gate

## Verdict

The first TG-MSA implementation was runnable, but it was not strong enough to support a paper-level claim about multi-target binding. The critical flaw was Stage C: it refined each flattened `[SEG]` independently and did not know which `[SEG]` tokens came from the same image. That makes it a text-conditioned query adapter, not a binding mechanism.

The current revision makes Stage C group-aware and decoder-visible:

- `mask_num` is passed into the mask decoder.
- Dynamic query binding returns a per-`[SEG]` `query_pos`, not only a residual `SEG_embedding` update.
- The decoder uses that `query_pos` in cross-attention when `use_seg_query=False`.
- The module reports four losses:
  - `loss_tgmsa_query_diversity`: separates candidate dynamic queries inside the bank.
  - `loss_tgmsa_segment_separation`: separates dynamic query positions among `[SEG]` tokens from the same image.
  - `loss_tgmsa_binding_entropy`: encourages confident bank selection.
  - `loss_tgmsa_peer_contrast`: makes a target query closer to its own text than to the hardest same-image sibling target.

This is now a defensible implementation of target-guided decoder binding. It still preserves strict ablation control because all TG-MSA flags default to off.


## Boundary from SET++

TG-MSA must not repeat the SET++ idea. The hard boundary is:

- No `[SET]` token.
- No union-mask target.
- No `query0` reserved for group prediction.
- No `Q = 1 + Kmax` joint decoder rewrite.
- No SET/SEG role split or SET-specific projector.
- No union auxiliary loss and no evaluation of a group-level mask.

The only shared information TG-MSA uses is `mask_num`, and only for two limited purposes:

1. Stage A averages the existing `[SEG]` embeddings to form an image-level conditioning vector before the pixel decoder.
2. Stage C computes same-image anti-collapse regularization among existing flattened `[SEG]` dynamic query positions.

The decoder output remains one mask per original `[SEG]`. Visual features are still repeated by `mask_num` before prediction, matching the baseline decoding topology. This makes TG-MSA a target-binding/query-calibration method, not a set-level union-query method.

## Stage roles

### Stage A: Swin-output target filtering

Purpose: target-aware multi-scale feature filtering before the pixel decoder without training Swin.

Implementation: `SwinOutputTargetFilter` builds an image-level text condition by averaging the `[SEG]` embeddings for each image according to `mask_num`. Shallow levels use spatial gates; deep levels use channel gates. All residual `alpha` parameters start at zero, so the baseline path is unchanged at initialization.

Limitation: Stage A is intentionally image-level. It should not be used as the main explanation for instance-level multi-`[SEG]` binding.

### Stage B: Pixel-output calibration

Purpose: target/background residual calibration after `pixel_decoder.forward_features`, without modifying MSDeformAttn or FPN internals.

Implementation: `PixelTargetBackgroundCalibrator` runs after features are repeated by `mask_num`, so each flattened `[SEG]` has its own target/background calibration. This avoids repeating the earlier TCPD mistake of strong internal injection.

Limitation: Stage B is a calibrated local refinement path. It can help target/background confusion, but it is not the main binding mechanism.

### Stage C: Dynamic query binding

Purpose: bind each `[SEG]` to a dynamic decoder query and prevent same-image query collapse.

Implementation: `DynamicQueryBinding` uses a learnable query bank, same-image peer context, hardest-sibling contrast, text-conditioned cross-attention over the bank, candidate scoring, and a selected query position. That query position is fed into the transformer decoder as `query_pos`, so it directly affects visual cross-attention.

The peer contrast path is the part that directly targets SegEarth-R2's structural weakness: baseline decoding flattens each `[SEG]` into an independent sample, so same-image sibling targets do not compete. TG-MSA keeps the same output topology but injects sibling-aware contrast into the query position and loss.

This is the main method claim.

## Recommended ablation schedule

Do not run A+B+C first.

1. Smoke only while GPU is busy:
   - CPU/static tests only, or `max_steps=1` only when GPU is safe.
   - Smoke checks import, shape, forward/backward, non-NaN loss, and trainable params.

2. First 50-step engineering checks when GPU is available:
   - C-only:
     `--use_tgmsa_decoder_binding --tgmsa_segment_separation_loss_weight 0.05 --tgmsa_peer_contrast_loss_weight 0.05 --tgmsa_binding_entropy_loss_weight 0.001`
   - A-only:
     `--use_tgmsa_swin_filter`
   - B-only:
     `--use_tgmsa_pixel_calibration`

3. 1000-step direction screen:
   - C-only
   - A+C
   - B+C

4. 5000-step decision:
   - Keep only the two strongest 1000-step variants.
   - Main evidence: LaSeRS `test_multi_cate`, `test_long_query`, K>1 instance, and no degradation on `test_single_cate` / `test_explicit`.

## Rejection criteria

Stop this method branch if any of the following happens after a clean same-backbone 1000-step run:

- Multi-cate drops by more than 1 gIoU versus the same warmstart baseline.
- Single/explicit drops while multi-cate is flat.
- `loss_tgmsa_segment_separation` goes to zero immediately while binding logits stay uniform.
- Binding logits collapse to the same bank index for all same-image `[SEG]` tokens.
- `loss_tgmsa_peer_contrast` is flat while multi-cate masks continue selecting sibling targets.
- Train loss behaves like the earlier TCPD mixed collapse.

## Audit commands already run

```bash
python -m py_compile segearth_r2/model/tgmsa/tgmsa.py segearth_r2/model/language_model/llava_phi.py segearth_r2/model/mask_decoder/Mask2Former_Simplify/modeling/transformer_decoder/mask2former_transformer_decoder.py segearth_r2/train/train.py segearth_r2/train/merge_lora_weights_and_save_hf_model.py
python tests/test_tgmsa_modules.py
git diff --check -- segearth_r2/model/language_model/llava_phi.py segearth_r2/model/mask_decoder/Mask2Former_Simplify/modeling/transformer_decoder/mask2former_transformer_decoder.py segearth_r2/model/tgmsa/tgmsa.py segearth_r2/model/tgmsa/__init__.py segearth_r2/train/train.py segearth_r2/train/merge_lora_weights_and_save_hf_model.py tests/test_tgmsa_modules.py
```

A decoder-level CPU smoke also verified that default-off returns no TG-MSA losses, while default-on returns `loss_tgmsa_query_diversity`, `loss_tgmsa_segment_separation`, `loss_tgmsa_binding_entropy`, and `loss_tgmsa_peer_contrast`.

## Fit to SegEarth-R2 after strict audit

### What TG-MSA now fits

SegEarth-R2's baseline mask path flattens `[SEG]` tokens into `sum(K_i)` independent mask samples after `mask_features` and `multi_scale_features` are repeated by `mask_num`. This is efficient and keeps one output mask per `[SEG]`, but it weakens same-image target competition. TG-MSA is now designed specifically for that structure instead of replacing it.

The current version fits the existing architecture in three ways:

1. It preserves the baseline output contract: one flattened `[SEG]` produces one mask.
2. It keeps pixel decoder internals unchanged, avoiding another TCPD-style strong injection into MSDeformAttn/FPN.
3. It adds same-image peer contrast only inside the dynamic query binding path, so it targets multi-cate binding without introducing a set-level query.

### Defects it can plausibly??

- Multi-cate / K>1 binding confusion: Stage C now uses same-image peer context and hardest sibling contrast, so target A's query is explicitly trained away from target B.
- Decoder query collapse: bank diversity, segment separation, entropy, and peer contrast give four observable diagnostics instead of one vague regularizer.
- Target/background ambiguity after pixel decoder: Stage B gives per-`[SEG]` target/background residual calibration after features are repeated.
- Backbone-level target blindness: Stage A gives a conservative image-level target filter while keeping Swin frozen.

### Defects it probably cannot??

- Pure boundary quality failure: TG-MSA does not add a boundary decoder or high-resolution refinement head.
- Dataset label noise or ambiguous language: peer contrast may amplify bad target descriptions if sibling annotations are noisy.
- Frozen language model semantic failure: if the `[SEG]` hidden state itself does not encode the correct referent, dynamic query binding cannot fully repair it.
- Global backbone mismatch: any comparison remains invalid if SigLIP/Swin/init differs from the baseline warmstart.

### Strongest falsifiable claim

The strongest honest claim is not "general image-text interaction improves segmentation". It is:

> In SegEarth-R2's flattened per-`[SEG]` decoding topology, target-conditioned dynamic query positions with same-image peer contrast reduce multi-target binding collapse without adding a set-level union query or modifying pixel-decoder internals.

This claim is testable by C-only versus baseline on the same warmstart, then A+C and B+C if C-only is stable.
