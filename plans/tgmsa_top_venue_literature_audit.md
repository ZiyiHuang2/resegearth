# TG-MSA top-venue literature and structure audit

## Audit question

Can TG-MSA become a top-tier method for SegEarth-R2 / LaSeRS, or is it only a stack of text-conditioned adapters?

## Evidence corpus

Local PDFs reviewed from the user-provided `feature-guided image` paper folder on drive E:

- LQMFormer, CVPR 2024: dynamic language-aware query bank, scoring network, query-mask margin loss for query collapse.
- SBANet, ISPRS / arXiv 2025: bidirectional alignment, dynamic feature selection, text-conditioned channel/spatial aggregator.
- STDNet, IEEE JSTARS 2026 local PDF: spatial multi-scale correlation, target-background twin-stream decoder, dual-modal object learning.
- VSPNet, IEEE TGRS 2026 local PDF: shallow text-guided texture interaction and deep text-guided semantic fusion.
- RSAM, Remote Sensing 2025: two-way guidance and multimodal mask decoder.
- RS2-SAM2, AAAI 2026: pseudo-mask dense prompt and text-guided boundary constraints.
- FarmSeg_VLM, ISPRS 2025: dense visual prompt to correct image-text alignment.
- T2ASeg, IEEE TGRS 2025 local PDF: text-to-image activation and scale-aware semantic activation.

## Structural diagnosis of SegEarth-R2

The live SegEarth-R2 path is not a generic RIS transformer. Its decisive constraint is:

1. `[SEG]` embeddings are extracted from the LLM and projected to mask hidden dim.
2. `pixel_decoder.forward_features(image_features)` is run before per-target repetition.
3. `mask_features` and `multi_scale_features` are repeated by `mask_num`.
4. The mask decoder receives a flattened `sum(K_i)` batch, one `[SEG]` per predicted mask.

This means SegEarth-R2 is strong at per-SEG independent decoding but weak at same-image sibling competition. Any top-tier claim must target that weakness without repeating SET++.

## Mechanism matrix

### LQMFormer

Useful idea: language-aware dynamic query bank, query scoring, explicit anti-collapse loss.

Not directly enough: LQMFormer handles many decoder queries for a single referring expression. SegEarth-R2 has one active query per flattened `[SEG]`, so a simple DQM copy only becomes a query adapter.

TG-MSA adaptation: dynamic query position must be decoder-visible and same-image peer-aware. The current peer-contrast Stage C is aligned with this adaptation.

### SBANet

Useful idea: bidirectional alignment and dynamic scale selection; text-conditioned channel/spatial aggregation bridges encoder-decoder gap.

Not directly enough: full bidirectional image-text token updating would be expensive and would undermine the frozen LLM/vision feature reuse that SegEarth-R2 relies on.

TG-MSA adaptation: Stage A/B can borrow scale-wise gating, but should remain residual and identity-initialized. The main claim should not depend on Stage A/B.

### STDNet

Useful idea: target/background separation is relevant to remote sensing clustered objects and ambiguous backgrounds.

Not directly enough: a full twin-stream decoder would duplicate much of Mask2Former and make ablations unclear.

TG-MSA adaptation: Stage B should remain post-pixel target/background calibration, not another internal MSDeformAttn/FPN conditioning path.

### VSPNet

Useful idea: shallow texture interaction and deep semantic fusion match Swin output hierarchy.

Not directly enough: VSPNet uses a hybrid CNN/Transformer/Mamba backbone. SegEarth-R2 should not retrain or replace Swin.

TG-MSA adaptation: Stage A can use shallow spatial gates and deep channel gates, but only as auxiliary enhancement.

### RSAM / RS2-SAM2 / FarmSeg_VLM

Useful idea: dense prompt or pseudo-mask priors can provide spatial evidence to decoder.

Not directly enough: migrating SAM/SAM2 or adding a second foundation model would change the system identity and break comparability.

TG-MSA adaptation: a future optional extension could generate a lightweight pseudo-mask prior from existing mask decoder attention, but it must be diagnostic first. Do not add it before C-only evidence.

## Verdict

The current TG-MSA is only top-tier defensible if Stage C is the main method:

> Target-Guided Peer-Contrast Dynamic Query Binding for flattened per-SEG decoding.

A/B are supportive alignment/calibration modules. They are not the paper's core novelty.

## Strong design recommendation

Keep TG-MSA as a three-stage method in code, but write the method around one central mechanism:

1. Problem: SegEarth-R2 flattens multiple `[SEG]` tokens, losing explicit same-image competition.
2. Core: same-image peer-contrast dynamic query binding creates decoder-visible query positions conditioned on target-vs-sibling differences.
3. Support: Stage A supplies frozen-backbone target filtering; Stage B supplies target/background calibration after pixel decoding.
4. Proof: C-only must improve K>1 / multi-cate. A+C and B+C may improve robustness. A+B+C is only full-model validation after attribution is clear.

## Rejections after literature audit

- Do not add SET/union query.
- Do not port SAM/SAM2 as a second pipeline.
- Do not modify MSDeformAttn internally before same-checkpoint evidence exists.
- Do not claim generic image-text interaction as novelty.
- Do not let Stage A/B become the main story unless C-only fails and a new hypothesis is stated.

## Next evidence to collect

When GPU is free, run only C-only first with:

```bash
--use_tgmsa_decoder_binding --tgmsa_segment_separation_loss_weight 0.05 --tgmsa_peer_contrast_loss_weight 0.05 --tgmsa_binding_entropy_loss_weight 0.001
```

Required diagnostics:

- trainable parameter list includes only intended TG-MSA / mask modules.
- `tgmsa_binding_logits` distribution by same-image group.
- `loss_tgmsa_peer_contrast`, `loss_tgmsa_segment_separation`, and baseline mask loss curves.
- LaSeRS `test_multi_cate`, `test_long_query`, K>1 subset, plus single/explicit regression check.

Hard stop: if C-only does not move multi-cate/K>1 or reduces single/explicit, do not compensate by stacking A+B+C. Revisit whether `[SEG]` hidden states carry enough sibling-distinguishing information.
