#!/usr/bin/env python3
"""
Shape test for SegEarth-R2 SET++ design.
Validates all critical tensor dimension paths without requiring actual model weights.

Usage:
    python scripts/shape_test_setpp.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

# ============================================================
# Test Config
# ============================================================
HIDDEN_DIM = 256                     # mask_decoder hidden_dim
LLM_HIDDEN = 3072                    # Phi-3B hidden_size
B = 4                                # batch size
T = 256                              # sequence length (tokenized)
K_list = [3, 1, 2, 2]               # mask_num per sample (batch)
Kmax = max(K_list)                   # 3
sum_K = sum(K_list)                  # 8
H, W = 200, 200                      # mask feature map (after pixel_decoder)
IMG_H, IMG_W = 800, 800              # original image size

# ============================================================
# Test 1: Dataset → Collate → Forward (SET token path)
# ============================================================
def test_dataset_to_collate():
    print("=" * 60)
    print("Test 1: Dataset → Collate → Forward (SET token path)")
    print("=" * 60)

    # Simulate per-sample dataset output
    for b_idx, k in enumerate(K_list):
        input_ids = torch.randint(1, 50000, (T,))
        # [SET] token at position idx 50, [SEG] tokens at positions 51, 52, ...
        SEG_token_id = 99999
        SET_token_id = 88888
        input_ids[50] = SET_token_id
        for i in range(k):
            input_ids[51 + i] = SEG_token_id

        SEG_indices = (input_ids == SEG_token_id).long()
        SET_indices = (input_ids == SET_token_id).long()

        print(f"  Sample {b_idx}: mask_num={k}, "
              f"SEG_count={SEG_indices.sum().item()}, "
              f"SET_count={SET_indices.sum().item()}")

        # Verify
        assert SEG_indices.sum().item() == k, f"Sample {b_idx}: SEG count mismatch"
        assert SET_indices.sum().item() == 1, f"Sample {b_idx}: SET count mismatch"

    # Simulate collate: pad SEG/SET indices
    # In practice, collate pads to batch_first=True
    SEG_padded = torch.zeros(B, T, dtype=torch.float)
    SET_padded = torch.zeros(B, T, dtype=torch.float)
    for b_idx, k in enumerate(K_list):
        SEG_padded[b_idx, 51:51+k] = 1.0
        SET_padded[b_idx, 50] = 1.0

    print(f"  Collated SEG_token_embedding_indices: {SEG_padded.shape}")  # [B, T]
    print(f"  Collated SET_token_embedding_indices: {SET_padded.shape}")  # [B, T]
    assert SEG_padded.shape == (B, T), "SEG padded shape error"
    assert SET_padded.shape == (B, T), "SET padded shape error"

    # Simulate get_SEG_embedding / get_SET_embedding
    hidden_states = torch.randn(B, T, LLM_HIDDEN)  # [B, T, LLM_hidden]

    # get_SEG_embedding: extract where SEG_indices == 1
    SEG_embedding_list = []
    for b_idx in range(B):
        mask = SEG_padded[b_idx].bool()
        SEG_embedding_list.append(hidden_states[b_idx, mask, :])
    SEG_embedding = torch.cat(SEG_embedding_list, dim=0).unsqueeze(1)  # [sum(K), 1, LLM_hidden]
    print(f"  SEG_embedding (raw): {SEG_embedding.shape}")  # [8, 1, 3072]

    # get_SET_embedding: extract where SET_indices == 1
    SET_embedding_list = []
    for b_idx in range(B):
        mask = SET_padded[b_idx].bool()
        SET_embedding_list.append(hidden_states[b_idx, mask, :])
    SET_embedding = torch.cat(SET_embedding_list, dim=0).unsqueeze(1)  # [B, 1, LLM_hidden]
    print(f"  SET_embedding (raw): {SET_embedding.shape}")  # [4, 1, 3072]

    # Project through independent SEG/SET token projectors (SET copy-init from SEG)
    seg_projector = nn.Linear(LLM_HIDDEN, HIDDEN_DIM)
    set_projector = nn.Linear(LLM_HIDDEN, HIDDEN_DIM)
    set_projector.load_state_dict(seg_projector.state_dict())
    SEG_embedding = seg_projector(SEG_embedding)  # [sum(K), 1, 256]
    SET_embedding = set_projector(SET_embedding)  # [B, 1, 256]
    print(f"  SEG_embedding (projected): {SEG_embedding.shape}")  # [8, 1, 256]
    print(f"  SET_embedding (projected): {SET_embedding.shape}")  # [4, 1, 256]

    assert SEG_embedding.shape == (sum_K, 1, HIDDEN_DIM), "SEG embedding shape error"
    assert SET_embedding.shape == (B, 1, HIDDEN_DIM), "SET embedding shape error"

    print("  ✅ Test 1 PASSED\n")
    return SEG_embedding, SET_embedding


# ============================================================
# Test 2: Decoder Query Layout (SET + SEG dual-line)
# ============================================================
def test_decoder_query_layout(SEG_embedding, SET_embedding):
    print("=" * 60)
    print("Test 2: Decoder Query Layout (SET + SEG dual-line)")
    print("=" * 60)

    # SEG_embedding: [sum(K), 1, C]
    # SET_embedding: [B, 1, C]

    # Step 1: Pad SEG to [B, Kmax, C]
    SEG_emb = SEG_embedding.squeeze(1)  # [sum(K), C]
    SEG_padded = torch.zeros(B, Kmax, HIDDEN_DIM)
    SEG_split = list(torch.split(SEG_emb, K_list, dim=0))
    for b_idx, seg in enumerate(SEG_split):
        SEG_padded[b_idx, :seg.shape[0]] = seg
    print(f"  SEG_padded: {SEG_padded.shape}")  # [4, 3, 256]

    # Step 2: Build SET query with role embedding
    SET_query_embed = nn.Embedding(1, HIDDEN_DIM)
    nn.init.constant_(SET_query_embed.weight, 0)
    # weight is [1, C]; unsqueeze(0) -> [1, 1, C] to broadcast with [B, 1, C]
    SET_query = SET_embedding + SET_query_embed.weight.unsqueeze(0)  # [B, 1, C]
    print(f"  SET_query: {SET_query.shape}")  # [4, 1, 256]

    # Step 3: Build SEG queries with role embedding
    SEG_role_embed = nn.Embedding(1, HIDDEN_DIM)
    nn.init.constant_(SEG_role_embed.weight, 0)
    # weight is [1, C]; unsqueeze(0) -> [1, 1, C] to broadcast with [B, Kmax, C]
    SEG_query = SEG_padded + SEG_role_embed.weight.unsqueeze(0)  # [B, Kmax, C]
    print(f"  SEG_query: {SEG_query.shape}")  # [4, 3, 256]

    # Step 4: Concat [SET, SEG_1, ..., SEG_Kmax]
    output = torch.cat([SET_query, SEG_query], dim=1)  # [B, 1+Kmax, C]
    output = output.permute(1, 0, 2)  # [1+Kmax, B, C]
    print(f"  output (permuted): {output.shape}")  # [4, 4, 256]

    # Step 5: combined_emb for SEG_class dot-product
    SET_emb_2d = SET_embedding.squeeze(1)  # [B, C]
    combined_emb = torch.cat([SET_emb_2d.unsqueeze(1), SEG_padded], dim=1)  # [B, 1+Kmax, C]
    print(f"  combined_emb: {combined_emb.shape}")  # [4, 4, 256]

    # Step 6: tgt_key_padding_mask
    tgt_key_padding_mask = torch.ones(B, 1 + Kmax, dtype=torch.bool)
    tgt_key_padding_mask[:, 0] = False  # SET always valid
    for b_idx, k in enumerate(K_list):
        tgt_key_padding_mask[b_idx, 1:1 + k] = False  # valid SEG queries
    print(f"  tgt_key_padding_mask: {tgt_key_padding_mask.shape}")  # [4, 4]
    print(f"  tgt_key_padding_mask values:\n{tgt_key_padding_mask.int()}")

    # Verify padding mask correctness
    for b_idx, k in enumerate(K_list):
        assert tgt_key_padding_mask[b_idx, 0] == False, f"Sample {b_idx}: SET should be valid"
        for i in range(1, 1 + k):
            assert tgt_key_padding_mask[b_idx, i] == False, f"Sample {b_idx}: SEG {i-1} should be valid"
        for i in range(1 + k, 1 + Kmax):
            assert tgt_key_padding_mask[b_idx, i] == True, f"Sample {b_idx}: padded SEG {i-1} should be masked"

    # Step 7: query_embed (positional) - zeros
    query_embed = torch.zeros(1 + Kmax, B, HIDDEN_DIM)
    print(f"  query_embed: {query_embed.shape}")  # [4, 4, 256]

    assert output.shape == (1 + Kmax, B, HIDDEN_DIM)
    assert combined_emb.shape == (B, 1 + Kmax, HIDDEN_DIM)
    assert tgt_key_padding_mask.shape == (B, 1 + Kmax)
    assert query_embed.shape == (1 + Kmax, B, HIDDEN_DIM)

    print("  ✅ Test 2 PASSED\n")
    return output, combined_emb, tgt_key_padding_mask, query_embed


# ============================================================
# Test 3: Self-Attention with tgt_key_padding_mask
# ============================================================
def test_self_attention_mask(output, tgt_key_padding_mask, query_embed):
    print("=" * 60)
    print("Test 3: Self-Attention with tgt_key_padding_mask")
    print("=" * 60)

    # Simulate SelfAttentionLayer
    nhead = 8
    self_attn = nn.MultiheadAttention(HIDDEN_DIM, nhead, dropout=0.0, batch_first=False)

    # Forward: output is [1+Kmax, B, C], key_padding_mask is [B, 1+Kmax]
    attn_out, attn_weights = self_attn(
        output, output, output,
        key_padding_mask=tgt_key_padding_mask,
        need_weights=True,
        average_attn_weights=False
    )
    print(f"  attn_out: {attn_out.shape}")  # [4, 4, 256]
    print(f"  attn_weights: {attn_weights.shape}")  # [4, 8, 4, 4]

    assert attn_out.shape == (1 + Kmax, B, HIDDEN_DIM), "Attention output shape error"
    assert attn_weights.shape == (B, nhead, 1 + Kmax, 1 + Kmax), "Attention weights shape error"

    print("  ✅ Test 3 PASSED\n")


# ============================================================
# Test 4: Forward Prediction Heads
# ============================================================
def test_forward_prediction_heads(output, combined_emb):
    print("=" * 60)
    print("Test 4: Forward Prediction Heads")
    print("=" * 60)

    # Simulate mask_features from pixel_decoder
    mask_features = torch.randn(B, HIDDEN_DIM, H // 4, W // 4)  # typical: 1/4 of input
    print(f"  mask_features: {mask_features.shape}")  # [4, 256, 50, 50]

    # decoder_output: [B, Q, C] where Q = 1+Kmax
    decoder_output = output.transpose(0, 1)  # [B, 1+Kmax, C]
    print(f"  decoder_output: {decoder_output.shape}")  # [4, 4, 256]

    # SEG_class: dot-product [B, Q, C] x [B, Q, C] -> [B, Q] -> [B, Q, 1]
    SEG_class = torch.einsum('bqc,bqc->bq', decoder_output, combined_emb).unsqueeze(-1)
    print(f"  SEG_class: {SEG_class.shape}")  # [4, 4, 1]

    # mask_embed: MLP on decoder_output
    mask_embed = nn.Sequential(
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
    )(decoder_output)
    print(f"  mask_embed: {mask_embed.shape}")  # [4, 4, 256]

    # outputs_mask: [B, Q, C] x [B, C, H, W] -> [B, Q, H, W]
    outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
    print(f"  outputs_mask: {outputs_mask.shape}")  # [4, 4, 50, 50]
    print(f"  pred_masks (full): {outputs_mask.shape}")  # [B, 1+Kmax, H, W]

    assert SEG_class.shape == (B, 1 + Kmax, 1), "SEG_class shape error"
    assert outputs_mask.shape == (B, 1 + Kmax, mask_features.shape[2], mask_features.shape[3]), "outputs_mask shape error"

    print("  ✅ Test 4 PASSED\n")
    return outputs_mask, SEG_class


# ============================================================
# Test 5: Criterion - Strip Query 0 + Hungarian Matcher
# ============================================================
def test_criterion_strip_and_match(pred_masks_full, SEG_class_full):
    print("=" * 60)
    print("Test 5: Criterion - Strip Query 0 + Hungarian Matcher")
    print("=" * 60)

    # Strip query 0 (SET union mask)
    instance_masks = pred_masks_full[:, 1:, :, :]  # [B, Kmax, H, W]
    instance_SEG_logits = SEG_class_full[:, 1:, :]  # [B, Kmax, 1]
    print(f"  instance_masks: {instance_masks.shape}")  # [4, 3, 50, 50]
    print(f"  instance_SEG_logits: {instance_SEG_logits.shape}")  # [4, 3, 1]

    # Union mask (query 0)
    union_mask = pred_masks_full[:, 0:1, :, :]  # [B, 1, H, W]
    print(f"  union_mask: {union_mask.shape}")  # [4, 1, 50, 50]

    # Simulate Hungarian matcher on instance_masks
    # In practice, matcher takes [B, num_queries, H, W] and targets
    # Here we just verify shapes
    B_match, Q_match, H_match, W_match = instance_masks.shape
    assert Q_match == Kmax, f"Instance queries should be Kmax={Kmax}, got {Q_match}"
    assert B_match == B

    # Simulate targets (for matcher verification)
    # Each sample has K_i instances
    targets = []
    for b_idx, k in enumerate(K_list):
        target_masks = torch.randn(k, H_match, W_match)  # [K_i, H, W]
        target_labels = torch.zeros(k, dtype=torch.long)
        targets.append({"masks": target_masks, "labels": target_labels})

    # Run matcher (simplified: just check it doesn't crash)
    from scipy.optimize import linear_sum_assignment

    def simple_matcher(pred_masks, targets):
        """Simplified matcher for shape testing."""
        indices = []
        for b in range(len(targets)):
            K_pred = pred_masks[b].shape[0]  # Kmax
            K_tgt = targets[b]["masks"].shape[0]  # K_i
            # Cost matrix: [K_pred, K_tgt] - use negative IOU approximation
            cost = torch.rand(K_pred, K_tgt)  # dummy cost
            cost_np = cost.cpu().numpy()
            # Only match valid predictions (first K_i of Kmax)
            cost_valid = cost_np[:K_tgt, :]
            row_ind, col_ind = linear_sum_assignment(cost_valid)
            indices.append((torch.tensor(row_ind), torch.tensor(col_ind)))
        return indices

    indices = simple_matcher(instance_masks, targets)
    print(f"  Matcher indices: {[(len(src), len(tgt)) for src, tgt in indices]}")

    # Verify indices are within valid range
    for b_idx, (src, tgt) in enumerate(indices):
        k = K_list[b_idx]
        assert src.max() < k, f"Sample {b_idx}: src index {src.max()} >= K_i={k}"
        assert tgt.max() < k, f"Sample {b_idx}: tgt index {tgt.max()} >= K_i={k}"

    # Build union targets
    union_targets = []
    for t in targets:
        masks = t["masks"]
        union = (masks.float().sum(dim=0) > 0).float().unsqueeze(0)  # [1, H, W]
        union_targets.append({"union_mask": union})

    print(f"  union_targets: {len(union_targets)} samples")
    print(f"  union_targets[0]['union_mask']: {union_targets[0]['union_mask'].shape}")

    assert union_mask.shape == (B, 1, H_match, W_match)
    assert instance_masks.shape == (B, Kmax, H_match, W_match)

    print("  ✅ Test 5 PASSED\n")
    return instance_masks, union_mask, indices


# ============================================================
# Test 6: Loss Computation (union + instance)
# ============================================================
def test_loss_computation(pred_masks_full, SEG_class_full, union_mask, instance_masks, indices):
    print("=" * 60)
    print("Test 6: Loss Computation (union + instance)")
    print("=" * 60)

    # --- Union loss ---
    # point sampling on union_mask [B, 1, H, W]
    num_points = 12544
    B_union, Q_union, H_u, W_u = union_mask.shape
    src_masks = union_mask.flatten(0, 1)  # [B, H, W]
    print(f"  union src_masks (flattened): {src_masks.shape}")

    # In practice: point sampling + sigmoid_ce + dice
    # Just verify shapes
    assert src_masks.shape == (B, H_u, W_u), "Union mask flatten error"

    # --- Instance loss ---
    # src_idx, tgt_idx from indices
    src_idx = torch.cat([src for src, _ in indices])  # [sum(K_i)]
    tgt_idx = torch.cat([tgt for _, tgt in indices])  # [sum(K_i)]
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])

    src_masks_instance = instance_masks[batch_idx, src_idx]  # [sum(K_i), H, W]
    print(f"  src_masks_instance: {src_masks_instance.shape}")  # [8, 50, 50]

    # Verify sum(K_i) = sum_K
    assert src_masks_instance.shape[0] == sum_K, f"Expected {sum_K} matched instances, got {src_masks_instance.shape[0]}"

    # --- SEG_class loss ---
    instance_SEG_logits = SEG_class_full[:, 1:, :]  # [B, Kmax, 1]
    target_query = torch.zeros_like(instance_SEG_logits)
    for i, (index_i, _) in enumerate(indices):
        target_query[i, index_i] = 1
    print(f"  target_query: {target_query.shape}")  # [4, 3, 1]

    # Verify target_query has correct number of positives
    assert target_query.sum().item() == sum_K, f"Expected {sum_K} positive labels, got {target_query.sum().item()}"

    num_sample = instance_SEG_logits.shape[0] * instance_SEG_logits.shape[1]
    pos_weight = (num_sample - sum_K) / sum_K
    print(f"  SEG_class pos_weight: {pos_weight:.2f}")

    print("  ✅ Test 6 PASSED\n")


# ============================================================
# Test 7: eval_seg Path (strip query 0 + padded queries)
# ============================================================
def test_eval_seg_path(pred_masks_full):
    print("=" * 60)
    print("Test 7: eval_seg Path (strip query 0 + padded queries)")
    print("=" * 60)

    # [B, 1+Kmax, H, W] -> upsample to original image size
    mask_pred_results = F.interpolate(
        pred_masks_full,
        size=(IMG_H, IMG_W),
        mode="bilinear",
        align_corners=False,
    )
    print(f"  mask_pred_results (upsampled): {mask_pred_results.shape}")  # [4, 4, 800, 800]

    # Strip query 0 (SET union mask)
    instance_masks = mask_pred_results[:, 1:, :, :]  # [B, Kmax, H, W]
    print(f"  instance_masks: {instance_masks.shape}")  # [4, 3, 800, 800]

    # Remove padded SEG queries: [B, Kmax, H, W] -> [sum(K_i), H, W]
    flattened_masks = []
    for b_idx, k in enumerate(K_list):
        flattened_masks.append(instance_masks[b_idx, :k, :, :])
    mask_pred_results = torch.cat(flattened_masks, dim=0)  # [sum(K_i), H, W]
    print(f"  mask_pred_results (flattened): {mask_pred_results.shape}")  # [8, 800, 800]

    assert mask_pred_results.shape[0] == sum_K, f"Expected {sum_K} flattened masks, got {mask_pred_results.shape[0]}"
    assert mask_pred_results.shape[1] == IMG_H
    assert mask_pred_results.shape[2] == IMG_W

    print("  ✅ Test 7 PASSED\n")


# ============================================================
# Test 8: Multi-scale features path (pixel_decoder integration)
# ============================================================
def test_multi_scale_features():
    print("=" * 60)
    print("Test 8: Multi-scale features path (pixel_decoder integration)")
    print("=" * 60)

    # Simulate image_features from vision tower (Swin)
    # Typical output: dict of {res2, res3, res4, res5}
    # After pixel_decoder: mask_features + multi_scale_features
    image_features = {
        "res2": torch.randn(B, 128, H, W),
        "res3": torch.randn(B, 256, H // 2, W // 2),
        "res4": torch.randn(B, 512, H // 4, W // 4),
        "res5": torch.randn(B, 1024, H // 8, W // 8),
    }
    print(f"  image_features: {[(k, v.shape) for k, v in image_features.items()]}")

    # Simulate pixel_decoder output
    mask_features = torch.randn(B, HIDDEN_DIM, H // 4, W // 4)  # [B, 256, 50, 50]
    multi_scale_features = [
        torch.randn(H // 4 * W // 4, B, HIDDEN_DIM),  # level 0: 2500, 4, 256
        torch.randn(H // 8 * W // 8, B, HIDDEN_DIM),  # level 1: 625, 4, 256
        torch.randn(H // 16 * W // 16, B, HIDDEN_DIM),  # level 2: 169, 4, 256
    ]
    print(f"  mask_features: {mask_features.shape}")
    print(f"  multi_scale_features: {[f.shape for f in multi_scale_features]}")

    # predictor input: multi_scale_features, mask_features, SEG_embedding, SET_embedding
    SEG_embedding = torch.randn(sum_K, 1, HIDDEN_DIM)
    SET_embedding = torch.randn(B, 1, HIDDEN_DIM)

    # Verify all shapes are compatible
    assert mask_features.shape[0] == B
    assert mask_features.shape[1] == HIDDEN_DIM
    for i, msf in enumerate(multi_scale_features):
        assert msf.shape[1] == B, f"Multi-scale feature {i}: batch dim mismatch"
        assert msf.shape[2] == HIDDEN_DIM, f"Multi-scale feature {i}: channel dim mismatch"

    print("  ✅ Test 8 PASSED\n")


# ============================================================
# Test 9: Edge Cases
# ============================================================
def test_edge_cases():
    print("=" * 60)
    print("Test 9: Edge Cases")
    print("=" * 60)

    # Edge case 1: Single sample, single mask (Kmax=1)
    K_edge = [1]
    B_edge = 1
    Kmax_edge = 1

    SEG_embedding = torch.randn(sum(K_edge), 1, HIDDEN_DIM)  # [1, 1, 256]
    SET_embedding = torch.randn(B_edge, 1, HIDDEN_DIM)  # [1, 1, 256]

    SEG_emb = SEG_embedding.squeeze(1)  # [1, 256]
    SEG_padded = torch.zeros(B_edge, Kmax_edge, HIDDEN_DIM)
    SEG_padded[0, 0] = SEG_emb[0]

    tgt_key_padding_mask = torch.ones(B_edge, 1 + Kmax_edge, dtype=torch.bool)
    tgt_key_padding_mask[:, 0] = False
    tgt_key_padding_mask[:, 1] = False

    print(f"  Edge case 1 (single sample, single mask):")
    print(f"    SEG_padded: {SEG_padded.shape}")
    print(f"    tgt_key_padding_mask: {tgt_key_padding_mask.int()}")
    assert tgt_key_padding_mask.sum() == 0, "All positions should be valid"

    # Edge case 2: Variable K_i across batch, 0 masks for some sample
    K_edge2 = [0, 3, 1]  # sample 0 has no masks
    B_edge2 = 3
    Kmax_edge2 = 3

    tgt_key_padding_mask2 = torch.ones(B_edge2, 1 + Kmax_edge2, dtype=torch.bool)
    tgt_key_padding_mask2[:, 0] = False  # SET always valid
    for b_idx, k in enumerate(K_edge2):
        tgt_key_padding_mask2[b_idx, 1:1 + k] = False

    print(f"  Edge case 2 (sample 0 has 0 masks):")
    print(f"    tgt_key_padding_mask:\n{tgt_key_padding_mask2.int()}")
    # Sample 0: only SET valid, all SEG padded
    assert tgt_key_padding_mask2[0].sum() == 3, "Sample 0: all SEG should be masked"
    assert tgt_key_padding_mask2[1].sum() == 0, "Sample 1: all should be valid"
    assert tgt_key_padding_mask2[2].sum() == 2, "Sample 2: 2 SEG should be masked"

    # Edge case 3: Kmax=0 (no SEG tokens at all)
    # This shouldn't happen in practice but let's verify it doesn't crash
    K_edge3 = [0, 0]
    B_edge3 = 2
    Kmax_edge3 = 0

    if Kmax_edge3 == 0:
        print(f"  Edge case 3 (Kmax=0): tgt_key_padding_mask would be [B, 1] (SET only)")
        tgt_key_padding_mask3 = torch.ones(B_edge3, 1, dtype=torch.bool)
        tgt_key_padding_mask3[:, 0] = False
        print(f"    tgt_key_padding_mask: {tgt_key_padding_mask3.int()}")
        assert tgt_key_padding_mask3.shape == (B_edge3, 1)

    print("  ✅ Test 9 PASSED\n")


# ============================================================
# Test 10: SET++-ClosedLoop structured outputs & losses
# ============================================================
def test_closed_loop_structured_outputs():
    print("=" * 60)
    print("Test 10: SET++-ClosedLoop structured decoder outputs")
    print("=" * 60)

    B_cl = 2
    Kmax_cl = 3
    mask_num = [2, 3]
    H_m, W_m = 50, 50

    pred_masks = torch.randn(B_cl, 1 + Kmax_cl, H_m, W_m)
    pred_SEG_logits = torch.randn(B_cl, 1 + Kmax_cl, 1)

    valid_seg_mask = torch.zeros(B_cl, Kmax_cl, dtype=torch.bool)
    for b_idx, k in enumerate(mask_num):
        valid_seg_mask[b_idx, :k] = True

    outputs = {
        "pred_masks": pred_masks,
        "pred_SEG_logits": pred_SEG_logits,
        "pred_set_union_mask": pred_masks[:, 0:1],
        "pred_seg_masks": pred_masks[:, 1:],
        "pred_seg_logits": pred_SEG_logits[:, 1:],
        "valid_seg_mask": valid_seg_mask,
    }

    assert outputs["pred_set_union_mask"].shape == (B_cl, 1, H_m, W_m)
    assert outputs["pred_seg_masks"].shape == (B_cl, Kmax_cl, H_m, W_m)
    assert outputs["pred_seg_logits"].shape == (B_cl, Kmax_cl, 1)
    assert outputs["valid_seg_mask"].shape == (B_cl, Kmax_cl)

    expected_valid = torch.tensor([
        [True, True, False],
        [True, True, True],
    ])
    assert torch.equal(outputs["valid_seg_mask"], expected_valid), (
        f"valid_seg_mask mismatch:\n{outputs['valid_seg_mask']}\nexpected:\n{expected_valid}"
    )

    print(f"  pred_set_union_mask: {outputs['pred_set_union_mask'].shape}")
    print(f"  pred_seg_masks: {outputs['pred_seg_masks'].shape}")
    print(f"  valid_seg_mask:\n{outputs['valid_seg_mask'].int()}")
    print("  ✅ Test 10 PASSED\n")
    return outputs


def test_closed_loop_criterion_helpers(outputs):
    print("=" * 60)
    print("Test 11: SET++-ClosedLoop criterion helpers")
    print("=" * 60)

    def soft_union(seg_logits, valid_seg_mask):
        seg_prob = seg_logits.sigmoid()
        seg_prob = seg_prob * valid_seg_mask[:, :, None, None].float()
        return 1.0 - torch.prod(1.0 - seg_prob, dim=1, keepdim=True)

    def dice_prob_loss(pred_prob, tgt_prob, sample_weight=None, eps=1.0):
        pred = pred_prob.flatten(1)
        tgt = tgt_prob.flatten(1)
        numerator = 2 * (pred * tgt).sum(dim=1) + eps
        denominator = pred.sum(dim=1) + tgt.sum(dim=1) + eps
        loss = 1.0 - numerator / denominator
        if sample_weight is not None:
            return (loss * sample_weight).mean()
        return loss.mean()

    def build_union_target_tensor(targets, size):
        union_list = []
        for t in targets:
            masks = t["masks"].float()
            union = (masks.sum(dim=0, keepdim=True) > 0).float()
            if union.shape[-2:] != size:
                union = F.interpolate(
                    union.unsqueeze(0),
                    size=size,
                    mode="nearest",
                ).squeeze(0)
            union_list.append(union)
        return torch.stack(union_list, dim=0)

    def build_sample_weights(targets, single_weight, multi_weight):
        weights = []
        for t in targets:
            k = len(t["labels"])
            weights.append(single_weight if k <= 1 else multi_weight)
        return torch.tensor(weights, dtype=torch.float32)

    B_cl = outputs["pred_seg_masks"].shape[0]
    H_m, W_m = outputs["pred_seg_masks"].shape[-2:]
    mask_num = [2, 3]
    targets = []
    for b_idx, k in enumerate(mask_num):
        gt_masks = torch.zeros(k, H_m, W_m)
        for i in range(k):
            gt_masks[i, 10 + i * 5:20 + i * 5, 10 + i * 5:20 + i * 5] = 1.0
        targets.append({"labels": torch.zeros(k, dtype=torch.long), "masks": gt_masks})

    # soft_union: padded SEG logits should not affect union
    seg_logits = outputs["pred_seg_masks"].clone()
    seg_logits[:, 2, :, :] = 100.0  # padded slot for sample 0 only
    seg_union = soft_union(seg_logits, outputs["valid_seg_mask"])
    seg_union_no_pad = soft_union(
        outputs["pred_seg_masks"], outputs["valid_seg_mask"]
    )
    assert torch.allclose(seg_union[0], seg_union_no_pad[0], atol=1e-5), \
        "Padded SEG logits must not affect soft union for sample 0"

    # build_union_target_tensor == OR of GT masks
    gt_union = build_union_target_tensor(targets, size=(H_m, W_m))
    for b_idx, t in enumerate(targets):
        manual_union = (t["masks"].float().sum(dim=0, keepdim=True) > 0).float()
        assert torch.equal(gt_union[b_idx], manual_union), f"Union target mismatch at sample {b_idx}"

    # consistency direction: seg_align_set detaches SET
    pred_set = outputs["pred_set_union_mask"].clone().requires_grad_(True)
    pred_seg = outputs["pred_seg_masks"].clone().requires_grad_(True)
    valid = outputs["valid_seg_mask"]
    set_prob = pred_set.sigmoid()
    seg_union = soft_union(pred_seg, valid)
    weights = build_sample_weights(targets, 0.005, 0.02)
    cons_loss = dice_prob_loss(seg_union, set_prob.detach(), sample_weight=weights)
    cons_loss.backward()
    assert pred_set.grad is None or torch.all(pred_set.grad == 0), \
        "seg_align_set: SET union must not receive consistency gradient"
    assert pred_seg.grad is not None, "seg_align_set: SEG masks must receive consistency gradient"

    # matcher still uses only SEG queries (shape-level check)
    instance_masks = outputs["pred_seg_masks"]
    for b_idx, k in enumerate(mask_num):
        assert instance_masks[b_idx].shape[0] == outputs["valid_seg_mask"].shape[1]
        assert outputs["valid_seg_mask"][b_idx, :k].all()
        if k < outputs["valid_seg_mask"].shape[1]:
            assert not outputs["valid_seg_mask"][b_idx, k:].any()

    print("  soft_union ignores padded SEG: OK")
    print("  build_union_target_tensor OR: OK")
    print("  seg_align_set grad direction: OK")
    print("  matcher uses pred_seg_masks only (shape check): OK")
    print("  ✅ Test 11 PASSED\n")


# ============================================================
# Test 12: CSQR block shapes
# ============================================================
def test_csqr_block():
    print("=" * 60)
    print("Test 12: CSQR block (Set-guided Query Refinement)")
    print("=" * 60)

    # Lightweight inline CSQR forward (same tensor contract as SetGuidedQueryRefinementBlock)
    B_cs, K_cs, C_cs, HW = 2, 3, HIDDEN_DIM, 50 * 50
    q_set = torch.randn(B_cs, 1, C_cs)
    q_seg = torch.randn(B_cs, K_cs, C_cs)
    img_memory = torch.randn(HW, B_cs, C_cs)
    valid_seg_mask = torch.tensor([[True, True, False], [True, True, True]])
    seg_key_padding_mask = ~valid_seg_mask

    seg_attn = nn.MultiheadAttention(C_cs, 8, batch_first=False)
    cross_attn = nn.MultiheadAttention(C_cs, 8, batch_first=False)
    fusion_mlp = nn.Sequential(nn.Linear(C_cs, C_cs), nn.ReLU(), nn.Linear(C_cs, C_cs))
    fusion_alpha = 0.99

    seg_t = q_seg.permute(1, 0, 2)
    q1_t, _ = seg_attn(seg_t, seg_t, seg_t, key_padding_mask=seg_key_padding_mask)
    q_set_t = q_set.permute(1, 0, 2)
    q_set_t, _ = cross_attn(q_set_t, q1_t, q1_t, key_padding_mask=seg_key_padding_mask)
    q2_t, _ = cross_attn(q1_t, q_set_t, q_set_t)
    pad_mask_t = seg_key_padding_mask.transpose(0, 1).unsqueeze(-1)
    q2_t = q2_t.masked_fill(pad_mask_t, 0.0)
    q3_t, _ = cross_attn(q2_t, img_memory, img_memory)
    q3_t = q3_t.masked_fill(pad_mask_t, 0.0)
    q3 = q3_t.permute(1, 0, 2)
    q_seg_out = fusion_alpha * q_seg + (1.0 - fusion_alpha) * fusion_mlp(q3)
    q_seg_out = q_seg_out * (~seg_key_padding_mask).unsqueeze(-1).float()
    q_set_out = q_set_t.permute(1, 0, 2)

    assert q_set_out.shape == (B_cs, 1, C_cs)
    assert q_seg_out.shape == (B_cs, K_cs, C_cs)
    assert torch.all(q_seg_out[0, 2] == 0), "Padded SEG slot should stay zero after CSQR"

    print(f"  q_set_out: {q_set_out.shape}, q_seg_out: {q_seg_out.shape}")
    print(f"  fusion_alpha (reference): {fusion_alpha:.4f}")
    print("  ✅ Test 12 PASSED\n")


def test_loss_sample_weight_scale():
    print("=" * 60)
    print("Test 13: ClosedLoop loss sample_weight scale (mean, not normalized)")
    print("=" * 60)

    def dice_prob_loss(pred_prob, tgt_prob, sample_weight=None, eps=1.0):
        pred = pred_prob.flatten(1)
        tgt = tgt_prob.flatten(1)
        numerator = 2 * (pred * tgt).sum(dim=1) + eps
        denominator = pred.sum(dim=1) + tgt.sum(dim=1) + eps
        loss = 1.0 - numerator / denominator
        if sample_weight is not None:
            return (loss * sample_weight).mean()
        return loss.mean()

    B_t, H_t, W_t = 2, 32, 32
    pred_prob = torch.rand(B_t, 1, H_t, W_t).clamp(1e-3, 1 - 1e-3)
    tgt_prob = torch.rand(B_t, 1, H_t, W_t).clamp(1e-3, 1 - 1e-3)

    base_dice = dice_prob_loss(pred_prob, tgt_prob, sample_weight=None)
    uniform_w = torch.full((B_t,), 0.02)
    weighted_dice = dice_prob_loss(pred_prob, tgt_prob, sample_weight=uniform_w)
    assert torch.allclose(weighted_dice, base_dice * 0.02, rtol=1e-3, atol=1e-5), (
        f"dice weighted scale mismatch: {weighted_dice.item():.6f} vs {base_dice.item() * 0.02:.6f}"
    )
    assert weighted_dice < base_dice * 0.05, "uniform lambda=0.02 should strongly downscale dice"

    pred_logits = torch.randn(B_t, 1, H_t, W_t)
    tgt = torch.randint(0, 2, (B_t, 1, H_t, W_t)).float()
    per_sample_bce = F.binary_cross_entropy_with_logits(
        pred_logits, tgt, reduction="none"
    ).flatten(1).mean(dim=1)
    base_bce = per_sample_bce.mean()
    weighted_bce = (per_sample_bce * uniform_w).mean()
    assert torch.allclose(weighted_bce, base_bce * 0.02, rtol=1e-3, atol=1e-5), (
        f"bce weighted scale mismatch: {weighted_bce.item():.6f} vs {base_bce.item() * 0.02:.6f}"
    )

    hetero_w = torch.tensor([0.01, 0.05])
    hetero_dice = dice_prob_loss(pred_prob, tgt_prob, sample_weight=hetero_w)
    per_sample_dice = 1.0 - (
        2 * (pred_prob.flatten(1) * tgt_prob.flatten(1)).sum(dim=1) + 1.0
    ) / (pred_prob.flatten(1).sum(dim=1) + tgt_prob.flatten(1).sum(dim=1) + 1.0)
    expected_hetero = (per_sample_dice * hetero_w).mean()
    assert torch.allclose(hetero_dice, expected_hetero, rtol=1e-5, atol=1e-6), \
        "heterogeneous sample_weight must use batch mean, not sum/normalize"

    print(f"  base_dice={base_dice.item():.6f}, weighted_dice(0.02)={weighted_dice.item():.6f}")
    print(f"  base_bce={base_bce.item():.6f}, weighted_bce(0.02)={weighted_bce.item():.6f}")
    print("  heterogeneous weights use (loss * w).mean(): OK")
    print("  ✅ Test 13 PASSED\n")


# ============================================================
# Run All Tests
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SegEarth-R2 SET++ Shape Test Suite")
    print("=" * 60)
    print(f"  Config: B={B}, Kmax={Kmax}, K_list={K_list}, sum(K)={sum_K}")
    print(f"  Config: HIDDEN_DIM={HIDDEN_DIM}, LLM_HIDDEN={LLM_HIDDEN}")
    print(f"  Config: H={H}, W={W}, IMG_H={IMG_H}, IMG_W={IMG_W}")
    print()

    SEG_embedding, SET_embedding = test_dataset_to_collate()
    output, combined_emb, tgt_key_padding_mask, query_embed = test_decoder_query_layout(SEG_embedding, SET_embedding)
    test_self_attention_mask(output, tgt_key_padding_mask, query_embed)
    pred_masks_full, SEG_class_full = test_forward_prediction_heads(output, combined_emb)
    instance_masks, union_mask, indices = test_criterion_strip_and_match(pred_masks_full, SEG_class_full)
    test_loss_computation(pred_masks_full, SEG_class_full, union_mask, instance_masks, indices)
    test_eval_seg_path(pred_masks_full)
    test_multi_scale_features()
    test_edge_cases()
    closed_loop_outputs = test_closed_loop_structured_outputs()
    test_closed_loop_criterion_helpers(closed_loop_outputs)
    test_csqr_block()
    test_loss_sample_weight_scale()

    print("=" * 60)
    print("  🎉 ALL TESTS PASSED!")
    print("=" * 60)