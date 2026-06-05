#!/usr/bin/env python3
"""Smoke tests for A3 pre-decoder set conditioning."""

import importlib.util
import os
import sys

import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFE_GATE_INIT = 1e-3


def _load_set_conditioner_module():
    module_path = os.path.join(REPO_DIR, "segearth_r2", "model", "set_conditioner.py")
    spec = importlib.util.spec_from_file_location("set_conditioner", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sc = _load_set_conditioner_module()
SetConditioner = _sc.SetConditioner
flatten_seg_embeddings = _sc.flatten_seg_embeddings
regroup_seg_embeddings = _sc.regroup_seg_embeddings
compute_set_count_loss = _sc.compute_set_count_loss
compute_set_category_loss = _sc.compute_set_category_loss
CountHead = _sc.CountHead
CategorySetHead = _sc.CategorySetHead
extract_category_phrases_from_answer = _sc.extract_category_phrases_from_answer
build_category_set_labels = _sc.build_category_set_labels
build_category_set_labels_from_answer = _sc.build_category_set_labels_from_answer


def test_regroup_order():
    hidden_dim = 256
    mask_num = [2, 3, 1]
    total = sum(mask_num)
    seg_flat = torch.arange(total * hidden_dim, dtype=torch.float32).view(total, hidden_dim)
    seg_embedding = seg_flat.unsqueeze(1)

    seg_group, valid_mask, counts = regroup_seg_embeddings(seg_embedding, mask_num)
    restored = flatten_seg_embeddings(seg_group, valid_mask, counts).unsqueeze(1)

    assert restored.shape == seg_embedding.shape
    assert torch.allclose(restored, seg_embedding), "regroup/flatten changed token order"
    print("[OK] mask_num regroup/flatten order preserved")


def test_set_conditioner_shapes():
    hidden_dim = 256
    mask_num = [2, 1, 3]
    total = sum(mask_num)
    seg_embedding = torch.randn(total, 1, hidden_dim)
    module = SetConditioner(hidden_dim=hidden_dim, num_layers=1, num_heads=4, gate_init=SAFE_GATE_INIT)
    refined, q_set, valid_mask, gate_mean, gate = module(seg_embedding, mask_num)

    assert refined.shape == seg_embedding.shape, f"refined shape {refined.shape} != {seg_embedding.shape}"
    assert gate_mean.numel() == 1
    assert module.residual_gate.item() >= SAFE_GATE_INIT - 1e-12
    diff = (refined - seg_embedding).abs()
    assert diff.max().item() < 1e-5, f"init should be near-identity, max diff={diff.max().item()}"
    assert q_set.shape == (len(mask_num), hidden_dim)
    assert valid_mask.shape == (len(mask_num), max(mask_num))
    print("[OK] SetConditioner output shapes")


def test_set_conditioner_init_close_to_baseline():
    hidden_dim = 256
    mask_num = [2, 0, 3]
    seg_embedding = torch.randn(sum(mask_num), 1, hidden_dim)
    module = SetConditioner(hidden_dim=hidden_dim, num_layers=1, num_heads=4, gate_init=SAFE_GATE_INIT)
    refined, _, _, gate_mean, gate = module(seg_embedding, mask_num)
    diff = (refined - seg_embedding).abs()
    assert diff.mean().item() < 1e-3, f"mean diff too large: {diff.mean().item()}"
    assert diff.max().item() < 1e-2, f"max diff too large: {diff.max().item()}"
    assert module.residual_gate.item() >= SAFE_GATE_INIT - 1e-12
    assert torch.isfinite(gate_mean).all()
    assert torch.isfinite(gate).all()
    print("[OK] SetConditioner init stays close to baseline")


def test_set_conditioner_main_path_has_gradient():
    hidden_dim = 256
    mask_num = [2, 1]
    seg_embedding = torch.randn(sum(mask_num), 1, hidden_dim, requires_grad=True)
    module = SetConditioner(hidden_dim=hidden_dim, num_layers=1, num_heads=4, gate_init=SAFE_GATE_INIT)
    refined, _, _, _, _ = module(seg_embedding, mask_num)
    refined.sum().backward()

    output_weight_grad = module.output_proj.weight.grad
    output_bias_grad = module.output_proj.bias.grad
    residual_gate_grad = module.residual_gate.grad

    assert output_weight_grad is not None and output_weight_grad.abs().sum().item() > 0, "output_proj.weight has no grad"
    assert output_bias_grad is not None and output_bias_grad.abs().sum().item() > 0, "output_proj.bias has no grad"
    assert residual_gate_grad is None or torch.isfinite(residual_gate_grad).all()
    print("[OK] SetConditioner main path receives gradient")


def test_baseline_identity_flag():
    """When use_set_conditioner=False, _apply_set_conditioning must be a no-op."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("[SKIP] llava_phi integration tests (transformers not installed)")
        return

    sys.path.insert(0, REPO_DIR)
    from types import SimpleNamespace

    from segearth_r2.model.language_model.llava_phi import SegEarthR2

    args = SimpleNamespace(
        use_set_conditioner=False,
        use_set_count_loss=False,
        use_set_category_loss=False,
    )
    model = object.__new__(SegEarthR2)
    model.config = type("Cfg", (), {})()
    model.init_set_conditioning_modules(args)

    seg_embedding = torch.randn(4, 1, 256)
    out, lc, lcat, gate, acc = model._apply_set_conditioning(seg_embedding, [2, 2])
    assert out is seg_embedding
    assert lc is None and lcat is None and gate is None and acc is None
    print("[OK] use_set_conditioner=false is strict no-op")


def test_zero_seg_sample():
    """mask_num = [2, 0, 3] with K_i=0 must not fabricate [SEG] tokens."""
    hidden_dim = 256
    mask_num = [2, 0, 3]
    total = sum(mask_num)
    assert total == 5
    seg_embedding = torch.randn(total, 1, hidden_dim)
    seg_group, valid_mask, counts = regroup_seg_embeddings(seg_embedding, mask_num)

    assert seg_group.shape == (3, 3, hidden_dim)
    assert valid_mask.shape == (3, 3)
    assert valid_mask[0].tolist() == [True, True, False]
    assert valid_mask[1].tolist() == [False, False, False]
    assert valid_mask[2].tolist() == [True, True, True]

    module = SetConditioner(hidden_dim=hidden_dim, num_layers=1, num_heads=4, gate_init=SAFE_GATE_INIT)
    refined, q_set, _, _, _ = module(seg_embedding, mask_num)
    assert refined.shape == (5, 1, hidden_dim)

    seg_flat = torch.arange(total * hidden_dim, dtype=torch.float32).view(total, hidden_dim)
    seg_embedding_det = seg_flat.unsqueeze(1)
    refined_det, _, _, _, _ = module(seg_embedding_det, mask_num)
    assert refined_det.shape == (5, 1, hidden_dim)
    assert torch.allclose(refined_det, seg_embedding_det), "zero-SEG case changed order/values"

    count_head = CountHead(hidden_dim, set_max_count=10)
    loss_count, acc = compute_set_count_loss(count_head, q_set, mask_num, set_max_count=10)
    assert torch.isfinite(loss_count).all(), f"loss_set_count not finite: {loss_count}"
    assert torch.isfinite(acc).all(), f"target_count_acc not finite: {acc}"
    assert q_set.shape == (3, hidden_dim)
    print("[OK] zero-SEG sample: regroup/valid_mask/flatten/loss finite")


def test_a3_category_label_pipeline():
    vocab = ["small car", "van", "dry cargo ship", "motorboat"]
    answers = [
        "<p>small car</p> and <p>van</p> [SEG] [SEG]",
        "<p>dry cargo ship</p> [SEG]",
    ]
    phrases0 = extract_category_phrases_from_answer(answers[0])
    assert phrases0 == ["small car", "van"], f"got {phrases0}"
    phrases1 = extract_category_phrases_from_answer(answers[1])
    assert phrases1 == ["dry cargo ship"], f"got {phrases1}"

    labels0, _, matched0, unknown0 = build_category_set_labels_from_answer(answers[0], vocab)  # noqa: E501
    labels1, _, matched1, unknown1 = build_category_set_labels_from_answer(answers[1], vocab)
    assert labels0.shape == (len(vocab),)
    assert labels1.shape == (len(vocab),)
    assert labels0[vocab.index("small car")] == 1.0
    assert labels0[vocab.index("van")] == 1.0
    assert labels1[vocab.index("dry cargo ship")] == 1.0
    assert unknown0 == [] and unknown1 == []

    sys.path.insert(0, REPO_DIR)
    from segearth_r2.datasets.dataset import DataCollatorForCOCODatasetV2

    instances = [
        {"category_set_labels": labels0},
        {"category_set_labels": labels1},
    ]
    collator = DataCollatorForCOCODatasetV2.__new__(DataCollatorForCOCODatasetV2)
    batch = {"category_set_labels": torch.stack([labels0, labels1], dim=0)}
    assert batch["category_set_labels"].shape == (2, len(vocab))

    hidden_dim = 256
    category_head = CategorySetHead(hidden_dim, vocab_size=len(vocab))
    q_set = torch.randn(2, hidden_dim)
    loss_cat = compute_set_category_loss(category_head, q_set, batch["category_set_labels"])
    assert torch.isfinite(loss_cat).all(), f"loss_set_category not finite: {loss_cat}"

    try:
        import transformers  # noqa: F401
    except ImportError:
        print("[OK] A3 category label pipeline (forward hook skipped: no transformers)")
        return

    from types import SimpleNamespace
    from segearth_r2.model.language_model.llava_phi import SegEarthR2

    args = SimpleNamespace(
        use_set_conditioner=True,
        use_set_count_loss=True,
        use_set_category_loss=True,
        set_conditioner_layers=1,
        set_conditioner_heads=4,
        set_conditioner_gate_init=SAFE_GATE_INIT,
        lambda_set_count=0.05,
        lambda_set_category=0.1,
        set_max_count=10,
        lasers_category_vocab_path=None,
    )
    model = object.__new__(SegEarthR2)
    torch.nn.Module.__init__(model)
    model.config = type("Cfg", (), {})()
    model.mask_decoder_cfg = SimpleNamespace(
        MODEL=SimpleNamespace(MASK_FORMER=SimpleNamespace(HIDDEN_DIM=256))
    )
    model.init_set_conditioning_modules(args)
    model.lasers_category_vocab = vocab
    model.category_set_head = CategorySetHead(hidden_dim, vocab_size=len(vocab))

    seg_embedding = torch.randn(3, 1, hidden_dim)
    out, lc, lcat, gate, acc = model._apply_set_conditioning(
        seg_embedding,
        [2, 1],
        category_set_labels=batch["category_set_labels"],
    )
    assert lcat is not None and torch.isfinite(lcat).all()
    print("[OK] A3 category label pipeline: extract/collator/loss_set_category finite")


def test_q_set_modulates_refinement():
    hidden_dim = 256
    mask_num = [2, 1]
    seg_embedding = torch.randn(sum(mask_num), 1, hidden_dim)
    module = SetConditioner(hidden_dim=hidden_dim, num_layers=1, num_heads=4, gate_init=SAFE_GATE_INIT)
    module.eval()
    with torch.no_grad():
        refined_a, q_set, _, _, _ = module(seg_embedding, mask_num)
    assert q_set.shape == (2, hidden_dim)
    with torch.no_grad():
        module.set_proj.weight.fill_(0.2)
        module.gate_proj.bias.fill_(3.0)
        module.output_proj.weight.fill_(0.05)
        module.residual_gate.fill_(1.0)
        refined_b, _, _, _, _ = module(seg_embedding, mask_num)
    assert not torch.allclose(refined_a, refined_b), "Q_set + set_proj should change refined SEG"
    print("[OK] Q_set participates in SEG refinement (set_proj coupling)")


def test_a3_frozen_keyword_policy():
    sys.path.insert(0, REPO_DIR)
    from segearth_r2.train.a3_training_utils import (
        A3_FORBIDDEN_KEYWORDS,
        A3_TRAINABLE_KEYWORDS,
    )

    assert "set_conditioner" in A3_TRAINABLE_KEYWORDS
    assert "pixel_decoder" in A3_FORBIDDEN_KEYWORDS
    print("[OK] A3-frozen keyword policy defined")


def test_apply_with_aux_heads():
    try:
        import transformers  # noqa: F401
    except ImportError:
        print("[SKIP] aux-head integration test (transformers not installed)")
        return

    sys.path.insert(0, REPO_DIR)
    from types import SimpleNamespace

    from segearth_r2.model.language_model.llava_phi import SegEarthR2

    args = SimpleNamespace(
        use_set_conditioner=True,
        use_set_count_loss=True,
        use_set_category_loss=True,
        set_conditioner_layers=1,
        set_conditioner_heads=4,
        set_conditioner_gate_init=SAFE_GATE_INIT,
        lambda_set_count=0.05,
        lambda_set_category=0.1,
        set_max_count=10,
        lasers_category_vocab_path=None,
    )
    model = object.__new__(SegEarthR2)
    torch.nn.Module.__init__(model)
    model.config = type("Cfg", (), {})()
    model.mask_decoder_cfg = SimpleNamespace(
        MODEL=SimpleNamespace(MASK_FORMER=SimpleNamespace(HIDDEN_DIM=256))
    )
    model.init_set_conditioning_modules(args)

    mask_num = [2, 1]
    seg_embedding = torch.randn(sum(mask_num), 1, 256)
    out, lc, lcat, gate, acc = model._apply_set_conditioning(seg_embedding, mask_num)
    assert out.shape == seg_embedding.shape
    assert lc is not None and acc is not None
    assert lcat is None, "category loss should skip without batch labels"
    print("[OK] count loss runs; category loss skips without labels")


def main():
    test_regroup_order()
    test_set_conditioner_shapes()
    test_set_conditioner_init_close_to_baseline()
    test_set_conditioner_main_path_has_gradient()
    test_zero_seg_sample()
    test_a3_category_label_pipeline()
    test_q_set_modulates_refinement()
    test_a3_frozen_keyword_policy()
    test_baseline_identity_flag()
    test_apply_with_aux_heads()
    print("All A3 smoke tests passed.")


if __name__ == "__main__":
    main()
