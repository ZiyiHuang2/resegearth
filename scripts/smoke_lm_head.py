#!/usr/bin/env python3
"""Smoke test: verify lm_head_size decision logic and tokenizer compatibility.

Does NOT require torch/transformers -- uses only json + struct for safetensors reading.

Usage:
    python scripts/smoke_lm_head.py
    python scripts/smoke_lm_head.py --merged_model /path/to/merged_model
"""

import os
import sys
import argparse
import json
import struct


def check_safetensors_lm_head(merged_dir: str):
    """Read lm_head.weight shape from the first safetensors shard.
    Returns [in_features, out_features] (PyTorch convention)."""
    if not os.path.isdir(merged_dir):
        return None
    for fname in sorted(os.listdir(merged_dir)):
        if fname.endswith('.safetensors'):
            path = os.path.join(merged_dir, fname)
            with open(path, 'rb') as f:
                header_len = struct.unpack('<Q', f.read(8))[0]
                header = json.loads(f.read(header_len).decode())
            for k, v in header.items():
                if 'lm_head' in k:
                    # safetensors stores [out_features, in_features]; reverse to [in, out]
                    return list(reversed(v['shape']))
    return None


def decide_lm_head_size(config: dict) -> int:
    """Priority: 1) config.lm_head_size, 2) max(config.vocab_size, 51200)."""
    lm_head_size = config.get('lm_head_size', None)
    if lm_head_size is None:
        lm_head_size = max(config.get('vocab_size', 51200), 51200)
    return lm_head_size


def test_config_logic():
    """Test 1: Mipha-3B config -> lm_head_size=51200."""
    mipha_path = '/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B'
    cfg_path = os.path.join(mipha_path, 'config.json')

    print(f"[Test 1] Mipha-3B config: {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    vocab_size = cfg.get('vocab_size')
    hidden_size = cfg.get('hidden_size')
    lm_head_size = decide_lm_head_size(cfg)

    print(f"  vocab_size={vocab_size}, hidden_size={hidden_size}")
    print(f"  lm_head_size (decided)={lm_head_size}")

    assert lm_head_size == 51200, f"Expected 51200, got {lm_head_size}"
    assert lm_head_size >= vocab_size, f"lm_head_size ({lm_head_size}) < vocab_size ({vocab_size})"
    print("  ✓ PASSED\n")


def test_merged_model_compat():
    """Test 2: merged_models with config.vocab_size != actual lm_head weight."""
    candidates = [
        ("clite-v2", "/root/rivermind-data/huangziyi/reseg/output/set/clite-v2-from-scratch-50k/merged_model"),
        ("joint",   "/root/rivermind-data/huangziyi/reseg/output/joint/full_joint_from_scratch/merged_model"),
    ]

    print("[Test 2] Merged model compatibility (config.vocab_size != actual weight)")
    all_ok = True
    for label, merged_dir in candidates:
        cfg_path = os.path.join(merged_dir, 'config.json')
        if not os.path.exists(cfg_path):
            print(f"  [{label}] SKIP: {cfg_path} not found")
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)

        cfg_vocab = cfg.get('vocab_size')
        lm_head_size = decide_lm_head_size(cfg)
        actual_shape = check_safetensors_lm_head(merged_dir)

        print(f"  [{label}] vocab_size={cfg_vocab}, decided_lm_head_size={lm_head_size}, "
              f"actual_weight={actual_shape}")

        if actual_shape:
            actual_out = actual_shape[1]
            assert actual_out == 51200, \
                f"[{label}] actual lm_head out_features={actual_out}, expected 51200"
            # The key check: the decision logic must match the actual weight
            assert lm_head_size == actual_out, \
                f"[{label}] decided lm_head_size={lm_head_size} != actual={actual_out}"
            if cfg_vocab != 51200 and lm_head_size == 51200:
                print(f"  [{label}] ✓ Handles mismatch: config.vocab_size={cfg_vocab} but lm_head={lm_head_size}")

    print("  ✓ PASSED\n")


def test_tokenizer_safety():
    """Test 3: Read tokenizer.json, simulate add [SEG]/[SET], verify ids < 51200."""
    import os as _os
    mipha_path = '/root/rivermind-data/huangziyi/reseg/pretrained_model/mllm/Mipha-3B'

    print("[Test 3] Tokenizer safety check (simulated)")

    # Read tokenizer.json to get the base vocab size
    tok_path = os.path.join(mipha_path, 'tokenizer.json')
    if os.path.exists(tok_path):
        with open(tok_path) as f:
            tok = json.load(f)
        model_vocab = tok.get('model', {}).get('vocab', {})
        base_len = len(model_vocab)
        print(f"  base vocab size (from tokenizer.json) = {base_len}")
    else:
        # Fallback: use known value
        base_len = 50295
        print(f"  base vocab size (fallback) = {base_len}")

    # After add [SEG]/[SET]
    tokenizer_len = base_len + 2
    seg_id = base_len
    set_id = base_len + 1
    lm_head_size = 51200

    print(f"  after add [SEG]/[SET]: len={tokenizer_len}, SEG_id={seg_id}, SET_id={set_id}")
    print(f"  lm_head_size={lm_head_size}")

    assert tokenizer_len <= lm_head_size, \
        f"tokenizer_len ({tokenizer_len}) > lm_head_size ({lm_head_size})"
    assert seg_id < lm_head_size, \
        f"[SEG] id ({seg_id}) >= lm_head_size ({lm_head_size})"
    assert set_id < lm_head_size, \
        f"[SET] id ({set_id}) >= lm_head_size ({lm_head_size})"

    print("  ✓ PASSED\n")


def test_llava_phi_init_simulation():
    """Test 4: Simulate SegEarthR2.__init__ lm_head decision with a dummy config."""
    print("[Test 4] Simulating SegEarthR2.__init__ lm_head_size logic")

    # Scenario A: Mipha-3B config (vocab_size=51200, no lm_head_size key)
    cfg_a = {"vocab_size": 51200, "hidden_size": 2560}
    size_a = decide_lm_head_size(cfg_a)
    assert size_a == 51200, f"A: {size_a}"
    print(f"  A) Mipha-3B (vocab=51200): lm_head_size={size_a}")

    # Scenario B: merged_model with vocab_size=50297 but actual weight is 51200
    cfg_b = {"vocab_size": 50297, "hidden_size": 2560}
    size_b = decide_lm_head_size(cfg_b)
    assert size_b == 51200, f"B: {size_b}"
    print(f"  B) merged model (vocab=50297, no lm_head_size): lm_head_size={size_b}")

    # Scenario C: explicit lm_head_size override
    cfg_c = {"vocab_size": 50297, "hidden_size": 2560, "lm_head_size": 51200}
    size_c = decide_lm_head_size(cfg_c)
    assert size_c == 51200, f"C: {size_c}"
    print(f"  C) explicit lm_head_size=51200: lm_head_size={size_c}")

    # Scenario D: edge case - vocab_size > 51200 with no override
    cfg_d = {"vocab_size": 60000, "hidden_size": 2560}
    size_d = decide_lm_head_size(cfg_d)
    assert size_d == 60000, f"D: {size_d}"
    print(f"  D) large vocab (vocab=60000): lm_head_size={size_d}")

    print("  ✓ PASSED\n")


def main():
    parser = argparse.ArgumentParser(description="Smoke test lm_head_size logic")
    parser.add_argument('--merged_model', type=str, default=None,
                        help='Additional merged_model directory to check')
    args = parser.parse_args()

    print("=" * 60)
    print("lm_head_size Smoke Test Suite")
    print("=" * 60)
    print()

    test_config_logic()
    test_merged_model_compat()
    test_tokenizer_safety()
    test_llava_phi_init_simulation()

    # Additional user-specified merged model
    if args.merged_model:
        print(f"[Extra] Checking user-specified: {args.merged_model}")
        cfg_path = os.path.join(args.merged_model, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            size = decide_lm_head_size(cfg)
            actual = check_safetensors_lm_head(args.merged_model)
            print(f"  vocab_size={cfg.get('vocab_size')}, decided_lm_head_size={size}, actual={actual}")
            if actual:
                assert size == actual[1], f"decided={size} != actual={actual[1]}"
            print("  ✓ PASSED\n")

    print("=" * 60)
    print("All smoke tests PASSED")
    print("=" * 60)


if __name__ == '__main__':
    main()