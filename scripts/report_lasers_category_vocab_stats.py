#!/usr/bin/env python3
"""Report LaSeRS category vocab coverage on train JSON."""

import argparse
import json
import os
import sys
from collections import Counter

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from segearth_r2.model.set_conditioner import (
    build_category_set_labels,
    extract_category_phrases_from_answer,
    load_lasers_category_vocab,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_json",
        default="/root/rivermind-data/huangziyi/data/LaSeRS/train/annotations/train_data.json",
    )
    parser.add_argument(
        "--vocab_path",
        default="segearth_r2/model/lasers_category_vocab.json",
    )
    args = parser.parse_args()

    vocab_path = args.vocab_path
    if not os.path.isabs(vocab_path):
        vocab_path = os.path.join(REPO_DIR, vocab_path)

    with open(args.train_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = load_lasers_category_vocab(vocab_path)
    vocab_set = set(vocab)

    all_phrases = Counter()
    known_phrases = Counter()
    unknown_phrases = Counter()
    empty_samples = 0
    total_samples = len(data)

    for item in data:
        answer = item.get("answer", "")
        phrases = extract_category_phrases_from_answer(answer)
        labels, matched, unknown = build_category_set_labels(phrases, vocab)
        for p in phrases:
            all_phrases[p] += 1
        for p in matched:
            known_phrases[p] += 1
        for p in unknown:
            unknown_phrases[p] += 1
        if labels.sum().item() == 0:
            empty_samples += 1

    total_extracted = sum(all_phrases.values())
    total_known = sum(known_phrases.values())
    total_unknown = sum(unknown_phrases.values())
    unknown_ratio = total_unknown / total_extracted if total_extracted else 0.0
    empty_ratio = empty_samples / total_samples if total_samples else 0.0

    print("=== LaSeRS Category Vocab Stats ===")
    print(f"vocab_path: {vocab_path}")
    print(f"vocab_size: {len(vocab)}")
    print(f"total_samples: {total_samples}")
    print(f"total_extracted_phrases: {total_extracted}")
    print(f"known_phrases: {total_known}")
    print(f"unknown_phrases: {total_unknown}")
    print(f"unknown_ratio: {unknown_ratio:.6f}")
    print(f"empty_label_samples: {empty_samples}")
    print(f"empty_label_sample_ratio: {empty_ratio:.6f}")
    print("top_unknown_phrases:")
    for phrase, cnt in unknown_phrases.most_common(20):
        print(f"  {phrase!r}: {cnt}")


if __name__ == "__main__":
    main()
