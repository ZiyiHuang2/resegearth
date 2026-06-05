#!/usr/bin/env python3
"""Build LaSeRS category vocabulary JSON from train_data.json <p>...</p> tags."""

import argparse
import json
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from segearth_r2.model.set_conditioner import extract_category_phrases_from_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_json",
        default="/root/rivermind-data/huangziyi/data/LaSeRS/train/annotations/train_data.json",
    )
    parser.add_argument(
        "--output",
        default="segearth_r2/model/lasers_category_vocab.json",
    )
    args = parser.parse_args()

    with open(args.train_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = set()
    for item in data:
        for phrase in extract_category_phrases_from_answer(item.get("answer", "")):
            categories.add(phrase)

    out_path = os.path.join(REPO_DIR, args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {"categories": sorted(categories)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(categories)} categories to {out_path}")


if __name__ == "__main__":
    main()
