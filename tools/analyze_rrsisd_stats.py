import pickle
import argparse
import statistics
from collections import Counter

import transformers


IMPLICIT_HINTS = [
    "where", "which", "what", "find", "locate", "identify",
    "used for", "designed for", "provides", "serves as",
    "awaiting", "celebrated for", "in case of", "should i do",
]


def percentile(values, q):
    if not values:
        return 0
    values = sorted(values)
    idx = int((len(values) - 1) * q)
    return values[idx]


def is_implicit(text: str) -> bool:
    t = text.lower().strip()
    return any(k in t for k in IMPLICIT_HINTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs_path", required=True, type=str)
    parser.add_argument("--tokenizer_path", required=True, type=str)
    parser.add_argument("--target_split", default="train", type=str)
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=False,
    )

    with open(args.refs_path, "rb") as f:
        refs = pickle.load(f)

    refs = [x for x in refs if x.get("split") == args.target_split]

    num_refs = len(refs)
    num_images = len(set(x["image_id"] for x in refs))
    num_anns = len(set(x["ann_id"] for x in refs))

    sent_num_list = []
    token_lens = []
    char_lens = []
    implicit_flags = []
    cat_counter = Counter()

    for ref in refs:
        sents = ref.get("sentences", [])
        sent_num_list.append(len(sents))
        cat_counter[ref.get("category_id")] += 1

        for s in sents:
            text = s.get("sent", "") or s.get("raw", "")
            text = text.strip()
            token_lens.append(len(tokenizer.encode(text, add_special_tokens=False)))
            char_lens.append(len(text))
            implicit_flags.append(1 if is_implicit(text) else 0)

    print(f"===== RRSISD refs stats ({args.target_split}) =====")
    print(f"num_refs: {num_refs}")
    print(f"num_images: {num_images}")
    print(f"num_anns: {num_anns}")
    print(f"avg_sentences_per_ref: {statistics.mean(sent_num_list):.4f}")
    print()

    print("----- sentence token length -----")
    print(f"avg: {statistics.mean(token_lens):.2f}")
    print(f"p50: {percentile(token_lens, 0.50)}")
    print(f"p90: {percentile(token_lens, 0.90)}")
    print(f"p95: {percentile(token_lens, 0.95)}")
    print(f"max: {max(token_lens) if token_lens else 0}")
    print()

    print("----- sentence char length -----")
    print(f"avg: {statistics.mean(char_lens):.2f}")
    print(f"p50: {percentile(char_lens, 0.50)}")
    print(f"p90: {percentile(char_lens, 0.90)}")
    print(f"p95: {percentile(char_lens, 0.95)}")
    print(f"max: {max(char_lens) if char_lens else 0}")
    print()

    print("----- query style -----")
    print(f"implicit_ratio(rule_based): {sum(implicit_flags) / len(implicit_flags):.4f}")
    print(f"top10_category_counts: {cat_counter.most_common(10)}")


if __name__ == "__main__":
    main()