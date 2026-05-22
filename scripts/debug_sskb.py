from segearth_r2.knowledge.structured_kb import StructuredRSKB, SSKBConfig


def main():
    kb = StructuredRSKB(
        SSKBConfig(
            min_query_tokens=3,
            max_prefix_chars=64,
            inject_mode="auto",
        )
    )

    samples = [
        "the bridge near the river",
        "the airplane on the right",
        "ship",
        "the small building at top left",
        "segment the runway",
    ]

    for query in samples:
        print(f"query:   {query}")
        print(f"aug:     {kb.augment(query)}")
        print(f"slots:   {kb.parse_slots(query)}")
        print("-" * 40)


if __name__ == "__main__":
    main()