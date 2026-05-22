from segearth_r2.knowledge.semantic_kb import SemanticRSKB, SemanticKBConfig


def main():
    kb = SemanticRSKB(
        SemanticKBConfig(
            inject_mode="hard",
            hard_query_max_tokens=7,
            max_prefix_chars=72,
            include_fields="cat,rel,ctx,shape,scale",
        )
    )

    samples = [
        "the bridge near the river",
        "the airplane on the right",
        "ship near coast",
        "the baseball field at center",
        "the object",
    ]

    for query in samples:
        parsed = kb.parse(query)
        print(f"query:   {query}")
        print(f"parsed:  {parsed}")
        print(f"inject:  {kb.should_inject(parsed)}")
        print(f"aug:     {kb.augment(query)}")
        print("-" * 60)


if __name__ == "__main__":
    main()