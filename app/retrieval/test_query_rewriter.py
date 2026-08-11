from app.retrieval.query_rewriter import rewrite_query


def main():
    queries = [
        "Does this candidate actually have enough backend stuff?",
        "What about deployment?",
        "Does the company look sketchy?",
        "Is their experience enough?",
    ]

    for query in queries:
        rewritten = rewrite_query(query)

        print("\n" + "=" * 80)
        print("ORIGINAL")
        print(query)

        print("\nREWRITTEN")
        print(rewritten)


if __name__ == "__main__":
    main()