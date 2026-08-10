from app.retrieval.vector_store import get_vector_store


def main():
    vector_store = get_vector_store()

    query = (
        "What evidence shows that the "
        "candidate has backend API experience?"
    )

    results = vector_store.similarity_search_with_score(
        query,
        k=4,
        filter={"doc_type": "candidate"}
    )

    print(f"\nQUERY:\n{query}\n")

    for i, (doc, score) in enumerate(results, start=1):
        print("=" * 80)
        print(f"RESULT {i}")
        print(f"Raw score: {score}")

        print("\nMetadata:")
        print(doc.metadata)

        print("\nContent:")
        print(doc.page_content)


if __name__ == "__main__":
    main()