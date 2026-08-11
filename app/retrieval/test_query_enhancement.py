from app.retrieval.vector_store import get_vector_store

from app.retrieval.queries import (CANDIDATE_EXPERIENCE_QUERY,)


def print_results(title, docs):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for i, doc in enumerate(docs, start=1):
        print(
            f"\n{i}. "
            f"{doc.metadata.get('chunk_id')}"
        )

        print(
            f"Section: "
            f"{doc.metadata.get('section')}"
        )

        print(doc.page_content)


def main():
    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4,
            "filter": {
                "doc_type": "candidate"
            },
        },
    )

    old_query = ("What professional or practical experience does the candidate have?")

    old_docs = retriever.invoke(old_query)

    enhanced_docs = retriever.invoke(CANDIDATE_EXPERIENCE_QUERY)

    print_results("OLD QUERY",old_docs)

    print_results("ENHANCED QUERY",enhanced_docs)


if __name__ == "__main__":
    main()