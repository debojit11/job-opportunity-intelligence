from app.retrieval.query_rewriter import rewrite_query
from app.retrieval.vector_store import get_vector_store


def print_docs(title, docs):
    print("\n" + title)
    print("-" * 80)

    for i, doc in enumerate(docs, start=1):
        print(
            f"{i}. {doc.metadata.get('chunk_id')}"
        )
        print(doc.page_content)
        print()


def main():
    user_query = (
        "Does this candidate actually "
        "have enough backend stuff?"
    )

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

    raw_docs = retriever.invoke(user_query)

    rewritten_query = rewrite_query(user_query)

    rewritten_docs = retriever.invoke(rewritten_query)

    print("ORIGINAL QUERY:")
    print(user_query)

    print("\nREWRITTEN QUERY:")
    print(rewritten_query)

    print_docs("RAW RETRIEVAL",raw_docs)

    print_docs("REWRITTEN RETRIEVAL",rewritten_docs)


if __name__ == "__main__":
    main()