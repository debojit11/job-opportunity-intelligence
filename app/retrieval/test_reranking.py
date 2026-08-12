from app.retrieval.hybrid import (hybrid_search_candidate,)

from app.retrieval.reranker import (get_reranker, rerank_documents,)

from app.retrieval.vector_store import (get_vector_store,)


def print_docs(title, docs):
    print("\n" + title)
    print("-" * 80)

    for i, doc in enumerate(docs, start=1,):
        print(
            f"{i}. "
            f"{doc.metadata.get('chunk_id')}"
        )


def print_reranked(title, results):
    print("\n" + title)
    print("-" * 80)

    for i, (doc, score) in enumerate(results, start=1,):
        print(
            f"{i}. "
            f"{doc.metadata.get('chunk_id')} "
            f"| score={score:.4f}"
        )


def main():
    vector_store = get_vector_store()

    reranker = get_reranker()

    queries = [
        ("What evidence demonstrates practical backend API experience?"),
        ("What evidence demonstrates Transformer model experience?"),
        ("What evidence shows Docker proficiency?"),
    ]

    for query in queries:

        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        results = hybrid_search_candidate(query=query, k=6, vector_store=vector_store,)

        hybrid_docs = results["hybrid"]

        reranked = rerank_documents(query=query, documents=hybrid_docs, reranker=reranker, top_k=3,)

        print_docs("HYBRID BEFORE RERANK", hybrid_docs)

        print_reranked("AFTER CROSS-ENCODER RERANK", reranked)


if __name__ == "__main__":
    main()