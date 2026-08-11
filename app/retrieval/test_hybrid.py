from app.retrieval.hybrid import (hybrid_search_candidate,)
from app.retrieval.vector_store import get_vector_store

def print_docs(title, docs):
    print("\n" + title)
    print("-" * 80)

    for i, doc in enumerate(docs, start=1,):
        print(
            f"{i}. "
            f"{doc.metadata.get('chunk_id')}"
        )

        print(
            f"   Section: "
            f"{doc.metadata.get('section')}"
        )


def main():
    vector_store = get_vector_store()
    queries = ["Docker experience", "FastAPI backend API experience", "Transformer model experience",]

    for query in queries:

        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        results = hybrid_search_candidate(query=query, k=3, vector_store=vector_store)

        print_docs("VECTOR", results["vector"])

        print_docs("BM25", results["bm25"])

        print_docs("HYBRID", results["hybrid"])


if __name__ == "__main__":
    main()