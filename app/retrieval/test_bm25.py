from app.retrieval.bm25_retriever import (get_bm25_retriever,)


def main():
    retriever = get_bm25_retriever(k=4, doc_type="candidate",)

    queries = ["FastAPI", "Docker", "Transformers", "backend AI systems",]

    for query in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        docs = retriever.invoke(query)

        for i, doc in enumerate(docs, start=1,):
            print(
                f"\n{i}. "
                f"{doc.metadata.get('chunk_id')}"
            )

            print(
                f"Section: "
                f"{doc.metadata.get('section')}"
            )

            print(doc.page_content)


if __name__ == "__main__":
    main()