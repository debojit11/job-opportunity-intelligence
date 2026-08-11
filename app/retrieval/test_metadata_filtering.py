from app.retrieval.vector_store import get_vector_store
from app.retrieval.retrievers import get_candidate_retriever


def print_docs(title, docs):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for i, doc in enumerate(docs, start=1):
        print(f"\n{i}. {doc.metadata.get('chunk_id')}")

        print(f"Section: {doc.metadata.get('section')}")

        print(doc.page_content)


def main():
    vector_store = get_vector_store()

    broad_retriever = get_candidate_retriever(k=4, vector_store=vector_store,)

    project_retriever = get_candidate_retriever(k=4, vector_store=vector_store, section="projects",)

    query = ("What candidate projects demonstrate backend API or AI engineering experience?")

    broad_docs = broad_retriever.invoke(query)

    filtered_docs = project_retriever.invoke(query)

    print_docs("DOC_TYPE FILTER ONLY",broad_docs)

    print_docs("DOC_TYPE + SECTION FILTER",filtered_docs)


if __name__ == "__main__":
    main()