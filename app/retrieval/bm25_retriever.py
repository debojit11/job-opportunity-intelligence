from langchain_community.retrievers import BM25Retriever

from app.ingestion.ingest import ingest_documents


def get_bm25_retriever(k: int = 4, doc_type: str | None = None,):
    documents = ingest_documents()

    if doc_type is not None:
        documents = [doc for doc in documents if doc.metadata.get("doc_type") == doc_type]

    retriever = BM25Retriever.from_documents(documents)

    retriever.k = k

    return retriever