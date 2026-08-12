from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def rerank_documents(query: str, documents: list[Document], reranker: CrossEncoder, top_k: int = 3,) -> list[tuple[Document, float]]:

    if not documents:
        return []

    pairs = [(query, doc.page_content,) for doc in documents]

    scores = reranker.predict(pairs)

    scored_documents = list(zip(documents, scores,))

    scored_documents.sort(key=lambda item: float(item[1]), reverse=True,)

    return [(doc, float(score),) for doc, score in scored_documents[:top_k]]


