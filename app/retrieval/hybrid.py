from langchain_core.documents import Document

from app.retrieval.vector_store import (get_vector_store,)

from app.retrieval.bm25_retriever import (get_bm25_retriever,)


def reciprocal_rank_fusion(result_lists: list[list[Document]], k: int = 60,) -> list[Document]:

    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1,):
            chunk_id = doc.metadata.get("chunk_id")

            if not chunk_id:
                continue

            documents[chunk_id] = doc

            scores[chunk_id] = (scores.get(chunk_id, 0.0) + 1.0 / (k + rank))

    ranked_ids = sorted(scores, key=scores.get, reverse=True,)

    return [documents[chunk_id] for chunk_id in ranked_ids]


def hybrid_search_candidate(query: str, k: int = 4,vector_store=None):

    if vector_store is None:
        vector_store = get_vector_store()

    vector_retriever = (
        vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k,
                "filter": {
                    "doc_type": "candidate"
                },
            },
        )
    )

    bm25_retriever = get_bm25_retriever(k=k, doc_type="candidate",)

    vector_docs = vector_retriever.invoke(query)

    bm25_docs = bm25_retriever.invoke(query)

    fused_docs = reciprocal_rank_fusion([vector_docs, bm25_docs,])

    return {
        "vector": vector_docs,
        "bm25": bm25_docs,
        "hybrid": fused_docs[:k],
    }