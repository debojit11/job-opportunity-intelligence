from app.retrieval.query_rewriter import rewrite_query
from app.retrieval.vector_store import get_vector_store


def search_candidate(user_query: str, k: int = 4,):
    vector_store = get_vector_store()

    enhanced_query = rewrite_query(user_query)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {
                "doc_type": "candidate"
            },
        },
    )

    docs = retriever.invoke(enhanced_query)

    return enhanced_query, docs