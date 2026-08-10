from langchain_chroma import Chroma

from app.retrieval.vector_store import get_vector_store


def get_candidate_retriever(
    k: int = 4,
    vector_store: Chroma | None = None,
):
    vector_store = vector_store or get_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"doc_type": "candidate"}
        }
    )


def get_job_retriever(
    k: int = 4,
    vector_store: Chroma | None = None,
):
    vector_store = vector_store or get_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"doc_type": "job"}
        }
    )


def get_company_retriever(
    k: int = 4,
    vector_store: Chroma | None = None,
):
    vector_store = vector_store or get_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": {"doc_type": "company"}
        }
    )