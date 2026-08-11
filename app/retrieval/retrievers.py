from langchain_chroma import Chroma

from app.retrieval.vector_store import get_vector_store

def build_filter(doc_type: str, section: str | None = None,):
    if section is None:
        return {"doc_type": doc_type}

    return {"$and": [{"doc_type": doc_type}, {"section": section},]}



def get_candidate_retriever(k: int = 4, vector_store: Chroma | None = None, section: str | None = None,):
    vector_store = vector_store or get_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": build_filter(doc_type="candidate", section=section)
        }
    )


def get_job_retriever(k: int = 4, vector_store: Chroma | None = None, section: str | None = None,):
    vector_store = vector_store or get_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": build_filter(doc_type="job", section=section)
        }
    )


def get_company_retriever(k: int = 4, vector_store: Chroma | None = None, section: str | None = None,):
    vector_store = vector_store or get_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
            "filter": build_filter(doc_type="company", section=section)
        }
    )