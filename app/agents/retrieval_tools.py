from langchain_core.tools import tool

from app.retrieval.vector_store import get_vector_store


vector_store = get_vector_store()


def format_docs(docs) -> str:
    if not docs:
        return "No relevant evidence found."

    blocks = []

    for doc in docs:
        blocks.append(
            "\n".join([
                f"Chunk ID: {doc.metadata.get('chunk_id')}",
                f"Source: {doc.metadata.get('source')}",
                f"Section: {doc.metadata.get('section')}",
                f"Content: {doc.page_content}",
            ])
        )

    return "\n\n".join(blocks)


@tool
def search_candidate_skills(query: str) -> str:
    """Search candidate skill evidence relevant to the query."""

    docs = vector_store.similarity_search(query, k=3,
        filter={
            "$and": [
                {"doc_type": "candidate"},
                {"section": "skills"},
            ]
        },
    )

    return format_docs(docs)


@tool
def search_candidate_projects(query: str) -> str:
    """Search candidate project evidence relevant to the query."""

    docs = vector_store.similarity_search(query, k=3,
        filter={
            "$and": [
                {"doc_type": "candidate"},
                {"section": "projects"},
            ]
        },
    )

    return format_docs(docs)


@tool
def search_candidate_experience(query: str) -> str:
    """Search candidate experience evidence relevant to the query."""

    docs = vector_store.similarity_search(query, k=3,
        filter={"doc_type": "candidate"},)

    return format_docs(docs)


@tool
def search_job_requirements(query: str) -> str:
    """Search the job's required technical skills and qualifications."""

    docs = vector_store.similarity_search(query, k=3,
        filter={
            "$and": [
                {"doc_type": "job"},
                {"section": "required_skills"},
            ]
        },
    )

    return format_docs(docs)


@tool
def search_job_responsibilities(query: str) -> str:
    """Search the job's stated responsibilities and expected work."""

    docs = vector_store.similarity_search(query, k=3,
        filter={
            "$and": [
                {"doc_type": "job"},
                {"section": "responsibilities"},
            ]
        },
    )

    return format_docs(docs)


@tool
def search_job_experience(query: str) -> str:
    """Search job experience and seniority requirements."""

    docs = vector_store.similarity_search(query, k=3,
        filter={
            "$and": [
                {"doc_type": "job"},
                {"section": "experience"},
            ]
        },
    )

    return format_docs(docs)


@tool
def search_company_risk(query: str) -> str:
    """Search company and recruitment-risk evidence."""

    docs = vector_store.similarity_search(query, k=4, filter={"doc_type": "company"},)

    return format_docs(docs)