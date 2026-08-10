from dataclasses import dataclass

from langchain_core.documents import Document
from app.retrieval.vector_store import get_vector_store
from app.retrieval.retrievers import (get_candidate_retriever, get_job_retriever, get_company_retriever,)


@dataclass
class EvidenceBundle:
    candidate_skills: list[Document]
    candidate_projects: list[Document]
    candidate_experience: list[Document]

    job_required_skills: list[Document]
    job_preferred_skills: list[Document]
    job_responsibilities: list[Document]
    job_experience: list[Document]

    company_background: list[Document]
    company_recruitment_risk: list[Document]


def deduplicate_documents(documents: list[Document]) -> list[Document]:

    seen = set()
    unique_docs = []

    for doc in documents:
        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id in seen:
            continue

        seen.add(chunk_id)
        unique_docs.append(doc)

    return unique_docs


def retrieve_evidence() -> EvidenceBundle:
    vector_store = get_vector_store()
    candidate_retriever = get_candidate_retriever(k=3, vector_store=vector_store)
    job_retriever = get_job_retriever(k=4, vector_store=vector_store)
    company_retriever = get_company_retriever(k=4, vector_store=vector_store)

    candidate_skills = candidate_retriever.invoke("What technical skills does the candidate have?")

    candidate_projects = candidate_retriever.invoke(
        "What projects demonstrate the candidate's AI, NLP, backend, "
        "API, RAG, and software engineering experience?"
    )

    candidate_experience = candidate_retriever.invoke("What professional or practical experience does the candidate have?")

    job_required_skills = job_retriever.invoke("What technical skills are required for this job?")

    job_preferred_skills = job_retriever.invoke("What preferred or nice-to-have skills are mentioned for this job?")

    job_responsibilities = job_retriever.invoke("What will the person in this role be responsible for doing?")

    job_experience = job_retriever.invoke("What experience or seniority level does this job require or prefer?")

    company_background = company_retriever.invoke(
        "What factual information do we have about the company "
        "and this job opening?"
    )

    company_recruitment_risk = company_retriever.invoke(
        "What evidence is available about recruitment legitimacy, "
        "suspicious payment requests, or other job-posting risk signals?"
    )

    return EvidenceBundle(
        candidate_skills=candidate_skills,
        candidate_projects=candidate_projects,
        candidate_experience=candidate_experience,

        job_required_skills=job_required_skills,
        job_preferred_skills=job_preferred_skills,
        job_responsibilities=job_responsibilities,
        job_experience=job_experience,

        company_background=company_background,
        company_recruitment_risk=company_recruitment_risk,
    )


def get_all_candidate_evidence(bundle: EvidenceBundle) -> list[Document]:

    documents = (
        bundle.candidate_skills
        + bundle.candidate_projects
        + bundle.candidate_experience
    )

    return deduplicate_documents(documents)

def get_all_job_evidence(bundle: EvidenceBundle) -> list[Document]:

    documents = (
        bundle.job_required_skills
        + bundle.job_preferred_skills
        + bundle.job_responsibilities
        + bundle.job_experience
    )

    return deduplicate_documents(documents)

def get_all_company_evidence(bundle: EvidenceBundle) -> list[Document]:

    documents = (
        bundle.company_background
        + bundle.company_recruitment_risk
    )

    return deduplicate_documents(documents)

def format_evidence(
    documents: list[Document]
) -> str:

    blocks = []

    for doc in documents:
        metadata = doc.metadata

        block = (
            f"[Chunk ID: {metadata.get('chunk_id')}]\n"
            f"[Source: {metadata.get('source')}]\n"
            f"[Type: {metadata.get('doc_type')}]\n"
            f"[Section: {metadata.get('section')}]\n"
            f"{doc.page_content}"
        )

        blocks.append(block)

    return "\n\n".join(blocks)