from dataclasses import dataclass

from langchain_core.documents import Document
from app.retrieval.vector_store import get_vector_store
from app.retrieval.retrievers import (get_candidate_retriever, get_job_retriever, get_company_retriever,)
from app.retrieval.queries import (
    CANDIDATE_SKILLS_QUERY,
    CANDIDATE_PROJECTS_QUERY,
    CANDIDATE_EXPERIENCE_QUERY,
    JOB_REQUIRED_SKILLS_QUERY,
    JOB_PREFERRED_SKILLS_QUERY,
    JOB_RESPONSIBILITIES_QUERY,
    JOB_EXPERIENCE_QUERY,
    COMPANY_BACKGROUND_QUERY,
    COMPANY_RISK_QUERY,
)

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

    candidate_skills_retriever = get_candidate_retriever(k=3, vector_store=vector_store, section="skills",)

    candidate_projects_retriever = get_candidate_retriever(k=3, vector_store=vector_store, section="projects",)

    # Keep broad for now because current candidate experience
    # evidence is sparse / may not have a useful experience section.
    candidate_experience_retriever = get_candidate_retriever(k=3, vector_store=vector_store,)

    job_required_retriever = get_job_retriever(k=4, vector_store=vector_store, section="required_skills",)

    job_preferred_retriever = get_job_retriever(k=4, vector_store=vector_store, section="preferred_skills",)

    job_responsibilities_retriever = get_job_retriever(k=4, vector_store=vector_store, section="responsibilities",)

    job_experience_retriever = get_job_retriever(k=4, vector_store=vector_store, section="experience",)

    # Keep company retrieval broad because useful evidence
    # can live across about/careers/known_information/notes.
    company_retriever = get_company_retriever(k=4, vector_store=vector_store,)

    candidate_skills = candidate_skills_retriever.invoke(CANDIDATE_SKILLS_QUERY)

    candidate_projects = candidate_projects_retriever.invoke(CANDIDATE_PROJECTS_QUERY)

    candidate_experience = candidate_experience_retriever.invoke(CANDIDATE_EXPERIENCE_QUERY)

    job_required_skills = job_required_retriever.invoke(JOB_REQUIRED_SKILLS_QUERY)

    job_preferred_skills = job_preferred_retriever.invoke(JOB_PREFERRED_SKILLS_QUERY)

    job_responsibilities = job_responsibilities_retriever.invoke(JOB_RESPONSIBILITIES_QUERY)

    job_experience = job_experience_retriever.invoke(JOB_EXPERIENCE_QUERY)

    company_background = company_retriever.invoke(COMPANY_BACKGROUND_QUERY)

    company_recruitment_risk = company_retriever.invoke(COMPANY_RISK_QUERY)

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