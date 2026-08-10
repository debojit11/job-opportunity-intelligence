from langchain_core.documents import Document
from app.analysis.schemas import JobOpportunityAnalysis

def build_chunk_registry(documents: list[Document],) -> dict[str, Document]:

    registry = {}

    for doc in documents:
        chunk_id = doc.metadata.get("chunk_id")

        if not chunk_id:
            continue

        registry[chunk_id] = doc

    return registry

def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def validate_analysis_evidence(analysis: JobOpportunityAnalysis, registry: dict[str, Document],) -> list[str]:

    errors = []

    # 1. Validate IDs referenced by skill matches
    for skill_match in analysis.skill_matches:
        for chunk_id in skill_match.evidence_chunk_ids:
            if chunk_id not in registry:
                errors.append(
                    f"Skill '{skill_match.skill}' "
                    f"references nonexistent chunk: {chunk_id}"
                )

    # 2. Validate IDs referenced by fit assessments
    fit_assessments = {
        "experience_fit": analysis.experience_fit,
        "project_relevance": analysis.project_relevance,
        "seniority_fit": analysis.seniority_fit,
    }

    for name, fit in fit_assessments.items():
        for chunk_id in fit.evidence_chunk_ids:
            if chunk_id not in registry:
                errors.append(
                    f"{name} references nonexistent chunk: "
                    f"{chunk_id}"
                )

    # 3. Validate EvidenceItem objects
    for item in analysis.evidence:

        doc = registry.get(item.chunk_id)

        if doc is None:
            errors.append(
                f"Evidence references nonexistent chunk: "
                f"{item.chunk_id}"
            )
            continue

        actual_source = doc.metadata.get("source")
        actual_type = doc.metadata.get("doc_type")

        if item.source != actual_source:
            errors.append(
                f"{item.chunk_id}: source mismatch. "
                f"Model={item.source}, actual={actual_source}"
            )

        if item.source_type != actual_type:
            errors.append(
                f"{item.chunk_id}: source_type mismatch. "
                f"Model={item.source_type}, actual={actual_type}"
            )

        evidence_text = normalize_text(item.evidence_text)
        source_text = normalize_text(doc.page_content)

        if evidence_text not in source_text:
            errors.append(
                f"{item.chunk_id}: evidence_text "
                f"does not appear in source chunk."
            )

    return errors