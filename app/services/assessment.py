from app.analysis.analyzer import analyze_evidence
from app.analysis.scoring import (build_final_assessment,)
from app.analysis.validation import (build_chunk_registry, validate_analysis_evidence,)
from app.analysis.schemas import FinalJobAssessment
from app.retrieval.evidence import (
    retrieve_evidence,
    get_all_candidate_evidence,
    get_all_job_evidence,
    get_all_company_evidence,
    format_evidence,
)


def assess_job_opportunity() -> FinalJobAssessment:

    
    # 1. Retrieve evidence
    bundle = retrieve_evidence()

    candidate_docs = get_all_candidate_evidence(
        bundle
    )

    job_docs = get_all_job_evidence(
        bundle
    )

    company_docs = get_all_company_evidence(
        bundle
    )


    # 2. Build registry for validation
    all_docs = (
        candidate_docs
        + job_docs
        + company_docs
    )

    registry = build_chunk_registry(
        all_docs
    )


    # 3. Format evidence for LLM
    candidate_evidence = format_evidence(
        candidate_docs
    )

    job_evidence = format_evidence(
        job_docs
    )

    company_evidence = format_evidence(
        company_docs
    )


    # 4. Structured LLM analysis
    analysis = analyze_evidence(
        candidate_evidence=candidate_evidence,
        job_evidence=job_evidence,
        company_evidence=company_evidence,
    )


    # 5. Validate grounding
    validation_errors = (
        validate_analysis_evidence(
            analysis,
            registry
        )
    )

    if validation_errors:
        print(
            "\nEVIDENCE VALIDATION WARNINGS:"
        )

        for error in validation_errors:
            print(f"- {error}")


    # 6. Deterministic scoring
    final_assessment = (
        build_final_assessment(
            analysis
        )
    )


    return final_assessment