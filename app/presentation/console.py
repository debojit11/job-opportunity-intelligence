from app.analysis.schemas import FinalJobAssessment


def print_skill_matches(assessment: FinalJobAssessment,) -> None:

    print("\nSKILL MATCHES")
    print("-" * 80)

    for item in assessment.analysis.skill_matches:
        print(
            f"{item.skill:<20} "
            f"{item.status.upper():<8} "
            f"[{item.importance.upper()}]"
        )

        print(f"  Reason: {item.reasoning}")

        if item.evidence_chunk_ids:
            print("  Evidence: " + ", ".join(item.evidence_chunk_ids))
        else:
            print("  Evidence: none found")

        print()


def print_fit_assessments(assessment: FinalJobAssessment,) -> None:

    analysis = assessment.analysis

    fits = [
        (
            "Experience Fit",
            analysis.experience_fit,
            assessment.scores.experience_fit,
            20,
        ),
        (
            "Project Relevance",
            analysis.project_relevance,
            assessment.scores.project_relevance,
            20,
        ),
        (
            "Seniority Fit",
            analysis.seniority_fit,
            assessment.scores.seniority_fit,
            10,
        ),
    ]

    print("\nFIT ANALYSIS")
    print("-" * 80)

    for name, fit, score, maximum in fits:
        print(
            f"{name}: "
            f"{fit.level.upper()} "
            f"({score}/{maximum})"
        )

        print(f"  Reason: {fit.reasoning}")

        if fit.evidence_chunk_ids:
            print("  Evidence: " + ", ".join(fit.evidence_chunk_ids))

        print()


def print_findings(assessment: FinalJobAssessment,) -> None:

    analysis = assessment.analysis

    print("\nSTRENGTHS")
    print("-" * 80)

    if analysis.strengths:
        for item in analysis.strengths:
            print(f"+ {item}")
    else:
        print("None identified.")

    print("\nGAPS")
    print("-" * 80)

    if analysis.gaps:
        for item in analysis.gaps:
            print(f"- {item}")
    else:
        print("None identified.")

    print("\nCONCERNS")
    print("-" * 80)

    if analysis.concerns:
        for item in analysis.concerns:
            print(f"! {item}")
    else:
        print("None identified.")


def print_evidence(assessment: FinalJobAssessment,) -> None:

    print("\nSUPPORTING EVIDENCE")
    print("-" * 80)

    evidence_items = assessment.analysis.evidence

    if not evidence_items:
        print("No evidence items returned.")
        return

    for item in evidence_items:
        print(
            f"\n[{item.chunk_id}] "
            f"{item.source_type.upper()} "
            f"— {item.source}"
        )

        print(f"Reasoning: {item.claim}")

        print("Source evidence:")
        print(f'  "{item.evidence_text}"')


def print_assessment_report(assessment: FinalJobAssessment,) -> None:

    print("\n" + "=" * 80)
    print("JOB OPPORTUNITY ASSESSMENT")
    print("=" * 80)

    print(
        f"\nScore: "
        f"{assessment.scores.final_score}/100"
    )

    print(
        "Recommendation:",
        assessment.final_recommendation.upper()
    )

    print(
        "Overall Fit:",
        assessment.analysis.fit_level.upper()
    )

    print(
        "Recruitment Risk:",
        assessment.analysis.fraud_risk.upper()
    )

    print("\nSCORE BREAKDOWN")
    print("-" * 80)

    print(
        f"Skill Match:              "
        f"{assessment.scores.skill_match}/35"
    )

    print(
        f"Experience Fit:           "
        f"{assessment.scores.experience_fit}/20"
    )

    print(
        f"Project Relevance:        "
        f"{assessment.scores.project_relevance}/20"
    )

    print(
        f"Seniority Fit:            "
        f"{assessment.scores.seniority_fit}/10"
    )

    print(
        f"Company / Risk:           "
        f"{assessment.scores.company_risk}/10"
    )

    print(
        f"Additional Requirements:  "
        f"{assessment.scores.additional_requirements}/5"
    )

    print_skill_matches(assessment)

    print_fit_assessments(assessment)

    print_findings(assessment)

    print_evidence(assessment)

    print("\nSUMMARY")
    print("-" * 80)

    print(assessment.analysis.summary)

    print("\n" + "=" * 80)