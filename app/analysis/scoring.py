from app.analysis.schemas import (JobOpportunityAnalysis, ScoreBreakdown,)
from app.analysis.schemas import FinalJobAssessment


STATUS_SCORE = {
    "matched": 1.0,
    "partial": 0.5,
    "missing": 0.0,
}

IMPORTANCE_WEIGHT = {
    "required": 2.0,
    "preferred": 1.0,
}


def calculate_skill_score(analysis: JobOpportunityAnalysis) -> float:

    earned = 0.0
    possible = 0.0

    for item in analysis.skill_matches:
        status_score = STATUS_SCORE[item.status]
        weight = IMPORTANCE_WEIGHT[item.importance]

        earned += status_score * weight
        possible += weight

    if possible == 0:
        return 0.0

    ratio = earned / possible

    return round(ratio * 35, 2)


FIT_SCORE = {
    "weak": 0.35,
    "partial": 0.65,
    "strong": 0.90,
}

FIT_RATIO = {
    "strong": 0.90,
    "partial": 0.65,
    "weak": 0.35,
    "unknown": 0.50,
}

def calculate_experience_fit(analysis: JobOpportunityAnalysis) -> float:

    ratio = FIT_RATIO[analysis.experience_fit.level]

    return round(ratio * 20, 2)


def calculate_project_relevance(analysis: JobOpportunityAnalysis) -> float:

    ratio = FIT_RATIO[analysis.project_relevance.level]

    return round(ratio * 20, 2)


def calculate_seniority_fit(analysis: JobOpportunityAnalysis) -> float:

    ratio = FIT_RATIO[analysis.seniority_fit.level]

    return round(ratio * 10, 2)


RISK_SCORE = {
    "low": 9.0,
    "medium": 5.0,
    "high": 0.0,
    "unknown": 6.0,
}


def calculate_company_risk(analysis: JobOpportunityAnalysis) -> float:

    return RISK_SCORE[analysis.fraud_risk]


def calculate_additional_requirements(analysis: JobOpportunityAnalysis) -> float:

    preferred = [
        item
        for item in analysis.skill_matches
        if item.importance == "preferred"
    ]

    if not preferred:
        return 5.0

    earned = 0.0

    for item in preferred:
        earned += STATUS_SCORE[item.status]

    ratio = earned / len(preferred)

    return round(ratio * 5, 2)


def calculate_scores(analysis: JobOpportunityAnalysis) -> ScoreBreakdown:

    skill_match = calculate_skill_score(analysis)

    experience_fit = calculate_experience_fit(analysis)

    project_relevance = calculate_project_relevance(analysis)

    seniority_fit = calculate_seniority_fit(analysis)

    company_risk = calculate_company_risk(analysis)

    additional_requirements = (calculate_additional_requirements(analysis))

    final_score = round(
        skill_match + experience_fit + project_relevance + seniority_fit + company_risk + additional_requirements, 2)

    return ScoreBreakdown(
        skill_match=skill_match,
        experience_fit=experience_fit,
        project_relevance=project_relevance,
        seniority_fit=seniority_fit,
        company_risk=company_risk,
        additional_requirements=additional_requirements,
        final_score=final_score,
    )


def recommendation_from_score(score: float, fraud_risk: str,) -> str:

    if fraud_risk == "high":
        return "skip"

    if score >= 75:
        return "apply"

    if score >= 55:
        return "maybe"

    return "skip"


def build_final_assessment(analysis: JobOpportunityAnalysis) -> FinalJobAssessment:

    scores = calculate_scores(analysis)

    recommendation = recommendation_from_score(scores.final_score, analysis.fraud_risk,)

    return FinalJobAssessment(analysis=analysis, scores=scores, final_recommendation=recommendation,)