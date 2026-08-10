from typing import Literal

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    chunk_id: str = Field(
        description="ID of the retrieved chunk supporting this claim."
    )

    source_type: Literal["candidate","job","company"]

    source: str

    claim: str = Field(description="What conclusion this evidence supports.")

    evidence_text: str = Field(description=("Exact supporting text copied from the referenced source chunk."))


class SkillMatch(BaseModel):
    skill: str

    status: Literal["matched","partial","missing"]

    importance: Literal["required","preferred"]

    reasoning: str

    evidence_chunk_ids: list[str]


class FitAssessment(BaseModel):
    level: Literal["weak", "partial", "strong", "unknown"]

    reasoning: str

    evidence_chunk_ids: list[str] = Field(default_factory=list)


class JobOpportunityAnalysis(BaseModel):
    fit_level: Literal["weak", "partial", "strong"]

    experience_fit: FitAssessment

    project_relevance: FitAssessment

    seniority_fit: FitAssessment

    fraud_risk: Literal["low", "medium", "high", "unknown"]

    skill_matches: list[SkillMatch] = Field(default_factory=list)

    strengths: list[str] = Field(default_factory=list)

    gaps: list[str] = Field(default_factory=list)

    concerns: list[str] = Field(default_factory=list)

    evidence: list[EvidenceItem] = Field(default_factory=list)

    summary: str


class ScoreBreakdown(BaseModel):
    skill_match: float = Field(ge=0, le=35)
    experience_fit: float = Field(ge=0, le=20)
    project_relevance: float = Field(ge=0, le=20)
    seniority_fit: float = Field(ge=0, le=10)
    company_risk: float = Field(ge=0, le=10)
    additional_requirements: float = Field(ge=0, le=5)
    final_score: float = Field(ge=0, le=100)


class FinalJobAssessment(BaseModel):
    analysis: JobOpportunityAnalysis
    scores: ScoreBreakdown
    final_recommendation: Literal["apply", "maybe", "skip"]