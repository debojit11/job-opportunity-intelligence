from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


class RetrievalSubquery(BaseModel):
    query: str

    source_type: Literal[
        "candidate",
        "job",
        "company",
    ]


class DecomposedQuery(BaseModel):
    subqueries: list[
        RetrievalSubquery
    ]

# class DecomposedQuery(BaseModel):
#     subqueries: list[RetrievalSubquery] = Field(
#         description=(
#             "Focused retrieval questions that together "
#             "cover the original information need."
#         )
#     )


model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0,)


structured_model = model.with_structured_output(DecomposedQuery)


decomposition_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You decompose complex questions for a job-opportunity
evidence retrieval system.

The corpus may contain:
- candidate skills
- candidate projects
- candidate experience
- job required skills
- job preferred skills
- job responsibilities
- job experience requirements
- company background
- recruitment-risk evidence

RULES:

1. Break the original question into focused retrieval subqueries.

2. Each subquery should represent one clear information need.

3. Preserve the user's original intent.

4. Do not answer the question.

5. Do not invent candidate, job, or company facts.

6. Avoid redundant subqueries.

7. Do not generate more subqueries than necessary.

8.  Assign each subquery exactly one source_type:
   candidate, job, or company.

9. source_type must indicate where the evidence for that subquery should come from.

10. Do not introduce a new information need that is not necessary
    to answer the original question.

11. Only include company or recruitment-risk subqueries when the
    original question asks about the company, job legitimacy,
    recruitment risk, safety, trustworthiness, or whether the
    opportunity is worth applying to in a way that requires such evidence.

12. Recruitment-risk, company legitimacy signals, careers-page evidence,
    suspicious payments, and recruitment warning signs must use
    source_type="company", never "candidate".

13. Candidate source queries must concern only candidate evidence such as
    skills, projects, experience, education, qualifications, or seniority.

14. Job source queries must concern only job evidence such as requirements,
    preferred qualifications, responsibilities, role details, or experience
    expectations.

15. Prefer the smallest set of subqueries that fully covers the original
    information need.
        """
    ),
    (
        "user",
        "{query}"
    ),
])


decomposition_chain = (decomposition_prompt | structured_model)


def decompose_query(query: str,) -> list[RetrievalSubquery]:

    result = decomposition_chain.invoke({"query": query})

    return result.subqueries

