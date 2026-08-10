# Evidence-Grounded Job Opportunity Intelligence Agent

An evidence-grounded RAG system that evaluates whether a job opportunity is worth applying to by comparing candidate evidence, job requirements, and company/recruitment evidence.

Instead of acting like a generic chatbot, the system retrieves relevant evidence, performs structured semantic analysis, validates the cited evidence, calculates a deterministic score, and returns an explainable **Apply / Maybe / Skip** recommendation.

---

## Overview

The project analyzes three main evidence sources:

- Candidate profile, skills, projects, and experience
- Job description, requirements, responsibilities, and seniority expectations
- Company information and recruitment-risk signals

The system then produces:

- Overall fit
- Recruitment-risk level
- Matched, partial, and missing skills
- Required vs preferred skill analysis
- Experience fit
- Project relevance
- Seniority fit
- Strengths
- Gaps
- Concerns
- Evidence chunk references
- Exact supporting evidence
- Weighted score out of 100
- Final Apply / Maybe / Skip recommendation

The goal is to make the recommendation inspectable rather than relying on an opaque LLM-generated score.

---

## Why This Project?

A simple keyword matcher cannot reliably evaluate whether a candidate's real project experience is relevant to a job.

For example, a candidate may have built:

> A FastAPI log classification system using Sentence Transformers and an LLM fallback.

while a job description says:

> Build AI APIs.

Those phrases are not identical, but they are semantically related.

This project uses RAG and an LLM for semantic understanding while keeping scoring and decision thresholds deterministic in Python.

The core design principle is:

```text
Retrieve evidence
        ↓
Reason from evidence
        ↓
Validate evidence
        ↓
Apply deterministic scoring
        ↓
Return an explainable recommendation
```

---

## Architecture

```text
Input Documents
    ↓
Document Loading
    ↓
Section-Aware Splitting
    ↓
Chunking
    ↓
Hugging Face Embeddings
    ↓
Chroma Vector Store
    ↓
Source-Specific Retrievers
    ↓
Evidence Bundle
    ↓
Structured LLM Analysis
    ↓
Evidence Validation
    ↓
Deterministic Scoring
    ↓
Apply / Maybe / Skip
    ↓
Trustworthy CLI Report
```

---

## Project Structure

```text
job-opportunity-intelligence/
│
├── app/
│   │
│   ├── analysis/
│   │   ├── analyzer.py
│   │   ├── schemas.py
│   │   ├── scoring.py
│   │   └── validation.py
│   │
│   ├── ingestion/
│   │   ├── indexer.py
│   │   ├── ingest.py
│   │   ├── loader.py
│   │   └── splitter.py
│   │
│   ├── retrieval/
│   │   ├── evidence.py
│   │   ├── retrievers.py
│   │   └── vector_store.py
│   │
│   ├── services/
│   │   └── assessment.py
│   │
│   ├── presentation/
│   │   └── console.py
│   │
│   └── main.py
│
├── data/
│   ├── candidate.txt
│   ├── job_description.txt
│   └── company.txt
│
├── chroma_db/
├── .env
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## Core Components

### 1. Section-Aware Ingestion

The input documents are not treated as one large block.

The system recognizes sections such as:

- `SUMMARY`
- `SKILLS`
- `PROJECTS`
- `EXPERIENCE`
- `REQUIRED SKILLS`
- `PREFERRED SKILLS`
- `RESPONSIBILITIES`
- `CAREERS`
- `KNOWN INFORMATION`
- `NOTES`

Each section becomes structured evidence with metadata.

Example chunk IDs:

```text
candidate_summary_0
candidate_skills_0
candidate_projects_0
job_required_skills_0
job_responsibilities_0
company_known_information_0
```

Stable chunk IDs are later used for evidence grounding.

---

### 2. Embeddings and Chroma

The current V1 uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

with normalized embeddings and a persistent Chroma vector store.

This allows retrieval based on semantic similarity rather than exact keyword overlap.

---

### 3. Source-Specific Retrieval

Candidate, job, and company evidence are retrieved separately.

This avoids a common RAG failure mode where, for example, a job requirement is accidentally treated as evidence that the candidate possesses that skill.

The project uses dedicated retrievers for:

```text
candidate
job
company
```

with metadata filtering.

---

### 4. Focused Evidence Retrieval

Instead of running one generic retrieval query, the system retrieves evidence for separate analytical goals.

Examples:

```text
Candidate
- technical skills
- projects
- practical experience

Job
- required skills
- preferred skills
- responsibilities
- experience expectations

Company
- company background
- recruitment-risk signals
```

The retrieved chunks are grouped into an `EvidenceBundle`.

---

### 5. Structured LLM Analysis

The retrieved evidence is sent to Gemini through LangChain.

The model returns a Pydantic-validated structured object rather than free-form output.

It evaluates:

```text
skill matches
experience fit
project relevance
seniority fit
recruitment risk
strengths
gaps
concerns
supporting evidence
summary
```

The LLM is explicitly instructed not to generate the final numerical score or final Apply / Maybe / Skip recommendation.

---

### 6. Evidence Validation

The system does not blindly trust the model's citations.

It validates:

- whether a referenced chunk ID exists
- whether the claimed source matches the actual source
- whether the claimed document type matches
- whether the evidence text actually appears in the referenced chunk
- whether skill-match chunk references exist
- whether fit-assessment chunk references exist

This creates two distinct validation layers:

```text
Pydantic validation
→ Is the output structure valid?

Evidence validation
→ Are the evidence references grounded in retrieved data?
```

---

## Scoring System

The final score is calculated deterministically in Python.

```text
Skill Match                 35 points
Experience Fit              20 points
Project Relevance           20 points
Seniority Fit               10 points
Company / Recruitment Risk  10 points
Additional Requirements      5 points
--------------------------------------
Total                       100 points
```

### Skill Status

```text
Matched = 1.0
Partial = 0.5
Missing = 0.0
```

### Skill Importance

```text
Required  = 2.0
Preferred = 1.0
```

Required skills therefore affect the score more strongly than preferred skills.

### Semantic Fit Mapping

For experience, project relevance, and seniority:

```text
Strong  = 0.90
Partial = 0.65
Weak    = 0.35
Unknown = 0.50
```

The LLM supplies the semantic label.

Python converts it to the numerical value.

This prevents free-form LLM wording from directly changing the score.

---

## Recommendation Logic

The final recommendation is deterministic:

```text
Score >= 75        → APPLY
Score >= 55        → MAYBE
Score < 55         → SKIP
```

There is also a recruitment-risk guardrail:

```text
fraud_risk == "high"
→ SKIP
```

So a technically strong job match cannot receive an Apply recommendation when the available evidence indicates high recruitment risk.

---

## Example Output

```text
Score: 65.42/100
Recommendation: MAYBE
Overall Fit: PARTIAL
Recruitment Risk: LOW

SCORE BREAKDOWN
Skill Match:              20.42/35
Experience Fit:           13.0/20
Project Relevance:        18.0/20
Seniority Fit:             5.0/10
Company / Risk:            9.0/10
Additional Requirements:   0.0/5
```

The CLI also displays:

- every skill match
- required/preferred classification
- reasoning
- evidence chunk IDs
- fit-assessment explanations
- strengths
- gaps
- concerns
- exact supporting evidence
- final summary

---

## Installation

This project uses `uv`.

Clone the repository:

```bash
git clone <your-repository-url>
cd job-opportunity-intelligence
```

Install dependencies:

```bash
uv sync
```

If dependencies are not yet recorded in `pyproject.toml`, add the main packages used by the project:

```bash
uv add langchain langchain-core langchain-chroma langchain-huggingface langchain-google-genai sentence-transformers chromadb python-dotenv pydantic
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
```

The application loads the environment using `python-dotenv`.

Do not commit `.env`.

Add it to `.gitignore`:

```gitignore
.env
```

---

## Input Data

The current V1 expects:

```text
data/candidate.txt
data/job_description.txt
data/company.txt
```

Example candidate structure:

```text
SUMMARY
AI/ML engineer focused on NLP, Transformers, RAG and FastAPI.

SKILLS
Python
FastAPI
PyTorch
Transformers

PROJECTS
...
```

Example job structure:

```text
ROLE
AI Engineer

REQUIRED SKILLS
Python
FastAPI
RAG
LangChain
Docker

PREFERRED SKILLS
LangGraph
AWS

RESPONSIBILITIES
Build AI APIs.
Develop retrieval pipelines.
Deploy AI services.
```

---

## Build / Rebuild the Vector Index

Run the indexer after:

- changing input files
- changing chunking logic
- changing the embedding model
- intentionally rebuilding the Chroma collection

```bash
uv run python -m app.ingestion.indexer
```

You do not need to rebuild the index before every application run.

---

## Run the Application

```bash
uv run python -m app.main
```

The application will:

```text
retrieve evidence
→ analyze it
→ validate grounding
→ calculate scores
→ produce a recommendation
→ print an explainable report
```

---

## Current Tech Stack

```text
Python
LangChain
Gemini
Pydantic
Hugging Face Sentence Transformers
Chroma
python-dotenv
uv
```

Current embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Current LLM configuration in the project:

```text
gemini-3-flash-preview
```

---

## Current V1 Limitations

The current system is intentionally a first version.

Known limitations include:

- input documents are manually prepared text files
- retrieval currently relies primarily on dense semantic search
- no hybrid lexical + vector retrieval yet
- no dedicated reranker yet
- no query decomposition yet
- no multimodal ingestion yet
- no automated company research yet
- no full entailment-validation layer yet
- semantic labels still depend on LLM reasoning
- scoring weights are manually designed heuristics
- recommendation thresholds are not statistically calibrated
- no production frontend yet
- no large-scale RAG evaluation dataset yet

---

## Planned Advanced RAG Upgrades

The project is being continued while learning Advanced RAG.

Planned upgrades include:

```text
Query Enhancement
Query Decomposition
Metadata Filtering
Hybrid Retrieval
Reranking
Multimodal RAG
Agentic RAG
Cache-Augmented Generation
RAG Evaluation
Hallucination Reduction
```

The goal is to improve the same project incrementally rather than replacing it with disconnected tutorial examples.

---

## Future Ideas

Possible future extensions:

- resume/PDF ingestion
- direct job-page ingestion
- company-website research
- automatic requirement extraction
- configurable scoring profiles
- multi-job comparison
- opportunity ranking
- hybrid retrieval
- reranking
- richer recruitment-risk analysis
- API layer
- web interface
- evidence-grounded recommendation dashboard

---

## Design Philosophy

This project follows three main principles:

```text
Use the LLM for semantic understanding.
Use deterministic code for policy and arithmetic.
Keep evidence inspectable.
```

The final score is not intended to be an unquestionable prediction of hiring success.

It is an evidence-grounded decision-support score designed to help a candidate understand whether a job opportunity appears worth applying to and, more importantly, why.
