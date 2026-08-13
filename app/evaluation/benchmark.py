RETRIEVAL_BENCHMARK = [
    {
        "query": "Does the candidate know Docker?",
        "doc_type": "candidate",
        "section": "skills",
        "expected_chunk_ids": {"candidate_skills_0"},
    },
    {
        "query": "What projects has the candidate built?",
        "doc_type": "candidate",
        "section": "projects",
        "expected_chunk_ids": {"candidate_projects_0"},
    },
    {
        "query": "What experience level is preferred?",
        "doc_type": "job",
        "section": "experience",
        "expected_chunk_ids": {"job_experience_0"},
    },
    {
        "query": "What technical skills are required for the role?",
        "doc_type": "job",
        "section": "required_skills",
        "expected_chunk_ids": {"job_required_skills_0"},
    },
    {
        "query": "Are there suspicious recruitment requests?",
        "doc_type": "company",
        "section": None,
        "expected_chunk_ids": {"company_notes_0"},
    },
]