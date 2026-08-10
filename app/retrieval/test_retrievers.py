from app.retrieval.retrievers import (
    get_candidate_retriever,
    get_job_retriever,
    get_company_retriever,
)


def print_results(title, docs):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for i, doc in enumerate(docs, start=1):
        print(f"\nRESULT {i}")
        print(doc.metadata)
        print(doc.page_content)


def main():
    candidate_retriever = get_candidate_retriever(k=3)
    job_retriever = get_job_retriever(k=3)
    company_retriever = get_company_retriever(k=3)

    candidate_docs = candidate_retriever.invoke(
        "What backend and API experience does the candidate have?"
    )

    job_docs = job_retriever.invoke(
        "What technical skills and experience does the job require?"
    )

    company_docs = company_retriever.invoke(
        "What do we know about the company and recruitment legitimacy?"
    )

    print_results(
        "CANDIDATE EVIDENCE",
        candidate_docs
    )

    print_results(
        "JOB EVIDENCE",
        job_docs
    )

    print_results(
        "COMPANY EVIDENCE",
        company_docs
    )


if __name__ == "__main__":
    main()