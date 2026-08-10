from app.retrieval.evidence import (
    retrieve_evidence,
    get_all_candidate_evidence,
    get_all_job_evidence,
    get_all_company_evidence,
    format_evidence,
)


def main():
    bundle = retrieve_evidence()

    candidate_docs = get_all_candidate_evidence(bundle)
    job_docs = get_all_job_evidence(bundle)
    company_docs = get_all_company_evidence(bundle)

    print("\n" + "=" * 80)
    print("CANDIDATE EVIDENCE")
    print("=" * 80)
    print(format_evidence(candidate_docs))

    print("\n" + "=" * 80)
    print("JOB EVIDENCE")
    print("=" * 80)
    print(format_evidence(job_docs))

    print("\n" + "=" * 80)
    print("COMPANY EVIDENCE")
    print("=" * 80)
    print(format_evidence(company_docs))


if __name__ == "__main__":
    main()