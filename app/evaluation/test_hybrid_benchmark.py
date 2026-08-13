from app.evaluation.benchmark import (RETRIEVAL_BENCHMARK,)
from app.evaluation.retrieval_eval import (evaluate_retrieval,)
from app.retrieval.vector_store import (get_vector_store,)
from app.retrieval.hybrid import (
    hybrid_search_candidate,
    hybrid_search_job,
    hybrid_search_company,)


def hybrid_search_for_type(query: str, doc_type: str, vector_store,):
    if doc_type == "candidate":
        result = hybrid_search_candidate(
            query=query,
            k=3,
            vector_store=vector_store,
        )

    elif doc_type == "job":
        result = hybrid_search_job(
            query=query,
            k=3,
            vector_store=vector_store,
        )

    elif doc_type == "company":
        result = hybrid_search_company(
            query=query,
            k=3,
            vector_store=vector_store,
        )

    else:
        raise ValueError(
            f"Unsupported doc_type: {doc_type}"
        )

    return result["hybrid"]


def main():
    vector_store = get_vector_store()

    total_recall = 0.0
    total_precision = 0.0
    total_rr = 0.0

    for case in RETRIEVAL_BENCHMARK:

        docs = hybrid_search_for_type(
            query=case["query"],
            doc_type=case["doc_type"],
            vector_store=vector_store,
        )

        metrics = evaluate_retrieval(
            retrieved_docs=docs,
            expected_chunk_ids=case[
                "expected_chunk_ids"
            ],
        )

        total_recall += metrics.recall_at_k
        total_precision += metrics.precision_at_k
        total_rr += metrics.reciprocal_rank

        print("\n" + "=" * 70)
        print("QUERY:", case["query"])

        print("\nRetrieved:")
        for doc in docs:
            print(
                "-",
                doc.metadata.get("chunk_id")
            )

        print("\nMetrics")
        print(
            "Recall@3:",
            metrics.recall_at_k
        )
        print(
            "Precision@3:",
            metrics.precision_at_k
        )
        print(
            "Reciprocal Rank:",
            metrics.reciprocal_rank
        )

    count = len(RETRIEVAL_BENCHMARK)

    print("\n" + "=" * 70)
    print("HYBRID BENCHMARK SUMMARY")
    print("=" * 70)

    print(
        "Mean Recall@3:",
        round(total_recall / count, 3)
    )

    print(
        "Mean Precision@3:",
        round(total_precision / count, 3)
    )

    print(
        "MRR:",
        round(total_rr / count, 3)
    )


if __name__ == "__main__":
    main()