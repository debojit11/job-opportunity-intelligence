from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float


def evaluate_retrieval(retrieved_docs, expected_chunk_ids: set[str],) -> RetrievalMetrics:

    retrieved_ids = [doc.metadata.get("chunk_id") for doc in retrieved_docs]

    retrieved_set = set(retrieved_ids)

    relevant_found = (retrieved_set & expected_chunk_ids)

    recall = (
        len(relevant_found)
        / len(expected_chunk_ids)
        if expected_chunk_ids
        else 0.0
    )

    precision = (
        len(relevant_found)
        / len(retrieved_ids)
        if retrieved_ids
        else 0.0
    )

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(retrieved_ids, start=1,):
        if chunk_id in expected_chunk_ids:
            reciprocal_rank = 1 / rank
            break

    return RetrievalMetrics(
        recall_at_k=recall,
        precision_at_k=precision,
        reciprocal_rank=reciprocal_rank,
    )