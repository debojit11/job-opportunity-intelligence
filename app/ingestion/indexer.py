from app.ingestion.ingest import ingest_documents
from app.retrieval.vector_store import get_vector_store


def build_index(
    rebuild: bool = True
):
    chunks = ingest_documents()

    if not chunks:
        raise ValueError(
            "No chunks were produced. "
            "Indexing aborted."
        )

    vector_store = get_vector_store()

    if rebuild:
        vector_store.reset_collection()

    ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    vector_store.add_documents(documents=chunks, ids=ids)

    print(f"Indexed chunks: {len(chunks)}")

    print("Stored records:", vector_store._collection.count())

    return vector_store


if __name__ == "__main__":
    build_index(rebuild=True)