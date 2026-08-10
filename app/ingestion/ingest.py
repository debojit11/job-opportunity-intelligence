from app.ingestion.loader import load_text_document
from app.ingestion.splitter import (split_into_sections,chunk_documents,)


DATA_SOURCES = [
    {
        "path": "data/candidate.txt",
        "doc_type": "candidate",
    },
    {
        "path": "data/job_description.txt",
        "doc_type": "job",
    },
    {
        "path": "data/company.txt",
        "doc_type": "company",
    },
]


def ingest_documents():
    all_sections = []

    for source in DATA_SOURCES:

        document = load_text_document(file_path=source["path"], doc_type=source["doc_type"],)

        sections = split_into_sections(document)

        all_sections.extend(sections)

    chunks = chunk_documents(all_sections, chunk_size=700, chunk_overlap=120,)

    return chunks