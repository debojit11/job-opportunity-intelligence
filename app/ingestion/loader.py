from pathlib import Path

from langchain_core.documents import Document


def load_text_document(file_path: str, doc_type: str) -> Document:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"File is empty: {file_path}")

    return Document(
        page_content=text,
        metadata={
            "source": path.name,
            "source_path": str(path),
            "doc_type": doc_type,
        }
    )