from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


KNOWN_SECTIONS = {
    "NAME",
    "SUMMARY",
    "SKILLS",
    "PROJECTS",
    "EXPERIENCE",
    "EDUCATION",

    "ROLE",
    "COMPANY",
    "LOCATION",
    "REQUIRED SKILLS",
    "PREFERRED SKILLS",
    "RESPONSIBILITIES",

    "ABOUT",
    "CAREERS",
    "KNOWN INFORMATION",
    "NOTES",
}

SKIP_INDEX_SECTIONS = {
    "name",
}

MIN_INDEX_CHARS = 10

def normalize_section_name(name: str) -> str:
    return (name.strip().lower().replace(" ", "_"))


def split_into_sections(document: Document) -> list[Document]:

    lines = document.page_content.splitlines()

    sections = []

    current_section = "general"
    current_lines = []

    def save_section():
        if not current_lines:
            return

        content = "\n".join(current_lines).strip()

        if not content:
            return

        sections.append(
            Document(page_content=content,
                metadata={**document.metadata, "section": current_section})
        )

    for line in lines:

        stripped = line.strip()

        if stripped.upper() in KNOWN_SECTIONS:

            save_section()

            current_section = (normalize_section_name(stripped))

            current_lines = []

        else:
            current_lines.append(line)

    save_section()

    return sections


def chunk_documents(documents: list[Document], chunk_size: int = 700, chunk_overlap: int = 120,) -> list[Document]:

    indexable_documents = [doc for doc in documents
        if (
            doc.metadata.get("section") not in SKIP_INDEX_SECTIONS
            and len(doc.page_content.strip()) >= MIN_INDEX_CHARS
        )
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(indexable_documents)

    counters = {}

    for chunk in chunks:
        doc_type = chunk.metadata.get("doc_type", "unknown")
        section = chunk.metadata.get("section", "general")

        key = f"{doc_type}_{section}"
        current = counters.get(key, 0)

        chunk.metadata["chunk_id"] = f"{key}_{current}"
        chunk.metadata["char_count"] = len(chunk.page_content)

        counters[key] = current + 1

    return chunks