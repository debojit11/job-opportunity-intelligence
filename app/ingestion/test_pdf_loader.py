from app.ingestion.pdf_loader import (load_pdf,)


def main():
    file_path = "data/sample_resume.pdf"

    documents = load_pdf(file_path)

    print(f"Loaded {len(documents)} document(s)")

    for i, doc in enumerate(documents, start=1,):
        print("\n" + "=" * 80)
        print(f"DOCUMENT {i}")
        print("=" * 80)

        print("METADATA:")
        print(doc.metadata)

        print("\nCONTENT:")
        print(doc.page_content[:2000])


if __name__ == "__main__":
    main()