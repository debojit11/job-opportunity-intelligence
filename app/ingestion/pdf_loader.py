from langchain_pymupdf4llm import PyMuPDF4LLMLoader


def load_pdf(file_path: str,):
    
    loader = PyMuPDF4LLMLoader(file_path=file_path, mode="page",)

    return loader.load()