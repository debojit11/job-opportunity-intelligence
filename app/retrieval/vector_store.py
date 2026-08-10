from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_PATH = Path("chroma_db")

COLLECTION_NAME = "job_opportunity_evidence"

EMBEDDING_MODEL = ("sentence-transformers/all-MiniLM-L6-v2")


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})


def get_vector_store():
    embeddings = get_embedding_model()

    return Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=str(CHROMA_PATH),)