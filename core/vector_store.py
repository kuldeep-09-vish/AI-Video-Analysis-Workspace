import os
from pathlib import Path
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DIR = Path("vector_db")
COLLECTION_PREFIX = "meeting_transcript"
CURRENT_COLLECTION_FILE = CHROMA_DIR / "current_collection.txt"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def build_vector_store(transcript: str) -> Chroma:
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Cannot build a vector store from an empty transcript.")

    print("Building vector store...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)
    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    collection_name = f"{COLLECTION_PREFIX}_{uuid4().hex}"
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
    )
    CURRENT_COLLECTION_FILE.write_text(collection_name, encoding="utf-8")
    return vector_store


def load_vector_store() -> Chroma:
    if not CURRENT_COLLECTION_FILE.is_file():
        raise FileNotFoundError("No saved meeting vector store was found. Run an analysis first.")

    collection_name = CURRENT_COLLECTION_FILE.read_text(encoding="utf-8").strip()
    if not collection_name:
        raise RuntimeError("Saved vector-store metadata is invalid.")

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    if k <= 0:
        raise ValueError("k must be greater than zero")
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
