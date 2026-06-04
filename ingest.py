from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    UPWORK_API_DOC_PATH,
)


def load_pdf(path: Path = UPWORK_API_DOC_PATH) -> List[Document]:
    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found at {path}. Set UPWORK_API_DOC_PATH in your .env file."
        )
    loader = PyPDFLoader(str(path))
    return loader.load()


def print_sanity_check(documents: List[Document]) -> None:
    full_text = "\n\n".join(doc.page_content for doc in documents)
    print("Sanity check")
    print(f"Total character count: {len(full_text)}")
    print("Sample text:")
    print(full_text[:1000])


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    return chunks


def build_vector_store(chunks: List[Document]) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(FAISS_INDEX_DIR))
    return store


def ingest() -> None:
    documents = load_pdf()
    print_sanity_check(documents)
    chunks = split_documents(documents)
    print(f"Created chunks: {len(chunks)}")
    build_vector_store(chunks)
    print(f"Vector store written to: {FAISS_INDEX_DIR}")


if __name__ == "__main__":
    ingest()
