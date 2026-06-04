from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    DEEPINFRA_API_KEY,
    DEEPINFRA_CHAT_URL,
    DEEPINFRA_MODEL,
    EMBEDDING_MODEL,
    FAISS_INDEX_DIR,
    TOP_K,
)
from ingest import ingest


HALLUCINATION_FALLBACK = (
    "I'm sorry, but the provided documentation does not contain that information."
)

SYSTEM_PROMPT = f"""You are a Senior Upwork API Consultant.
Answer developer questions using only the retrieved documentation snippets.
Do not use outside knowledge.
If the retrieved snippets do not contain the answer, reply exactly:
{HALLUCINATION_FALLBACK}
Keep answers concise and cite the relevant source numbers when possible."""


@dataclass
class RagResponse:
    answer: str
    sources: List[Document]
    latency_seconds: float


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


import streamlit as st

@st.cache_resource
def get_vector_store() -> FAISS:
    if not Path(FAISS_INDEX_DIR).exists():
        ingest()
    return FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings=_get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def retrieve_relevant_chunks(query: str, k: int = TOP_K) -> List[Document]:
    store = get_vector_store()
    return store.similarity_search(query, k=k)


def format_context(sources: List[Document]) -> str:
    blocks = []
    for idx, doc in enumerate(sources, start=1):
        page = doc.metadata.get("page")
        page_label = f"page {page + 1}" if isinstance(page, int) else "unknown page"
        blocks.append(f"[Source {idx} | {page_label}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def call_deepinfra(question: str, sources: List[Document]) -> tuple[str, float]:
    if not DEEPINFRA_API_KEY:
        raise RuntimeError("DEEPINFRA_API_KEY is missing. Add it to your .env file.")

    context = format_context(sources)
    user_prompt = f"""Retrieved documentation:
{context}

Question:
{question}

Answer using only the retrieved documentation."""

    payload = {
        "model": DEEPINFRA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 500,
    }
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json",
    }

    start = time.perf_counter()
    response = requests.post(
        DEEPINFRA_CHAT_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    latency = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    answer = data["choices"][0]["message"]["content"].strip()
    return answer, latency


def answer_question(question: str) -> RagResponse:
    sources = retrieve_relevant_chunks(question)
    answer, latency = call_deepinfra(question, sources)
    return RagResponse(answer=answer, sources=sources, latency_seconds=latency)
