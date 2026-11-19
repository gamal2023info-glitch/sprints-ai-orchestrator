"""Curriculum ingestion + retrieval (Sprint 1: LangChain loaders, chunking, ChromaDB).

Curriculum mapping: "RAG fundamentals" - load -> split (500 tokens / 50 overlap)
-> embed -> persist in Chroma -> timed retrieval (SLA 1000 ms).
"""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import (CHROMA_DIR, CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, CURRICULUM_DIR,
                        EMBED_MODEL, RAG_TOP_K, RETRIEVAL_SLA_MS, UNABLE_TO_VERIFY)

COLLECTION = "sprints_curriculum"


@lru_cache(maxsize=1)
def get_embeddings():
    """Local ONNX embeddings (no API call, no cost, fast enough for the 1000 ms SLA)."""
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name=EMBED_MODEL)


def load_and_chunk(docs_dir: Path = CURRICULUM_DIR):
    """Load .md/.txt curriculum files and split into 500-token chunks, 50-token overlap."""
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = []
    for pattern in ("**/*.md", "**/*.txt"):
        docs += DirectoryLoader(str(docs_dir), glob=pattern, loader_cls=TextLoader,
                                loader_kwargs={"encoding": "utf-8"}).load()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=CHUNK_SIZE_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS)
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["source"] = Path(c.metadata.get("source", "unknown")).name
    return chunks


def ingest_curriculum(docs_dir: Path = CURRICULUM_DIR, persist_dir: Path = CHROMA_DIR) -> int:
    """Embed all chunks and persist them to ChromaDB. Returns the number of chunks stored."""
    from langchain_chroma import Chroma

    chunks = load_and_chunk(docs_dir)
    store = Chroma(collection_name=COLLECTION, embedding_function=get_embeddings(),
                   persist_directory=str(persist_dir))
    try:
        store.delete_collection()
    except Exception:
        pass
    Chroma.from_documents(chunks, get_embeddings(),
                          collection_name=COLLECTION, persist_directory=str(persist_dir))
    get_store.cache_clear()
    return len(chunks)


@lru_cache(maxsize=1)
def get_store():
    """Cached vector store handle (avoids re-opening Chroma on every query - latency)."""
    from langchain_chroma import Chroma

    return Chroma(collection_name=COLLECTION, embedding_function=get_embeddings(),
                  persist_directory=str(CHROMA_DIR))


def warm_up() -> None:
    """Load the embedding model + open Chroma once so the 1000 ms SLA measures steady state, not cold start."""
    try:
        get_store().similarity_search("warm up", k=1)
    except Exception:
        pass


def retrieve_curriculum(query: str, k: int = RAG_TOP_K) -> dict[str, Any]:
    """Timed similarity search over the Sprints curriculum.

    Returns passages + latency_ms + within_sla (<= 1000 ms). Never raises: on any
    failure returns the structured 'Unable to verify' payload.
    """
    t0 = time.perf_counter()
    try:
        hits = get_store().similarity_search(query, k=k)
    except Exception as exc:
        return {**UNABLE_TO_VERIFY, "error": type(exc).__name__, "passages": []}
    ms = round((time.perf_counter() - t0) * 1000, 1)
    return {"status": "ok", "latency_ms": ms, "within_sla": ms <= RETRIEVAL_SLA_MS,
            "passages": [{"source": h.metadata.get("source", "?"), "text": h.page_content} for h in hits]}
