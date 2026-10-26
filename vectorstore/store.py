"""
ChromaDB access.

One collection per (strategy, embedder[, label]) so strategies can be compared
side by side. Each collection records the embedder that built it in its metadata;
queries always re-use that embedder, so a vector of the wrong size is never sent.
"""
import json
import logging
import re
from functools import lru_cache
from typing import List

import chromadb

from chunking.character_splitter import Chunk

log = logging.getLogger(__name__)


def _serialize_metadata(metadata: dict) -> dict:
    """Convert None values and non-primitive types to strings for ChromaDB compatibility."""
    return {k: (str(v) if v is None else v) for k, v in metadata.items()}


@lru_cache(maxsize=None)
def get_client(chroma_path: str) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=chroma_path)


def collection_name(strategy: str, embedder_alias: str, label: str | None = None) -> str:
    """Build a valid Chroma collection name: '<strategy>__<embedder>[__<label>]'."""
    parts = [strategy, embedder_alias] + ([label] if label else [])
    name = "__".join(re.sub(r"[^a-zA-Z0-9._-]", "-", p) for p in parts)
    if len(name) > 63:
        raise ValueError(f"Collection name too long (max 63 chars): {name}")
    return name


def get_or_create_collection(
    chroma_path: str,
    name: str,
    strategy: str,
    embedder_alias: str,
    embedder_model: str,
    params: dict,
) -> chromadb.Collection:
    """
    Get or create the named collection and record how it was built.
    Raise ValueError if it already exists but was built with a different embedder
    (vectors would differ in size/meaning) or different chunking params (chunks
    from two configurations would be mixed).
    """
    client = get_client(chroma_path)
    params_json = json.dumps(params, sort_keys=True)
    collection = client.get_or_create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine",
            "strategy": strategy,
            "embedder": embedder_alias,
            "embedder_model": embedder_model,
            "params": params_json,
        },
    )
    existing = collection.metadata or {}
    if existing.get("embedder") != embedder_alias or existing.get("params") != params_json:
        raise ValueError(
            f"Collection '{name}' was built with embedder={existing.get('embedder')} "
            f"params={existing.get('params')}; requested embedder={embedder_alias} params={params_json}. "
            "Delete it first or pass a different label."
        )
    return collection


def replace_document_chunks(
    collection: chromadb.Collection,
    doc_title: str,
    chunks: List[Chunk],
    embeddings: List[List[float]],
) -> None:
    """
    Delete every chunk previously stored for doc_title, then insert the new ones.
    Avoids stale chunks when a re-ingest produces fewer chunks than before.
    Each document ID is: f"{doc_title}_{chunk_index}".
    """
    collection.delete(where={"doc_title": doc_title})
    if not chunks:
        return
    collection.upsert(
        ids=[f"{doc_title}_{chunk.metadata['chunk_index']}" for chunk in chunks],
        documents=[chunk.content for chunk in chunks],
        embeddings=embeddings,
        metadatas=[_serialize_metadata(chunk.metadata) for chunk in chunks],
    )
    log.info("Stored %d chunks for %s in %s", len(chunks), doc_title, collection.name)


def load_collection(chroma_path: str, name: str) -> chromadb.Collection:
    """
    Load an existing ChromaDB collection from disk.
    Raise a clear RuntimeError if the collection does not exist yet.
    """
    try:
        return get_client(chroma_path).get_collection(name=name)
    except Exception:
        raise RuntimeError(f"Collection '{name}' not found. Ingest documents with a strategy first.")


def list_collections(chroma_path: str) -> List[chromadb.Collection]:
    client = get_client(chroma_path)
    return [client.get_collection(c if isinstance(c, str) else c.name) for c in client.list_collections()]


def delete_collection(chroma_path: str, name: str) -> None:
    get_client(chroma_path).delete_collection(name=name)
