"""Pick a retrieval strategy by name; the embedder comes from the collection's own metadata."""
from typing import Dict, List

import chromadb

from retrieval import bm25_search, hybrid_search, semantic_search
from retrieval.semantic_search import RetrievedChunk
from vectorstore.embedder import Embedder, get_embedder

RETRIEVERS: Dict[str, str] = {
    "semantic": "Dense vector similarity (cosine) using the collection's embedder.",
    "bm25": "Sparse keyword ranking (BM25 Okapi) over chunk text. No embeddings.",
    "hybrid": "BM25 + semantic merged with Reciprocal Rank Fusion (k=60).",
}


def embedder_for_collection(collection: chromadb.Collection) -> Embedder:
    alias = (collection.metadata or {}).get("embedder")
    if not alias:
        raise RuntimeError(f"Collection '{collection.name}' has no embedder recorded; re-ingest it.")
    return get_embedder(alias)


def retrieve(query: str, collection: chromadb.Collection, retriever: str, top_k: int) -> List[RetrievedChunk]:
    if retriever == "bm25":
        return bm25_search.search(query, collection, top_k)
    if retriever == "semantic":
        return semantic_search.search(query, collection, embedder_for_collection(collection), top_k)
    if retriever == "hybrid":
        return hybrid_search.search(query, collection, embedder_for_collection(collection), top_k)
    raise ValueError(f"Unknown retriever '{retriever}'. Available: {sorted(RETRIEVERS)}")
