from dataclasses import dataclass
from typing import List

import chromadb

from vectorstore.embedder import Embedder


@dataclass
class RetrievedChunk:
    id: str
    content: str
    metadata: dict
    score: float   # semantic: cosine distance (lower = more similar); bm25/hybrid: higher = better
    rank: int      # 1-indexed position in results


def search(
    query: str,
    collection: chromadb.Collection,
    embedder: Embedder,
    top_k: int,
) -> List[RetrievedChunk]:
    """
    Embed the query, query the ChromaDB collection, return top_k results
    as RetrievedChunk objects ranked by similarity score.
    """
    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[embedder.embed_query(query)],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    chunks: List[RetrievedChunk] = []
    for rank, (doc_id, doc, meta, dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        chunks.append(RetrievedChunk(
            id=doc_id,
            content=doc,
            metadata=meta,
            score=dist,
            rank=rank + 1,
        ))

    return chunks
