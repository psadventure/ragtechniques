from typing import List

import chromadb
from rank_bm25 import BM25Okapi

from retrieval.semantic_search import RetrievedChunk


def tokenize(text: str) -> List[str]:
    """Naive lowercase whitespace tokenizer (no stemming or stop-word removal)."""
    return text.lower().split()


def search(
    query: str,
    collection: chromadb.Collection,
    top_k: int | None,
) -> List[RetrievedChunk]:
    """
    Keyword search: build a BM25 index over every chunk in the collection
    and rank them against the query. top_k=None returns the full ranking.
    No embeddings are involved.
    """
    all_docs = collection.get(include=["documents", "metadatas"])
    documents: List[str] = all_docs["documents"]
    if not documents:
        return []

    bm25 = BM25Okapi([tokenize(doc) for doc in documents])
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        RetrievedChunk(
            id=all_docs["ids"][idx],
            content=documents[idx],
            metadata=all_docs["metadatas"][idx],
            score=float(scores[idx]),
            rank=rank + 1,
        )
        for rank, idx in enumerate(ranked)
    ]
