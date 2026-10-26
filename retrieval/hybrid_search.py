from dataclasses import dataclass
from typing import Dict, List

import chromadb

from config import settings
from retrieval import bm25_search, semantic_search
from retrieval.semantic_search import RetrievedChunk
from vectorstore.embedder import Embedder


@dataclass
class HybridTrace:
    bm25: List[RetrievedChunk]     # top_k keyword matches
    vector: List[RetrievedChunk]   # top_k semantic matches
    fused: List[RetrievedChunk]    # top_k after Reciprocal Rank Fusion


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = settings.RRF_K) -> Dict[str, float]:
    """
    Merge several ranked id lists: rrf_score(d) = sum over lists of 1 / (rank_in_list(d) + k),
    with rank 0-indexed. Ids missing from a list get no contribution from it.
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + k)
    return scores


def search_with_trace(
    query: str,
    collection: chromadb.Collection,
    embedder: Embedder,
    top_k: int,
) -> HybridTrace:
    """
    Run BM25 keyword search over all chunk texts in the collection.
    Run vector similarity search via the collection (2 * top_k candidates).
    Merge both ranked lists using Reciprocal Rank Fusion (RRF).
    Return the intermediate rankings alongside the fused top_k.
    """
    bm25_ranked = bm25_search.search(query, collection, top_k=None)
    if not bm25_ranked:
        return HybridTrace(bm25=[], vector=[], fused=[])
    vector_ranked = semantic_search.search(query, collection, embedder, top_k * 2)

    rrf_scores = reciprocal_rank_fusion([
        [c.id for c in bm25_ranked],
        [c.id for c in vector_ranked],
    ])
    by_id = {c.id: c for c in bm25_ranked}
    fused_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)[:top_k]

    fused = [
        RetrievedChunk(
            id=doc_id,
            content=by_id[doc_id].content,
            metadata=by_id[doc_id].metadata,
            score=rrf_scores[doc_id],
            rank=rank + 1,
        )
        for rank, doc_id in enumerate(fused_ids)
    ]
    return HybridTrace(bm25=bm25_ranked[:top_k], vector=vector_ranked[:top_k], fused=fused)


def search(
    query: str,
    collection: chromadb.Collection,
    embedder: Embedder,
    top_k: int,
) -> List[RetrievedChunk]:
    """Hybrid BM25 + vector search; returns top_k results sorted by RRF score descending."""
    return search_with_trace(query, collection, embedder, top_k).fused
