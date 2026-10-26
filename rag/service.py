"""
Service layer: every RAG capability as a plain function returning JSON-friendly data.

Used by the demo scripts and exposed as tools to the ADK agent (agent/rag_agent).
Functions raise on bad input; callers decide how to surface errors.
"""
import json
import os
from pathlib import Path
from typing import List

from chunking.comparison import _compute_stats
from chunking.strategies import STRATEGIES, get_strategy
from config import settings
from pipeline.graph import run_query
from retrieval import hybrid_search
from retrieval.retrievers import RETRIEVERS, embedder_for_collection, retrieve
from retrieval.semantic_search import RetrievedChunk
from vectorstore import store
from vectorstore.embedder import EMBEDDERS, embed_chunks, get_embedder

PREVIEW_CHARS = 300


# ── helpers ──────────────────────────────────────────────────────────────

def _section_label(meta: dict) -> str:
    # The intro of an H3-split section carries its own H2 title as subsection; show it once.
    if meta.get("subsection_title") and meta["subsection_title"] != meta.get("section_title"):
        return f"{meta.get('section_title', '')} > {meta['subsection_title']}"
    if meta.get("section_title"):
        return meta["section_title"]
    if "char_start" in meta:
        return f"chars {meta['char_start']}-{meta['char_end']}"
    return "?"


def _preview(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def _chunk_view(chunk: RetrievedChunk, max_chars: int = PREVIEW_CHARS) -> dict:
    return {
        "rank": chunk.rank,
        "id": chunk.id,
        "score": round(chunk.score, 6),
        "doc_title": chunk.metadata.get("doc_title", "?"),
        "section": _section_label(chunk.metadata),
        "content": _preview(chunk.content, max_chars),
    }


def _doc_paths(documents: List[str] | None = None) -> List[Path]:
    paths = sorted(Path(settings.DOCS_PATH).glob("*.md"))
    if documents:
        wanted = set(documents)
        missing = wanted - {p.name for p in paths}
        if missing:
            raise ValueError(f"Documents not found in {settings.DOCS_PATH}/: {sorted(missing)}")
        paths = [p for p in paths if p.name in wanted]
    return paths


# ── discovery ────────────────────────────────────────────────────────────

def list_strategies() -> List[dict]:
    """Chunking strategies available for ingestion, with their tunable params and defaults."""
    return [{"name": s.name, "description": s.description, "defaults": s.defaults} for s in STRATEGIES.values()]


def list_embedders() -> List[dict]:
    """Embedding models available for ingestion."""
    return [
        {"alias": e.alias, "provider": e.provider, "model": e.model, "description": e.description}
        for e in EMBEDDERS.values() if e.provider != "test"
    ]


def list_retrievers() -> List[dict]:
    """Retrieval strategies available for querying."""
    return [{"name": name, "description": desc} for name, desc in RETRIEVERS.items()]


def list_documents() -> List[dict]:
    """Markdown documents in the docs folder that can be ingested."""
    return [{"name": p.name, "bytes": p.stat().st_size} for p in _doc_paths()]


# ── documents ────────────────────────────────────────────────────────────

def upload_document(
    filename: str,
    content: str | None = None,
    source_path: str | None = None,
    overwrite: bool = False,
) -> dict:
    """
    Save a Markdown document into the docs folder, from inline content or a local .md file.
    It is not indexed until ingest() is run.
    """
    if (content is None) == (source_path is None):
        raise ValueError("Provide exactly one of content or source_path.")
    name = os.path.basename(filename)
    if name != filename or not name.endswith(".md") or name.startswith("."):
        raise ValueError("filename must be a plain name ending in .md, with no directories (e.g. notes.md).")

    if source_path is not None:
        src = Path(source_path).expanduser()
        if not src.is_file() or src.suffix != ".md":
            raise ValueError(f"source_path must be an existing .md file: {source_path}")
        if src.stat().st_size > settings.MAX_UPLOAD_BYTES:
            raise ValueError(f"File exceeds {settings.MAX_UPLOAD_BYTES} bytes.")
        content = src.read_text(encoding="utf-8")
    elif len(content.encode("utf-8")) > settings.MAX_UPLOAD_BYTES:
        raise ValueError(f"Content exceeds {settings.MAX_UPLOAD_BYTES} bytes.")

    target = Path(settings.DOCS_PATH) / name
    if target.exists() and not overwrite:
        raise ValueError(f"{name} already exists; pass overwrite=True to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"saved": str(target), "bytes": target.stat().st_size, "next": "Run ingest to index this document."}


# ── chunking (no DB) ─────────────────────────────────────────────────────

def compare_chunking(document: str, params: dict | None = None) -> dict:
    """
    Chunk one document with every strategy and report stats (chunk count, avg words,
    broken code blocks, broken tables). Nothing is embedded or stored.
    params: optional {strategy_name: {param: value}} overrides.
    """
    path = _doc_paths([document])[0]
    text = path.read_text(encoding="utf-8")
    results = {}
    for name, strategy in STRATEGIES.items():
        chunks = strategy.split(text, path.name, (params or {}).get(name))
        results[name] = {"params": {**strategy.defaults, **(params or {}).get(name, {})}, **_compute_stats(chunks)}
    return {"document": path.name, "strategies": results}


# ── ingestion ────────────────────────────────────────────────────────────

def ingest(
    strategy: str,
    embedder: str | None = None,
    params: dict | None = None,
    label: str | None = None,
    documents: List[str] | None = None,
) -> dict:
    """
    Chunk docs with the strategy, embed with the embedder, and store them in the
    collection '<strategy>__<embedder>[__<label>]'. Re-ingesting a document replaces
    its previous chunks in that collection. A full ingest (documents=None) also
    removes chunks of documents that no longer exist in the docs folder.
    """
    strat = get_strategy(strategy)
    emb_alias = embedder or settings.DEFAULT_EMBEDDER
    emb_spec = EMBEDDERS.get(emb_alias)
    if emb_spec is None:
        raise ValueError(f"Unknown embedder '{emb_alias}'. Available: {sorted(EMBEDDERS)}")
    merged_params = {**strat.defaults, **(params or {})}
    paths = _doc_paths(documents)
    if not paths:
        raise ValueError(f"No .md documents found in {settings.DOCS_PATH}/")

    name = store.collection_name(strategy, emb_alias, label)
    collection = store.get_or_create_collection(
        settings.CHROMA_PATH, name, strategy, emb_alias, emb_spec.model, merged_params
    )
    emb = get_embedder(emb_alias)

    per_doc = {}
    all_chunks = []
    for path in paths:
        chunks = strat.split(path.read_text(encoding="utf-8"), path.name, params)
        store.replace_document_chunks(collection, path.name, chunks, embed_chunks(chunks, emb) if chunks else [])
        per_doc[path.name] = len(chunks)
        all_chunks.extend(chunks)

    removed: List[str] = []
    if documents is None:
        stored = {m.get("doc_title") for m in collection.get(include=["metadatas"])["metadatas"]}
        removed = sorted(t for t in stored - set(per_doc) if t)
        for title in removed:
            collection.delete(where={"doc_title": title})

    return {
        "collection": name,
        "strategy": strategy,
        "embedder": emb_alias,
        "params": merged_params,
        "chunks_per_document": per_doc,
        "removed_documents": removed,
        "total_chunks_stored": collection.count(),
        "stats": _compute_stats(all_chunks),
    }


def ingest_many(
    strategies: List[str] | None = None,
    embedders: List[str] | None = None,
    documents: List[str] | None = None,
    label: str | None = None,
    params: dict | None = None,
) -> dict:
    """
    Ingest with every strategy x embedder combination (defaults: all strategies,
    the default embedder), one collection each, so they can be compared.
    params: optional {strategy_name: {param: value}} overrides.
    A failing combination is reported in 'errors' and does not stop the others.
    """
    results, errors = [], []
    for strategy in strategies or list(STRATEGIES):
        for emb in embedders or [settings.DEFAULT_EMBEDDER]:
            try:
                r = ingest(strategy, emb, (params or {}).get(strategy), label, documents)
                results.append({
                    "collection": r["collection"],
                    "strategy": strategy,
                    "embedder": emb,
                    "params": r["params"],
                    "total_chunks_stored": r["total_chunks_stored"],
                    "removed_documents": r["removed_documents"],
                    **r["stats"],
                })
            except Exception as exc:
                errors.append({"strategy": strategy, "embedder": emb, "error": str(exc)})
    return {"collections": results, "errors": errors}


# ── collections ──────────────────────────────────────────────────────────

def list_collections() -> List[dict]:
    """Every Chroma collection with its chunk count and how it was built."""
    out = []
    for c in store.list_collections(settings.CHROMA_PATH):
        meta = c.metadata or {}
        out.append({
            "name": c.name,
            "count": c.count(),
            "strategy": meta.get("strategy"),
            "embedder": meta.get("embedder"),
            "embedder_model": meta.get("embedder_model"),
            "params": json.loads(meta["params"]) if meta.get("params") else None,
        })
    return sorted(out, key=lambda c: c["name"])


def inspect_collection(
    name: str,
    limit: int = 5,
    offset: int = 0,
    doc_title: str | None = None,
    max_chars: int = PREVIEW_CHARS,
) -> dict:
    """Stats for a collection plus a page of stored chunks (content + metadata)."""
    collection = store.load_collection(settings.CHROMA_PATH, name)
    everything = collection.get(include=["documents", "metadatas"])
    docs, metas = everything["documents"], everything["metadatas"]

    per_doc: dict = {}
    for m in metas:
        per_doc[m.get("doc_title", "?")] = per_doc.get(m.get("doc_title", "?"), 0) + 1
    words = [len(d.split()) for d in docs]
    first = collection.get(limit=1, include=["embeddings"])
    dim = len(first["embeddings"][0]) if len(first["ids"]) else None

    where = {"doc_title": doc_title} if doc_title else None
    page = collection.get(where=where, limit=limit, offset=offset, include=["documents", "metadatas"])
    meta = collection.metadata or {}
    return {
        "name": name,
        "strategy": meta.get("strategy"),
        "embedder": meta.get("embedder"),
        "embedding_dim": dim,
        "params": json.loads(meta["params"]) if meta.get("params") else None,
        "count": len(docs),
        "chunks_per_document": per_doc,
        "avg_words": round(sum(words) / len(words), 1) if words else 0,
        "min_words": min(words, default=0),
        "max_words": max(words, default=0),
        "broken_code_blocks": sum(1 for d in docs if d.count("```") % 2 != 0),
        "chunks": [
            {"id": i, "section": _section_label(m), "metadata": m, "content": _preview(d, max_chars)}
            for i, d, m in zip(page["ids"], page["documents"], page["metadatas"])
        ],
    }


def delete_collection(name: str) -> dict:
    store.load_collection(settings.CHROMA_PATH, name)   # raises a clear error if missing
    store.delete_collection(settings.CHROMA_PATH, name)
    return {"deleted": name}


# ── retrieval & answering ────────────────────────────────────────────────

def search(
    query: str,
    collection: str,
    retriever: str = "hybrid",
    top_k: int = settings.TOP_K,
    max_chars: int = PREVIEW_CHARS,
) -> dict:
    """
    Retrieve top_k chunks only (no LLM). For hybrid, also returns the BM25 and
    vector rankings that were fused, so the effect of RRF is visible.
    """
    coll = store.load_collection(settings.CHROMA_PATH, collection)
    result = {"query": query, "collection": collection, "retriever": retriever}
    if retriever == "hybrid":
        trace = hybrid_search.search_with_trace(query, coll, embedder_for_collection(coll), top_k)
        result["results"] = [_chunk_view(c, max_chars) for c in trace.fused]
        result["trace"] = {
            "bm25": [_chunk_view(c, 120) for c in trace.bm25],
            "vector": [_chunk_view(c, 120) for c in trace.vector],
        }
    else:
        result["results"] = [_chunk_view(c, max_chars) for c in retrieve(query, coll, retriever, top_k)]
    return result


def answer(
    query: str,
    collection: str,
    retriever: str = "hybrid",
    top_k: int = settings.TOP_K,
    chat_model: str | None = None,
) -> dict:
    """Full RAG: retrieve, build a grounded prompt, generate a cited answer."""
    result, chunks = run_query(query, collection, retriever, top_k, chat_model)
    return {
        "query": query,
        "collection": collection,
        "retriever": retriever,
        "chat_model": chat_model or settings.DEFAULT_CHAT_MODEL,
        "answer": result.text,
        "citations": result.citations,
        "chunks": [_chunk_view(c, 120) for c in chunks],
    }


def compare(
    query: str,
    collections: List[str] | None = None,
    retrievers: List[str] | None = None,
    top_k: int = settings.TOP_K,
    strategies: List[str] | None = None,
    embedder: str | None = None,
) -> dict:
    """
    Run one query across collections x retrievers and return the top_k for each cell.
    Collections: the explicit list if given, else every collection matching the
    strategies / embedder filters (no filters = all collections).
    """
    if collections:
        names = collections
    else:
        names = [
            c["name"] for c in list_collections()
            if (not strategies or c["strategy"] in strategies) and (not embedder or c["embedder"] == embedder)
        ]
        if not names:
            raise ValueError(f"No collections match strategies={strategies} embedder={embedder}. Ingest first.")
    grid = []
    for name in names:
        coll = store.load_collection(settings.CHROMA_PATH, name)
        for r in retrievers or list(RETRIEVERS):
            grid.append({
                "collection": name,
                "retriever": r,
                "results": [_chunk_view(c, 120) for c in retrieve(query, coll, r, top_k)],
            })
    return {"query": query, "grid": grid}
