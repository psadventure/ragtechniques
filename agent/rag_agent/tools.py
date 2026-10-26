"""
ADK tool wrappers around rag/service.py.

Each tool returns the service result, or {"error": "..."} instead of raising, so
the model can read the problem and recover (e.g. ingest first, fix a name).
Docstrings and type hints become the tool schema the model sees.
"""
import functools
from typing import Optional

from google.adk.tools import ToolContext

from rag import service


def _safe(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
    return wrapper


def _params(chunk_size: Optional[int], overlap: Optional[int], max_words: Optional[int]) -> dict:
    """Route per-strategy chunking overrides to the strategy that owns them."""
    out: dict = {}
    char = {k: v for k, v in {"chunk_size": chunk_size, "overlap": overlap}.items() if v is not None}
    if char:
        out["character"] = char
    if max_words is not None:
        out["structure"] = {"max_words": max_words}
    return out


@_safe
def list_options() -> dict:
    """
    Show everything available: chunking strategies (with default params), embedders,
    retrievers, documents in docs/, and existing collections. Call this first when
    unsure what exists.
    """
    return {
        "strategies": service.list_strategies(),
        "embedders": service.list_embedders(),
        "retrievers": service.list_retrievers(),
        "documents": service.list_documents(),
        "collections": service.list_collections(),
    }


@_safe
def compare_chunking(
    document: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    max_words: Optional[int] = None,
) -> dict:
    """
    Chunk one document (e.g. "kestrel_api_reference.md") with every strategy and return
    stats: chunk count, avg words, broken code blocks, broken tables. Nothing is stored.
    Optional overrides: chunk_size/overlap (character strategy), max_words (structure strategy).
    """
    return service.compare_chunking(document, _params(chunk_size, overlap, max_words))


@_safe
def ingest_documents(
    strategies: Optional[list[str]] = None,
    embedders: Optional[list[str]] = None,
    documents: Optional[list[str]] = None,
    label: Optional[str] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    max_words: Optional[int] = None,
) -> dict:
    """
    Chunk, embed and store documents. Creates one collection per strategy x embedder,
    named "<strategy>__<embedder>[__<label>]".
    strategies: e.g. ["character", "structure"]; omit for all strategies.
    embedders: e.g. ["text-embedding-005"]; omit for the default embedder.
    documents: file names in docs/; omit for all.
    label: suffix to keep an experiment separate, required when changing chunk params
      of an already-ingested strategy (e.g. label="cs300" with chunk_size=300).
    chunk_size/overlap apply to "character"; max_words applies to "structure".
    Hosted embedders take a few seconds per strategy (gemini-embedding-001 is slowest).
    """
    return service.ingest_many(strategies, embedders, documents, label, _params(chunk_size, overlap, max_words))


@_safe
def list_collections() -> list:
    """List stored collections with chunk count, strategy, embedder and chunking params."""
    return service.list_collections()


@_safe
def inspect_collection(
    name: str,
    limit: int = 5,
    offset: int = 0,
    doc_title: Optional[str] = None,
) -> dict:
    """
    Look inside a collection: stats (count, embedding dimension, chunks per document,
    avg/min/max words, broken code blocks) plus a page of stored chunks with metadata.
    Use offset to page; doc_title to filter to one document.
    """
    return service.inspect_collection(name, limit, offset, doc_title)


@_safe
def search(query: str, collection: str, retriever: str = "hybrid", top_k: int = 5) -> dict:
    """
    Retrieve the top_k chunks for a query from one collection, without generating an answer.
    retriever: "semantic" (dense vectors), "bm25" (keywords) or "hybrid" (RRF fusion of both).
    For hybrid, 'trace' shows the BM25 and vector rankings that were fused.
    """
    return service.search(query, collection, retriever, top_k)


@_safe
def answer_question(query: str, collection: str, retriever: str = "hybrid", top_k: int = 5) -> dict:
    """
    Full RAG: retrieve from the collection, then have the LLM answer using only those
    chunks, with citations. Returns the answer, citations and the chunks used.
    """
    return service.answer(query, collection, retriever, top_k)


@_safe
def compare_retrieval(
    query: str,
    collections: Optional[list[str]] = None,
    retrievers: Optional[list[str]] = None,
    strategies: Optional[list[str]] = None,
    embedder: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """
    Run one query across several collections x retrievers and return the top_k for each,
    for side-by-side comparison.
    Pick collections explicitly, or filter by strategies (e.g. ["character", "structure"])
    and/or embedder (e.g. "text-embedding-005"). No filters = every collection.
    retrievers: omit for all three.
    """
    return service.compare(query, collections, retrievers, top_k, strategies, embedder)


@_safe
def upload_document(filename: str, content: str, overwrite: bool = False) -> dict:
    """
    Save Markdown text the user pasted into docs/<filename> (must end in .md).
    It is not searchable until ingested.
    """
    return service.upload_document(filename, content=content, overwrite=overwrite)


async def save_uploaded_file(artifact_name: str, tool_context: ToolContext, filename: Optional[str] = None,
                             overwrite: bool = False) -> dict:
    """
    Save a file the user attached in the chat (shown to you as [Uploaded Artifact: "<name>"])
    into docs/. filename defaults to the artifact name and must end in .md.
    It is not searchable until ingested.
    """
    try:
        part = await tool_context.load_artifact(artifact_name)
        if part is None:
            return {"error": f"No uploaded file named '{artifact_name}' in this session."}
        if part.inline_data is not None:
            content = part.inline_data.data.decode("utf-8")
        elif part.text is not None:
            content = part.text
        else:
            return {"error": "Uploaded file has no readable content."}
        return service.upload_document(filename or artifact_name, content=content, overwrite=overwrite)
    except UnicodeDecodeError:
        return {"error": "Only UTF-8 Markdown (.md) files are supported."}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@_safe
def delete_collection(name: str) -> dict:
    """Permanently delete a collection and its embeddings. Source docs are untouched."""
    return service.delete_collection(name)
