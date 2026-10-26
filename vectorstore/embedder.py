"""
Embedders, selectable by alias.

- "minilm"               : local sentence-transformers model, no network needed.
- "text-embedding-005"   : Vertex AI, via the LiteLLM gateway.
- "gemini-embedding-001" : Vertex AI, via the LiteLLM gateway.

The LiteLLM aliases must match `model_name` entries in infra/litellm/config/config.yaml.
"""
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Protocol

from sentence_transformers import SentenceTransformer

from chunking.character_splitter import Chunk
from config import settings

log = logging.getLogger(__name__)


class Embedder(Protocol):
    alias: str
    model: str

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...


class LocalEmbedder:
    def __init__(self, alias: str, model: str):
        self.alias = alias
        self.model = model
        self._st = SentenceTransformer(model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [emb.tolist() for emb in self._st.encode(texts, show_progress_bar=False)]

    def embed_query(self, text: str) -> List[float]:
        return self._st.encode([text])[0].tolist()


class LiteLLMEmbedder:
    """Calls the LiteLLM gateway's OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self, alias: str, model: str, batch_size: int):
        from openai import OpenAI

        self.alias = alias
        self.model = model
        self.batch_size = batch_size
        self._client = OpenAI(base_url=settings.LITELLM_BASE_URL, api_key=settings.LITELLM_API_KEY or "missing")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


@dataclass
class EmbedderSpec:
    alias: str
    provider: str          # "local" | "litellm" | "test"
    model: str
    description: str
    factory: Callable[[], Embedder]


EMBEDDERS: Dict[str, EmbedderSpec] = {}


def register_embedder(spec: EmbedderSpec) -> None:
    EMBEDDERS[spec.alias] = spec
    get_embedder.cache_clear()


@lru_cache(maxsize=None)
def get_embedder(alias: str) -> Embedder:
    """Return a cached embedder instance for the alias (models load once per process)."""
    if alias not in EMBEDDERS:
        raise ValueError(f"Unknown embedder '{alias}'. Available: {sorted(EMBEDDERS)}")
    return EMBEDDERS[alias].factory()


register_embedder(EmbedderSpec(
    alias="minilm",
    provider="local",
    model=settings.EMBED_MODEL,
    description="Local sentence-transformers all-MiniLM-L6-v2 (384 dims). Offline, fast, small.",
    factory=lambda: LocalEmbedder("minilm", settings.EMBED_MODEL),
))
register_embedder(EmbedderSpec(
    alias="text-embedding-005",
    provider="litellm",
    model="text-embedding-005",
    description="Vertex AI text-embedding-005 (768 dims) via LiteLLM. English/code tuned.",
    factory=lambda: LiteLLMEmbedder("text-embedding-005", "text-embedding-005", batch_size=100),
))
register_embedder(EmbedderSpec(
    alias="gemini-embedding-001",
    provider="litellm",
    model="gemini-embedding-001",
    description="Vertex AI gemini-embedding-001 (3072 dims) via LiteLLM. Strongest quality, slower.",
    # Vertex accepts a single input per request for this model.
    factory=lambda: LiteLLMEmbedder("gemini-embedding-001", "gemini-embedding-001", batch_size=1),
))


def embed_chunks(chunks: List[Chunk], embedder: Embedder) -> List[List[float]]:
    """Embed a list of Chunk objects; each vector corresponds to the Chunk at the same index."""
    log.info("Embedding %d chunks with %s (%s)", len(chunks), embedder.alias, embedder.model)
    return embedder.embed_documents([chunk.content for chunk in chunks])
