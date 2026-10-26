import os
from pathlib import Path

from dotenv import load_dotenv

# Anchor paths to the project root so demos, tests and `adk web` work from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


DOCS_PATH = str(PROJECT_ROOT / "docs")
CHROMA_PATH = str(PROJECT_ROOT / "chroma_data")
EMBED_MODEL = "all-MiniLM-L6-v2"

# Defaults below can be overridden in .env (see .env.example); per-experiment
# overrides (agent / demo flags) take precedence over both.

# Chunking
CHAR_CHUNK_SIZE = _env_int("RAG_CHAR_CHUNK_SIZE", 500)
CHAR_CHUNK_OVERLAP = _env_int("RAG_CHAR_CHUNK_OVERLAP", 50)
SEMANTIC_MAX_WORDS = _env_int("RAG_STRUCTURE_MAX_WORDS", 400)   # sections above this trigger H3 splitting

# Retrieval
TOP_K = _env_int("RAG_TOP_K", 5)
RRF_K = _env_int("RAG_RRF_K", 60)                               # Reciprocal Rank Fusion constant

# LiteLLM gateway (see infra/litellm/). Model names are the gateway's `model_name` aliases.
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")
DEFAULT_EMBEDDER = os.getenv("RAG_EMBEDDER", "minilm")
DEFAULT_CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gemini-2.5-flash")
GENERATOR = os.getenv("RAG_GENERATOR", "litellm")   # litellm | anthropic | openai

# Uploads
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
