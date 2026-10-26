"""
Registry of chunking strategies.

Each strategy wraps one splitter with its default parameters so callers
(demo scripts, the service layer, the ADK agent) can pick a strategy by name
and optionally override its parameters.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List

from chunking.character_splitter import Chunk, split_by_characters
from chunking.semantic_splitter import split_by_headers
from config import settings


@dataclass
class Strategy:
    name: str
    description: str
    defaults: Dict[str, int]
    splitter: Callable[..., List[Chunk]] = field(repr=False)

    def split(self, text: str, doc_title: str, params: Dict[str, int] | None = None) -> List[Chunk]:
        """Split text with the strategy's defaults, overridden by params. Stamps metadata['strategy']."""
        merged = {**self.defaults, **(params or {})}
        unknown = set(merged) - set(self.defaults)
        if unknown:
            raise ValueError(f"Unknown params for strategy '{self.name}': {sorted(unknown)}")
        chunks = self.splitter(text, doc_title, **merged)
        for chunk in chunks:
            chunk.metadata["strategy"] = self.name
        return chunks


STRATEGIES: Dict[str, Strategy] = {
    "character": Strategy(
        name="character",
        description=(
            "Fixed-size character windows with overlap. Ignores sentences, sections, "
            "code blocks and tables — the 'bad' baseline."
        ),
        defaults={"chunk_size": settings.CHAR_CHUNK_SIZE, "overlap": settings.CHAR_CHUNK_OVERLAP},
        splitter=split_by_characters,
    ),
    "structure": Strategy(
        name="structure",
        description=(
            "Markdown structure-based: split at ## headers, and split sections longer "
            "than max_words again at ### headers. Keeps code blocks and tables intact."
        ),
        defaults={"max_words": settings.SEMANTIC_MAX_WORDS},
        splitter=split_by_headers,
    ),
}


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Available: {sorted(STRATEGIES)}")
    return STRATEGIES[name]
