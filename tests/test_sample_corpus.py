"""
Tests against the real sample corpus in docs/ (fictional Kestrel bank platform).

They pin down the lessons the corpus is designed to teach, so editing the docs
can't silently remove them. Retrieval runs offline: BM25 needs no embeddings.
"""
from pathlib import Path

import pytest

from chunking.comparison import _compute_stats
from chunking.strategies import STRATEGIES
from config import settings
from rag import service
from retrieval.retrievers import retrieve
from vectorstore.store import load_collection

DOCS = Path(settings.PROJECT_ROOT) / "docs"

# (question, a fact that only a chunk answering it contains)
GOLDEN = [
    ("How long is an access token valid?", "3600 seconds"),
    ("How do I verify the webhook signature and prevent replay attacks?", "more than 300 seconds old"),
    ("What should I do when I hit the rate limit and get HTTP 429?", "Retry-After"),
    ("Can I cancel a transfer that is already processing?", "only while it is `pending`"),
    ("What does error idempotency_conflict mean?", "reused with a different body"),
    ("Why did the Fraud Shield pilot start late?", "fraud history"),
    ("What is the rollback decision point during cutover weekend?", "Saturday 20:00"),
    ("Which loan applications are auto-approved by credit score and debt-to-income?", "700 or higher"),
    ("When does the end-of-day close run and how is interest accrued?", "23:30 Eastern"),
    ("How many points does the dormant reactivation rule add?", "+350"),
    ("What was the reconciliation variance in each mock conversion?", "$41,207.13"),
    ("Why was go-live moved to October?", "PCI DSS audit found"),
]


@pytest.fixture
def corpus(monkeypatch):
    """Use the real docs/ folder (Chroma stays in the per-test temp dir)."""
    monkeypatch.setattr(settings, "DOCS_PATH", str(DOCS))
    return sorted(DOCS.glob("*.md"))


def _hits_at_3(collection_name: str) -> int:
    coll = load_collection(settings.CHROMA_PATH, collection_name)
    return sum(
        any(fact in c.content for c in retrieve(question, coll, "bm25", 3))
        for question, fact in GOLDEN
    )


def test_corpus_is_marked_fictional(corpus):
    assert len(corpus) == 3
    for path in corpus:
        assert "Fictional sample document" in path.read_text()


def test_every_golden_fact_exists_in_the_corpus(corpus):
    text = "".join(p.read_text() for p in corpus)
    missing = [fact for _, fact in GOLDEN if fact not in text]
    assert not missing


def test_structure_keeps_code_blocks_intact_character_breaks_them(corpus):
    broken = {name: 0 for name in STRATEGIES}
    for path in corpus:
        for name, strategy in STRATEGIES.items():
            broken[name] += _compute_stats(strategy.split(path.read_text(), path.name))["broken_code_blocks"]
    assert broken["structure"] == 0
    assert broken["character"] >= 5


def test_each_document_has_a_section_split_at_h3(corpus):
    for path in corpus:
        chunks = STRATEGIES["structure"].split(path.read_text(), path.name)
        assert any(c.metadata["subsection_title"] for c in chunks), path.name


def test_structure_chunks_retrieve_answers_better_than_character(corpus):
    service.ingest_many(embedders=["fake"])
    structure, character = _hits_at_3("structure__fake"), _hits_at_3("character__fake")
    assert structure >= 10, f"structure hit@3 {structure}/{len(GOLDEN)}"
    assert structure > character, f"structure {structure} vs character {character}"
