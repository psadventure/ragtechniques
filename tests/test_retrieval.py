import pytest

from rag import service
from retrieval.hybrid_search import reciprocal_rank_fusion


def test_rrf_rewards_agreement():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"]], k=60)
    assert scores["a"] == pytest.approx(1 / 60 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 61 + 1 / 60)
    assert scores["c"] == pytest.approx(1 / 62)   # only in one list


@pytest.fixture
def collection():
    return service.ingest("structure", "fake")["collection"]


def test_bm25_ranks_keyword_match_first(collection):
    top = service.search("idempotency key retries", collection, "bm25", top_k=1)["results"][0]
    assert top["section"] == "Creating Transfers"


@pytest.mark.parametrize("retriever", ["semantic", "bm25", "hybrid"])
def test_each_retriever_returns_ranked_results(collection, retriever):
    out = service.search("data quality risk", collection, retriever, top_k=3)
    assert [r["rank"] for r in out["results"]] == list(range(1, len(out["results"]) + 1))
    assert 0 < len(out["results"]) <= 3


def test_hybrid_exposes_trace(collection):
    out = service.search("cutover weekend", collection, "hybrid", top_k=2)
    assert set(out["trace"]) == {"bm25", "vector"}


def test_compare_grid(collection):
    out = service.compare("go-live timeline", top_k=2)
    assert {(c["collection"], c["retriever"]) for c in out["grid"]} == {
        (collection, r) for r in ("semantic", "bm25", "hybrid")
    }
