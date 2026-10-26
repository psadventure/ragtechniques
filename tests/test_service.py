import pytest

from rag import service


def test_ingest_creates_collection_per_strategy_and_embedder():
    a = service.ingest("character", "fake")
    b = service.ingest("structure", "fake")
    c = service.ingest("structure", "fake32", label="small")
    assert (a["collection"], b["collection"], c["collection"]) == (
        "character__fake", "structure__fake", "structure__fake32__small"
    )
    listed = {x["name"]: x for x in service.list_collections()}
    assert listed["structure__fake32__small"]["embedder"] == "fake32"
    assert service.inspect_collection("structure__fake32__small")["embedding_dim"] == 32


def test_reingest_replaces_document_chunks(sandbox):
    first = service.ingest("structure", "fake")["total_chunks_stored"]
    (sandbox / "docs" / "beta.md").write_text("# Beta\n\n## Only\nOne section now.\n")
    second = service.ingest("structure", "fake")["total_chunks_stored"]
    assert second < first
    beta = service.inspect_collection("structure__fake", doc_title="beta.md", limit=50)
    assert len(beta["chunks"]) == beta["chunks_per_document"]["beta.md"]


def test_reingest_with_different_params_is_refused():
    service.ingest("character", "fake")
    with pytest.raises(ValueError, match="params"):
        service.ingest("character", "fake", params={"chunk_size": 100})
    assert service.ingest("character", "fake", params={"chunk_size": 100}, label="cs100")["collection"] == "character__fake__cs100"


def test_query_uses_collection_embedder():
    service.ingest("structure", "fake32")
    # A 32-dim collection queried without the caller naming an embedder must not error.
    assert service.search("transfers", "structure__fake32", "semantic")["results"]


def test_inspect_collection_pages_chunks():
    service.ingest("character", "fake", params={"chunk_size": 100, "overlap": 10})
    out = service.inspect_collection("character__fake", limit=2, offset=1)
    assert len(out["chunks"]) == 2 and out["count"] >= 3
    assert out["chunks"][0]["section"].startswith("chars ")


def test_upload_document(sandbox):
    service.upload_document("new.md", content="# New\n\n## S\nhello\n")
    assert "new.md" in {d["name"] for d in service.list_documents()}
    with pytest.raises(ValueError):
        service.upload_document("new.md", content="again")          # no silent overwrite
    service.upload_document("new.md", content="again", overwrite=True)


@pytest.mark.parametrize("bad", ["../evil.md", "sub/x.md", "x.txt", ".hidden.md"])
def test_upload_rejects_bad_names(bad):
    with pytest.raises(ValueError):
        service.upload_document(bad, content="x")


def test_upload_from_source_path(sandbox):
    src = sandbox / "outside.md"
    src.write_text("# Outside\n")
    service.upload_document("copied.md", source_path=str(src))
    with pytest.raises(ValueError):
        service.upload_document("copied2.md", source_path=str(sandbox / "missing.md"))


def test_compare_chunking_reports_both_strategies():
    out = service.compare_chunking("alpha.md")
    assert set(out["strategies"]) == {"character", "structure"}
    assert out["strategies"]["structure"]["broken_code_blocks"] == 0


def test_delete_collection():
    service.ingest("structure", "fake")
    service.delete_collection("structure__fake")
    assert service.list_collections() == []
    with pytest.raises(RuntimeError):
        service.delete_collection("structure__fake")


def test_ingest_many_builds_every_combination():
    out = service.ingest_many(embedders=["fake", "fake32"])
    assert {c["collection"] for c in out["collections"]} == {
        "character__fake", "structure__fake", "character__fake32", "structure__fake32"
    }
    assert out["errors"] == []


def test_ingest_many_reports_errors_without_stopping():
    service.ingest("character", "fake")
    out = service.ingest_many(embedders=["fake"], params={"character": {"chunk_size": 99}})
    assert [e["strategy"] for e in out["errors"]] == ["character"]
    assert [c["collection"] for c in out["collections"]] == ["structure__fake"]


def test_compare_filters_by_strategy_and_embedder():
    service.ingest_many(embedders=["fake", "fake32"])
    out = service.compare("transfers", retrievers=["bm25"], embedder="fake32")
    assert {c["collection"] for c in out["grid"]} == {"character__fake32", "structure__fake32"}
    out = service.compare("transfers", retrievers=["bm25"], strategies=["structure"], embedder="fake")
    assert [c["collection"] for c in out["grid"]] == ["structure__fake"]
    with pytest.raises(ValueError):
        service.compare("x", embedder="nope")


def test_full_ingest_prunes_deleted_documents(sandbox):
    service.ingest("structure", "fake")
    (sandbox / "docs" / "beta.md").unlink()
    out = service.ingest("structure", "fake")
    assert out["removed_documents"] == ["beta.md"]
    assert set(service.inspect_collection("structure__fake")["chunks_per_document"]) == {"alpha.md"}


def test_partial_ingest_does_not_prune(sandbox):
    service.ingest("structure", "fake")
    out = service.ingest("structure", "fake", documents=["alpha.md"])
    assert out["removed_documents"] == []
    assert "beta.md" in service.inspect_collection("structure__fake")["chunks_per_document"]
