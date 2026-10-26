from agent.rag_agent import tools


def test_params_routes_overrides_to_owning_strategy():
    assert tools._params(None, None, None) == {}
    assert tools._params(300, 30, 200) == {
        "character": {"chunk_size": 300, "overlap": 30},
        "structure": {"max_words": 200},
    }


def test_tools_return_error_dict_instead_of_raising():
    out = tools.search("q", collection="does_not_exist")
    assert "error" in out and "not found" in out["error"]


def test_ingest_and_compare_via_tools():
    out = tools.ingest_documents(embedders=["fake"])
    assert {c["strategy"] for c in out["collections"]} == {"character", "structure"}
    grid = tools.compare_retrieval("transfers", embedder="fake", retrievers=["bm25"])["grid"]
    assert len(grid) == 2


def test_delete_requires_confirmation():
    from agent.rag_agent.agent import root_agent
    delete = [t for t in root_agent.tools if getattr(t, "name", "") == "delete_collection"]
    assert delete and delete[0]._require_confirmation is True
