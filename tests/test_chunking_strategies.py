import pytest

from chunking.strategies import get_strategy
from tests.conftest import DOC_A


def test_strategy_stamps_name_on_chunks():
    for name in ("character", "structure"):
        chunks = get_strategy(name).split(DOC_A, "alpha.md")
        assert chunks and all(c.metadata["strategy"] == name for c in chunks)


def test_param_override_changes_chunking():
    char = get_strategy("character")
    assert len(char.split(DOC_A, "a.md", {"chunk_size": 50, "overlap": 0})) > len(char.split(DOC_A, "a.md"))


def test_unknown_param_and_strategy_rejected():
    with pytest.raises(ValueError):
        get_strategy("character").split(DOC_A, "a.md", {"max_words": 10})
    with pytest.raises(ValueError):
        get_strategy("nope")


def test_structure_keeps_code_block_intact():
    chunks = get_strategy("structure").split(DOC_A, "a.md")
    assert all(c.content.count("```") % 2 == 0 for c in chunks)
