import hashlib
import math

import pytest

from config import settings
from vectorstore.embedder import EMBEDDERS, EmbedderSpec, register_embedder

DOC_A = """# Mini Kestrel API

## Authentication
Exchange the client id and secret for an access token at the token endpoint.

## Creating Transfers
Send POST /transfers with an idempotency key header so retries never pay twice.

```python
client.transfers.create(idempotency_key=key, amount=amount, rail="ach")
```
"""

DOC_B = """# Mini Heron Plan

## Timeline
Kickoff is in January and go-live is the October cutover weekend.

## Risks
Legacy data quality is the highest risk.
"""


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for tests."""

    def __init__(self, alias="fake", dim=64):
        self.alias = alias
        self.model = f"hash-{dim}"
        self.dim = dim

    def _vec(self, text):
        v = [0.0] * self.dim
        for tok in text.lower().split():
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Point docs/chroma at a temp dir and register fake embedders."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "alpha.md").write_text(DOC_A)
    (docs / "beta.md").write_text(DOC_B)
    monkeypatch.setattr(settings, "DOCS_PATH", str(docs))
    monkeypatch.setattr(settings, "CHROMA_PATH", str(tmp_path / "chroma"))
    for alias, dim in [("fake", 64), ("fake32", 32)]:
        register_embedder(EmbedderSpec(alias, "test", f"hash-{dim}", "test", lambda a=alias, d=dim: HashEmbedder(a, d)))
    yield tmp_path
    for alias in ("fake", "fake32"):
        EMBEDDERS.pop(alias, None)
