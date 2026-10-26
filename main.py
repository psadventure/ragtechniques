"""
RAG Documentation Assistant — Demo Sequence
============================================

Each chunking strategy is stored in its own collection: <strategy>__<embedder>[__<label>].

  Step 1 — Chunking stats comparison (no DB required):
      python demo/chunking_comparison.py

  Step 2 — Ingest with CHARACTER splitting (broken chunks):
      python demo/ingest_char.py

  Step 3 — Ingest with STRUCTURE-based splitting (clean sections):
      python demo/ingest_structure_based.py

  Step 4 — Same query against both, observe the difference:
      python demo/user_query.py "How do I create a transfer with the Python SDK?" --collection character__minilm
      python demo/user_query.py "How do I create a transfer with the Python SDK?" --collection structure__minilm

  Add --embedder text-embedding-005 to ingest via the LiteLLM gateway (Vertex AI),
  and --retriever semantic|bm25|hybrid to the query to compare retrievers.

  Shortcut — every strategy at once, then a side-by-side comparison:
      python demo/ingest_all.py --embedders minilm text-embedding-005
      python demo/compare.py "How do I create a transfer with the Python SDK?" --embedder text-embedding-005

  Or drive everything conversationally:  adk web agent   (see README)
"""
