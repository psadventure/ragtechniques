# RAG Concepts

**Audience:** Software engineers moving into AI engineering
**Companion to:** [README.md](README.md), which covers how to run things. This file explains *why* they behave as they do.

Each concept below maps to something you can try in the lab. Look for the **▶ Try it** lines.

---

## Table of Contents

1. [What Is RAG and Why Does It Matter](#1-what-is-rag-and-why-does-it-matter)
2. [Core Concepts](#2-core-concepts)
   - 2.1 [Chunking](#21-chunking)
   - 2.2 [Embeddings](#22-embeddings)
   - 2.3 [Vector Databases and HNSW](#23-vector-databases-and-hnsw)
   - 2.4 [BM25 Keyword Search](#24-bm25-keyword-search)
   - 2.5 [Hybrid Search and Reciprocal Rank Fusion](#25-hybrid-search-and-reciprocal-rank-fusion-rrf)
   - 2.6 [LangGraph](#26-langgraph)
   - 2.7 [Prompt Engineering for RAG](#27-prompt-engineering-for-rag)
   - 2.8 [LLM Gateways](#28-llm-gateways)
   - 2.9 [Agents and Tools](#29-agents-and-tools)
3. [Design Decisions and Trade-offs](#3-design-decisions-and-trade-offs)
4. [How to Extend This Project](#4-how-to-extend-this-project)

---

## 1. What Is RAG and Why Does It Matter

Large Language Models (LLMs) are trained on a snapshot of public data up to a certain date. They have two fundamental limitations:

- **They do not know your private data**: your internal documentation, codebase, or proprietary knowledge base.
- **They hallucinate**: when they don't know an answer, they confidently make one up.

**Retrieval-Augmented Generation (RAG)** addresses both. Instead of asking the LLM to recall an answer from training, you:

1. **Retrieve** the most relevant passages from your own documents.
2. **Inject** those passages into the prompt as grounding context.
3. **Generate** an answer constrained to that context.

The LLM becomes a reasoning engine over your documents, not a knowledge store. If the answer is not in the retrieved context, a well-prompted RAG system says so instead of inventing one.

```
Without RAG:   User Question ──────────────────► LLM ──► Answer (may hallucinate)

With RAG:      User Question ──► Retrieval ──► Relevant Chunks
                                                      │
                                               Injected into Prompt
                                                      │
                               User Question ──────► LLM ──► Grounded Answer + Citations
```

**Retrieval quality caps answer quality.** If the right chunk isn't retrieved, no LLM can answer correctly. That's why most of this lab is about chunking and retrieval rather than generation.

---

## 2. Core Concepts

### 2.1 Chunking

An LLM cannot read your whole document library in one prompt: context windows are limited and large prompts are expensive and noisy. Chunking splits documents into smaller, retrievable pieces.

Bad chunks mean bad answers, however good your LLM is.

#### Character splitting (`character` strategy)

The naive approach. A window of fixed character count (`chunk_size`, default 500) slides across the text with some `overlap` (default 50), cutting wherever the count is reached.

A real chunk from `kestrel_api_reference.md` (characters 6750–7250):

```
import hashlib, hmac, time

def verify_kestrel_signature(raw_body: bytes, header: str, secret: str, tolerance: int = 300) -> bool:
    parts = dict(item.split("=", 1) for item in header.split(","))
    timestamp, received = parts["t"], parts["v1"]
    if abs(time.time() - int(timestamp)) > tolerance:
        return False                                    # too old: possible replay
    signed_payload = f"{timestamp}.".encode() + raw_body
    expected = hmac.new(secret.encode(), signed_payload, h      ← chunk ends mid-line
```

The function's last two lines, including the constant-time comparison that is the security point of the example, sit in the next chunk. Neither chunk alone shows a correct implementation.

**Problems:**
- Code blocks are cut mid-function.
- Tables are split across two chunks, breaking their structure.
- A single chunk often mixes two unrelated sections.
- The LLM receives incomplete, incoherent context.

#### Structure-based splitting (`structure` strategy)

Splits at meaningful boundaries. In Markdown that means `##` section headers: each H2 section becomes one chunk, keeping its code blocks and tables intact.

```
Structure chunk "5. Webhooks":
  Everything from "## 5. Webhooks" to the next "## 6." ← event table, signature rules,
                                                         the full verify function, retry policy
```

This is sometimes called "semantic splitting". In this project, **semantic** refers to the vector retriever (§2.2) and **structure** to this chunker.

#### Size gate (hybrid chunking)

A word-count gate on top of structure splitting. If a section exceeds `max_words` (default 400), it is split again at `###` sub-headers. This keeps any single chunk from getting too large while still respecting meaning boundaries. Each sample document has exactly one section that triggers it: "3. Core Features" (overview), "4. Transfers API" (API reference) and "4. Phase Details" (timeline). Each of their `###` subsections becomes its own chunk.

| On the sample docs | `character` (500/50) | `structure` (400 words) |
|---|---|---|
| Chunks | 66 | 40 |
| Avg words per chunk | 77 | 114 |
| Broken code blocks | 12 | 0 |

**Breaking text isn't always fatal.** For exact-fact questions ("How long is an access token valid?"), BM25 found the answer in the top 3 for 10 of 12 test questions with character chunks, against 11 of 12 with structure chunks. A fact usually survives inside a 500-character slice. What gets lost is *completeness*: the surrounding conditions, the rest of a code sample, the table header that explains a row.

**▶ Try it:** *"Compare chunking stats for kestrel_api_reference.md"*, then re-ingest `character` with `chunk_size 1000` under a `label` and compare retrieval again.

---

### 2.2 Embeddings

An embedding is a fixed-size array of numbers that represents the **meaning** of a piece of text. Texts with similar meaning have embeddings that are close together in vector space, even when they use different words.

```
"How do I register a tool?"    → [0.21, -0.44, 0.87, ...]
"What is the tool decorator?"  → [0.19, -0.41, 0.85, ...]  (close → similar meaning)
"What is the weather today?"   → [-0.72, 0.33, -0.12, ...] (far away → different meaning)
```

At ingest time every chunk is embedded and stored with its text. At query time the query is embedded **with the same model**, and the store returns the chunks whose vectors are closest.

This lab has three embedders:

| Alias | Runs | Dimensions | Notes |
|---|---|---|---|
| `minilm` (`all-MiniLM-L6-v2`) | locally, sentence-transformers | 384 | Free, offline, fast; weaker on nuance |
| `text-embedding-005` | Vertex AI via LiteLLM | 768 | Strong on English and code |
| `gemini-embedding-001` | Vertex AI via LiteLLM | 3072 | Highest quality; one input per request, so the slowest to ingest |

**Embeddings from different models are not compatible.** The dimensions differ, and even at the same size the coordinate systems mean different things. That's why every collection records its embedder, and queries always reuse it.

**▶ Try it:** ingest `structure` with `minilm` and with `text-embedding-005`, then compare `semantic` retrieval on a paraphrased question that shares few words with the docs.

---

### 2.3 Vector Databases and HNSW

A vector database is built for **approximate nearest neighbor search**: finding the N vectors closest to a query vector among potentially millions.

This project uses **ChromaDB** with a `PersistentClient`, so data is written to `chroma_data/` and survives restarts. Each experiment is its own **collection**, named `<strategy>__<embedder>[__<label>]`, which lets you compare them side by side. Distance is **cosine**: it compares the direction of vectors and ignores their length, the usual choice for text embeddings. Lower distance means more similar.

ChromaDB indexes vectors with **HNSW (Hierarchical Navigable Small World)**, a multi-layer graph where each node links to its nearest neighbors. To find the closest vector to a query:

1. Start at an entry point in the top (sparse) layer.
2. Greedily move to the neighbor closest to the query.
3. Drop to the next, denser layer and repeat.
4. At the bottom layer, the current neighborhood is the approximate nearest-neighbor set.

This runs in roughly `O(log N)` rather than `O(N)`. The trade-off is that it is **approximate**: it can occasionally miss the true nearest neighbor. In RAG this rarely matters, because the second-closest chunk is usually just as useful.

**▶ Try it:** *"Inspect structure__text-embedding-005"* shows the embedding dimension and the metadata stored with each chunk.

---

### 2.4 BM25 Keyword Search

BM25 (Best Match 25) is the classic search-engine ranking function. It scores chunks by:

- **Term frequency (TF):** how often the query word appears in the chunk.
- **Inverse document frequency (IDF):** how rare the word is across all chunks. Rare words (e.g. `get_patient_risk_summary`) count far more than common ones (`the`, `is`).
- **Length normalization:** long chunks are penalized so they can't win just by containing more words.

```
BM25(chunk, query) = Σ  IDF(word) × TF(word, chunk) × (k+1)
                    words          ─────────────────────────────────
                                   TF(word, chunk) + k × (1 - b + b × len(chunk)/avglen)
```

`k` and `b` are tuning constants (BM25Okapi defaults: `k=1.5`, `b=0.75`). Tokenization here is deliberately naive: lowercase and split on whitespace, with no stemming and no punctuation stripping. So `webhook` does not match `Webhooks`, and the query word `signature?` (with its question mark) does not match `signature`.

**BM25 is strong when** the query contains rare, specific tokens (`idempotency_conflict`, `KST-4090`, `verify_kestrel_signature`) that appear in only one or two chunks.

**BM25 is weak when** the query is a conceptual question in everyday words (*"how do I stop someone replaying a webhook?"*), or it uses synonyms the docs don't use.

**▶ Try it:** compare `bm25` against `semantic` once with an exact function name and once with a paraphrase.

---

### 2.5 Hybrid Search and Reciprocal Rank Fusion (RRF)

Neither BM25 nor vector search is better everywhere. BM25 wins on exact tokens; vector search wins on meaning. Hybrid search runs both and merges the rankings.

**Reciprocal Rank Fusion (RRF)** merges by rank position, not by raw score. The raw scores aren't comparable: a BM25 score of 15 and a cosine distance of 0.4 are on different scales.

```
rrf_score(chunk) = Σ over each ranked list   1 / (60 + position)
                   (position 0 = top hit, so the best possible contribution is 1/60)
```

A chunk missing from one list gets nothing from that list. The constant `60` smooths the curve: top ranks still matter most, but a chunk ranked well in *both* lists can beat one ranked first in only one.

**Example** (rank 1 = position 0):

| Chunk | BM25 rank | Vector rank | RRF score |
|-------|-----------|-------------|-----------|
| A     | 1         | 4           | 1/60 + 1/63 = 0.03254 |
| B     | 6         | 1           | 1/65 + 1/60 = 0.03205 |
| C     | 2         | 2           | 1/61 + 1/61 = 0.03279 ← wins |

C wins without topping either list, because both retrievers agree on it. **Agreement between retrieval methods is more reliable than dominance in one.**

In this project, BM25 ranks every chunk and vector search contributes its top `2 × top_k`. The hybrid result includes a `trace` with both input rankings, so you can see the fusion at work.

**When fusion hurts.** RRF assumes both retrievers are reasonable. Ask *"How do I verify a webhook signature?"* on `structure__text-embedding-005`:

| | Top 3 |
|---|---|
| Vector | **5. Webhooks** (distance 0.29), 8. SDK Quickstart, 2. Authentication |
| BM25 | Platform overview intro, 4. Transfers API, API reference intro |
| Hybrid (RRF) | 4. Transfers API, 8. SDK Quickstart, API reference intro: **Webhooks is gone** |

BM25's naive tokenizer can't match `webhook` or `signature?` to the Webhooks section. Because BM25 ranks *every* chunk, the chunks it likes collect points from both lists, and they outvote the single correct vector hit. Fusing with a weak retriever can make results worse than using the strong one alone.

**▶ Try it:** `search` with `hybrid` and read the trace. Find a chunk that is mid-ranked in both lists but ends up in the fused top 3, then run the webhook question above with each retriever.

---

### 2.6 LangGraph

LangGraph builds stateful, graph-based AI pipelines. Your application is a **directed graph**:

- **Nodes** are functions that do one job (retrieve, build prompt, generate).
- **Edges** define the execution order.
- **State** is a shared dict that flows through every node and accumulates results.

```
         ┌──────────────────────────────────────────────────────────────────┐
         │                        RAGState (shared dict)                     │
         │  query | collection | retriever | top_k | chat_model              │
         │  retrieved_chunks | prompt | answer | error                       │
         └──────────────────────────────────────────────────────────────────┘
                  │               │              │             │
                  ▼               ▼              ▼             ▼
               START ──► retrieve_node ──► build_prompt_node ──► generate_node ──► END
                               │
                         (if error)
                               └──────────────────────────────────────► END
```

Each node receives the current state, does one job, and returns a dict containing only the keys it changed. LangGraph merges that back into the state. Nodes **never call each other**; the graph wires them, so each node can be tested and swapped on its own.

The conditional edge after `retrieve_node` checks `state["error"]`. If retrieval failed (for example, the collection doesn't exist), the pipeline goes straight to END instead of prompting the LLM with no context.

---

### 2.7 Prompt Engineering for RAG

The system prompt is where hallucination is controlled. It sets three explicit constraints:

```
1. Answer ONLY using the information in the provided document chunks.

2. If the answer is not present in the chunks, respond with:
   "I could not find an answer to this question in the provided documentation."

3. For every claim in your answer, cite the source using this format:
   [Source: <doc_title>, Section: <section_title>]
```

- Constraint 1 stops the LLM from drawing on its training knowledge.
- Constraint 2 handles "not in the docs": the LLM refuses rather than invents.
- Constraint 3 makes every claim traceable to a source chunk, so the answer can be audited.

The user message then injects the retrieved chunks with labels:

```
--- Retrieved Documentation ---

[Chunk 1 - kestrel_api_reference.md | 2. Authentication]
<full chunk text>

[Chunk 2 - kestrel_api_reference.md | 7. Error Codes]
<full chunk text>
...
---
Question: How long is an access token valid?
```

Character chunks have no section title, so their citations are weaker. That's another hidden cost of naive chunking.

**Grounding is not truth-checking.** The prompt makes the model answer *from the chunks*, not decide which chunk is right. Ask *"When does Harbor & Pine go live on Kestrel?"*: the overview still says September 2026, while the project timeline records the move to 17–18 October 2026 (decision D-11). Gemini cites both without flagging the conflict. Real corpora are full of stale pages; fixes include metadata such as dates and authority, prompt rules ("prefer the most recent source, and flag conflicts"), or curating the corpus.

**▶ Try it:** ask *"How much does a Kestrel Cloud license cost?"*. No document covers pricing, so the correct answer is the refusal.

---

### 2.8 LLM Gateways

An **LLM gateway** is a proxy between your application and model providers. This lab runs **LiteLLM** locally (`localhost:4000`): it exposes Vertex AI's Gemini and embedding models through one OpenAI-compatible API.

Why put one in the middle:

- **One API, many providers.** The app calls `model="gemini-2.5-flash"` or `model="text-embedding-005"` using the plain OpenAI client. Switching provider is a gateway config change, not a code change.
- **Credentials stay in one place.** GCP credentials live only in the gateway. The app holds a **virtual key** scoped to three models, which can be revoked without touching GCP.
- **Visibility.** Spend, latency and request logs per key appear in the gateway UI (`/ui`).
- **Policy.** Rate limits, budgets, fallbacks and retries can be configured centrally.

The costs: another moving part to run, an extra network hop, and provider-specific features (such as Vertex embedding *task types*) only work if the gateway passes them through.

---

### 2.9 Agents and Tools

An **agent** is an LLM in a loop that decides which **tools** (functions) to call, reads their results, and decides what to do next. The RAG Lab agent is built with Google's **Agent Development Kit (ADK)**. Its model calls go through the LiteLLM gateway, and its tools are the functions in `rag/service.py`.

Points worth noticing in this project:

- **Tool schemas come from code.** ADK builds each tool's description from the Python signature and docstring, so a clear docstring is effectively prompt engineering.
- **Errors are data, not crashes.** Tools return `{"error": "..."}`, so the agent can read the problem and recover, for example by ingesting first.
- **Guardrails belong in code, not in prompts.** `delete_collection` uses ADK's `require_confirmation`, so the UI needs a human click. An instruction like "ask before deleting" can be talked around; the confirmation gate cannot.
- **Agentic RAG vs pipeline RAG.** The LangGraph pipeline always runs retrieve → generate in a fixed order. The agent decides at run time which collection, retriever and comparison to use. That's more flexible, but less predictable and harder to evaluate.

---

## 3. Design Decisions and Trade-offs

| Decision | Why | Trade-off |
|----------|-----|-----------|
| One Chroma collection per strategy × embedder | Experiments sit side by side; no wiping between runs | More disk; changing chunk params needs a `label` or a delete |
| Three embedders (local + 2 hosted) | Compare cost, speed and quality directly | Hosted ones need the gateway and GCP access |
| ChromaDB with HNSW, cosine distance | Zero config, persists to disk, good for prototyping | HNSW parameters (`m`, `ef_construct`) aren't tuned. For production, use Qdrant or Weaviate with explicit index config |
| BM25 built in memory at query time | Simple, no extra infrastructure | Fetches every chunk from Chroma on every query. Slow beyond ~10k chunks; persist the index at ingest time to scale |
| RRF constant of 60 | Standard default, robust across domains | Lower values (e.g. 10) let top ranks dominate; higher (e.g. 100) smooth more. Tune against retrieval metrics |
| `gemini-2.5-flash` via LiteLLM for generation | Fast, cheap, follows grounding instructions well | Lower reasoning quality than Pro-tier models. Swap with `RAG_CHAT_MODEL` once another alias is in the gateway |
| Citation extraction via regex | Simple, no structured-output overhead | Brittle: a differently formatted citation is missed. Use structured output for production |
| LangGraph for the pipeline | Explicit state flow, easy to add nodes, conditional routing | Overhead for a linear 3-node pipeline; pays off once there are loops, branches or human-in-the-loop |
| ADK agent with in-process tools | Conversational experiments with no extra services | Less deterministic than scripts; use the demos when you need repeatable runs |

---

## 4. How to Extend This Project

**Add an embedding model:**
Add it to `infra/litellm/config/config.yaml`, restart the gateway, allow it on the virtual key, and register an alias in `vectorstore/embedder.py`. Then ingest with `embedders=["<alias>"]`. A new collection is created automatically, and nothing needs deleting.

**Add more documents:**
Drop `.md` files into `docs/` (or upload them through the agent) and re-ingest. Re-ingesting replaces a document's old chunks.

**Add a chunking strategy:**
Write a splitter that returns `Chunk` objects and register it in `chunking/strategies.py` with its default params. The agent, demos and comparisons pick it up from the registry.

**Add a PDF loader:**
Before chunking, convert PDF to Markdown with `pymupdf4llm` or `pdfplumber`. The rest of the pipeline is unchanged.

**Add a reranker:**
Add a LangGraph node between retrieval and prompt building that re-scores the top-K chunks with a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`). Cross-encoders read the query and chunk together, so they're more accurate than comparing separate embeddings. They're also slower, which is why they only rerank a short list.

**Use embedding task types:**
Vertex embedding models accept `RETRIEVAL_DOCUMENT` for chunks and `RETRIEVAL_QUERY` for queries, which can improve retrieval. Pass them through LiteLLM and compare against the current setup.

**Persist the BM25 index:**
Serialize the index at ingest time and load it at query time, instead of rebuilding it from Chroma on every query.

**Replace ChromaDB with Qdrant:**
Only `vectorstore/store.py` changes; the rest of the code goes through its functions.

**Add evaluation:**
Build a small question set with the expected source section for each question, and measure hit@k and MRR across every collection × retriever. Then add answer-level metrics (faithfulness, answer relevancy, context precision) with RAGAS or `adk eval`. That turns "this looks better" into a number.
