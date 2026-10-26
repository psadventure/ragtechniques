# RAGAdvanced — a RAG techniques lab

A hands-on playground for learning Retrieval-Augmented Generation. You ingest the same Markdown docs with different **chunking strategies** and **embedders**, query them with different **retrievers**, and compare the results side by side. You can drive it two ways:

- **RAG Lab agent** (Google ADK): you describe the experiment in chat; it runs the tools and explains why results differ.
- **Demo scripts**: plain CLI commands for each step.

All hosted models (Gemini chat, Vertex AI embeddings) go through a local **LiteLLM gateway**. Without GCP, you can still run every chunking and retrieval experiment with a local embedder.

```
                ┌──────────────────────────┐
  you ──chat──► │ RAG Lab agent (ADK)      │──LLM calls──┐
                │ tools → rag/service.py   │             │
                └────────────┬─────────────┘             ▼
  you ──CLI───► demo/*.py ───┤                ┌─────────────────────┐    ┌───────────┐
                             │  embeddings ──►│ LiteLLM gateway     │───►│ Vertex AI │
                             │  answers    ──►│ localhost:4000      │    │ (GCP)     │
                             ▼                └─────────────────────┘    └───────────┘
                ┌──────────────────────────┐
                │ ChromaDB (chroma_data/)  │  one collection per strategy × embedder
                └──────────────────────────┘
```

---

## What you can learn here

How embeddings, HNSW, BM25 and RRF work, why chunking matters, and the design trade-offs are explained in **[CONCEPTS.md](CONCEPTS.md)**.

| Topic | What to compare |
|---|---|
| **Chunking** | `character` (fixed 500-char windows) vs `structure` (Markdown `##`/`###` sections). On the sample docs, character splitting breaks 12 code blocks; structure splitting breaks none. |
| **Chunking params** | `chunk_size` / `overlap` for character, `max_words` for structure. Keep each variant with a `label`. |
| **Embedders** | `minilm` (local, 384-d) vs `text-embedding-005` (768-d) vs `gemini-embedding-001` (3072-d). |
| **Retrievers** | `semantic` (dense vectors), `bm25` (keywords), `hybrid` (Reciprocal Rank Fusion of both). Hybrid also shows the two rankings it fused. |
| **Grounded generation** | Answers cite `[Source, Section]` and refuse when the docs don't contain the answer. |
| **Pipelines & agents** | LangGraph `retrieve → prompt → generate`; an ADK agent with tools, a confirmation gate and file uploads. |

---

## Sample documents

`docs/` holds three **fictional** documents about an invented core-banking product called **Kestrel**, and an invented credit union's project to adopt it. Every name, number and date is made up, and hostnames use the reserved `.example` domain.

| File | What's in it | Good for testing |
|---|---|---|
| `kestrel_platform_overview.md` | Services, architecture, features (ledger, payments, Fraud Shield, lending, onboarding), scoring rules, deployment, security | Prose, tables, a YAML config block, an oversized section that splits at `###` |
| `kestrel_api_reference.md` | Auth, accounts and transfers endpoints, webhooks, rate limits, error codes, SDK | Long code blocks, exact identifiers (`KST-4090`, `Idempotency-Key`) for BM25 |
| `project_heron_timeline.md` | Rollout plan: phases, milestones, go/no-go criteria, risks, decision log, cutover | Dates and numbers, questions that need the right row of the right table |

Lessons built into the corpus:

- **Conflicting sources.** The overview says go-live is *September 2026*; the timeline's decision D-11 moved it to *17–18 October 2026*. Ask *"When does Harbor & Pine go live on Kestrel?"*: Gemini cites both dates without flagging the conflict.
- **Out of scope.** Pricing isn't in any document. *"How much does a Kestrel Cloud license cost?"* should be refused.
- **Hybrid isn't always better.** For *"How do I verify a webhook signature?"*, semantic search ranks **5. Webhooks** first, but BM25's naive tokenizer misses it (`signature?` ≠ `signature`, `webhook` ≠ `Webhooks`), and RRF pulls it out of the fused top 3.
- **Facts vs context.** For exact-fact questions, BM25 finds the answer about as often in character chunks as in structure chunks (10 vs 11 of 12 in the test set). Character chunking hurts *completeness*: code blocks cut in half, answers missing their surrounding context.

You can replace these files with your own `.md` docs at any time. A full re-ingest prunes chunks of deleted files.

---

## Setup

**Prerequisites:** [uv](https://docs.astral.sh/uv/) and Python 3. For hosted models you also need Docker Desktop, the `gcloud` CLI, and a GCP project with the Vertex AI API enabled.

### Quick start (script)

```bash
gcloud auth application-default login                              # once, for Vertex AI
gcloud auth application-default set-quota-project <your-project-id>

scripts/setup.sh                 # full setup: venv, .env files, gateway, scoped key, smoke test
scripts/setup.sh --no-gateway    # offline: venv + .env only (local minilm embedder, no answers)
```

The script is safe to re-run: existing `.env` values are kept. For a non-interactive run, export `VERTEX_PROJECT` (and optionally `VERTEX_LOCATION`) first.

### Manual setup

```bash
# 1. Python environment
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2. Gateway: configure and start it, then mint a virtual key
#    (see "LiteLLM gateway" below: "First start by hand" and "Mint the app's key by hand")

# 3. App settings
cp .env.example .env      # set LITELLM_API_KEY to the virtual key from step 2
```

Check it works:

```bash
.venv/bin/python -m pytest -q       # offline tests, no gateway needed
```

---

## LiteLLM gateway

The gateway exposes Vertex AI models behind one OpenAI-compatible endpoint at `http://localhost:4000`. It runs in Docker as two containers: `litellm` (the proxy) and `db` (Postgres, which stores virtual keys and spend logs).

| Alias | Type | Vertex model | Region |
|---|---|---|---|
| `gemini-2.5-flash` | chat | `gemini-2.5-flash` | us-east1 |
| `text-embedding-005` | embedding (768-d) | `text-embedding-005` | us-east1 |
| `gemini-embedding-001` | embedding (3072-d) | `gemini-embedding-001` | us-east1 |

`gemini-2.5-pro` ships commented out in `infra/litellm/config/config.yaml`, because not every project or region has access to it. Uncomment it if yours does.

### Start, stop and check

Run these from `infra/litellm/`. Every command needs both compose files: the upstream quickstart plus this project's overlay.

```bash
cd infra/litellm

docker compose -f docker-compose.quickstart.yml -f docker-compose.rag.yml up -d            # start (or restart after a stop)
docker compose -f docker-compose.quickstart.yml -f docker-compose.rag.yml ps               # status of both containers
docker compose -f docker-compose.quickstart.yml -f docker-compose.rag.yml logs -f litellm  # follow gateway logs (Ctrl+C to exit)
docker compose -f docker-compose.quickstart.yml -f docker-compose.rag.yml restart litellm  # apply config.yaml changes
docker compose -f docker-compose.quickstart.yml -f docker-compose.rag.yml stop             # pause; containers kept
docker compose -f docker-compose.quickstart.yml -f docker-compose.rag.yml down             # stop and remove containers; keys and logs kept
docker compose -f docker-compose.quickstart.yml -f docker-compose.rag.yml down -v          # stop and DELETE the database (virtual keys, spend logs)

curl -s http://localhost:4000/health/liveliness                                            # "I'm alive!" when up
```

- After `down -v`, start again and mint a new key: re-running `scripts/setup.sh` does both, once you clear `LITELLM_API_KEY` in `.env`.
- Docker Desktop must be running for any of these. Quitting Docker Desktop stops the gateway too.
- The agent (`adk web`) and demos are separate processes. Stop `adk web` with Ctrl+C.

**Admin UI:** http://localhost:4000/ui. Log in with `LITELLM_MASTER_KEY` from `infra/litellm/.env` to see spend, request logs and keys.

### First start by hand

`scripts/setup.sh` does all of this. To do it yourself:

```bash
cd infra/litellm
cp .env.example .env
# fill in: LITELLM_MASTER_KEY and LITELLM_SALT_KEY (each: echo "sk-$(openssl rand -hex 32)")
#          VERTEX_PROJECT, and VERTEX_LOCATION if not us-east1
docker compose -f docker-compose.quickstart.yml -f docker-compose.rag.yml up -d
```

Keep `infra/litellm/.env` once the database exists: changing `LITELLM_SALT_KEY` later makes the credentials stored in Postgres unreadable.

### Mint the app's key by hand

The app never uses the master key. Create a **virtual key** limited to the configured models, and put it in the root `.env` as `LITELLM_API_KEY`:

```bash
export LITELLM_MASTER_KEY=$(sed -n 's/^LITELLM_MASTER_KEY=//p' infra/litellm/.env)   # from the project root
curl -s -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  http://localhost:4000/key/generate \
  -d '{"key_alias":"rag-advanced-local","models":["gemini-2.5-flash","text-embedding-005","gemini-embedding-001"]}'
```

Keys can be revoked or edited in the admin UI.

### Add a model

1. Add an entry to `infra/litellm/config/config.yaml` and run the `restart litellm` command above.
2. Allow the new alias on the app's virtual key, in the admin UI or with `/key/update`.
3. For an embedding model, also register the alias in `vectorstore/embedder.py`.

### Gateway files

| File | Purpose |
|---|---|
| `infra/litellm/docker-compose.quickstart.yml` | Upstream LiteLLM quickstart, unmodified, so it can be re-downloaded |
| `infra/litellm/docker-compose.rag.yml` | This project's overlay: mounts the `config/` folder and your gcloud credentials (read-only), sets the Vertex env |
| `infra/litellm/config/config.yaml` | Model aliases → Vertex models |
| `infra/litellm/.env.example` | Every setting the gateway reads; copy to `.env` (gitignored) |

---

## Path A: RAG Lab agent

```bash
.venv/bin/adk web agent     # from the project root → http://localhost:8000, pick "rag_agent"
```

Try these prompts in order:

1. *"What can I experiment with?"*: lists strategies, embedders, retrievers, documents and collections.
2. *"Compare chunking stats for kestrel_api_reference.md."*: chunk counts and broken code blocks, nothing stored.
3. *"Ingest all docs with both strategies using text-embedding-005."*: creates `character__text-embedding-005` and `structure__text-embedding-005`.
4. *"Compare character vs structure with text-embedding-005 for 'How do I create a transfer with the Python SDK?' using hybrid."*
5. *"Now compare bm25, semantic and hybrid on the structure collection for 'How do I verify a webhook signature?'"*: watch hybrid lose to semantic.
6. *"Show me the first 3 chunks of character__text-embedding-005."*
7. *"When does Harbor & Pine go live on Kestrel?"*: two documents disagree. Does the answer notice?
8. *"How much does a Kestrel Cloud license cost?"*: it should refuse, because no document covers pricing.
9. *"Ingest the character strategy again with chunk_size 1000, label cs1000, and compare it with the default."*
10. Attach a `.md` file in the chat: *"Save this to my docs and ingest it with structure."*

The agent's tools:

| Tool | Does |
|---|---|
| `list_options` | strategies, embedders, retrievers, documents, collections |
| `compare_chunking` | chunk stats per strategy for one document (no DB) |
| `ingest_documents` | strategies × embedders → one collection each |
| `list_collections` / `inspect_collection` | what's stored: stats, embedding dim, paged chunks |
| `search` | retrieve only, with the hybrid trace |
| `answer_question` | full RAG with citations |
| `compare_retrieval` | one query across collections × retrievers |
| `upload_document` / `save_uploaded_file` | pasted Markdown / chat attachment → `docs/` |
| `delete_collection` | needs you to click **Approve** in the UI. Typing "yes" is not enough. |

---

## Path B: Demo scripts

```bash
# Chunking stats only, no DB
.venv/bin/python demo/chunking_comparison.py [document.md]

# Ingest: every strategy at once, or one at a time
.venv/bin/python demo/ingest_all.py --embedders minilm text-embedding-005
.venv/bin/python demo/ingest_char.py --embedder text-embedding-005 --chunk-size 300 --label cs300
.venv/bin/python demo/ingest_structure_based.py --embedder gemini-embedding-001

# Compare retrieval across collections × retrievers
.venv/bin/python demo/compare.py "How do I create a transfer with the Python SDK?" --embedder text-embedding-005

# One collection: see BM25 / vector / fused rankings, then the cited answer
.venv/bin/python demo/user_query.py "How long is an access token valid?" --collection structure__text-embedding-005
.venv/bin/python demo/user_query.py "What does error KST-4090 mean?" --collection character__minilm --retriever bm25 --retrieve-only
```

Sample result for *"How do I create a transfer with the Python SDK?"* using `text-embedding-005` and hybrid retrieval:

| Rank | `character__…` | `structure__…` |
|---|---|---|
| 1 | kestrel_api_reference.md \| chars 3150-3650 | **kestrel_api_reference.md \| 4. Transfers API > 4.1 Create a transfer** |
| 2 | kestrel_api_reference.md \| chars 5400-5900 | kestrel_api_reference.md \| 8. SDK Quickstart |
| 3 | kestrel_api_reference.md \| chars 2700-3200 | kestrel_api_reference.md \| 4. Transfers API |

The character chunks are 500-character slices that start and end mid-sentence or mid-code. The structure chunk is the complete section, with the full SDK example intact.

---

## How the data is organised

- **Documents:** `.md` files in `docs/`.
- **Collections:** one per experiment, named `<strategy>__<embedder>[__<label>]`. Each stores its strategy, embedder and chunking params as metadata.
- **Queries reuse the collection's embedder,** so a query vector always matches the stored vectors.
- **Re-ingesting a document** replaces its old chunks. A full re-ingest also prunes chunks of documents you've deleted from `docs/`.
- **Changing chunk params for an existing collection is refused.** Use a new `label`, or delete the old collection.

---

## Project structure

```
RAGAdvanced/
├── agent/rag_agent/        ADK agent: agent.py (model, instructions, app), tools.py
├── rag/service.py          every capability as a function, the single API for agent + demos
├── chunking/               character_splitter, semantic_splitter (structure), strategies registry
├── vectorstore/            embedder registry (local + LiteLLM), Chroma collections
├── retrieval/              semantic, bm25, hybrid (RRF), retrievers dispatcher
├── generation/             grounded prompt builder, LiteLLM / Anthropic / OpenAI generators
├── pipeline/               LangGraph state, nodes, graph
├── config/settings.py      defaults, overridable from .env
├── demo/                   CLI demos
├── infra/litellm/          LiteLLM gateway: compose overlay, model config, .env.example, README
├── scripts/setup.sh        one-command setup
├── tests/                  offline pytest suite (unit, sample-corpus, env-example drift)
└── docs/                   source documents (fictional samples)
```

---

## Configuration

Two env files, each with a committed `.env.example` that lists every setting:

| File | For | Key settings |
|---|---|---|
| `.env` | the app | `LITELLM_BASE_URL`, `LITELLM_API_KEY` (a scoped virtual key, never the master key) |
| `infra/litellm/.env` | the gateway | `LITELLM_MASTER_KEY`, `LITELLM_SALT_KEY`, `VERTEX_PROJECT`, `VERTEX_LOCATION` |

Optional app settings (defaults in brackets):

| Variable | Default | Meaning |
|---|---|---|
| `RAG_EMBEDDER` | `minilm` | default embedder for ingestion |
| `RAG_CHAT_MODEL` | `gemini-2.5-flash` | gateway alias for answers and the agent |
| `RAG_GENERATOR` | `litellm` | `litellm`, `anthropic` or `openai` |
| `RAG_CHAR_CHUNK_SIZE` / `RAG_CHAR_CHUNK_OVERLAP` | `500` / `50` | character chunking defaults |
| `RAG_STRUCTURE_MAX_WORDS` | `400` | size gate for `###` splitting |
| `RAG_TOP_K` / `RAG_RRF_K` | `5` / `60` | results per query / RRF constant |

Per-experiment values (agent requests, demo flags) override these. A test (`tests/test_env_examples.py`) fails if the code reads a variable that's missing from an `.env.example`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` on `localhost:4000` | Start Docker Desktop, then the gateway (see "Start, stop and check"). |
| Gateway starts but calls fail with 401/403 | `LITELLM_API_KEY` in `.env` is missing or was revoked (e.g. after `down -v`). Mint a new key. |
| Port 4000 already in use | Another gateway is running (e.g. a second clone). Stop it, or set `LITELLM_PORT` in `infra/litellm/.env` and `LITELLM_BASE_URL` in `.env`. |
| `Collection '…' not found` | Ingest first. The agent's `list_options` shows what exists. |
| `…was built with … params …` | You changed chunk params. Use a `label`, or delete the collection. |
| Vertex `404 … model was not found` | The model isn't available to your project or region. Try another region, or remove the model from `infra/litellm/config/config.yaml`. |
| `503 … authentication database is temporarily unreachable` while Postgres is healthy | Usually the gateway can't read its config or credentials, e.g. after re-running `gcloud auth application-default login`. Check `logs litellm` for `FileNotFoundError`, then recreate the container: `docker compose -f docker-compose.quickstart.yml -f docker-compose.rag.yml up -d --force-recreate --no-deps litellm` (keeps keys and the database). |
| `No module named 'mcp'` | `uv pip install -r requirements.txt` (litellm needs it for tool calls). |
| `gemini-embedding-001` ingest is slow | Vertex accepts one input per request for this model. That's expected. |

---

## Ideas for next experiments

- A smarter BM25 tokenizer (strip punctuation, stem plurals) and see whether hybrid recovers on the webhook question
- Query-side task types (`RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT`) for Vertex embeddings
- A reranker (cross-encoder) after hybrid retrieval
- Query rewriting / HyDE in the LangGraph pipeline
- Turn the test set in `tests/test_sample_corpus.py` into an eval report (hit@k, MRR) across every collection × retriever
- Swap Chroma for Qdrant to tune HNSW `m` / `ef_construct`

---

## License

[MIT](LICENSE). The sample documents in `docs/` are fictional and are covered by the same license.
