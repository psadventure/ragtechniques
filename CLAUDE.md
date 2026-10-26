# RAGAdvanced

Learning playground for RAG techniques: compare chunking strategies, embedders and retrievers
on the same Markdown docs. Driven by demo scripts or a Google ADK agent; all hosted models
(Vertex AI embeddings + Gemini chat) go through a local LiteLLM gateway.

## Stack & commands
- One-command setup: `scripts/setup.sh` (or `--no-gateway` for offline). Manual: `uv venv --python 3.13 .venv && uv pip install -r requirements.txt`
- Tests: `.venv/bin/python -m pytest -q` (offline — fake hashing embedder, temp Chroma dir)
- Demos: see `main.py` docstring.
- Agent: `.venv/bin/adk web agent` from the project root → http://localhost:8000, pick `rag_agent`.
  Needs the LiteLLM gateway running.
- LiteLLM gateway: `infra/litellm/` — start/stop/keys documented in the root README ("LiteLLM gateway"). App auth is a scoped virtual key in `.env` (`LITELLM_API_KEY`), never the master key.

## Layout
- `chunking/strategies.py` — strategy registry (`character`, `structure`) with overridable params.
- `vectorstore/embedder.py` — embedder registry (`minilm` local; `text-embedding-005`, `gemini-embedding-001` via LiteLLM).
- `vectorstore/store.py` — Chroma collections, one per strategy+embedder.
- `retrieval/` — `semantic`, `bm25`, `hybrid` (RRF); `retrievers.py` dispatches by name.
- `rag/service.py` — every capability as a JSON-returning function; the single API for demos and the agent.
- `pipeline/` — LangGraph retrieve → prompt → generate.
- `agent/rag_agent/` — ADK agent (`app` + `root_agent`), tools in `tools.py` wrap `rag/service.py`.

## Decisions
- **One collection per strategy and embedder** (`<strategy>__<embedder>[__<label>]`). The original
  single `rag_docs` collection forced wiping `chroma_data/` between experiments; separate collections
  let the agent compare strategies side by side. Each collection stores `strategy`, `embedder`,
  `embedder_model` and chunking `params` in its metadata. Queries always use the recorded embedder,
  so a query vector can't mismatch the collection's dimension. Re-ingesting with different params
  into the same name is refused — pass a `label` instead. Re-ingesting a document deletes its old
  chunks first (no stale chunks when chunk count shrinks). Distance metric is cosine.
- **Service functions raise; callers format.** Demo scripts print teaching output; agent tools
  (Phase 3) wrap errors into `{"error": ...}` so the LLM can recover.
- **Gateway overlay, not a fork.** `infra/litellm/docker-compose.quickstart.yml` is upstream verbatim;
  our changes (config mount, ADC mount, Vertex env) live in `docker-compose.rag.yml`. The config is
  mounted as a folder (`infra/litellm/config/`), not a single file: on Docker Desktop a single-file bind
  mount goes stale when git or an editor replaces the file, and LiteLLM then reports a misleading
  "authentication database unreachable" 503. Vertex auth is
  the developer's gcloud ADC mounted read-only — fine for local learning; use a service account or
  Workload Identity if this ever runs anywhere shared. `gemini-2.5-pro` is disabled (404 for the project).
- **ADK agent via `adk create` layout, not the agents-cli scaffold.** agents-cli generates a deployable
  app (FastAPI, Dockerfile, eval suites, own pyproject); for an in-repo learning agent the ADK docs point
  to the plain `adk create` layout. The agent's model is `LiteLlm("openai/<alias>")` against the gateway,
  so all LLM traffic goes through LiteLLM. Tools run in-process (no MCP hop). `delete_collection` uses
  `FunctionTool(require_confirmation=True)` — a UI approve/reject gate the model cannot bypass by text.
  Chat attachments become session artifacts via `SaveFilesAsArtifactsPlugin(attach_file_reference=False)`
  and `save_uploaded_file` reads them, so file bytes never go through LiteLLM.
- **`mcp` is a direct dependency** only because litellm imports it when tools are passed.
- **Public repo hygiene.** MIT licensed. `docs/` holds only original, fictional sample docs (Kestrel bank
  platform, API, Project Heron timeline) with planted lessons: a stale go-live date (overview vs timeline),
  an unanswerable pricing question, and a webhook query where hybrid loses to semantic because of the naive
  BM25 tokenizer. Don't "fix" the tokenizer or the stale date without updating README/CONCEPTS, which teach
  from them. `tests/test_sample_corpus.py` pins these lessons; `tests/test_env_examples.py` fails if code
  reads an env var missing from an `.env.example`. A full ingest prunes chunks of deleted docs.
