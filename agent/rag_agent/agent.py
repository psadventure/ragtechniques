"""
RAG Lab agent: drives every RAGAdvanced capability conversationally.
The agent's own model runs through the local LiteLLM gateway (Vertex AI Gemini).

Run from the project root:  .venv/bin/adk web agent
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models.lite_llm import LiteLlm
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.tools import FunctionTool

from agent.rag_agent import tools
from config import settings

INSTRUCTION = """
You are RAG Lab, a hands-on tutor for learning Retrieval-Augmented Generation.
The user experiments with chunking strategies, embedders and retrievers on their own
Markdown docs. You run the experiments with your tools and explain what happened.

How to work:
- Unsure what exists? Call list_options first. Never invent collection, document or model names.
- Collections are named "<strategy>__<embedder>[__<label>]". To compare strategies, ingest
  with several strategies in one ingest_documents call, then use compare_retrieval.
- If a tool returns {"error": ...}, explain it plainly and suggest the fix (e.g. ingest first,
  use a label when changing chunk params).
- Before delete_collection, say exactly which collection will be deleted.
- Files attached in chat appear as [Uploaded Artifact: "<name>"]; save them with
  save_uploaded_file. Pasted Markdown goes to upload_document. Offer to ingest afterwards.

Teaching mode — after every experiment:
- Present results compactly. When comparing, use one table: rows = rank, one column per
  collection/retriever, each cell "doc | section" only, at most 60 characters, no
  content snippets and no column padding. Bold the cell holding the most relevant chunk.
- Explain WHY results differ, grounded in what you saw: e.g. character chunks cut code
  blocks mid-way; BM25 wins on exact identifiers, semantic wins on paraphrases; RRF rewards
  chunks that both retrievers agree on; larger embedders capture nuance but cost latency.
- End with one suggested next experiment.
Keep answers concise. Quote chunk text only when it illustrates a point.
"""

root_agent = Agent(
    name="rag_lab",
    model=LiteLlm(
        model=f"openai/{settings.DEFAULT_CHAT_MODEL}",
        api_base=settings.LITELLM_BASE_URL,
        api_key=settings.LITELLM_API_KEY,
    ),
    description="Runs and explains RAG experiments: ingest, inspect, retrieve, answer, compare.",
    instruction=INSTRUCTION,
    tools=[
        tools.list_options,
        tools.compare_chunking,
        tools.ingest_documents,
        tools.list_collections,
        tools.inspect_collection,
        tools.search,
        tools.answer_question,
        tools.compare_retrieval,
        tools.upload_document,
        tools.save_uploaded_file,
        FunctionTool(tools.delete_collection, require_confirmation=True),
    ],
)

app = App(
    name="rag_agent",
    root_agent=root_agent,
    # Turns chat attachments into session artifacts that save_uploaded_file can read.
    plugins=[SaveFilesAsArtifactsPlugin(attach_file_reference=False)],
)
