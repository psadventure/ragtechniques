from typing import List, Optional
from typing_extensions import TypedDict

from generation.answer_generator import Answer
from generation.prompt_builder import PromptMessages
from retrieval.semantic_search import RetrievedChunk


class RAGState(TypedDict):
    query: str
    collection: str      # Chroma collection name, e.g. "structure__minilm"
    retriever: str       # semantic | bm25 | hybrid
    top_k: int
    chat_model: Optional[str]   # LiteLLM alias; None = settings default
    retrieved_chunks: List[RetrievedChunk]
    prompt: Optional[PromptMessages]
    answer: Optional[Answer]
    error: str   # empty string if no error; populated if a node fails
