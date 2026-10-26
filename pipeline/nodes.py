from config import settings
from generation.answer_generator import generate_answer
from generation.prompt_builder import build_prompt
from pipeline.state import RAGState
from retrieval.retrievers import retrieve
from vectorstore.store import load_collection


def retrieve_node(state: RAGState) -> dict:
    """
    Load the ChromaDB collection named in state.
    Run the chosen retriever (semantic | bm25 | hybrid) with query from state.
    Return: {"retrieved_chunks": [...]}
    On error: return {"error": "Retrieval failed: <message>"}
    """
    try:
        collection = load_collection(settings.CHROMA_PATH, state["collection"])
        chunks = retrieve(state["query"], collection, state["retriever"], state["top_k"])
        return {"retrieved_chunks": chunks}
    except Exception as exc:
        return {"error": f"Retrieval failed: {exc}"}


def build_prompt_node(state: RAGState) -> dict:
    """
    Call prompt_builder.build_prompt() with query and retrieved_chunks from state.
    Return: {"prompt": <PromptMessages>}
    On error: return {"error": "Prompt build failed: <message>"}
    """
    try:
        prompt = build_prompt(state["query"], state["retrieved_chunks"])
        return {"prompt": prompt}
    except Exception as exc:
        return {"error": f"Prompt build failed: {exc}"}


def generate_node(state: RAGState) -> dict:
    """
    Call answer_generator.generate_answer() with prompt from state.
    Return: {"answer": <Answer>}
    On error: return {"error": "Generation failed: <message>"}
    """
    try:
        answer = generate_answer(
            state["prompt"], state["query"], len(state["retrieved_chunks"]), state.get("chat_model")
        )
        return {"answer": answer}
    except Exception as exc:
        return {"error": f"Generation failed: {exc}"}
