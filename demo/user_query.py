"""
Demo 3: RAG Query Pipeline
Run a question through the full RAG pipeline:
retrieve → build prompt → generate answer with citations.

Usage:
    python demo/user_query.py "How long is an access token valid?" --collection structure__minilm
    python demo/user_query.py "What does error KST-4090 mean?" --collection character__minilm --retriever bm25
    python demo/user_query.py "How much does a Kestrel Cloud license cost?" --collection structure__minilm
    python demo/user_query.py "..." --collection structure__minilm --retrieve-only

List collections with: python -c "from rag import service; print(service.list_collections())"
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from rag import service

WIDTH = 67


def print_ranking(title: str, rows: list) -> None:
    print(f"\n{'─' * WIDTH}\n{title}\n{'─' * WIDTH}")
    for r in rows:
        preview = r["content"][:80].replace("\n", " ")
        print(f"  #{r['rank']}  score={r['score']:<10}  [{r['doc_title']} | {r['section']}]")
        print(f"        \"{preview}...\"")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--retriever", default="hybrid", choices=["semantic", "bm25", "hybrid"])
    parser.add_argument("--top-k", type=int, default=settings.TOP_K)
    parser.add_argument("--chat-model")
    parser.add_argument("--retrieve-only", action="store_true")
    args = parser.parse_args()

    print(f"\nQuery: {args.query}   [{args.collection} / {args.retriever}]")

    found = service.search(args.query, args.collection, args.retriever, args.top_k)
    if "trace" in found:
        print_ranking(f"BM25 TOP {args.top_k} (keyword match)", found["trace"]["bm25"])
        print_ranking(f"VECTOR TOP {args.top_k} (semantic similarity, cosine distance)", found["trace"]["vector"])
    print_ranking(f"{args.retriever.upper()} — {args.top_k} chunks sent to LLM", found["results"])
    if args.retrieve_only:
        return

    result = service.answer(args.query, args.collection, args.retriever, args.top_k, args.chat_model)
    print(f"\n{'─' * WIDTH}\nANSWER ({result['chat_model']})\n{'─' * WIDTH}")
    print(result["answer"])
    print(f"\n{'─' * WIDTH}\nCITATIONS\n{'─' * WIDTH}")
    for citation in result["citations"] or ["No citations found."]:
        print(f"  {citation}")
    print(f"{'─' * WIDTH}\n")


if __name__ == "__main__":
    main()
