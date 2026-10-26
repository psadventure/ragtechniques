"""
Run one query across collections x retrievers and print the top hits side by side.

Usage:
    python demo/compare.py "How do I create a transfer with the Python SDK?"
    python demo/compare.py "How do I verify a webhook signature?" --embedder text-embedding-005 --retrievers hybrid
    python demo/compare.py "..." --collections structure__minilm structure__text-embedding-005
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from rag import service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("--collections", nargs="+")
    parser.add_argument("--retrievers", nargs="+", choices=["semantic", "bm25", "hybrid"])
    parser.add_argument("--strategies", nargs="+")
    parser.add_argument("--embedder")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    out = service.compare(args.query, args.collections, args.retrievers, args.top_k, args.strategies, args.embedder)
    print(f"\nQuery: {args.query}")
    for cell in out["grid"]:
        print(f"\n── {cell['collection']} / {cell['retriever']} " + "─" * 20)
        for r in cell["results"]:
            print(f"  #{r['rank']}  [{r['doc_title']} | {r['section'][:55]}]")


if __name__ == "__main__":
    main()
