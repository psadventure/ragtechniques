"""
Demo 2: Document Ingestion Pipeline
Loads documents, applies structure-based (Markdown header) chunking, embeds chunks,
and persists them to their own ChromaDB collection on disk.

Usage:
    python demo/ingest_structure_based.py [--embedder minilm|text-embedding-005|gemini-embedding-001]
                                          [--max-words 400] [--label exp1]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from rag import service


def print_summary(result: dict) -> None:
    for doc, n in result["chunks_per_document"].items():
        print(f"  {doc}: {n} {result['strategy']} chunks")
    stats = result["stats"]
    print(f"\nIngestion complete ({result['strategy']} splitting):")
    print(f"  Collection          : {result['collection']}")
    print(f"  Embedder            : {result['embedder']}")
    print(f"  Params              : {result['params']}")
    print(f"  Total chunks stored : {result['total_chunks_stored']}")
    print(f"  Avg words/chunk     : {stats['avg_words']}")
    print(f"  Broken code blocks  : {stats['broken_code_blocks']}")
    print(f"  ChromaDB path       : {settings.CHROMA_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedder", default=None)
    parser.add_argument("--max-words", type=int)
    parser.add_argument("--label")
    args = parser.parse_args()

    params = {"max_words": args.max_words} if args.max_words else None
    print_summary(service.ingest("structure", args.embedder, params, args.label))


if __name__ == "__main__":
    main()
