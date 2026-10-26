"""
Ingest documents using CHARACTER splitting.

Chunks are cut at fixed character intervals (no respect for sections,
code blocks, or tables). Use this to demonstrate poor RAG quality,
then run ingest_structure_based.py and compare — each strategy gets
its own collection, so nothing needs to be deleted in between.

Usage:
    python demo/ingest_char.py [--embedder minilm|text-embedding-005|gemini-embedding-001]
                               [--chunk-size 500] [--overlap 50] [--label exp1]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.ingest_structure_based import print_summary
from rag import service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedder", default=None)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--label")
    args = parser.parse_args()

    params = {k: v for k, v in {"chunk_size": args.chunk_size, "overlap": args.overlap}.items() if v is not None}
    print_summary(service.ingest("character", args.embedder, params, args.label))


if __name__ == "__main__":
    main()
