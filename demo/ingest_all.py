"""
Ingest documents with EVERY chunking strategy (one collection each), optionally for
several embedders, then compare retrieval across them with demo/compare.py.

Usage:
    python demo/ingest_all.py                                   # all strategies, default embedder
    python demo/ingest_all.py --embedders minilm text-embedding-005
    python demo/ingest_all.py --strategies character --chunk-size 300 --label cs300
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategies", nargs="+")
    parser.add_argument("--embedders", nargs="+")
    parser.add_argument("--label")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--max-words", type=int)
    args = parser.parse_args()

    params = {}
    char = {k: v for k, v in {"chunk_size": args.chunk_size, "overlap": args.overlap}.items() if v is not None}
    if char:
        params["character"] = char
    if args.max_words:
        params["structure"] = {"max_words": args.max_words}

    result = service.ingest_many(args.strategies, args.embedders, None, args.label, params)

    print(f"\n{'COLLECTION':<36}{'CHUNKS':>7}{'AVG WORDS':>11}{'BROKEN CODE':>13}")
    print("─" * 67)
    for c in result["collections"]:
        print(f"{c['collection']:<36}{c['total_chunks_stored']:>7}{c['avg_words']:>11}{c['broken_code_blocks']:>13}")
    for c in result["collections"]:
        if c["removed_documents"]:
            print(f"\nPruned from {c['collection']} (no longer in docs/): {', '.join(c['removed_documents'])}")
    for e in result["errors"]:
        print(f"\nFAILED {e['strategy']} / {e['embedder']}: {e['error']}")


if __name__ == "__main__":
    main()
