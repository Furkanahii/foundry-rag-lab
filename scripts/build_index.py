"""Build the hybrid index from the corpus.

This is the first command to run on a fresh clone: every other script assumes
`data/index/bogazici.db` exists. The repository ships with that file already
built, so the usual reason to run this is that something upstream changed -
a new document in `data/corpus/`, a different chunking strategy, or a different
embedding model.

## When a rebuild is actually required

`RunConfig` hashes chunking settings and the embedding model separately from
everything else (`index_fingerprint`), because only those two change what is
*stored*. Retrieval and generation settings - fusion, alpha, top_k, thresholds -
are applied at query time and need no re-embedding. That separation is what
turns most experiments from a 20-minute rebuild into a 2-second config reload,
and it is why `--force` exists as an explicit flag rather than a default:
re-embedding 347 chunks is the single slowest operation in the project.

Usage:
    PYTHONPATH=src python scripts/build_index.py
    PYTHONPATH=src python scripts/build_index.py --force
    PYTHONPATH=src python scripts/build_index.py --strategy sentence_window
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, "src")

from frag.config import RunConfig  # noqa: E402
from frag.index.builder import build_index  # noqa: E402
from frag.runtime.foundry import FoundryRuntime  # noqa: E402

DEFAULT_CORPUS = "data/corpus"
DEFAULT_INDEX = "data/index/bogazici.db"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the dense + BM25 index from the document corpus."
    )
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--force", action="store_true",
                        help="rebuild even when the stored fingerprint matches")
    parser.add_argument("--strategy", default=None,
                        choices=["fixed", "recursive", "sentence_window", "semantic"],
                        help="override the chunking strategy")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--embedding-model", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = RunConfig()
    if args.strategy:
        cfg.chunking.strategy = args.strategy
    if args.chunk_size:
        cfg.chunking.chunk_size = args.chunk_size
    if args.overlap is not None:
        cfg.chunking.overlap = args.overlap
    if args.embedding_model:
        cfg.runtime.embedding_model = args.embedding_model

    print(f"corpus:    {args.corpus}")
    print(f"index:     {args.index}")
    print(f"chunking:  {cfg.chunking.strategy} "
          f"size={cfg.chunking.chunk_size} overlap={cfg.chunking.overlap}")
    print(f"embedding: {cfg.runtime.embedding_model}")
    print(f"fingerprint: {cfg.index_fingerprint()}\n")

    runtime = FoundryRuntime(cfg.runtime)
    report = build_index(
        cfg, runtime, args.corpus, args.index,
        force=args.force, progress=lambda m: print(f"  {m}", flush=True),
    )

    print(f"\n=== index ready ===")
    print(f"  documents      {report.n_documents}")
    print(f"  chunks         {report.n_chunks}")
    print(f"  embedding dim  {report.embedding_dim}")
    print(f"  strategy       {report.chunking_strategy}")
    print(f"  size           {report.index_size_mb} MB")
    print(f"  elapsed        {report.elapsed_s:.1f}s "
          f"(embedding {report.embed_s:.1f}s)")

    if report.chunk_length_stats:
        print("\n  chunk uzunlukları:")
        for key, value in report.chunk_length_stats.items():
            print(f"    {key:<10} {value}")

    # A rebuild invalidates every stored result: the gold chunk ids in the eval
    # set point at chunk ids from the previous build, and a benchmark compared
    # across two indexes measures the chunker, not the retriever.
    if args.force or report.embed_s > 0:
        print(
            "\nNOT: indeks yeniden kuruldu. data/eval/eval_set.json içindeki gold\n"
            "chunk id'leri eski indekse ait olabilir - eval setini yeniden üretin\n"
            "(scripts/build_eval_set.py), aksi halde benchmark sonuçları anlamsızdır."
        )

    runtime.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
