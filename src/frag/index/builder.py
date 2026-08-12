"""Corpus -> chunks -> embeddings -> hybrid index.

Kept separate from `HybridStore` on purpose: the store knows how to persist and
query, this knows how to *build*. That split is what lets the benchmark rebuild
an index with different chunking while reusing every query path unchanged.

The index records its own `index_fingerprint` (a hash of chunking settings plus
the embedding model). `needs_rebuild()` compares that against the current config,
so switching retrieval or generation settings reuses the existing index while
switching chunk size correctly forces a rebuild. Re-embedding this corpus takes
minutes; getting that distinction wrong means either stale results or a lot of
wasted time.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import RunConfig
from ..ingest.chunkers import Chunk, build_chunker
from ..ingest.loaders import Document, corpus_stats, load_corpus
from ..runtime.foundry import FoundryRuntime
from .store import HybridStore

logger = logging.getLogger(__name__)


@dataclass
class BuildReport:
    """What happened during a build - goes straight into the write-up."""

    n_documents: int
    n_chunks: int
    embedding_dim: int
    chunking_strategy: str
    elapsed_s: float
    embed_s: float
    index_size_mb: float
    chunk_length_stats: dict[str, float]
    corpus: dict[str, Any]

    def summary(self) -> str:
        return (
            f"{self.n_documents} docs -> {self.n_chunks} chunks "
            f"({self.chunking_strategy}, dim={self.embedding_dim}) in "
            f"{self.elapsed_s:.1f}s (embedding {self.embed_s:.1f}s), "
            f"{self.index_size_mb} MB"
        )


def _length_stats(chunks: list[Chunk]) -> dict[str, float]:
    """Chunk length distribution.

    Reported because it is the fastest way to catch a broken chunker. A strategy
    whose median lands far from the configured `chunk_size`, or whose minimum
    sits at a handful of characters, is misbehaving - and that shows up here
    long before it shows up as a mysterious drop in retrieval quality.
    """
    if not chunks:
        return {}
    lengths = sorted(len(c.text) for c in chunks)
    n = len(lengths)
    return {
        "min": float(lengths[0]),
        "p25": float(lengths[n // 4]),
        "median": float(lengths[n // 2]),
        "p75": float(lengths[(3 * n) // 4]),
        "max": float(lengths[-1]),
        "mean": round(sum(lengths) / n, 1),
    }


def build_index(
    cfg: RunConfig,
    runtime: FoundryRuntime,
    corpus_dir: str | Path,
    index_path: str | Path,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> BuildReport:
    """Build (or reuse) the hybrid index for this configuration."""
    say = progress or (lambda m: logger.info(m))
    started = time.perf_counter()

    store = HybridStore(index_path)
    fingerprint = cfg.index_fingerprint()

    if not force and not needs_rebuild(store, cfg):
        say(f"Index is current ({store.count_chunks()} chunks) - reusing.")
        return BuildReport(
            n_documents=store.count_documents(),
            n_chunks=store.count_chunks(),
            embedding_dim=store.embedding_dim or 0,
            chunking_strategy=store.get_meta("chunking_strategy") or "?",
            elapsed_s=time.perf_counter() - started,
            embed_s=0.0,
            index_size_mb=store.stats()["size_mb"],
            chunk_length_stats={},
            corpus={},
        )

    say("Loading corpus...")
    docs: list[Document] = load_corpus(corpus_dir)
    if not docs:
        raise ValueError(f"No readable documents in {corpus_dir}")
    stats = corpus_stats(docs)
    say(f"  {stats['n_docs']} documents, {stats['total_chars']:,} characters")

    # The semantic chunker needs to embed sentences as it splits, so it gets the
    # runtime's embed function. The others ignore it.
    chunker = build_chunker(cfg.chunking, embed_fn=runtime.embed)
    say(f"Chunking ({cfg.chunking.strategy})...")

    all_chunks: list[Chunk] = []
    for doc in docs:
        chunks = chunker.split(
            doc.text, doc_id=doc.doc_id, source=doc.source,
            meta={"language": doc.language, "path": doc.path},
        )
        all_chunks.extend(chunks)
    say(f"  {len(all_chunks)} chunks")

    if not all_chunks:
        raise ValueError("Chunking produced nothing - check chunk_size vs corpus")

    say(f"Embedding {len(all_chunks)} chunks...")
    embed_started = time.perf_counter()
    vectors = runtime.embed([c.text for c in all_chunks])
    embed_s = time.perf_counter() - embed_started
    say(f"  done in {embed_s:.1f}s ({len(all_chunks) / max(embed_s, 1e-9):.1f} chunks/s)")

    say("Writing index...")
    store.reset()
    store.add_documents(docs)
    store.add_chunks(all_chunks, vectors)

    # Stamp the index so a later run can tell whether it is still valid.
    store.set_meta("index_fingerprint", fingerprint)
    store.set_meta("chunking_strategy", cfg.chunking.strategy)
    store.set_meta("embedding_model", cfg.runtime.embedding_model)
    store.set_meta("chunk_size", str(cfg.chunking.chunk_size))
    store.set_meta("overlap", str(cfg.chunking.overlap))

    report = BuildReport(
        n_documents=len(docs),
        n_chunks=len(all_chunks),
        embedding_dim=int(vectors.shape[1]),
        chunking_strategy=cfg.chunking.strategy,
        elapsed_s=time.perf_counter() - started,
        embed_s=embed_s,
        index_size_mb=store.stats()["size_mb"],
        chunk_length_stats=_length_stats(all_chunks),
        corpus=stats,
    )
    say(report.summary())
    store.close()
    return report


def needs_rebuild(store: HybridStore, cfg: RunConfig) -> bool:
    """True when the stored index no longer matches the configuration.

    Only chunking and the embedding model can invalidate an index. Retrieval and
    generation settings are applied at query time, so changing `top_k` or the
    fusion strategy must *not* trigger a costly re-embed - that distinction is
    what makes a parameter sweep practical on a laptop.
    """
    try:
        if store.count_chunks() == 0:
            return True
    except Exception:  # noqa: BLE001 - a missing/corrupt DB means rebuild
        return True
    return store.get_meta("index_fingerprint") != cfg.index_fingerprint()
