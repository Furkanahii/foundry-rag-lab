"""The full retrieval pipeline, from question to a ranked, deduplicated context.

    query
      │
      ├─► embed (optionally instruction-prefixed) ─► dense search  ─┐
      │                                                             ├─► fuse
      └─► fold + stem ────────────────────────────► BM25 search ───┘     │
                                                                          ▼
                                                              rerank (optional)
                                                                          │
                                                                          ▼
                                                                 MMR (optional)
                                                                          │
                                                                          ▼
                                                                      top_k

Every stage is optional and every stage is recorded. `RetrievalResult` keeps the
candidate list at each step, which is what makes the dashboard able to show
*why* a chunk ended up where it did - and what makes an ablation study possible
without re-plumbing the code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import RetrievalConfig
from ..index.fusion import FusedHit, fuse, maximal_marginal_relevance
from ..index.store import HybridStore
from ..runtime.foundry import FoundryRuntime
from .rerank import apply_query_instruction, rerank

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Everything the pipeline produced, including intermediate stages."""

    query: str
    hits: list[FusedHit]                    # final top_k
    n_dense: int = 0
    n_lexical: int = 0
    n_fused: int = 0
    reranked: bool = False
    mmr_applied: bool = False
    elapsed_ms: float = 0.0
    stage_ms: dict[str, float] = field(default_factory=dict)
    query_vector: np.ndarray | None = None

    @property
    def top_score(self) -> float:
        """Fused score of the best hit - the abstention signal.

        Returns -inf rather than 0.0 for an empty result so that "found nothing"
        can never accidentally clear a threshold at or below zero.
        """
        return self.hits[0].fused_score if self.hits else float("-inf")

    def context_blocks(self) -> list[tuple[int, str, str]]:
        """(citation_number, text, source) triples for the prompt builder."""
        return [
            (i, h.read_text.strip(), h.hit.citation_label()
             if hasattr(h.hit, "citation_label") else h.source)
            for i, h in enumerate(self.hits, start=1)
        ]

    def explain(self) -> str:
        lines = [f"query={self.query!r}  ({self.elapsed_ms:.0f} ms)"]
        lines.append(
            f"  dense={self.n_dense} lexical={self.n_lexical} fused={self.n_fused}"
            f" rerank={self.reranked} mmr={self.mmr_applied}"
        )
        for i, hit in enumerate(self.hits, 1):
            lines.append(f"  {i}. [{hit.hit.doc_id}] {hit.explain()}")
        return "\n".join(lines)


class Retriever:
    """Owns the retrieval half of the system.

    Deliberately knows nothing about prompts or generation: this class answers
    "which passages", and `generate/` answers "what to say about them". Keeping
    them apart is what lets the benchmark measure retrieval quality on its own,
    without a language model's variance contaminating the number.
    """

    def __init__(
        self,
        cfg: RetrievalConfig,
        runtime: FoundryRuntime,
        store: HybridStore,
    ) -> None:
        self.cfg = cfg
        self.runtime = runtime
        self.store = store
        self._vector_cache: dict[str, np.ndarray] = {}

    def retrieve(self, query: str) -> RetrievalResult:
        started = time.perf_counter()
        stage: dict[str, float] = {}

        if not query.strip():
            return RetrievalResult(query=query, hits=[])

        # --- dense arm ------------------------------------------------------
        t0 = time.perf_counter()
        embed_input = apply_query_instruction(query, self.cfg.query_instruction)
        query_vector = self.runtime.embed_one(embed_input)
        dense = self.store.search_dense(query_vector, top_k=self.cfg.dense_top_k)
        stage["dense_ms"] = (time.perf_counter() - t0) * 1000

        # --- lexical arm ----------------------------------------------------
        t0 = time.perf_counter()
        lexical = self.store.search_lexical(
            query, top_k=self.cfg.lexical_top_k, stem=self.cfg.lexical_stem
        )
        stage["lexical_ms"] = (time.perf_counter() - t0) * 1000

        # --- fusion ---------------------------------------------------------
        t0 = time.perf_counter()
        fused = fuse(
            dense, lexical,
            strategy=self.cfg.fusion, rrf_k=self.cfg.rrf_k, alpha=self.cfg.alpha,
        )
        stage["fusion_ms"] = (time.perf_counter() - t0) * 1000
        n_fused = len(fused)

        # --- rerank ---------------------------------------------------------
        reranked = False
        if self.cfg.rerank != "none" and len(fused) > 1:
            t0 = time.perf_counter()
            fused = rerank(
                self.runtime, query, fused,
                strategy=self.cfg.rerank, max_candidates=self.cfg.rerank_candidates,
            )
            stage["rerank_ms"] = (time.perf_counter() - t0) * 1000
            reranked = True

        # --- MMR ------------------------------------------------------------
        mmr_applied = False
        if self.cfg.use_mmr and len(fused) > self.cfg.top_k:
            t0 = time.perf_counter()
            # MMR needs chunk vectors; fetch only for the candidates in play.
            pool = fused[: max(self.cfg.rerank_candidates, self.cfg.top_k * 3)]
            vectors = self._chunk_vectors([h.chunk_id for h in pool])
            if vectors:
                fused = maximal_marginal_relevance(
                    pool, vectors, query_vector,
                    lambda_param=self.cfg.mmr_lambda, top_k=self.cfg.top_k,
                )
                mmr_applied = True
            stage["mmr_ms"] = (time.perf_counter() - t0) * 1000

        top = fused[: self.cfg.top_k]
        return RetrievalResult(
            query=query, hits=top,
            n_dense=len(dense), n_lexical=len(lexical), n_fused=n_fused,
            reranked=reranked, mmr_applied=mmr_applied,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            stage_ms=stage, query_vector=query_vector,
        )

    def _chunk_vectors(self, chunk_ids: list[str]) -> dict[str, np.ndarray]:
        """Load stored embeddings for specific chunks, with a session cache.

        Re-embedding here would be both slow and *wrong*: MMR must compare the
        vectors that are actually in the index, not freshly computed ones that
        could differ if the model or its settings changed.
        """
        missing = [c for c in chunk_ids if c not in self._vector_cache]
        if missing:
            placeholders = ",".join("?" * len(missing))
            rows = self.store.conn.execute(
                f"""
                SELECT c.chunk_id AS chunk_id, v.embedding AS embedding
                FROM chunks c JOIN chunk_vec v ON v.chunk_rowid = c.rowid
                WHERE c.chunk_id IN ({placeholders})
                """,
                missing,
            ).fetchall()
            for row in rows:
                self._vector_cache[row["chunk_id"]] = np.frombuffer(
                    row["embedding"], dtype=np.float32
                )
        return {c: self._vector_cache[c] for c in chunk_ids if c in self._vector_cache}

    def describe(self) -> dict[str, Any]:
        return {
            "fusion": self.cfg.fusion,
            "dense_top_k": self.cfg.dense_top_k,
            "lexical_top_k": self.cfg.lexical_top_k,
            "rerank": self.cfg.rerank,
            "mmr": self.cfg.use_mmr,
            "top_k": self.cfg.top_k,
            "query_instruction": self.cfg.query_instruction,
        }
