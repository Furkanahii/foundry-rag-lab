"""Combining two ranked lists into one.

The two retrieval arms produce scores that are not comparable. Dense returns
cosine similarity, bounded in [-1, 1] and - as measured on this corpus -
concentrated in a narrow band around 0.3-0.6. BM25 returns an unbounded sum of
per-term weights whose scale depends on document length, corpus size, and how
rare the query terms happen to be. On the query "CMPE 150" this project observed
BM25 = 0.998 against dense = 0.368 for the *same, correct* chunk.

Averaging those two numbers is meaningless. There are two principled ways out,
and both are implemented here.

## 1. Reciprocal Rank Fusion (rank-based)

Throw the scores away and keep only the ranks:

    RRF(d) = sum over arms of  1 / (k + rank_arm(d))

Ranks are ordinal and therefore comparable across arms by construction, which
sidesteps the calibration problem entirely rather than trying to solve it. The
constant k (60 by convention, from Cormack et al. 2009) controls how sharply
the top of a list dominates: as k -> 0 the top rank takes everything, and as
k -> infinity all ranks contribute equally and the fusion degenerates into
counting how many arms retrieved the document at all.

The 1/(k+r) shape matters. It is steeply decreasing, so moving from rank 1 to 2
costs much more than moving from rank 20 to 21 - which matches how relevance
actually decays down a result list. It also means a document ranked
moderately-well by *both* arms can beat a document ranked first by one arm and
missed by the other. That consensus-seeking behaviour is exactly what we want
when the two arms have independent failure modes.

This is Borda-count style rank aggregation, and the connection to social choice
theory is real: fusing rankers is formally the same problem as combining voter
preferences, and it inherits the same impossibility results.

## 2. Weighted score fusion (score-based)

Normalise each arm's scores onto [0, 1] within the retrieved set, then take a
convex combination:

    score(d) = alpha * dense_norm(d) + (1 - alpha) * lexical_norm(d)

This keeps magnitude information that RRF discards - the gap between rank 1 and
rank 2 is visible - at the cost of being sensitive to outliers and to how many
candidates were retrieved. alpha becomes a tunable knob, which is useful for
the benchmark and dangerous in production.

Which one wins is corpus-dependent, so both ship and the harness measures them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .store import SearchHit


@dataclass
class FusedHit:
    """A chunk after fusion, keeping the provenance of every contributing arm."""

    chunk_id: str
    hit: SearchHit
    fused_score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.hit.text

    @property
    def read_text(self) -> str:
        return self.hit.read_text

    @property
    def source(self) -> str:
        return self.hit.source

    def explain(self) -> str:
        """One-line trace of why this chunk surfaced.

        Shown in the dashboard. Being able to see that a chunk arrived at rank 2
        because BM25 loved it while dense ignored it is the difference between
        debugging a retrieval system and guessing at it.
        """
        parts = []
        if self.dense_rank is not None:
            parts.append(f"dense #{self.dense_rank} ({self.dense_score:.3f})")
        if self.lexical_rank is not None:
            parts.append(f"bm25 #{self.lexical_rank} ({self.lexical_score:.3f})")
        if self.rerank_score is not None:
            parts.append(f"rerank {self.rerank_score:.3f}")
        return " | ".join(parts) if parts else "no arm"


def reciprocal_rank_fusion(
    dense: Sequence[SearchHit],
    lexical: Sequence[SearchHit],
    k: int = 60,
    weights: tuple[float, float] = (1.0, 1.0),
) -> list[FusedHit]:
    """Fuse two ranked lists by reciprocal rank.

    `weights` scales each arm's contribution, letting the benchmark test whether
    one arm deserves more say on this corpus. Defaults to equal weight, which is
    the honest starting point when we have no evidence either way.
    """
    by_id: dict[str, FusedHit] = {}
    w_dense, w_lexical = weights

    for hit in dense:
        contribution = w_dense / (k + hit.rank)
        by_id[hit.chunk_id] = FusedHit(
            chunk_id=hit.chunk_id, hit=hit, fused_score=contribution,
            dense_rank=hit.rank, dense_score=hit.score,
            contributions={"dense": contribution},
        )

    for hit in lexical:
        contribution = w_lexical / (k + hit.rank)
        existing = by_id.get(hit.chunk_id)
        if existing is None:
            by_id[hit.chunk_id] = FusedHit(
                chunk_id=hit.chunk_id, hit=hit, fused_score=contribution,
                lexical_rank=hit.rank, lexical_score=hit.score,
                contributions={"lexical": contribution},
            )
        else:
            existing.fused_score += contribution
            existing.lexical_rank = hit.rank
            existing.lexical_score = hit.score
            existing.contributions["lexical"] = contribution

    return sorted(by_id.values(), key=lambda f: f.fused_score, reverse=True)


def _min_max_normalize(values: Sequence[float]) -> list[float]:
    """Scale to [0, 1] within this result set.

    Degenerate case handled deliberately: when every score is identical the
    range is zero, and we return 1.0 for all rather than 0.0. These are all
    retrieved documents - treating them as maximally relevant-but-tied is
    correct, whereas zeroing them would silently erase a whole arm from the
    fusion.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def weighted_fusion(
    dense: Sequence[SearchHit],
    lexical: Sequence[SearchHit],
    alpha: float = 0.5,
) -> list[FusedHit]:
    """Convex combination of min-max normalised scores.

    alpha = 1.0 is dense-only, alpha = 0.0 is lexical-only, so this function
    also implements the two single-arm baselines the benchmark needs.

    Caveat worth stating in the write-up: min-max normalisation is computed over
    the *retrieved* candidates, not the whole corpus. Change `dense_top_k` and
    every normalised score changes, so alpha tuned at one candidate depth does
    not transfer to another. RRF has no such coupling.
    """
    by_id: dict[str, FusedHit] = {}

    dense_norm = _min_max_normalize([h.score for h in dense])
    lexical_norm = _min_max_normalize([h.score for h in lexical])

    for hit, norm in zip(dense, dense_norm):
        contribution = alpha * norm
        by_id[hit.chunk_id] = FusedHit(
            chunk_id=hit.chunk_id, hit=hit, fused_score=contribution,
            dense_rank=hit.rank, dense_score=hit.score,
            contributions={"dense": contribution},
        )

    for hit, norm in zip(lexical, lexical_norm):
        contribution = (1.0 - alpha) * norm
        existing = by_id.get(hit.chunk_id)
        if existing is None:
            by_id[hit.chunk_id] = FusedHit(
                chunk_id=hit.chunk_id, hit=hit, fused_score=contribution,
                lexical_rank=hit.rank, lexical_score=hit.score,
                contributions={"lexical": contribution},
            )
        else:
            existing.fused_score += contribution
            existing.lexical_rank = hit.rank
            existing.lexical_score = hit.score
            existing.contributions["lexical"] = contribution

    return sorted(by_id.values(), key=lambda f: f.fused_score, reverse=True)


def maximal_marginal_relevance(
    candidates: Sequence[FusedHit],
    embeddings: dict[str, np.ndarray],
    query_vector: np.ndarray,
    lambda_param: float = 0.7,
    top_k: int = 5,
) -> list[FusedHit]:
    """Greedy MMR: trade relevance against redundancy.

    At each step pick the candidate maximising

        lambda * sim(d, query) - (1 - lambda) * max_{s in selected} sim(d, s)

    The problem it solves is specific and common: with overlapping chunks, the
    top-5 by pure relevance are frequently five near-duplicates of the same
    passage. That wastes the generator's whole context window on one fact and
    guarantees that anything requiring a second fact cannot be answered.

    lambda = 1.0 reduces to plain relevance ranking; lambda = 0.0 selects for
    pure diversity and ignores the query. The default 0.7 leans towards
    relevance, since on a small corpus over-diversifying pulls in genuinely
    irrelevant material.

    Greedy, not optimal: the exact problem is NP-hard (it contains max
    dispersion), but the objective is submodular, so the greedy solution carries
    the standard (1 - 1/e) approximation guarantee. That is far more than we
    need for k = 5.
    """
    if not candidates:
        return []

    query_vector = np.asarray(query_vector, dtype=np.float32).ravel()
    pool = [c for c in candidates if c.chunk_id in embeddings]
    # Candidates without a stored vector cannot be diversified against; keep
    # them in relevance order at the back rather than dropping them.
    orphans = [c for c in candidates if c.chunk_id not in embeddings]

    selected: list[FusedHit] = []
    while pool and len(selected) < top_k:
        best: FusedHit | None = None
        best_value = -np.inf

        for candidate in pool:
            vector = embeddings[candidate.chunk_id]
            relevance = float(np.dot(vector, query_vector))
            if selected:
                redundancy = max(
                    float(np.dot(vector, embeddings[s.chunk_id])) for s in selected
                )
            else:
                redundancy = 0.0
            value = lambda_param * relevance - (1.0 - lambda_param) * redundancy
            if value > best_value:
                best_value, best = value, candidate

        if best is None:
            break
        selected.append(best)
        pool.remove(best)

    if len(selected) < top_k:
        selected.extend(orphans[: top_k - len(selected)])
    return selected


def fuse(
    dense: Sequence[SearchHit],
    lexical: Sequence[SearchHit],
    strategy: str = "rrf",
    rrf_k: int = 60,
    alpha: float = 0.5,
) -> list[FusedHit]:
    """Dispatch to the configured fusion strategy.

    `dense_only` and `lexical_only` are expressed through weighted_fusion rather
    than special-cased, so all four strategies travel the same code path and a
    bug in normalisation cannot hide in a baseline.
    """
    if strategy == "rrf":
        return reciprocal_rank_fusion(dense, lexical, k=rrf_k)
    if strategy == "weighted":
        return weighted_fusion(dense, lexical, alpha=alpha)
    if strategy == "dense_only":
        return weighted_fusion(dense, [], alpha=1.0)
    if strategy == "lexical_only":
        return weighted_fusion([], lexical, alpha=0.0)
    raise ValueError(f"Unknown fusion strategy: {strategy!r}")
