"""Retrieval quality metrics.

Each metric answers a different question, and using the wrong one produces
confident conclusions about the wrong thing.

  Recall@k     Did we retrieve the answer at all, anywhere in the top k?
               The ceiling on everything downstream: a passage the retriever
               never returns cannot be cited, reranked, or read. Optimise this
               first.

  Precision@k  What fraction of the k we returned were relevant?
               Matters because context is expensive - both in latency and,
               as this project measured, in instruction-following degradation.

  MRR          1 / (rank of the first relevant result), averaged.
               Right metric when the user needs *one* answer and reads from the
               top. Ignores everything after the first hit, on purpose.

  nDCG@k       Graded, position-discounted gain, normalised by the best
               possible ordering. The only one here that rewards putting a
               relevant passage at rank 1 over rank 4 while still crediting
               rank 4 over absence.

## Why nDCG discounts by log2(rank + 1)

DCG = sum over positions of  gain_i / log2(i + 1)

The logarithm encodes a claim about user behaviour: attention decays with rank,
but slowly - roughly like 1/log(rank), not 1/rank. A linear discount would say
position 10 is worth a tenth of position 1, which overstates how sharply real
users give up. The log is the standard compromise and, importantly, it is the
same for every system we compare, so it cannot favour one of ours.

Normalising by the *ideal* DCG is what makes scores comparable across queries.
A query with five relevant passages can accumulate more raw gain than one with a
single relevant passage; dividing by the best achievable ordering removes that,
so averaging across a query set is meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class RetrievalMetrics:
    """Metrics for one query."""

    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    first_relevant_rank: int | None
    n_relevant_retrieved: int
    n_relevant_total: int

    def as_dict(self) -> dict[str, float]:
        return {
            "recall@k": self.recall_at_k,
            "precision@k": self.precision_at_k,
            "mrr": self.reciprocal_rank,
            "ndcg@k": self.ndcg_at_k,
        }


def _relevance_vector(
    retrieved: Sequence[str], relevant: set[str], k: int
) -> list[int]:
    return [1 if cid in relevant else 0 for cid in retrieved[:k]]


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain.

    Position i (1-based) is discounted by log2(i + 1), so position 1 has
    discount log2(2) = 1 - i.e. no discount - and everything after is worth
    progressively less.
    """
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(
    retrieved: Sequence[str], relevant: set[str], k: int
) -> float:
    """Normalised DCG.

    The ideal ranking places every relevant document first, so the ideal DCG
    uses min(len(relevant), k) ones. When there are no relevant documents at
    all the metric is undefined; we return 0.0, and the caller is expected to
    exclude such queries from the average rather than let them drag it down.
    """
    if not relevant:
        return 0.0
    gains = _relevance_vector(retrieved, relevant, k)
    ideal = [1] * min(len(relevant), k)
    ideal_dcg = dcg(ideal)
    return dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_retrieval(
    retrieved: Sequence[str], relevant: Iterable[str], k: int = 5
) -> RetrievalMetrics:
    """Compute all retrieval metrics for a single query."""
    relevant_set = set(relevant)
    top = list(retrieved[:k])
    hits = [cid for cid in top if cid in relevant_set]

    first_rank: int | None = None
    for i, cid in enumerate(top, start=1):
        if cid in relevant_set:
            first_rank = i
            break

    n_relevant = len(relevant_set)
    return RetrievalMetrics(
        # Denominator is min(n_relevant, k): with 8 relevant passages and k=5,
        # perfect retrieval still only returns 5, and scoring that as 0.625
        # would penalise a flawless system for the value of k.
        recall_at_k=len(hits) / min(n_relevant, k) if n_relevant else 0.0,
        precision_at_k=len(hits) / len(top) if top else 0.0,
        reciprocal_rank=1.0 / first_rank if first_rank else 0.0,
        ndcg_at_k=ndcg_at_k(top, relevant_set, k),
        first_relevant_rank=first_rank,
        n_relevant_retrieved=len(hits),
        n_relevant_total=n_relevant,
    )


def aggregate(metrics: Sequence[RetrievalMetrics]) -> dict[str, float]:
    """Mean of each metric across queries.

    A plain mean is correct here (every query counts once) but it is only half
    the story - see `stats.bootstrap_ci` for the uncertainty that must be
    reported alongside it. A mean without an interval invites reading noise as
    improvement.
    """
    if not metrics:
        return {}
    n = len(metrics)
    return {
        "recall@k": sum(m.recall_at_k for m in metrics) / n,
        "precision@k": sum(m.precision_at_k for m in metrics) / n,
        "mrr": sum(m.reciprocal_rank for m in metrics) / n,
        "ndcg@k": sum(m.ndcg_at_k for m in metrics) / n,
        "hit_rate": sum(1 for m in metrics if m.first_relevant_rank) / n,
        "n_queries": float(n),
    }


# ---------------------------------------------------------------------------
# answer-level metrics
# ---------------------------------------------------------------------------


@dataclass
class AnswerMetrics:
    """Metrics for a generated answer."""

    abstained: bool
    should_abstain: bool
    has_citations: bool
    has_invalid_citations: bool
    citation_precision: float  # fraction of cited passages that were relevant

    @property
    def abstention_correct(self) -> bool:
        return self.abstained == self.should_abstain


def evaluate_answer(
    abstained: bool,
    should_abstain: bool,
    citations: Sequence[int],
    invalid_citations: Sequence[int],
    retrieved_ids: Sequence[str],
    relevant: Iterable[str],
) -> AnswerMetrics:
    """Score one answer.

    `citation_precision` checks something a text-similarity score cannot: that
    the passages the model *pointed at* are the ones that actually contain the
    answer. A model can produce a correct-sounding answer while citing the wrong
    passage, and that is a failure worth catching - it means the reasoning was
    not grounded even though the output looks fine.
    """
    relevant_set = set(relevant)
    cited_ids = [
        retrieved_ids[c - 1] for c in citations
        if 1 <= c <= len(retrieved_ids)
    ]
    correct_cites = [cid for cid in cited_ids if cid in relevant_set]

    return AnswerMetrics(
        abstained=abstained,
        should_abstain=should_abstain,
        has_citations=bool(citations),
        has_invalid_citations=bool(invalid_citations),
        citation_precision=(
            len(correct_cites) / len(cited_ids) if cited_ids else 0.0
        ),
    )


def aggregate_answers(metrics: Sequence[AnswerMetrics]) -> dict[str, float]:
    """Aggregate answer metrics, splitting abstention by class.

    Reporting a single "abstention accuracy" would hide the trade-off that
    matters. A system that refuses everything scores perfectly on unanswerable
    questions and uselessly on answerable ones. Both rates are reported so the
    trade is visible.
    """
    if not metrics:
        return {}
    n = len(metrics)
    should = [m for m in metrics if m.should_abstain]
    should_not = [m for m in metrics if not m.should_abstain]

    out: dict[str, float] = {
        "n_answers": float(n),
        "abstention_accuracy": sum(1 for m in metrics if m.abstention_correct) / n,
        "citation_rate": sum(1 for m in metrics if m.has_citations) / n,
        "invalid_citation_rate": sum(
            1 for m in metrics if m.has_invalid_citations
        ) / n,
    }
    if should:
        # True negative rate: correctly refused when it should have.
        out["refusal_recall"] = sum(1 for m in should if m.abstained) / len(should)
    if should_not:
        # Answered when it could have - the cost side of being cautious.
        out["answer_rate"] = sum(1 for m in should_not if not m.abstained) / len(should_not)
        out["citation_precision"] = sum(
            m.citation_precision for m in should_not
        ) / len(should_not)
    return out
