"""Tests for the retrieval metrics, against values worked out by hand.

Every number this project reports passes through these four functions. A silent
error here would not announce itself: nDCG computed with the wrong discount
still produces plausible values between 0 and 1, still orders configurations,
and still yields significant p-values - it would simply be measuring something
other than ranking quality. So the assertions below are closed-form values
derived from the definitions rather than snapshots of current behaviour, which
would only pin the bug in place.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frag.evaluation.metrics import (  # noqa: E402
    aggregate,
    dcg,
    evaluate_retrieval,
    ndcg_at_k,
)


# ---- DCG -------------------------------------------------------------------


def test_dcg_first_position_is_undiscounted():
    """Position 1 divides by log2(2) = 1."""
    assert dcg([1.0]) == pytest.approx(1.0)


def test_dcg_applies_log2_discount_by_position():
    # 1/log2(2) + 1/log2(3) + 1/log2(4)
    expected = 1.0 + 1 / math.log2(3) + 1 / math.log2(4)
    assert dcg([1.0, 1.0, 1.0]) == pytest.approx(expected)


def test_dcg_rewards_earlier_placement():
    assert dcg([1.0, 0.0, 0.0]) > dcg([0.0, 1.0, 0.0]) > dcg([0.0, 0.0, 1.0])


# ---- nDCG ------------------------------------------------------------------


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a"}, k=5) == pytest.approx(1.0)


def test_ndcg_single_relevant_at_rank_two():
    """One relevant document at position 2: DCG = 1/log2(3), ideal = 1."""
    assert ndcg_at_k(["x", "a"], {"a"}, k=5) == pytest.approx(1 / math.log2(3))


def test_ndcg_is_zero_when_nothing_relevant_retrieved():
    assert ndcg_at_k(["x", "y"], {"a"}, k=5) == 0.0


def test_ndcg_without_relevant_set_is_zero_not_error():
    """Undefined by definition; the contract is 0.0, not an exception."""
    assert ndcg_at_k(["a"], set(), k=5) == 0.0


def test_ndcg_respects_k():
    """A hit outside the cut-off does not count."""
    assert ndcg_at_k(["x", "x", "x", "a"], {"a"}, k=3) == 0.0
    assert ndcg_at_k(["x", "x", "x", "a"], {"a"}, k=4) > 0.0


def test_ndcg_ideal_uses_min_relevant_k():
    """With more relevant documents than k, the ideal is k ones, not n ones.

    Otherwise a system that returns k relevant documents - everything it
    possibly could - would score below 1.0 for a reason that has nothing to do
    with its ranking.
    """
    assert ndcg_at_k(["a", "b"], {"a", "b", "c", "d"}, k=2) == pytest.approx(1.0)


# ---- recall / precision / MRR ----------------------------------------------


def test_recall_denominator_is_capped_at_k():
    """8 relevant passages, k=5: retrieving 5 of them is recall 1.0."""
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "b", "c", "d", "e", "f", "g", "h"}
    m = evaluate_retrieval(retrieved, relevant, k=5)
    assert m.recall_at_k == pytest.approx(1.0)


def test_precision_counts_against_returned_length():
    m = evaluate_retrieval(["a", "x", "y", "z"], {"a"}, k=4)
    assert m.precision_at_k == pytest.approx(0.25)


def test_reciprocal_rank_is_one_over_first_hit():
    assert evaluate_retrieval(["x", "x", "a"], {"a"}, k=5).reciprocal_rank == \
        pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_on_miss():
    assert evaluate_retrieval(["x", "y"], {"a"}, k=5).reciprocal_rank == 0.0


def test_first_relevant_rank_is_one_based():
    """Off-by-one here would silently inflate MRR by ranking from zero."""
    assert evaluate_retrieval(["a"], {"a"}, k=5).first_relevant_rank == 1


def test_empty_retrieval_is_all_zeros_not_an_error():
    m = evaluate_retrieval([], {"a"}, k=5)
    assert m.recall_at_k == 0.0
    assert m.precision_at_k == 0.0
    assert m.reciprocal_rank == 0.0
    assert m.ndcg_at_k == 0.0
    assert m.first_relevant_rank is None


def test_metrics_are_consistent_on_a_worked_example():
    """One relevant document, retrieved at rank 3, k=5.

    recall    = 1/1  = 1.0     (one relevant, found)
    precision = 1/5  = 0.2
    MRR       = 1/3
    nDCG      = (1/log2(4)) / 1 = 0.5
    """
    m = evaluate_retrieval(["x", "y", "a", "z", "w"], {"a"}, k=5)
    assert m.recall_at_k == pytest.approx(1.0)
    assert m.precision_at_k == pytest.approx(0.2)
    assert m.reciprocal_rank == pytest.approx(1 / 3)
    assert m.ndcg_at_k == pytest.approx(0.5)


# ---- aggregation -----------------------------------------------------------


def test_aggregate_averages_each_metric():
    a = evaluate_retrieval(["a"], {"a"}, k=5)          # perfect
    b = evaluate_retrieval(["x"], {"a"}, k=5)          # miss
    out = aggregate([a, b])
    assert out["recall@k"] == pytest.approx(0.5)
    assert out["mrr"] == pytest.approx(0.5)
    assert out["ndcg@k"] == pytest.approx(0.5)


def test_aggregate_of_nothing_is_empty_not_a_crash():
    assert aggregate([]) == {} or all(v == 0 for v in aggregate([]).values())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
