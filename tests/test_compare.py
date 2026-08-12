"""Tests for the pairing rules in report comparison.

Both comparison functions exist to prevent the same class of error: producing a
plausible-looking p-value from two runs that were never comparable. The tests
below therefore care less about the arithmetic - `stats.py` owns that - than
about whether a misaligned pair is refused.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frag.evaluation.runner import (  # noqa: E402
    EvalReport,
    QueryRecord,
    compare,
    compare_by_gold,
)


def make_report(
    name: str,
    questions: list[str],
    golds: list[str],
    scores: list[float],
) -> EvalReport:
    records = [
        QueryRecord(
            question=q, answerable=True, retrieved_ids=[g],
            metrics={"ndcg@k": s}, latency_ms=1.0, top_score=s,
            gold_chunk_ids=[g],
        )
        for q, g, s in zip(questions, golds, scores)
    ]
    return EvalReport(
        config_name=name, fingerprint="test", n_queries=len(records),
        retrieval_metrics={}, answer_metrics={}, latency={}, records=records,
    )


ORIGINALS = ["Kayıt ne zaman yenilenir?", "Burs kesilir mi?", "Disiplin cezası nedir?"]
PARAPHRASES = ["Kaydı ne zaman yenilicem?", "Bursum uçar mı?", "Ceza yerse ne olur?"]
GOLDS = ["c1", "c2", "c3"]


def test_compare_refuses_different_question_sets():
    a = make_report("a", ORIGINALS, GOLDS, [1.0, 0.5, 0.0])
    b = make_report("b", PARAPHRASES, GOLDS, [1.0, 0.5, 0.0])
    with pytest.raises(ValueError, match="identical question sets"):
        compare(a, b)


def test_compare_by_gold_pairs_across_wordings():
    """Different questions, same gold: this is the variant comparison."""
    a = make_report("rrf", ORIGINALS, GOLDS, [1.0, 1.0, 1.0])
    b = make_report("rrf", PARAPHRASES, GOLDS, [0.0, 0.0, 0.0])
    result = compare_by_gold(a, b)
    assert result.test.mean_a == 1.0
    assert result.test.mean_b == 0.0
    assert result.test.difference == -1.0


def test_compare_by_gold_refuses_misaligned_gold():
    a = make_report("a", ORIGINALS, ["c1", "c2", "c3"], [1.0, 0.5, 0.0])
    b = make_report("b", PARAPHRASES, ["c1", "c3", "c2"], [1.0, 0.5, 0.0])
    with pytest.raises(ValueError, match="not aligned"):
        compare_by_gold(a, b)


def test_compare_by_gold_refuses_records_without_gold():
    """Results saved before gold ids were recorded must not be paired silently."""
    a = make_report("a", ORIGINALS, GOLDS, [1.0, 0.5, 0.0])
    for record in a.records:
        record.gold_chunk_ids = []
    b = make_report("b", PARAPHRASES, GOLDS, [1.0, 0.5, 0.0])
    with pytest.raises(ValueError, match="gold chunk ids"):
        compare_by_gold(a, b)


def test_identical_runs_give_p_one():
    a = make_report("a", ORIGINALS, GOLDS, [1.0, 0.5, 0.0])
    b = make_report("b", PARAPHRASES, GOLDS, [1.0, 0.5, 0.0])
    result = compare_by_gold(a, b)
    assert result.test.p_value == 1.0
    assert result.test.difference == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
