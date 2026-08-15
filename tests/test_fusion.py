"""Tests for rank fusion.

Fusion is where the project's central claim lives - that combining a dense arm
with a lexical arm beats either alone - so the arithmetic behind that claim
should not be taken on trust. Two properties matter most and neither is visible
from a benchmark score:

  * a document found by *both* arms must outrank one found by a single arm at
    the same position, since agreement between independent rankers is the whole
    reason to fuse;
  * the min-max normalisation in `weighted_fusion` is computed per query, which
    is why the fused score cannot serve as an abstention signal (docs/05). That
    is asserted here so the property is pinned rather than remembered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frag.index.fusion import (  # noqa: E402
    fuse,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from frag.index.store import SearchHit  # noqa: E402


def hit(chunk_id: str, rank: int, score: float, arm: str = "dense") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id, rowid=rank, text=f"metin {chunk_id}",
        source="doc.pdf", doc_id="doc", ordinal=rank,
        score=score, rank=rank, arm=arm,
    )


def ranked(hits: list) -> list[str]:
    return [h.chunk_id for h in hits]


# ---- RRF -------------------------------------------------------------------


def test_rrf_score_is_one_over_k_plus_rank():
    out = reciprocal_rank_fusion([hit("a", 1, 0.9)], [], k=60)
    assert out[0].fused_score == pytest.approx(1 / 61)


def test_rrf_sums_both_arms_for_a_shared_document():
    out = reciprocal_rank_fusion(
        [hit("a", 1, 0.9)], [hit("a", 3, 12.0, "lexical")], k=60
    )
    assert len(out) == 1
    assert out[0].fused_score == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_prefers_agreement_over_a_single_strong_arm():
    """`b` is top of one list; `a` is second in both. Agreement should win.

    1/62 + 1/62 = 0.03226  vs  1/61 = 0.01639
    """
    dense = [hit("b", 1, 0.95), hit("a", 2, 0.90)]
    lexical = [hit("c", 1, 20.0, "lexical"), hit("a", 2, 18.0, "lexical")]
    assert ranked(reciprocal_rank_fusion(dense, lexical, k=60))[0] == "a"


def test_rrf_k_controls_how_much_top_rank_dominates():
    """Small k sharpens the top of each list; large k flattens it."""
    dense = [hit("b", 1, 0.95), hit("a", 2, 0.90)]
    lexical = [hit("c", 1, 20.0, "lexical"), hit("a", 2, 18.0, "lexical")]

    # k=1: rank 1 scores 1/2 per arm, rank 2 scores 1/3 twice = 0.667 > 0.5.
    assert ranked(reciprocal_rank_fusion(dense, lexical, k=1))[0] == "a"
    # The ordering is stable at the default too - shown above - so the knob
    # changes margins here rather than the winner; that it is a knob at all is
    # what the benchmark sweeps.
    assert ranked(reciprocal_rank_fusion(dense, lexical, k=120))[0] == "a"


def test_rrf_records_provenance_from_both_arms():
    out = reciprocal_rank_fusion(
        [hit("a", 1, 0.9)], [hit("a", 4, 15.0, "lexical")], k=60
    )[0]
    assert out.dense_rank == 1 and out.dense_score == pytest.approx(0.9)
    assert out.lexical_rank == 4 and out.lexical_score == pytest.approx(15.0)
    assert "dense" in out.explain() and "bm25" in out.explain()


def test_rrf_ignores_raw_score_scale():
    """BM25 scores in the tens must not outweigh cosines below one.

    This is the reason fusion is done on ranks: the two arms produce numbers
    that are not on a common scale, and adding them directly would let BM25
    decide every query.
    """
    dense = [hit("a", 1, 0.4)]
    lexical = [hit("b", 1, 900.0, "lexical")]
    out = reciprocal_rank_fusion(dense, lexical, k=60)
    assert out[0].fused_score == pytest.approx(out[1].fused_score)


# ---- weighted --------------------------------------------------------------


def test_weighted_normalises_within_the_query():
    """Top of each arm always becomes 1.0 - the property docs/05 relies on."""
    out = weighted_fusion([hit("a", 1, 0.55), hit("b", 2, 0.10)], [], alpha=1.0)
    assert out[0].fused_score == pytest.approx(1.0)
    assert out[1].fused_score == pytest.approx(0.0)


def test_weighted_top_score_is_one_regardless_of_absolute_quality():
    """A weak best hit and a strong best hit both normalise to 1.0.

    This is exactly why `abstention_signal` defaults to a raw arm score: the
    fused number cannot distinguish "found something excellent" from "found the
    least bad of a poor set".
    """
    strong = weighted_fusion([hit("a", 1, 0.95), hit("b", 2, 0.10)], [], alpha=1.0)
    weak = weighted_fusion([hit("a", 1, 0.21), hit("b", 2, 0.05)], [], alpha=1.0)
    assert strong[0].fused_score == pytest.approx(weak[0].fused_score) == 1.0


def test_weighted_raw_scores_survive_normalisation():
    """The raw values must remain readable - abstention depends on them."""
    out = weighted_fusion([hit("a", 1, 0.21), hit("b", 2, 0.05)], [], alpha=1.0)
    assert out[0].dense_score == pytest.approx(0.21)


def test_weighted_alpha_splits_the_arms():
    dense = [hit("a", 1, 0.9)]
    lexical = [hit("b", 1, 30.0, "lexical")]
    out = {h.chunk_id: h.fused_score for h in weighted_fusion(dense, lexical, alpha=0.7)}
    assert out["a"] == pytest.approx(0.7)
    assert out["b"] == pytest.approx(0.3)


def test_weighted_identical_scores_normalise_to_one_not_zero():
    """Degenerate range: all retrieved, so all maximally relevant-but-tied.

    Returning 0.0 instead would erase a whole arm from the fusion whenever its
    scores happened to be flat.
    """
    out = weighted_fusion([hit("a", 1, 0.5), hit("b", 2, 0.5)], [], alpha=1.0)
    assert all(h.fused_score == pytest.approx(1.0) for h in out)


# ---- dispatch --------------------------------------------------------------


def test_dense_only_ignores_the_lexical_arm():
    out = fuse([hit("a", 1, 0.9)], [hit("b", 1, 30.0, "lexical")],
               strategy="dense_only")
    assert ranked(out) == ["a"]


def test_lexical_only_ignores_the_dense_arm():
    out = fuse([hit("a", 1, 0.9)], [hit("b", 1, 30.0, "lexical")],
               strategy="lexical_only")
    assert ranked(out) == ["b"]


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown fusion strategy"):
        fuse([], [], strategy="magic")


def test_results_are_sorted_by_fused_score():
    dense = [hit("a", 1, 0.9), hit("b", 2, 0.5), hit("c", 3, 0.1)]
    out = fuse(dense, [], strategy="rrf")
    scores = [h.fused_score for h in out]
    assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
