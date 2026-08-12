"""Tests for eval-set construction.

These run against a fake runtime instead of Foundry Local. That is not only for
speed: the properties being checked here - target sample size reached, no
duplicates, paraphrases aligned one-to-one - are exactly the ones whose failure
made the project's first benchmark invalid, and they must be verifiable without
an hour of model calls before every change.

    ./.venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frag.evaluation.qagen import (  # noqa: E402
    generate_questions,
    paraphrase_questions,
    question_gold_overlap,
    EvalQuestion,
)
from frag.ingest.normalize import lexical_tokens  # noqa: E402


class FakeRuntime:
    """Returns scripted replies, counting calls.

    `replies` is consumed in order; when it runs out the last one repeats, so a
    test only has to script the behaviour it cares about.
    """

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, temperature=None, max_tokens=None, **kwargs) -> str:
        self.calls += 1
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0]


class FakeStore:
    """Minimal stand-in for HybridStore: only `conn.execute(...).fetchall()`."""

    def __init__(self, n_chunks: int = 50, text: str | None = None) -> None:
        body = text or (
            "Öğrencinin kaydı, ilgili yönetmelikte belirtilen süreler içinde "
            "yenilenmediği takdirde askıya alınır ve durum öğrenciye yazılı "
            "olarak bildirilir. " * 3
        )
        self.rows = [
            {"chunk_id": f"c{i}", "doc_id": f"doc{i % 3}", "text": body}
            for i in range(n_chunks)
        ]
        self.conn = self

    def execute(self, sql, params=()):  # noqa: ARG002 - signature parity only
        return self

    def fetchall(self):
        return self.rows


GOOD = "Kaydımı yenilemezsem ne olur?"
BAD_SHORT = "Ne?"
BAD_REFERENTIAL = "Bu maddede kayıt yenileme süresi nedir?"


def test_overlap_is_high_when_question_quotes_the_chunk():
    chunk = "Öğrencinin kaydı süresi içinde yenilenmediği takdirde askıya alınır."
    quoting = "Öğrencinin kaydı yenilenmediği takdirde askıya alınır mı?"
    unrelated = "Yemekhanede bugün ne var?"
    assert question_gold_overlap(quoting, chunk) > question_gold_overlap(unrelated, chunk)
    assert question_gold_overlap(unrelated, chunk) < 0.15


def test_overlap_is_containment_not_jaccard():
    """A short question fully inside a long chunk must score 1.0.

    Under Jaccard the same pair scores near zero purely because the chunk is
    long, which is what made the first measurement of the eval set look like
    nothing was happening. The distinction is the point of the metric, so it is
    pinned here rather than left to the docstring.
    """
    question = "Takdirde askıya alınır?"          # every token occurs below
    long_chunk = (
        "Öğrencinin kaydı süresi içinde yenilenmediği takdirde askıya alınır. "
        + "Bu durum ilgili birime yazılı olarak bildirilir ve dosyasına işlenir. " * 8
    )
    assert question_gold_overlap(question, long_chunk) == pytest.approx(1.0)

    # The same pair under Jaccard, which the chunk's length drags towards zero.
    q = set(lexical_tokens(question))
    g = set(lexical_tokens(long_chunk))
    jaccard = len(q & g) / len(q | g)
    assert jaccard < 0.3


def test_overlap_is_zero_for_empty_input():
    assert question_gold_overlap("", "bir metin") == 0.0
    assert question_gold_overlap("bir soru", "") == 0.0


def test_generation_reaches_the_target_despite_rejections():
    """Every other reply is unusable; we must still get the number we asked for.

    This is the bug that produced the discarded 15-question benchmark: asking
    for n and sampling exactly n chunks yields n minus the rejections.
    """
    # Alternate a usable question with an unusable one, each usable one distinct
    # so the duplicate filter does not interfere.
    replies: list[str] = []
    for i in range(60):
        replies.append(f"Kaydımı {i} gün içinde yenilemezsem ne olur?")
        replies.append(BAD_SHORT)
    runtime = FakeRuntime(replies)

    questions = generate_questions(
        runtime, FakeStore(n_chunks=200), n_questions=20, progress=False
    )
    assert len(questions) == 20
    assert runtime.calls > 20, "should have retried past the rejected replies"


def test_generation_stops_at_the_attempt_cap():
    """A model that never produces anything usable must fail short, not loop."""
    runtime = FakeRuntime([BAD_SHORT])
    questions = generate_questions(
        runtime, FakeStore(n_chunks=500), n_questions=10,
        progress=False, max_attempts_factor=2.0,
    )
    assert questions == []
    assert runtime.calls <= 20


def test_duplicate_questions_are_dropped():
    """Two chunks yielding the same question would make one gold label wrong."""
    runtime = FakeRuntime([GOOD])
    questions = generate_questions(
        runtime, FakeStore(n_chunks=100), n_questions=5, progress=False
    )
    assert len(questions) == 1


def test_referential_questions_are_rejected():
    runtime = FakeRuntime([BAD_REFERENTIAL])
    questions = generate_questions(
        runtime, FakeStore(n_chunks=20), n_questions=3, progress=False
    )
    assert questions == []


def test_generated_questions_carry_gold_overlap():
    runtime = FakeRuntime([GOOD])
    q = generate_questions(runtime, FakeStore(), n_questions=1, progress=False)[0]
    assert "gold_overlap" in q.meta
    assert 0.0 <= q.meta["gold_overlap"] <= 1.0


# ---- paraphrasing ----------------------------------------------------------


def _originals(n: int = 4) -> list[EvalQuestion]:
    return [
        EvalQuestion(
            question=f"Kayıt yenileme süresi {i} gün müdür?",
            gold_chunk_ids=[f"c{i}"], answerable=True,
            source_doc="doc0", gold_answer="Kayıt yenileme süresi yönetmelikte belirtilir.",
        )
        for i in range(n)
    ]


def test_paraphrasing_preserves_alignment_when_the_model_fails():
    """A dropped paraphrase would unpair the two variant sets."""
    runtime = FakeRuntime([BAD_SHORT])
    originals = _originals(4)
    out = paraphrase_questions(runtime, originals, progress=False)

    assert len(out) == len(originals)
    for original, paraphrase in zip(originals, out):
        assert paraphrase.gold_chunk_ids == original.gold_chunk_ids
        # Nothing validated, so the original wording is kept and flagged.
        assert paraphrase.question == original.question
        assert paraphrase.meta["echo"] is True


def test_echoed_paraphrases_are_retried():
    """Returning the input verbatim must not be accepted on the first try."""
    originals = _originals(1)
    echo = originals[0].question
    runtime = FakeRuntime([echo, echo, "Kaydı kaç günde yenilemem gerekiyor abi?"])

    out = paraphrase_questions(runtime, originals, progress=False)
    assert runtime.calls == 3, "should retry while the reply echoes the input"
    assert out[0].question != echo
    assert out[0].meta["echo"] is False


def test_paraphrase_accepted_immediately_when_sufficiently_different():
    originals = _originals(1)
    runtime = FakeRuntime(["Kaydı kaç günde yenilemem gerekiyor abi?"])
    out = paraphrase_questions(runtime, originals, progress=False)
    assert runtime.calls == 1
    assert out[0].variant == "paraphrased"
    assert out[0].meta["original"] == originals[0].question


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
