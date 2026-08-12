"""Running configurations against the eval set and comparing them properly.

Two levels of evaluation, kept separate because they cost wildly different
amounts and answer different questions.

**Retrieval-only** (`evaluate_retrieval_config`) makes no LLM calls. On this
machine that is ~0.2 s per query versus ~30 s for generation, so a 12-point
parameter sweep is minutes instead of hours. Most design decisions - chunking,
fusion, stemming, top_k - only affect retrieval, and measuring them without a
language model in the loop also removes that model's variance from the number.

**End-to-end** (`evaluate_end_to_end`) adds generation, and measures the things
retrieval metrics cannot see: abstention behaviour, citation validity, latency
as the user experiences it.

Everything is stored per-query, never pre-aggregated. Paired significance
testing needs the individual scores, and a mean cannot be un-averaged later.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..config import RunConfig
from ..pipeline import RagPipeline
from .metrics import (
    AnswerMetrics,
    RetrievalMetrics,
    aggregate,
    aggregate_answers,
    evaluate_answer,
    evaluate_retrieval,
)
from .qagen import EvalQuestion
from .stats import ConfidenceInterval, TestResult, bootstrap_ci, paired_permutation_test

logger = logging.getLogger(__name__)


@dataclass
class QueryRecord:
    """Everything measured for one query under one configuration."""

    question: str
    answerable: bool
    retrieved_ids: list[str]
    metrics: dict[str, float]
    latency_ms: float
    top_score: float
    # Stored alongside the retrieved ids so a saved result is self-contained:
    # the gold label is what makes the metrics checkable after the fact, and it
    # is what lets two runs on *differently worded* questions be verified as
    # covering the same underlying items (see `compare_by_gold`).
    gold_chunk_ids: list[str] = field(default_factory=list)
    variant: str = ""
    answer_text: str = ""
    abstained: bool = False
    citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    generation_ms: float = 0.0


@dataclass
class EvalReport:
    """Aggregate results for one configuration."""

    config_name: str
    fingerprint: str
    n_queries: int
    retrieval_metrics: dict[str, float]
    answer_metrics: dict[str, float]
    latency: dict[str, float]
    records: list[QueryRecord]
    intervals: dict[str, ConfidenceInterval] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def per_query(self, metric: str) -> list[float]:
        """Per-query values, needed for paired testing."""
        return [r.metrics.get(metric, 0.0) for r in self.records]

    def summary(self) -> str:
        lines = [f"=== {self.config_name} ({self.fingerprint}) ==="]
        for name, ci in self.intervals.items():
            lines.append(f"  {name:<14} {ci}")
        for key, value in self.answer_metrics.items():
            lines.append(f"  {key:<14} {value:.4f}")
        lines.append(
            f"  latency        p50={self.latency.get('p50_ms', 0):.0f}ms "
            f"p95={self.latency.get('p95_ms', 0):.0f}ms"
        )
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config_name": self.config_name,
            "fingerprint": self.fingerprint,
            "n_queries": self.n_queries,
            "retrieval_metrics": self.retrieval_metrics,
            "answer_metrics": self.answer_metrics,
            "latency": self.latency,
            "intervals": {
                k: {"estimate": v.estimate, "lower": v.lower, "upper": v.upper,
                    "n": v.n}
                for k, v in self.intervals.items()
            },
            "config": self.config,
            "elapsed_s": self.elapsed_s,
            "records": [asdict(r) for r in self.records],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "p50_ms": ordered[n // 2],
        "p95_ms": ordered[min(n - 1, int(round(0.95 * n)) - 1 if n > 1 else 0)],
        "max_ms": ordered[-1],
        "mean_ms": sum(ordered) / n,
    }


def evaluate_retrieval_config(
    cfg: RunConfig,
    index_path: str | Path,
    questions: Sequence[EvalQuestion],
    runtime: Any = None,
    progress: bool = True,
) -> EvalReport:
    """Retrieval-only evaluation. No generation, no LLM calls."""
    started = time.perf_counter()
    pipeline = RagPipeline(cfg, index_path, runtime=runtime)

    records: list[QueryRecord] = []
    metric_objects: list[RetrievalMetrics] = []
    latencies: list[float] = []

    answerable = [q for q in questions if q.answerable]
    for i, question in enumerate(answerable, start=1):
        result = pipeline.retrieve(question.question)
        retrieved_ids = [h.chunk_id for h in result.hits]
        m = evaluate_retrieval(retrieved_ids, question.gold_chunk_ids, k=cfg.retrieval.top_k)
        metric_objects.append(m)
        latencies.append(result.elapsed_ms)
        records.append(
            QueryRecord(
                question=question.question, answerable=True,
                retrieved_ids=retrieved_ids, metrics=m.as_dict(),
                latency_ms=result.elapsed_ms, top_score=result.top_score,
                gold_chunk_ids=list(question.gold_chunk_ids),
                variant=question.variant,
            )
        )
        if progress and i % 10 == 0:
            print(f"  {i}/{len(answerable)}...", flush=True)

    aggregated = aggregate(metric_objects)
    intervals = {
        name: bootstrap_ci([r.metrics[name] for r in records])
        for name in ("recall@k", "mrr", "ndcg@k")
        if records
    }

    pipeline.close()
    return EvalReport(
        config_name=cfg.name, fingerprint=cfg.fingerprint(),
        n_queries=len(records), retrieval_metrics=aggregated,
        answer_metrics={}, latency=_latency_summary(latencies),
        records=records, intervals=intervals,
        config=cfg.to_dict(), elapsed_s=time.perf_counter() - started,
    )


def evaluate_end_to_end(
    cfg: RunConfig,
    index_path: str | Path,
    questions: Sequence[EvalQuestion],
    runtime: Any = None,
    progress: bool = True,
) -> EvalReport:
    """Full evaluation including generation, abstention, and citations."""
    started = time.perf_counter()
    pipeline = RagPipeline(cfg, index_path, runtime=runtime)

    records: list[QueryRecord] = []
    retrieval_objects: list[RetrievalMetrics] = []
    answer_objects: list[AnswerMetrics] = []
    latencies: list[float] = []

    for i, question in enumerate(questions, start=1):
        answer = pipeline.ask(question.question)
        result = answer.retrieval
        retrieved_ids = [h.chunk_id for h in result.hits]

        # Retrieval metrics only make sense for answerable questions: there is
        # no gold passage to find for the others.
        metrics: dict[str, float] = {}
        if question.answerable:
            m = evaluate_retrieval(
                retrieved_ids, question.gold_chunk_ids, k=cfg.retrieval.top_k
            )
            retrieval_objects.append(m)
            metrics = m.as_dict()

        answer_objects.append(
            evaluate_answer(
                abstained=answer.abstained,
                should_abstain=not question.answerable,
                citations=answer.citations,
                invalid_citations=answer.invalid_citations,
                retrieved_ids=retrieved_ids,
                relevant=question.gold_chunk_ids,
            )
        )
        latencies.append(answer.elapsed_ms)
        records.append(
            QueryRecord(
                question=question.question, answerable=question.answerable,
                retrieved_ids=retrieved_ids, metrics=metrics,
                latency_ms=answer.elapsed_ms, top_score=result.top_score,
                gold_chunk_ids=list(question.gold_chunk_ids),
                variant=question.variant,
                answer_text=answer.text[:500], abstained=answer.abstained,
                citations=answer.citations,
                invalid_citations=answer.invalid_citations,
                generation_ms=answer.generation_ms,
            )
        )
        if progress:
            print(f"  {i}/{len(questions)} ({answer.elapsed_ms/1000:.1f}s)", flush=True)

    answerable_records = [r for r in records if r.answerable]
    intervals = {
        name: bootstrap_ci([r.metrics.get(name, 0.0) for r in answerable_records])
        for name in ("recall@k", "mrr", "ndcg@k")
        if answerable_records
    }

    pipeline.close()
    return EvalReport(
        config_name=cfg.name, fingerprint=cfg.fingerprint(),
        n_queries=len(records), retrieval_metrics=aggregate(retrieval_objects),
        answer_metrics=aggregate_answers(answer_objects),
        latency=_latency_summary(latencies), records=records,
        intervals=intervals, config=cfg.to_dict(),
        elapsed_s=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    """Statistical comparison of two configurations on one metric."""

    metric: str
    name_a: str
    name_b: str
    test: TestResult
    ci_a: ConfidenceInterval
    ci_b: ConfidenceInterval

    def report(self) -> str:
        return (
            f"{self.metric}\n"
            f"  A={self.name_a}: {self.ci_a}\n"
            f"  B={self.name_b}: {self.ci_b}\n"
            f"  -> {self.test.interpret()}"
        )


def compare(
    report_a: EvalReport, report_b: EvalReport, metric: str = "ndcg@k"
) -> Comparison:
    """Paired comparison of two reports.

    Requires that both were run on the same questions in the same order - that
    is the whole basis of the pairing. We check it rather than trust it, because
    silently comparing misaligned runs would produce a plausible-looking number
    that means nothing.
    """
    questions_a = [r.question for r in report_a.records if r.answerable]
    questions_b = [r.question for r in report_b.records if r.answerable]
    if questions_a != questions_b:
        raise ValueError(
            "Paired comparison requires identical question sets in identical "
            "order. Run both configurations against the same saved eval set."
        )

    a_values = [r.metrics.get(metric, 0.0) for r in report_a.records if r.answerable]
    b_values = [r.metrics.get(metric, 0.0) for r in report_b.records if r.answerable]

    return Comparison(
        metric=metric,
        name_a=report_a.config_name, name_b=report_b.config_name,
        test=paired_permutation_test(a_values, b_values),
        ci_a=bootstrap_ci(a_values), ci_b=bootstrap_ci(b_values),
    )


def compare_by_gold(
    report_a: EvalReport, report_b: EvalReport, metric: str = "ndcg@k"
) -> Comparison:
    """Paired comparison of two runs whose questions differ but whose gold does not.

    `compare` pairs on the question text, which is right for two configurations
    facing the same queries and wrong for the comparison this project needs
    most: the *same* configuration facing the original and the paraphrased
    wording of the same items. Those runs are paired by construction - item i of
    each asks about the same passage - but no two question strings match.

    Pairing on the gold chunk ids checks exactly that construction, and the
    difference it measures is the one the eval set was rebuilt to expose: how
    much of a configuration's score comes from sharing vocabulary with the
    passage rather than from finding it.
    """
    gold_a = [tuple(r.gold_chunk_ids) for r in report_a.records if r.answerable]
    gold_b = [tuple(r.gold_chunk_ids) for r in report_b.records if r.answerable]
    if not gold_a or any(not g for g in gold_a):
        raise ValueError(
            "compare_by_gold needs gold chunk ids on every record. Results saved "
            "before gold ids were recorded cannot be paired this way."
        )
    if gold_a != gold_b:
        raise ValueError(
            "Variant runs are not aligned: item i of each must carry the same "
            "gold chunk. Rebuild the eval set - paraphrasing preserves order and "
            "labels by construction, so a mismatch means the sets came from "
            "different builds."
        )

    a_values = [r.metrics.get(metric, 0.0) for r in report_a.records if r.answerable]
    b_values = [r.metrics.get(metric, 0.0) for r in report_b.records if r.answerable]
    return Comparison(
        metric=metric,
        name_a=report_a.config_name, name_b=report_b.config_name,
        test=paired_permutation_test(a_values, b_values),
        ci_a=bootstrap_ci(a_values), ci_b=bootstrap_ci(b_values),
    )


def compare_many(
    reports: Sequence[EvalReport],
    baseline: EvalReport,
    metric: str = "ndcg@k",
    alpha: float = 0.05,
) -> list[tuple[Comparison, bool]]:
    """Compare several configurations against one baseline, correcting for
    multiple testing.

    Returns (comparison, survives_correction) pairs. Reporting the raw p-values
    from a sweep is how a parameter search turns into a false discovery: with 12
    configurations at alpha=0.05 there is a ~46% chance at least one looks
    significant by chance alone.
    """
    from .stats import holm_bonferroni

    comparisons = [compare(baseline, r, metric) for r in reports]
    flags = holm_bonferroni([c.test.p_value for c in comparisons], alpha=alpha)
    return list(zip(comparisons, flags))
