"""Sweep retrieval configurations and compare them with paired statistics.

Retrieval-only by design. On this machine a retrieval query costs ~0.2 s while a
generated answer costs 40-90 s, so a 10-configuration sweep is minutes rather
than most of a day - and removing the language model also removes its sampling
variance from the measurement. Generation is evaluated separately, once, on the
configuration this sweep selects.

Every configuration runs against the same saved eval set in the same order, so
the comparisons are genuinely paired.

## Why the sweep runs once per wording variant

The eval set contains each item twice: as the question generated from its gold
chunk, and as a colloquial paraphrase of that question. Pooling the two would
average away the thing they were built to expose. Questions generated from a
passage reuse its vocabulary, which hands the BM25 arm the answer - the first
run of this project reported lexical retrieval at 0.86 nDCG against dense
retrieval's 0.58, a gap that was mostly an artefact of how the questions were
written. Scoring the variants separately turns that artefact into a measurement:
the drop from the generated to the paraphrased set *is* the size of the wording
advantage, and it is tested per configuration at the end of this script.

Usage:
    PYTHONPATH=src python scripts/run_benchmark.py [--metric ndcg@k]
    PYTHONPATH=src python scripts/run_benchmark.py --variant generated
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import OrderedDict

sys.path.insert(0, "src")

from frag.config import RunConfig  # noqa: E402
from frag.evaluation.qagen import EvalQuestion, load_eval_set  # noqa: E402
from frag.evaluation.runner import (  # noqa: E402
    EvalReport,
    compare_by_gold,
    compare_many,
    evaluate_retrieval_config,
)
from frag.evaluation.stats import holm_bonferroni, required_sample_size  # noqa: E402
from frag.runtime.foundry import FoundryRuntime  # noqa: E402

DEFAULT_INDEX = "data/index/bogazici.db"
DEFAULT_EVAL = "data/eval/eval_set.json"

# Order matters for reporting: the generated set is the optimistic reading and
# the paraphrased set is the honest one, so they read best in that order.
VARIANT_ORDER = ["generated", "paraphrased"]


def make_configs() -> list[RunConfig]:
    """The configurations to compare.

    The first entry is the baseline - deliberately the naive setup the project
    plan describes (dense-only, no stemming, no fusion), so that every later
    number is stated as an improvement over the obvious thing rather than over
    another tuned system.
    """
    configs: list[RunConfig] = []

    def build(name: str, **over) -> RunConfig:
        cfg = RunConfig(name=name)
        for key, value in over.items():
            setattr(cfg.retrieval, key, value)
        return cfg

    configs.append(build("baseline-dense", fusion="dense_only", lexical_stem=False))
    configs.append(build("lexical-only", fusion="lexical_only"))
    configs.append(build("lexical-nostem", fusion="lexical_only", lexical_stem=False))
    configs.append(build("rrf", fusion="rrf"))
    configs.append(build("rrf-k20", fusion="rrf", rrf_k=20))
    configs.append(build("rrf-k120", fusion="rrf", rrf_k=120))
    configs.append(build("weighted-0.5", fusion="weighted", alpha=0.5))
    configs.append(build("weighted-0.7", fusion="weighted", alpha=0.7))
    configs.append(build("rrf-instruct", fusion="rrf", query_instruction=True))
    configs.append(build("rrf-truncate6", fusion="rrf", lexical_truncate=6))
    configs.append(build("rrf-mmr", fusion="rrf", use_mmr=True))
    return configs


def split_variants(
    questions: list[EvalQuestion], wanted: str | None
) -> "OrderedDict[str, list[EvalQuestion]]":
    """Group answerable questions by wording variant, in reporting order.

    Alignment between variants is not assumed here - `compare_by_gold` verifies
    it from the gold labels before any cross-variant test runs.
    """
    answerable = [q for q in questions if q.answerable]
    # Known variants first, then anything else the set happens to contain, so a
    # future variant is still reported rather than silently dropped.
    extra = sorted({q.variant for q in answerable} - set(VARIANT_ORDER))
    groups: OrderedDict[str, list[EvalQuestion]] = OrderedDict()
    for name in VARIANT_ORDER + extra:
        if wanted and name != wanted:
            continue
        items = [q for q in answerable if q.variant == name]
        if items:
            groups[name] = items
    return groups


def sweep(
    configs: list[RunConfig],
    questions: list[EvalQuestion],
    index: str,
    runtime: FoundryRuntime,
    save_dir: str,
) -> list[EvalReport]:
    reports: list[EvalReport] = []
    for cfg in configs:
        t0 = time.perf_counter()
        report = evaluate_retrieval_config(
            cfg, index, questions, runtime=runtime, progress=False
        )
        reports.append(report)
        m = report.retrieval_metrics
        print(
            f"  {cfg.name:<16} ndcg={m.get('ndcg@k', 0):.4f} "
            f"recall={m.get('recall@k', 0):.4f} mrr={m.get('mrr', 0):.4f} "
            f"hit={m.get('hit_rate', 0):.3f}  ({time.perf_counter() - t0:.1f}s)"
        )
        report.save(f"{save_dir}/{cfg.name}.json")
    return reports


def print_comparisons(reports: list[EvalReport], metric: str) -> None:
    """Holm-corrected comparison of every configuration against the baseline."""
    baseline, others = reports[0], reports[1:]
    print(f"\n  Paired comparison against '{baseline.config_name}' on {metric}")
    print(f"  baseline: {baseline.intervals.get(metric, '')}\n")
    print(f"  {'config':<16}{'mean':>9}{'diff':>10}{'p':>10}{'d':>8}  verdict")
    print("  " + "-" * 74)
    for comparison, survives in compare_many(others, baseline, metric=metric):
        t = comparison.test
        verdict = "SIGNIFICANT" if survives else (
            "n.s. (raw p<0.05, lost to Holm)" if t.p_value < 0.05 else "n.s."
        )
        print(
            f"  {comparison.name_b:<16}{t.mean_b:>9.4f}{t.difference:>+10.4f}"
            f"{t.p_value:>10.4f}{t.effect_size:>8.2f}  {verdict}"
        )


def print_wording_gap(
    by_variant: "OrderedDict[str, list[EvalReport]]", metric: str
) -> None:
    """Test, per configuration, how much score the paraphrase costs.

    A configuration that survives rewording is finding passages; one that
    collapses was matching strings. Holm is applied across configurations here
    for the same reason as in the main sweep - this is a family of hypotheses
    tested at once, not one planned comparison.
    """
    if len(by_variant) < 2:
        return
    (name_a, reports_a), (name_b, reports_b) = list(by_variant.items())[:2]

    print(f"\n{'=' * 78}")
    print(f"Wording sensitivity: {name_a} -> {name_b} ({metric}, paired by gold chunk)")
    print(f"{'=' * 78}")

    comparisons = []
    for report_a, report_b in zip(reports_a, reports_b):
        try:
            comparisons.append(compare_by_gold(report_a, report_b, metric=metric))
        except ValueError as exc:
            print(f"  {report_a.config_name:<16} skipped: {exc}")
    if not comparisons:
        return

    flags = holm_bonferroni([c.test.p_value for c in comparisons])
    print(f"{'config':<16}{name_a[:9]:>10}{name_b[:11]:>12}{'drop':>10}{'p':>10}  verdict")
    print("-" * 78)
    for comparison, survives in zip(comparisons, flags):
        t = comparison.test
        verdict = "significant drop" if survives and t.difference < 0 else (
            "significant gain" if survives else "no measurable change"
        )
        print(
            f"{comparison.name_a:<16}{t.mean_a:>10.4f}{t.mean_b:>12.4f}"
            f"{t.difference:>+10.4f}{t.p_value:>10.4f}  {verdict}"
        )

    print("\nA large drop means the configuration was scoring on shared vocabulary")
    print("rather than on retrieval. Read the paraphrased column as the estimate of")
    print("how the system behaves on questions a user actually types.")


def print_power(n: int) -> None:
    print(f"\nPower: this eval set has n={n} questions per variant.")
    for d, label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        need = required_sample_size(d)
        status = "sufficient" if n >= need else f"UNDERPOWERED (need {need})"
        print(f"  {label:<7} effect (d={d}): {status}")
    print("\nA non-significant result on an underpowered set does not mean the")
    print("configurations are equivalent - only that this set could not tell.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="ndcg@k",
                        choices=["ndcg@k", "recall@k", "mrr", "precision@k"])
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--eval", default=DEFAULT_EVAL)
    parser.add_argument("--save", default="data/results")
    parser.add_argument("--variant", default=None,
                        help="restrict to one wording variant (default: all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    questions = load_eval_set(args.eval)
    groups = split_variants(questions, args.variant)
    if not groups:
        print(f"No answerable questions matching variant={args.variant!r}")
        return 1

    print("eval set: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()) + "\n")

    # One runtime shared across configurations: model loading costs tens of
    # seconds and none of these configs change the embedding model.
    cfg0 = RunConfig()
    runtime = FoundryRuntime(cfg0.runtime)

    configs = make_configs()
    by_variant: OrderedDict[str, list[EvalReport]] = OrderedDict()
    started = time.perf_counter()

    for variant, items in groups.items():
        print(f"{'=' * 78}")
        print(f"variant: {variant}  (n={len(items)})")
        print(f"{'=' * 78}")
        reports = sweep(
            configs, items, args.index, runtime, f"{args.save}/{variant}"
        )
        by_variant[variant] = reports
        print_comparisons(reports, args.metric)
        print()

    print(f"total {time.perf_counter() - started:.1f}s")

    print_wording_gap(by_variant, args.metric)
    print("\nHolm-Bonferroni applied within each family of comparisons: with 10")
    print("comparisons at alpha=0.05 there is a ~40% chance of at least one false")
    print("positive without correction.")
    print_power(min(len(v) for v in groups.values()))

    runtime.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
