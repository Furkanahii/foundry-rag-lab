"""Statistics for comparing RAG configurations honestly.

The failure this module exists to prevent: running config A, getting nDCG 0.71,
running config B, getting 0.74, and reporting that B is better. With 40 eval
questions, a 0.03 difference is well inside the noise. Half the "improvements"
reported in RAG blog posts are that mistake.

Three tools, each for a specific question.

## 1. Bootstrap confidence intervals - "how precise is this number?"

We have one sample of queries, drawn from the space of questions users might
ask. The mean nDCG over that sample estimates the mean over the whole space,
with sampling error. The bootstrap estimates that error without assuming
normality: resample the queries *with replacement* many times, recompute the
mean each time, and read the spread of those means directly.

This works because the empirical distribution is a consistent estimate of the
true one, so resampling from it mimics drawing fresh samples from the
population. It is especially appropriate here because per-query metrics are
badly non-normal - MRR only takes values 1, 1/2, 1/3, ..., 0, and recall@5 is
supported on six points. A t-interval assumes a shape this data does not have.

## 2. Paired permutation test - "is B actually better than A?"

Both configs are run on *the same* queries, which is a paired design and must be
analysed as one. Pairing removes query difficulty from the comparison: some
questions are hard for every system, and treating the two runs as independent
samples throws away the fact that we know exactly which questions those were.

Under the null hypothesis "the two configs are exchangeable", swapping A and B
within any query is equally likely to have produced what we saw. So we
enumerate (or sample) those swaps, build the null distribution of the mean
difference, and ask where the observed difference falls. No distributional
assumption at all.

## 3. Expected Calibration Error - "can I trust the confidence score?"

Needed for the abstention threshold. If we refuse to answer when the retrieval
score is low, that score has to mean something. ECE bins predictions by
confidence and measures the gap between confidence and observed accuracy in
each bin. A well-calibrated score of 0.8 should be right 80% of the time.

## A note on multiple comparisons

Sweeping 12 configurations and reporting the best p-value is a specification
search: at alpha = 0.05, testing 12 hypotheses gives roughly a 46% chance of at
least one false positive (1 - 0.95^12). `holm_bonferroni` corrects for that and
is applied whenever the harness compares more than two configs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class ConfidenceInterval:
    """A point estimate with an interval and the level it was computed at."""

    estimate: float
    lower: float
    upper: float
    confidence: float = 0.95
    n: int = 0

    @property
    def half_width(self) -> float:
        return (self.upper - self.lower) / 2.0

    def __str__(self) -> str:
        pct = int(self.confidence * 100)
        return f"{self.estimate:.4f} [{self.lower:.4f}, {self.upper:.4f}] ({pct}% CI, n={self.n})"

    def overlaps(self, other: ConfidenceInterval) -> bool:
        """Whether two intervals overlap.

        Deliberately *not* a significance test. Non-overlapping intervals do
        imply a significant difference, but overlapping ones do not imply the
        absence of one - a well-known trap. Use `paired_permutation_test` to
        decide; use this only for a quick visual read.
        """
        return not (self.upper < other.lower or other.upper < self.lower)


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    statistic: Callable[[np.ndarray], float] = np.mean,
    seed: int = 20260801,
) -> ConfidenceInterval:
    """Percentile bootstrap interval for a statistic of `values`.

    `n_resamples = 10_000` is chosen so that Monte-Carlo error in the interval
    endpoints is small relative to the sampling error we are trying to measure;
    below ~2000 the endpoints themselves become noticeably unstable between
    runs, which defeats the purpose.
    """
    data = np.asarray(values, dtype=np.float64)
    n = len(data)
    if n == 0:
        return ConfidenceInterval(float("nan"), float("nan"), float("nan"), confidence, 0)
    if n == 1:
        v = float(data[0])
        return ConfidenceInterval(v, v, v, confidence, 1)

    rng = np.random.default_rng(seed)
    # Vectorised: draw all resamples at once rather than looping.
    indices = rng.integers(0, n, size=(n_resamples, n))
    resampled = data[indices]
    stats = np.apply_along_axis(statistic, 1, resampled) if statistic is not np.mean \
        else resampled.mean(axis=1)

    alpha = 1.0 - confidence
    lower = float(np.percentile(stats, 100 * alpha / 2))
    upper = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return ConfidenceInterval(float(statistic(data)), lower, upper, confidence, n)


@dataclass
class TestResult:
    """Outcome of a paired comparison between two configurations."""

    mean_a: float
    mean_b: float
    difference: float          # b - a
    p_value: float
    effect_size: float         # Cohen's d for paired data
    n: int
    n_resamples: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def interpret(self, alpha: float = 0.05) -> str:
        direction = "better" if self.difference > 0 else "worse"
        if self.p_value >= alpha:
            return (
                f"no significant difference (diff={self.difference:+.4f}, "
                f"p={self.p_value:.4f}, n={self.n})"
            )
        magnitude = (
            "negligible" if abs(self.effect_size) < 0.2
            else "small" if abs(self.effect_size) < 0.5
            else "medium" if abs(self.effect_size) < 0.8
            else "large"
        )
        return (
            f"B is {direction} (diff={self.difference:+.4f}, p={self.p_value:.4f}, "
            f"Cohen's d={self.effect_size:.2f} [{magnitude}], n={self.n})"
        )


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 20260801,
) -> TestResult:
    """Two-sided paired permutation test on the mean difference.

    `a` and `b` must be per-query scores in the same order - element i of each
    is the same question. Randomly flipping the sign of each per-query
    difference generates the null distribution, because under exchangeability a
    difference of +0.2 was as likely to have come out -0.2.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError(f"paired test needs equal lengths, got {len(x)} and {len(y)}")
    n = len(x)
    if n == 0:
        return TestResult(float("nan"), float("nan"), float("nan"), 1.0, 0.0, 0, 0)

    differences = y - x
    observed = float(differences.mean())

    sd = float(differences.std(ddof=1)) if n > 1 else 0.0
    effect_size = observed / sd if sd > 1e-12 else 0.0

    # Every difference is exactly zero: the configs produced identical scores,
    # so no permutation can produce a larger deviation and p = 1.
    if np.allclose(differences, 0.0):
        return TestResult(float(x.mean()), float(y.mean()), 0.0, 1.0, 0.0, n, 0)

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, n))
    null_means = (signs * differences).mean(axis=1)

    # +1 in numerator and denominator: the observed arrangement is itself one of
    # the permutations, which keeps p strictly positive and the test valid.
    extreme = int(np.sum(np.abs(null_means) >= abs(observed) - 1e-15))
    p_value = (extreme + 1) / (n_resamples + 1)

    return TestResult(
        mean_a=float(x.mean()), mean_b=float(y.mean()), difference=observed,
        p_value=float(p_value), effect_size=float(effect_size),
        n=n, n_resamples=n_resamples,
    )


def holm_bonferroni(
    p_values: Sequence[float], alpha: float = 0.05
) -> list[bool]:
    """Holm-Bonferroni step-down correction. Returns per-test reject flags.

    Preferred over plain Bonferroni because it is uniformly more powerful while
    controlling the same family-wise error rate: sort the p-values ascending and
    compare the i-th against alpha / (m - i), stopping at the first failure.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        if p_values[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break  # step-down: once one fails, all larger p-values fail too
    return reject


def required_sample_size(
    effect_size: float, power: float = 0.8, alpha: float = 0.05
) -> int:
    """Roughly how many queries are needed to detect `effect_size` (Cohen's d).

    Answers the question the harness should ask before anyone trusts a null
    result: "was this eval set even large enough to see the difference we care
    about?" A non-significant result on 15 questions usually means the study was
    underpowered, not that the configs are equivalent.

    Normal-approximation formula for a paired design:
        n = ((z_{1-alpha/2} + z_{1-beta}) / d)^2
    """
    from scipy import stats as sps

    if effect_size <= 0:
        return 0
    z_alpha = sps.norm.ppf(1 - alpha / 2)
    z_beta = sps.norm.ppf(power)
    return int(np.ceil(((z_alpha + z_beta) / effect_size) ** 2))


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """Reliability of a confidence score."""

    ece: float
    mce: float                      # worst single-bin gap
    brier: float
    bin_edges: list[float]
    bin_confidence: list[float]
    bin_accuracy: list[float]
    bin_counts: list[int]

    def __str__(self) -> str:
        return f"ECE={self.ece:.4f} MCE={self.mce:.4f} Brier={self.brier:.4f}"


def expected_calibration_error(
    confidences: Sequence[float],
    outcomes: Sequence[bool],
    n_bins: int = 10,
) -> CalibrationResult:
    """Bin predictions by confidence and measure confidence-vs-accuracy gaps.

        ECE = sum over bins of  (n_bin / N) * |accuracy_bin - confidence_bin|

    The weighting by bin population matters: a wildly miscalibrated bin holding
    two samples should not dominate a well-calibrated bin holding two hundred.

    MCE reports the worst bin instead, because for a *safety* decision like
    abstention the worst case can matter more than the average.

    Brier score is the mean squared error of the probabilities, included because
    it is a proper scoring rule - unlike ECE it cannot be gamed by a model that
    always predicts the base rate.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(outcomes, dtype=bool)
    if len(conf) == 0:
        return CalibrationResult(0.0, 0.0, 0.0, [], [], [], [])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_conf: list[float] = []
    bin_acc: list[float] = []
    counts: list[int] = []
    n = len(conf)

    for lo, hi in zip(edges[:-1], edges[1:]):
        # Upper-inclusive on the last bin so confidence == 1.0 is not dropped.
        mask = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= 1.0)
        count = int(mask.sum())
        counts.append(count)
        if count == 0:
            bin_conf.append(0.0)
            bin_acc.append(0.0)
            continue
        avg_conf = float(conf[mask].mean())
        avg_acc = float(correct[mask].mean())
        bin_conf.append(avg_conf)
        bin_acc.append(avg_acc)
        gap = abs(avg_acc - avg_conf)
        ece += (count / n) * gap
        mce = max(mce, gap)

    brier = float(np.mean((conf - correct.astype(np.float64)) ** 2))
    return CalibrationResult(
        ece=float(ece), mce=float(mce), brier=brier,
        bin_edges=[float(e) for e in edges],
        bin_confidence=bin_conf, bin_accuracy=bin_acc, bin_counts=counts,
    )


def find_best_threshold(
    scores: Sequence[float],
    should_answer: Sequence[bool],
    objective: str = "f1",
) -> tuple[float, dict[str, float]]:
    """Pick the abstention threshold from labelled data instead of guessing.

    Sweeps every candidate cut-off and returns the one maximising the chosen
    objective, together with its full confusion statistics.

    `objective="f1"` balances the two error types. That default is a judgement
    call worth stating out loud: refusing an answerable question annoys the
    user, while answering an unanswerable one fabricates a regulation. For a
    system people might rely on, the second error is worse - so a deployment
    could reasonably optimise recall of refusals instead, accepting more
    unnecessary refusals to buy fewer fabrications.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(should_answer, dtype=bool)
    if len(s) == 0:
        return 0.0, {}

    # Candidate thresholds: midpoints between observed scores, plus the extremes.
    unique = np.unique(s)
    candidates = np.concatenate([
        [unique[0] - 1e-6],
        (unique[:-1] + unique[1:]) / 2.0 if len(unique) > 1 else [],
        [unique[-1] + 1e-6],
    ])

    best_threshold = float(candidates[0])
    best_value = -np.inf
    best_stats: dict[str, float] = {}

    for threshold in candidates:
        answered = s >= threshold
        tp = int(np.sum(answered & y))
        fp = int(np.sum(answered & ~y))
        fn = int(np.sum(~answered & y))
        tn = int(np.sum(~answered & ~y))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / len(s)

        value = {"f1": f1, "accuracy": accuracy, "precision": precision,
                 "recall": recall}[objective]
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
            best_stats = {
                "threshold": float(threshold), "f1": f1, "accuracy": accuracy,
                "precision": precision, "recall": recall,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }

    return best_threshold, best_stats
