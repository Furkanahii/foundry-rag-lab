"""Latency and throughput instrumentation.

Why this exists as its own module: the project plan asks us to check that
answers come back in a "reasonable" time. Reporting a *mean* latency would be
close to meaningless here. Local inference latency is heavy-tailed — a cold
model load, a page fault, or a thermal throttle produces occasional outliers
several times the median. The mean chases those outliers; the median does not.

So we keep every individual measurement and report quantiles. See
docs/06-istatistik-degerlendirme.md for why p95 is the number that matters for
a user-facing system.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterator


@dataclass
class Measurement:
    """A single timed operation."""

    op: str
    duration_ms: float
    meta: dict[str, Any] = field(default_factory=dict)


class Telemetry:
    """Collects timings, grouped by operation name.

    Usage:
        tel = Telemetry()
        with tel.track("embed", n_texts=16):
            ...
        print(tel.summary())
    """

    def __init__(self) -> None:
        self._records: list[Measurement] = []

    # ---- recording --------------------------------------------------------

    @contextmanager
    def track(self, op: str, **meta: Any) -> Iterator[dict[str, Any]]:
        """Time a block. The yielded dict can be mutated to attach extra meta.

        The yielded handle matters for cases where you only learn the
        interesting metadata *during* the call — e.g. how many tokens the model
        actually produced.
        """
        handle: dict[str, Any] = dict(meta)
        start = time.perf_counter()
        try:
            yield handle
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self._records.append(Measurement(op=op, duration_ms=elapsed, meta=handle))

    def record(self, op: str, duration_ms: float, **meta: Any) -> None:
        self._records.append(Measurement(op=op, duration_ms=duration_ms, meta=meta))

    # ---- reading ----------------------------------------------------------

    @property
    def records(self) -> list[Measurement]:
        return list(self._records)

    def durations(self, op: str) -> list[float]:
        return [r.duration_ms for r in self._records if r.op == op]

    @staticmethod
    def quantile(values: list[float], q: float) -> float:
        """Nearest-rank quantile.

        Deliberately not interpolating: with the small sample sizes we get from
        a benchmark run (tens to low hundreds of queries), interpolation invents
        a precision the data does not support. Nearest-rank always returns a
        value that was actually observed.
        """
        if not values:
            return float("nan")
        ordered = sorted(values)
        if q <= 0:
            return ordered[0]
        if q >= 1:
            return ordered[-1]
        # Nearest-rank: smallest value with at least q of the data at or below it.
        rank = max(1, int(round(q * len(ordered))))
        return ordered[rank - 1]

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-operation latency summary: count, median, p95, max."""
        grouped: dict[str, list[float]] = defaultdict(list)
        for r in self._records:
            grouped[r.op].append(r.duration_ms)

        out: dict[str, dict[str, float]] = {}
        for op, values in grouped.items():
            out[op] = {
                "count": float(len(values)),
                "total_ms": float(sum(values)),
                "p50_ms": float(median(values)),
                "p95_ms": self.quantile(values, 0.95),
                "max_ms": float(max(values)),
            }
        return out

    def reset(self) -> None:
        self._records.clear()
