"""The assembled system: one object, one `ask()` method.

Everything below this file is a component that can be tested and benchmarked in
isolation. This is where they are wired together, and it is the only public
entry point the dashboard, the CLI, and the evaluation harness use - so all
three exercise exactly the same code path. A benchmark that measured a
different pipeline than the demo runs would be worthless.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

from .config import RunConfig
from .generate.answerer import Answer, Answerer
from .index.store import HybridStore
from .retrieve.retriever import RetrievalResult, Retriever
from .runtime.foundry import FoundryRuntime
from .runtime.telemetry import Telemetry

logger = logging.getLogger(__name__)


class RagPipeline:
    """Retrieval + generation, configured by a single `RunConfig`."""

    def __init__(
        self,
        cfg: RunConfig,
        index_path: str | Path,
        runtime: FoundryRuntime | None = None,
    ) -> None:
        self.cfg = cfg
        self.telemetry = Telemetry()
        # An injected runtime is reused rather than re-created: model loading
        # costs tens of seconds, and a benchmark sweeping 12 configurations must
        # not pay that 12 times.
        self.runtime = runtime or FoundryRuntime(cfg.runtime, telemetry=self.telemetry)
        self.store = HybridStore(index_path)
        self.retriever = Retriever(cfg.retrieval, self.runtime, self.store)
        self.answerer = Answerer(cfg.generation, self.runtime)

    # ---- main entry points -------------------------------------------------

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieval only - what the retrieval benchmark measures."""
        with self.telemetry.track("retrieve"):
            return self.retriever.retrieve(query)

    def ask(self, query: str) -> Answer:
        """Full pipeline: retrieve, then answer or abstain."""
        with self.telemetry.track("ask"):
            result = self.retriever.retrieve(query)
            return self.answerer.answer(query, result)

    def ask_stream(self, query: str) -> tuple[RetrievalResult, Iterator[str]]:
        """Streaming variant.

        Returns the retrieval result immediately alongside the token iterator,
        so the UI can render sources while the answer is still being written.
        """
        result = self.retriever.retrieve(query)
        return result, self.answerer.answer_stream(query, result)

    # ---- introspection -----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Full experiment record: config, index, hardware path."""
        return {
            "config_name": self.cfg.name,
            "fingerprint": self.cfg.fingerprint(),
            "index_fingerprint": self.cfg.index_fingerprint(),
            "retrieval": self.retriever.describe(),
            "generation": {
                "model": self.cfg.runtime.chat_model,
                "temperature": self.cfg.generation.temperature,
                "abstention_threshold": self.cfg.generation.abstention_threshold,
                "abstention_enabled": self.cfg.generation.enable_abstention,
            },
            "index": self.store.stats(),
            "runtime": self.runtime.describe(),
        }

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> RagPipeline:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
