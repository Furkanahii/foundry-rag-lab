"""The only module in this project that knows Foundry Local exists.

Everything above this layer talks to `FoundryRuntime` and would keep working if
the backend were swapped for Ollama, llama.cpp, or a cloud endpoint. That
isolation is not architectural vanity: it is what makes the benchmark possible.
To compare two chat models we change one config field; nothing else moves.

## What Foundry Local actually is (verified against SDK 1.2.4 on this machine)

`pip install foundry-local-sdk` pulls the native runtime with it
(`foundry-local-core`, `onnxruntime-genai-core`). No separate service install is
required - notably, the Homebrew `foundrylocal` formula is *not* needed on
Apple Silicon. Inference happens in-process through ONNX Runtime.

Two inference surfaces exist, and this project deliberately uses both:

  * **Native clients** - `model.get_embedding_client()` /
    `model.get_chat_client()`. Simple and dependency-free, but
    `complete_chat(messages, tools)` exposes *no* sampling parameters. There is
    no way to pin temperature.

  * **OpenAI-compatible web service** - `manager.start_web_service()` binds a
    local HTTP port speaking the OpenAI protocol, which accepts `temperature`,
    `max_tokens`, `logprobs`, and friends.

We embed through the native client (embeddings are deterministic; there is
nothing to tune) and generate through the web service (where `temperature=0.1`
and reproducibility genuinely matter). If the web service cannot start we fall
back to the native chat client and record that fact, because a benchmark run
with uncontrolled sampling is not comparable to one without.

## Measured on an Apple M2, 16 GB

    embedding : qwen3-embedding-0.6b-generic-gpu   dim=1024
    chat      : qwen2.5-0.5b-instruct-generic-gpu
    provider  : GPU

The `-generic-gpu` suffix in the resolved model id is the execution provider
the runtime chose for this hardware. It is recorded with every benchmark result
because latency numbers are meaningless without it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np

from ..config import RuntimeConfig
from .telemetry import Telemetry

logger = logging.getLogger(__name__)

# The SDK manager is a process-wide singleton (`FoundryLocalManager.initialize`
# then `.instance`). Calling initialize twice in one process is not safe, so we
# track it here rather than per-FoundryRuntime.
_MANAGER_READY = False


class FoundryRuntimeError(RuntimeError):
    """Raised when the local runtime cannot be reached or a model cannot load."""


@dataclass
class LoadedModel:
    """What we know about a model the runtime has resident."""

    alias: str
    model_id: str
    context_length: int
    handle: Any  # the SDK IModel

    @property
    def execution_provider(self) -> str:
        """Execution provider, recovered from the resolved model id.

        Foundry encodes it in the id, e.g. `qwen2.5-0.5b-instruct-generic-gpu:4`
        -> `gpu`. There is no public accessor for this, so we parse it; a
        missing value degrades to "unknown" rather than raising, because this is
        reporting metadata, not control flow.
        """
        base = self.model_id.split(":")[0]
        for candidate in ("npu", "gpu", "cpu"):
            if base.endswith(f"-{candidate}"):
                return candidate
        return "unknown"

    def describe(self) -> str:
        return (
            f"{self.alias} (id={self.model_id}, ep={self.execution_provider}, "
            f"ctx={self.context_length})"
        )


class FoundryRuntime:
    """Owns the Foundry Local runtime and both model roles.

    The two models have different lifecycles: the embedding model is called
    constantly during indexing (throughput-bound), the chat model once per
    question (latency-bound). They load lazily and independently, so a
    retrieval-only benchmark never pays to load a chat model it will not call.
    """

    def __init__(self, cfg: RuntimeConfig, telemetry: Telemetry | None = None) -> None:
        self.cfg = cfg
        self.telemetry = telemetry or Telemetry()
        self._manager: Any | None = None
        self._loaded: dict[str, LoadedModel] = {}
        self._openai_client: Any | None = None
        self._web_service_failed = False

    # ---- runtime lifecycle -------------------------------------------------

    def _ensure_manager(self) -> Any:
        """Initialise the SDK manager once per process."""
        global _MANAGER_READY

        if self._manager is not None:
            return self._manager

        try:
            from foundry_local_sdk import Configuration, FoundryLocalManager
        except ImportError as exc:  # pragma: no cover - environment problem
            raise FoundryRuntimeError(
                "foundry-local-sdk is not installed. Run: pip install foundry-local-sdk\n"
                "Note: on Apple Silicon this also installs the native runtime; "
                "no Homebrew install is needed."
            ) from exc

        with self.telemetry.track("runtime_init"):
            if not _MANAGER_READY:
                config = Configuration(app_name=self.cfg.app_name)
                try:
                    FoundryLocalManager.initialize(config)
                except Exception as exc:
                    raise FoundryRuntimeError(
                        "Could not initialise the Foundry Local runtime."
                    ) from exc
                _MANAGER_READY = True

            manager = FoundryLocalManager.instance

            # Registering execution providers is a no-op after the first run,
            # but on a cold machine it downloads hardware-specific packages.
            if self.cfg.register_execution_providers:
                try:
                    manager.download_and_register_eps()
                except Exception as exc:  # noqa: BLE001 - non-fatal
                    logger.warning(
                        "Execution provider registration failed (%s); "
                        "falling back to whatever is already registered.",
                        exc,
                    )

        self._manager = manager
        return manager

    # ---- model management --------------------------------------------------

    def ensure_model(self, alias: str) -> LoadedModel:
        """Download (if absent) and load a model by alias.

        Aliases like `qwen2.5-0.5b` resolve to a hardware-specific build such as
        `qwen2.5-0.5b-instruct-generic-gpu:4`. Downstream code needs the
        resolved id, never the alias.
        """
        if alias in self._loaded:
            return self._loaded[alias]

        manager = self._ensure_manager()
        catalog = manager.catalog

        model = catalog.get_model(alias)
        if model is None:
            available = ", ".join(
                sorted(getattr(m, "alias", "?") for m in catalog.list_models())[:20]
            )
            raise FoundryRuntimeError(
                f"Model alias '{alias}' is not in the catalog on this machine.\n"
                f"Some available aliases: {available}"
            )

        with self.telemetry.track("model_load", alias=alias) as meta:
            if not model.is_cached:
                logger.info("Downloading %s (first run only)...", alias)
                model.download(
                    lambda p: logger.debug("  %s: %.1f%%", alias, p)
                )
            if not model.is_loaded:
                model.load()

            loaded = LoadedModel(
                alias=alias,
                model_id=model.id,
                context_length=int(getattr(model, "context_length", 0) or 0),
                handle=model,
            )
            meta["model_id"] = loaded.model_id
            meta["execution_provider"] = loaded.execution_provider

        self._loaded[alias] = loaded
        logger.info("Loaded %s", loaded.describe())
        return loaded

    # ---- OpenAI-compatible surface ----------------------------------------

    @property
    def openai_client(self) -> Any | None:
        """Client for the local web service, or None if it is unavailable.

        Returning None rather than raising is intentional: the caller decides
        whether to degrade to the native client or to abort. A benchmark should
        abort; an interactive demo should degrade.
        """
        if self._openai_client is not None:
            return self._openai_client
        if self._web_service_failed or not self.cfg.use_web_service:
            return None

        manager = self._ensure_manager()
        try:
            manager.start_web_service()
            urls = getattr(manager, "urls", None)
            if not urls:
                raise FoundryRuntimeError("web service reported no bound URL")

            base = (urls[0] if isinstance(urls, (list, tuple)) else str(urls)).rstrip("/")
            if not base.endswith("/v1"):
                base += "/v1"

            from openai import OpenAI

            self._openai_client = OpenAI(
                base_url=base,
                api_key="local",  # the local service does not authenticate
                timeout=self.cfg.request_timeout,
                max_retries=0,  # we do our own backoff below
            )
            logger.info("OpenAI-compatible web service ready at %s", base)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Web service unavailable (%s). Falling back to native clients; "
                "sampling parameters will NOT be controllable.",
                exc,
            )
            self._web_service_failed = True
            return None

        return self._openai_client

    @property
    def sampling_is_controlled(self) -> bool:
        """True when generation honours temperature/max_tokens.

        Recorded in benchmark output. A run where this is False cannot be
        compared against one where it is True.
        """
        return self.openai_client is not None

    # ---- retry helper ------------------------------------------------------

    def _with_retry(self, op_name: str, fn, *args, **kwargs):
        """Retry transient runtime failures with exponential backoff.

        Local inference still fails transiently: a model may be mid-load, or a
        request may land while memory is being reclaimed. Those are
        recoverable. A malformed request is not, so the retry count is bounded
        and the original error is re-raised as the cause.
        """
        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - re-raised below
                last_exc = exc
                if attempt == self.cfg.max_retries - 1:
                    break
                backoff = 1.5**attempt
                logger.warning(
                    "%s failed (%d/%d): %s - retrying in %.1fs",
                    op_name, attempt + 1, self.cfg.max_retries, exc, backoff,
                )
                time.sleep(backoff)
        raise FoundryRuntimeError(
            f"{op_name} failed after {self.cfg.max_retries} attempts"
        ) from last_exc

    # ---- embeddings --------------------------------------------------------

    def embed(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        """Embed texts. Returns shape (len(texts), dim), float32.

        `normalize=True` scales every vector to unit L2 length. This is not
        cosmetic. Once vectors are unit-length, cosine similarity equals the dot
        product, because

            cos(a, b) = (a . b) / (|a| |b|)   and   |a| = |b| = 1

        which turns top-k search into a single matrix multiply. It also makes
        sqlite-vec's L2 distance rank *identically* to cosine, since for unit
        vectors

            |a - b|^2 = 2 - 2(a . b)

        is monotone decreasing in the dot product. One normalisation buys
        consistent ranking across three different backends. See docs/01.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        model = self.ensure_model(self.cfg.embedding_model)
        client = model.handle.get_embedding_client()

        vectors: list[list[float]] = []
        batch = self.cfg.embed_batch_size

        for start in range(0, len(texts), batch):
            chunk = list(texts[start : start + batch])
            with self.telemetry.track("embed", n_texts=len(chunk)):
                response = self._with_retry(
                    "embed", client.generate_embeddings, chunk
                )
            # Ordering is not contractually guaranteed; `index` is authoritative.
            ordered = sorted(response.data, key=lambda d: d.index)
            vectors.extend(d.embedding for d in ordered)

        arr = np.asarray(vectors, dtype=np.float32)
        return l2_normalize(arr) if normalize else arr

    def embed_one(self, text: str, normalize: bool = True) -> np.ndarray:
        """Embed a single string, returning a 1-D vector."""
        return self.embed([text], normalize=normalize)[0]

    # ---- chat --------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Single-shot chat completion, returning the assistant text."""
        model = self.ensure_model(self.cfg.chat_model)
        client = self.openai_client

        with self.telemetry.track("chat", n_messages=len(messages)) as meta:
            if client is not None:
                response = self._with_retry(
                    "chat",
                    client.chat.completions.create,
                    model=model.model_id,
                    messages=messages,
                    temperature=0.1 if temperature is None else temperature,
                    max_tokens=700 if max_tokens is None else max_tokens,
                    **kwargs,
                )
                usage = getattr(response, "usage", None)
                if usage is not None:
                    meta["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
                    meta["completion_tokens"] = getattr(usage, "completion_tokens", None)
                return response.choices[0].message.content or ""

            # Degraded path: no sampling control.
            meta["degraded"] = True
            native = model.handle.get_chat_client()
            response = self._with_retry("chat_native", native.complete_chat, messages)
            return response.choices[0].message.content or ""

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Streaming chat completion, yielding text deltas.

        Streaming changes *perceived* latency: a 3-second answer that starts
        printing at 400 ms feels faster than a 2-second answer that appears all
        at once. We therefore record time-to-first-token separately from total
        time - they answer different questions, and the project plan's "is it
        fast enough?" is really a question about the former.
        """
        model = self.ensure_model(self.cfg.chat_model)
        client = self.openai_client
        start = time.perf_counter()
        first_token_ms: float | None = None

        if client is not None:
            stream = self._with_retry(
                "chat_stream",
                client.chat.completions.create,
                model=model.model_id,
                messages=messages,
                temperature=0.1 if temperature is None else temperature,
                max_tokens=700 if max_tokens is None else max_tokens,
                stream=True,
                **kwargs,
            )
        else:
            native = model.handle.get_chat_client()
            stream = self._with_retry(
                "chat_stream_native", native.complete_streaming_chat, messages
            )

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta else None
            if content:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - start) * 1000.0
                    self.telemetry.record("time_to_first_token", first_token_ms)
                yield content

        self.telemetry.record("chat_stream", (time.perf_counter() - start) * 1000.0)

    # ---- introspection -----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Everything about the runtime worth storing next to a benchmark.

        Results are only comparable when the hardware path matches, so the
        execution provider is part of the experiment record, not a footnote.
        """
        return {
            "models": {a: m.describe() for a, m in self._loaded.items()},
            "execution_providers": {
                a: m.execution_provider for a, m in self._loaded.items()
            },
            "sampling_controlled": self.sampling_is_controlled,
            "configured_device": self.cfg.device or "auto",
        }

    def list_catalog(self) -> list[dict[str, Any]]:
        """Models available on this machine."""
        manager = self._ensure_manager()
        return [
            {
                "alias": getattr(m, "alias", ""),
                "id": getattr(m, "id", ""),
                "context_length": getattr(m, "context_length", 0),
                "cached": getattr(m, "is_cached", False),
            }
            for m in manager.catalog.list_models()
        ]

    def close(self) -> None:
        """Unload models so the machine gets its RAM back."""
        for alias, loaded in list(self._loaded.items()):
            try:
                loaded.handle.unload()
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                logger.debug("Could not unload %s: %s", alias, exc)
        self._loaded.clear()

    def __enter__(self) -> FoundryRuntime:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# ---- vector helpers --------------------------------------------------------


def l2_normalize(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each row to unit length.

    The `eps` guard matters: a model can emit an all-zero embedding for empty or
    degenerate input, and dividing by its zero norm would poison every
    downstream similarity with NaN. Clamping leaves the zero vector as zero,
    which then has similarity 0 with everything - the correct reading of "this
    chunk carries no signal".
    """
    if arr.size == 0:
        return arr
    if arr.ndim == 1:
        return arr / max(float(np.linalg.norm(arr)), eps)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


def cosine_similarity_matrix(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Cosine similarity between every query and every doc.

    Assumes both inputs are already L2-normalised, so this is one matrix
    multiply of shape (n_queries, n_docs). Doing it as a single BLAS call rather
    than a Python loop is ~2 orders of magnitude faster, which is the difference
    between a benchmark sweep taking seconds and taking minutes.
    """
    if queries.ndim == 1:
        queries = queries[None, :]
    return queries @ docs.T
