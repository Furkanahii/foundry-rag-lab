"""Configuration objects for the RAG lab.

Every knob that could plausibly change a benchmark result lives here, in one
place. That is deliberate: an experiment is only reproducible if the full
configuration can be serialised, hashed, and stored alongside its results.

The `RunConfig.fingerprint()` hash is what lets the evaluation harness say
"these two result sets came from genuinely different pipelines" rather than
relying on a human to remember what they changed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

# Repository root: src/frag/config.py -> src/frag -> src -> <root>
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
EVAL_DIR = DATA_DIR / "eval"
INDEX_DIR = DATA_DIR / "index"
RESULTS_DIR = DATA_DIR / "results"


@dataclass
class RuntimeConfig:
    """How we talk to Foundry Local.

    `device` maps to Foundry's DeviceType (CPU/GPU/NPU). Leaving it None lets
    the service pick the best available execution provider for the machine,
    which is what we want for a cross-platform demo: the same config runs on an
    Apple Silicon Mac and on a Windows laptop with an NPU.
    """

    app_name: str = "foundry_rag_lab"
    # Default chosen by measurement, not by guessing - see
    # scripts/compare_chat_models.py and docs/07-model-secimi.md.
    # On an Apple M2 (WebGPU EP), grounded Turkish QA:
    #   qwen2.5-0.5b   660 ms  - fabricates content, never refuses. Unsafe.
    #   qwen2.5-1.5b  10.8 s   - repetition score 0.315, degenerate. Unusable.
    #   phi-3.5-mini   7.2 s   - clean, refuses correctly.        <- default
    #   qwen2.5-7b    23.8 s   - best quality, too slow to demo interactively.
    chat_model: str = "phi-3.5-mini"
    embedding_model: str = "qwen3-embedding-0.6b"
    device: Literal["CPU", "GPU", "NPU"] | None = None
    request_timeout: float = 180.0
    max_retries: int = 3
    # Embedding calls are batched; too large a batch can stall the local runtime.
    embed_batch_size: int = 16
    # Generate through the OpenAI-compatible web service so that temperature and
    # max_tokens are honoured. Disabling this falls back to the native client,
    # which exposes no sampling parameters - fine for a demo, not for a
    # benchmark. See runtime/foundry.py for why this distinction is enforced.
    use_web_service: bool = True
    # First run on a cold machine downloads hardware-specific EP packages.
    register_execution_providers: bool = True


@dataclass
class ChunkConfig:
    """Document splitting strategy.

    `strategy` selects the algorithm; the remaining fields are its parameters.
    Chunk size is expressed in *characters* rather than tokens because we do not
    have a tokenizer that is guaranteed to match the local model. The character
    heuristic is documented in docs/02-chunking.md along with its bias for
    Turkish text (Turkish words are longer on average than English ones, so a
    fixed character budget holds fewer words).
    """

    strategy: Literal["fixed", "recursive", "sentence_window", "semantic"] = "recursive"
    chunk_size: int = 900
    overlap: int = 150
    # sentence_window: how many neighbouring sentences to attach as context
    window_size: int = 2
    # semantic: percentile of embedding-distance used as a split threshold
    breakpoint_percentile: float = 90.0
    min_chunk_chars: int = 120


@dataclass
class RetrievalConfig:
    """Hybrid retrieval settings.

    The pipeline is: dense search + lexical (BM25) search -> fusion -> optional
    rerank -> optional MMR diversification -> top_k passed to the generator.
    """

    dense_top_k: int = 30
    lexical_top_k: int = 30
    fusion: Literal["rrf", "weighted", "dense_only", "lexical_only"] = "rrf"
    # RRF smoothing constant. Larger values flatten the contribution of rank.
    rrf_k: int = 60
    # weighted fusion: final = alpha * dense_norm + (1 - alpha) * lexical_norm
    alpha: float = 0.5
    # "none" | "llm" (listwise, single call - see retrieve/rerank.py)
    rerank: Literal["none", "llm"] = "none"
    rerank_candidates: int = 12
    use_mmr: bool = False
    mmr_lambda: float = 0.7
    top_k: int = 5
    # Wrap the query in the embedding model's training-time instruction
    # template. Costs one extra embedding call; measured separately.
    query_instruction: bool = False
    # Turkish suffix stripping for the BM25 arm.
    lexical_stem: bool = True
    lexical_truncate: int | None = None


@dataclass
class GenerationConfig:
    """Answer generation and grounding behaviour."""

    temperature: float = 0.1
    max_tokens: int = 700
    require_citations: bool = True
    # Below this fused retrieval score the system refuses to answer.
    # The default is a placeholder: docs/05-kalibrasyon-abstention.md derives a
    # calibrated value from the evaluation set instead of guessing.
    abstention_threshold: float = 0.0
    enable_abstention: bool = True
    language: Literal["tr", "en", "auto"] = "auto"


@dataclass
class RunConfig:
    """A complete, hashable description of one pipeline configuration."""

    name: str = "default"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 20260801

    # ---- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunConfig:
        return cls(
            name=raw.get("name", "default"),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
            chunking=ChunkConfig(**raw.get("chunking", {})),
            retrieval=RetrievalConfig(**raw.get("retrieval", {})),
            generation=GenerationConfig(**raw.get("generation", {})),
            seed=raw.get("seed", 20260801),
        )

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, allow_unicode=True, sort_keys=False)

    # ---- identity ---------------------------------------------------------

    def fingerprint(self) -> str:
        """Stable short hash of everything except the human-facing name.

        Two runs with the same fingerprint are the same experiment; comparing
        their metrics measures noise, not a design change.
        """
        payload = self.to_dict()
        payload.pop("name", None)
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def index_fingerprint(self) -> str:
        """Hash of only the settings that change the *stored index*.

        Retrieval and generation settings can be swapped without re-embedding
        the corpus; chunking and the embedding model cannot. Separating the two
        turns a 20-minute re-index into a 2-second config reload for most
        experiments.
        """
        payload = {
            "chunking": asdict(self.chunking),
            "embedding_model": self.runtime.embedding_model,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def default_config() -> RunConfig:
    return RunConfig()
