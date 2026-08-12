"""Grounded answer generation: cite sources, or refuse.

Three behaviours here go beyond "put the chunks in the prompt".

**1. Abstention happens before the model is called.** If the best retrieval
score is below threshold, we return a refusal without generating anything. That
is not just a safety property, it is a latency property: the cheapest possible
answer to an unanswerable question is one that never touches the LLM. On this
machine that turns a 7-second wrong answer into a 300 ms honest one.

The threshold is not guessed. `evaluation/` calibrates it against a labelled
set, and docs/05 works through why a fixed cosine cut-off cannot work: the
score distribution is corpus- and query-dependent, and the measured
concentration effect means absolute similarity values carry little information
on their own.

**2. Citations are validated after generation.** Asking a model to cite is easy;
verifying it cited something that exists is the part that catches fabrication.
We parse the [n] markers out of the answer and check every one against the
context we actually supplied. A citation pointing at [7] when five passages were
given is a hallucination the system can detect on its own, with no judge model
and no ground truth.

**3. The prompt language follows the question, not the corpus.** A Turkish
question about an English document should still be answered in Turkish.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterator

from ..config import GenerationConfig
from ..ingest.normalize import detect_language
from ..retrieve.retriever import RetrievalResult
from ..runtime.foundry import FoundryRuntime

logger = logging.getLogger(__name__)

REFUSAL_TR = "Bu bilgi verilen belgelerde bulunmuyor."
REFUSAL_EN = "This information is not present in the provided documents."

_SYSTEM_TR = """Sen bir üniversite yönetmelik asistanısın. Soruları SADECE aşağıda verilen bağlamı kullanarak yanıtla.

Kurallar:
1. Yalnızca bağlamdaki bilgiyi kullan. Kendi bilgini ekleme.
2. Kullandığın her bilgi için kaynağı [1], [2] biçiminde belirt.
3. Bağlam soruyu yanıtlamaya yetmiyorsa SADECE şunu yaz: "{refusal}"
4. Kısa ve net yanıtla. Gereksiz tekrar yapma.
5. Emin olmadığın bir şeyi tahmin etme.

Bağlam:
{context}"""

_SYSTEM_EN = """You are a university regulations assistant. Answer questions using ONLY the context below.

Rules:
1. Use only information from the context. Do not add outside knowledge.
2. Cite the source of every claim as [1], [2].
3. If the context is insufficient, reply with exactly: "{refusal}"
4. Be concise. Do not repeat yourself.
5. Never guess.

Context:
{context}"""


@dataclass
class Answer:
    """A generated answer plus everything needed to audit it."""

    text: str
    abstained: bool
    retrieval: RetrievalResult
    citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    language: str = "tr"
    elapsed_ms: float = 0.0
    generation_ms: float = 0.0
    prompt_chars: int = 0

    @property
    def is_grounded(self) -> bool:
        """True when the answer cites at least one real passage.

        An answer with no citations at all is not necessarily wrong, but it is
        unverifiable - and for this system unverifiable is a failure mode, since
        the whole point is that claims trace back to a regulation.
        """
        return bool(self.citations) and not self.invalid_citations

    def cited_sources(self) -> list[str]:
        labels = []
        for n in self.citations:
            if 1 <= n <= len(self.retrieval.hits):
                labels.append(self.retrieval.hits[n - 1].hit.citation_label())
        return labels

    def summary(self) -> str:
        if self.abstained:
            return f"ABSTAINED ({self.elapsed_ms:.0f} ms)"
        flag = "grounded" if self.is_grounded else "UNGROUNDED"
        return (
            f"{flag} | citations={self.citations} "
            f"invalid={self.invalid_citations} | {self.elapsed_ms:.0f} ms"
        )


def _build_context(result: RetrievalResult, max_chars: int) -> str:
    """Numbered context block, truncated to a character budget.

    Budgeting by characters rather than tokens is a deliberate approximation: we
    have no tokenizer guaranteed to match the served model. The budget is set
    conservatively low so the approximation error cannot push us past the
    context window - overflowing it silently truncates the *system prompt*,
    which would remove the grounding rules and produce exactly the ungrounded
    behaviour we are trying to prevent.
    """
    blocks: list[str] = []
    used = 0
    for i, hit in enumerate(result.hits, start=1):
        text = hit.hit.display_text()
        label = hit.hit.citation_label()
        block = f"[{i}] ({label})\n{text}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _extract_citations(text: str, n_available: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) citation numbers found in the answer."""
    found: list[int] = []
    for token in re.findall(r"\[(\d{1,2})\]", text):
        value = int(token)
        if value not in found:
            found.append(value)
    valid = [c for c in found if 1 <= c <= n_available]
    invalid = [c for c in found if not (1 <= c <= n_available)]
    return valid, invalid


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "bulunmuyor", "belgelerde yok", "bilgi yok", "mevcut değil",
        "not present", "not contain", "cannot answer", "yanıtlayamıyorum",
    )
    return any(m in lowered for m in markers)


class Answerer:
    """Turns retrieved passages into a cited answer, or an honest refusal."""

    def __init__(
        self,
        cfg: GenerationConfig,
        runtime: FoundryRuntime,
        # Measured, not guessed. scripts/prompt_ablation.py swept this budget
        # against phi-3.5-mini on grounded Turkish QA and found instruction
        # following collapses as it grows:
        #
        #   1200 chars -> refused 2/3 unanswerable, answered 2/2 answerable, 39 s
        #   2400 chars -> 1/3, 1/2, 60 s
        #   4000 chars -> 0/3, 0/2, 57 s   (completely broken)
        #   6000 chars -> 0/3, 1/2, 94 s   (and leaked its own rules)
        #
        # This inverts the usual assumption that more retrieved context helps.
        # For a small model, context is a cost, not a resource: the grounding
        # rules sit at the top of the system message and get diluted by
        # everything that follows.
        #
        # Note the deliberate tension with retrieval: `top_k` stays at 5 because
        # recall@5 is what the retrieval metrics measure, but the budget here
        # means only the top ~2 chunks actually reach the model. Retrieve wide,
        # feed narrow.
        max_context_chars: int = 1600,
    ) -> None:
        self.cfg = cfg
        self.runtime = runtime
        self.max_context_chars = max_context_chars

    def _language_for(self, query: str) -> str:
        if self.cfg.language != "auto":
            return self.cfg.language
        detected = detect_language(query)
        # Turkish is the corpus default; "unknown" (very short queries) falls
        # back to it rather than silently switching the user's language.
        return "en" if detected == "en" else "tr"

    def _should_abstain(self, result: RetrievalResult) -> bool:
        if not self.cfg.enable_abstention:
            return False
        if not result.hits:
            return True
        return result.top_score < self.cfg.abstention_threshold

    def _messages(self, query: str, result: RetrievalResult, language: str):
        context = _build_context(result, self.max_context_chars)
        template = _SYSTEM_TR if language == "tr" else _SYSTEM_EN
        refusal = REFUSAL_TR if language == "tr" else REFUSAL_EN
        system = template.format(context=context, refusal=refusal)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ], len(system)

    def answer(self, query: str, result: RetrievalResult) -> Answer:
        import time

        started = time.perf_counter()
        language = self._language_for(query)
        refusal = REFUSAL_TR if language == "tr" else REFUSAL_EN

        if self._should_abstain(result):
            logger.debug(
                "Abstaining: top_score=%.4f < threshold=%.4f",
                result.top_score, self.cfg.abstention_threshold,
            )
            return Answer(
                text=refusal, abstained=True, retrieval=result, language=language,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        messages, prompt_chars = self._messages(query, result, language)
        gen_started = time.perf_counter()
        try:
            text = self.runtime.chat(
                messages,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            ).strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("Generation failed: %s", exc)
            return Answer(
                text=refusal, abstained=True, retrieval=result, language=language,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
        generation_ms = (time.perf_counter() - gen_started) * 1000

        valid, invalid = _extract_citations(text, len(result.hits))
        # The model may refuse on its own even when retrieval cleared the
        # threshold. That counts as an abstention: the outcome the user sees is
        # a refusal, and the evaluation must score it as one.
        model_refused = _looks_like_refusal(text) and not valid

        return Answer(
            text=text, abstained=model_refused, retrieval=result,
            citations=valid, invalid_citations=invalid, language=language,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            generation_ms=generation_ms, prompt_chars=prompt_chars,
        )

    def answer_stream(
        self, query: str, result: RetrievalResult
    ) -> Iterator[str]:
        """Streaming variant for the interactive UI.

        Abstention short-circuits here too, so a refusal appears instantly
        rather than after a model round-trip.
        """
        language = self._language_for(query)
        if self._should_abstain(result):
            yield REFUSAL_TR if language == "tr" else REFUSAL_EN
            return

        messages, _ = self._messages(query, result, language)
        yield from self.runtime.chat_stream(
            messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
