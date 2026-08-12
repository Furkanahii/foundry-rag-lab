"""Reranking: a second, more expensive opinion on the top candidates.

First retrieval optimises recall - cast a wide net cheaply. Reranking optimises
precision on what the net caught. The split exists because the accurate scorer
is too slow to run over the whole corpus but affordable over 20 candidates.

## What is *not* available here

The standard answer is a cross-encoder (bge-reranker and friends): encode query
and passage *together* so attention can compare them token by token, rather than
comparing two independently-produced vectors. Foundry Local's catalogue has no
such model, and pulling an ONNX cross-encoder from elsewhere would break the
"everything comes from Foundry Local" premise of the project.

So we implement two rerankers that work with what is on the machine.

### 1. Listwise LLM reranking (`llm`)

Show the model all candidates at once and ask for an ordering. Crucially this is
**one** call, not one per candidate: with phi-3.5-mini at ~7 s per call,
pointwise scoring of 20 candidates would cost ~140 s and be unusable, while
listwise costs one call.

Listwise also has a quality argument in its favour: the model sees candidates
side by side, so it judges *relative* relevance directly instead of trying to
produce calibrated absolute scores - which small models are notoriously bad at.

Its weakness is position bias: LLMs over-favour items near the start of a list.
We mitigate by feeding candidates in retrieval order and only accepting a
reordering the model actually justifies with an id; anything it omits keeps its
original relative order.

### 2. Instruction-prefixed query embedding (`instruct`)

Not reranking in the strict sense - it changes the *query* representation rather
than rescoring candidates - but it competes for the same budget and costs one
extra embedding call (~0.2 s), so it belongs in the same comparison.

Qwen3 embedding models are trained with an asymmetric instruction template on
the query side. Feeding a bare question therefore uses the model off the
template it was tuned for. Adding the prefix costs almost nothing and is a
measurable candidate for "cheapest thing that improves ranking".
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from ..index.fusion import FusedHit
from ..runtime.foundry import FoundryRuntime

logger = logging.getLogger(__name__)

# The template Qwen3 embedding models were trained with on the query side.
QWEN3_QUERY_INSTRUCTION = (
    "Instruct: Given a question, retrieve passages from official university "
    "regulations that answer the question\nQuery: {query}"
)

_RERANK_SYSTEM = """Sen bir belge sıralama uzmanısın. Sana bir soru ve numaralı pasajlar verilecek.
Pasajları soruyu yanıtlama gücüne göre EN İYİDEN EN KÖTÜYE sırala.

Kurallar:
- SADECE numaraları virgülle ayırarak yaz. Başka hiçbir şey yazma.
- Her numarayı tam olarak bir kez kullan.
- Açıklama yapma, cümle kurma.

Örnek çıktı: 3,1,5,2,4"""


def apply_query_instruction(query: str, enabled: bool = True) -> str:
    """Wrap a bare question in the embedding model's query template."""
    return QWEN3_QUERY_INSTRUCTION.format(query=query) if enabled else query


def _parse_ordering(raw: str, n: int) -> list[int]:
    """Pull a permutation of 0..n-1 out of the model's reply.

    Small models do not reliably honour "output only numbers". We therefore
    extract every integer in order, keep the ones in range, drop duplicates, and
    append whatever the model omitted in its original position. The result is
    always a valid permutation, so a malformed reply degrades to "no reordering"
    rather than losing candidates.
    """
    seen: set[int] = set()
    order: list[int] = []
    for token in re.findall(r"\d+", raw):
        idx = int(token) - 1  # prompt is 1-based
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            order.append(idx)
    order.extend(i for i in range(n) if i not in seen)
    return order


def llm_rerank(
    runtime: FoundryRuntime,
    query: str,
    candidates: Sequence[FusedHit],
    max_candidates: int = 12,
    max_chars_per_candidate: int = 400,
) -> list[FusedHit]:
    """Reorder candidates with a single listwise LLM call.

    `max_candidates` is a real constraint, not a tuning knob: the prompt must fit
    the model's context alongside the passages, and listwise quality degrades as
    the list grows. Candidates beyond the cut keep their fusion order and are
    appended after the reranked head.
    """
    if len(candidates) < 2:
        return list(candidates)

    head = list(candidates[:max_candidates])
    tail = list(candidates[max_candidates:])

    passages = "\n\n".join(
        f"[{i + 1}] {c.read_text[:max_chars_per_candidate].strip()}"
        for i, c in enumerate(head)
    )
    messages = [
        {"role": "system", "content": _RERANK_SYSTEM},
        {"role": "user", "content": f"Soru: {query}\n\nPasajlar:\n{passages}"},
    ]

    try:
        reply = runtime.chat(messages, temperature=0.0, max_tokens=100)
    except Exception as exc:  # noqa: BLE001 - reranking is an enhancement
        logger.warning("LLM rerank failed (%s); keeping fusion order", exc)
        return list(candidates)

    order = _parse_ordering(reply, len(head))
    reordered = []
    for new_rank, idx in enumerate(order):
        hit = head[idx]
        # Record the rerank position so the dashboard can show what moved.
        hit.rerank_score = float(len(order) - new_rank)
        reordered.append(hit)

    logger.debug("Rerank reply=%r -> order=%s", reply.strip()[:60], order)
    return reordered + tail


def rerank(
    runtime: FoundryRuntime,
    query: str,
    candidates: Sequence[FusedHit],
    strategy: str = "none",
    max_candidates: int = 12,
) -> list[FusedHit]:
    """Dispatch to the configured reranking strategy."""
    if strategy in ("none", None, False):
        return list(candidates)
    if strategy == "llm":
        return llm_rerank(runtime, query, candidates, max_candidates=max_candidates)
    raise ValueError(f"Unknown rerank strategy: {strategy!r}")
