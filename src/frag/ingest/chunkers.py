"""Document splitting strategies, swappable so they can be benchmarked.

Chunking is the highest-leverage and least-discussed decision in a RAG system.
It is upstream of everything: a fact split across two chunks can never be
retrieved intact, no matter how good the embedding model or the reranker is.
The project plan treats chunking as a one-liner ("split by paragraphs"). Here it
is a first-class variable with four implementations, so the write-up can report
which one wins on our corpus instead of asserting one.

Empirically motivated constraint - `min_chunk_chars`: on this machine, very
short strings embed into a degenerate region of the vector space where
unrelated texts score ~0.62 cosine against each other, while the same model
separates full sentences cleanly (0.64 relevant vs 0.27 irrelevant). Tiny
chunks are therefore not merely uninformative, they are actively harmful: they
inject high-scoring noise into every result list. Fragments below the threshold
are merged into their neighbour rather than indexed alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from ..config import ChunkConfig
from .normalize import clean_text, split_sentences


@dataclass
class Chunk:
    """One retrievable unit of text, plus everything needed to cite it."""

    text: str
    doc_id: str
    source: str
    ordinal: int  # position within the document, 0-based
    char_start: int
    char_end: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}::{self.ordinal}"

    def citation_label(self) -> str:
        """Human-readable provenance shown next to an answer."""
        page = self.meta.get("page")
        heading = self.meta.get("heading")
        parts = [self.source]
        if heading:
            parts.append(str(heading))
        if page is not None:
            parts.append(f"s.{page}")
        return " · ".join(parts)


class Chunker(Protocol):
    """Anything that turns a document's text into chunks."""

    name: str

    def split(self, text: str, doc_id: str, source: str,
              meta: dict[str, Any] | None = None) -> list[Chunk]: ...


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _merge_short_chunks(chunks: list[Chunk], min_chars: int) -> list[Chunk]:
    """Fold sub-threshold chunks into the previous one.

    Merging backwards keeps a trailing fragment ("Yürürlük" at the end of a
    regulation) attached to the text it belongs with, instead of orphaning it
    as its own noisy vector.
    """
    if not chunks:
        return []

    merged: list[Chunk] = []
    for chunk in chunks:
        if merged and len(chunk.text) < min_chars:
            prev = merged[-1]
            prev.text = f"{prev.text}\n{chunk.text}".strip()
            prev.char_end = chunk.char_end
        else:
            merged.append(chunk)

    # A leading fragment can survive the pass above; fold it forward.
    if len(merged) > 1 and len(merged[0].text) < min_chars:
        merged[1].text = f"{merged[0].text}\n{merged[1].text}".strip()
        merged[1].char_start = merged[0].char_start
        merged.pop(0)

    for i, chunk in enumerate(merged):
        chunk.ordinal = i
    return merged


def _make(text: str, doc_id: str, source: str, ordinal: int,
          start: int, end: int, meta: dict[str, Any] | None,
          strategy: str) -> Chunk:
    payload = dict(meta or {})
    payload["strategy"] = strategy
    return Chunk(
        text=text.strip(), doc_id=doc_id, source=source, ordinal=ordinal,
        char_start=start, char_end=end, meta=payload,
    )


# ---------------------------------------------------------------------------
# 1. fixed-size
# ---------------------------------------------------------------------------


class FixedChunker:
    """Fixed character windows with overlap. The baseline to beat.

    Overlap exists to hedge against splitting a fact in half: with a stride of
    `chunk_size - overlap`, any span shorter than `overlap` is guaranteed to
    appear intact in at least one chunk. That guarantee is the entire argument
    for overlap, and it also tells you how to size it - set `overlap` to the
    length of the longest fact you cannot afford to lose.

    The cost is duplication: total indexed text grows by roughly
    `chunk_size / (chunk_size - overlap)`, so 900/150 stores ~1.2x the corpus.
    """

    name = "fixed"

    def __init__(self, cfg: ChunkConfig) -> None:
        self.cfg = cfg

    def split(self, text: str, doc_id: str, source: str,
              meta: dict[str, Any] | None = None) -> list[Chunk]:
        text = clean_text(text)
        if not text:
            return []

        size = self.cfg.chunk_size
        stride = max(1, size - self.cfg.overlap)
        chunks: list[Chunk] = []
        ordinal = 0

        for start in range(0, len(text), stride):
            window = text[start : start + size]
            if not window.strip():
                continue
            chunks.append(
                _make(window, doc_id, source, ordinal, start,
                      min(start + size, len(text)), meta, self.name)
            )
            ordinal += 1
            if start + size >= len(text):
                break

        return _merge_short_chunks(chunks, self.cfg.min_chunk_chars)


# ---------------------------------------------------------------------------
# 2. recursive (structure-aware)
# ---------------------------------------------------------------------------


class RecursiveChunker:
    """Split on the most semantic boundary that fits, then fall back.

    The separator ladder runs from strongest to weakest: article headings ->
    paragraph breaks -> line breaks -> sentence ends -> spaces. We only descend
    when a piece is still too large, so a document that has clean paragraph
    structure never gets cut mid-sentence.

    The Turkish-specific rung is the regulation-article pattern ("MADDE 7 -").
    Boğaziçi yönetmelikleri and most Turkish academic regulations are organised
    that way, and each article is a self-contained rule - exactly the unit a
    question is asked about. Splitting there is the single highest-value
    domain adaptation in this file.
    """

    name = "recursive"

    # Ordered strongest -> weakest.
    SEPARATORS = [
        re.compile(r"\n(?=MADDE\s+\d+)", re.IGNORECASE),  # regulation articles
        re.compile(r"\n(?=Madde\s+\d+)"),
        re.compile(r"\n(?=\d+\.\s+[A-ZÇĞİÖŞÜ])"),          # numbered headings
        re.compile(r"\n\n+"),                               # paragraphs
        re.compile(r"\n"),                                  # lines
        re.compile(r"(?<=[.!?])\s+"),                       # sentences
        re.compile(r"\s+"),                                 # words
    ]

    def __init__(self, cfg: ChunkConfig) -> None:
        self.cfg = cfg

    def split(self, text: str, doc_id: str, source: str,
              meta: dict[str, Any] | None = None) -> list[Chunk]:
        text = clean_text(text)
        if not text:
            return []

        pieces = self._recurse(text, 0)
        chunks: list[Chunk] = []
        cursor = 0
        for ordinal, piece in enumerate(pieces):
            # Locate the piece to keep character offsets honest for citations.
            idx = text.find(piece, cursor)
            if idx == -1:
                idx = cursor
            chunks.append(
                _make(piece, doc_id, source, ordinal, idx, idx + len(piece),
                      meta, self.name)
            )
            cursor = idx + len(piece)

        chunks = self._apply_overlap(chunks, text)
        return _merge_short_chunks(chunks, self.cfg.min_chunk_chars)

    def _recurse(self, text: str, depth: int) -> list[str]:
        if len(text) <= self.cfg.chunk_size:
            return [text] if text.strip() else []
        if depth >= len(self.SEPARATORS):
            # Out of separators: hard-cut. Reaching here means a single
            # unbroken run longer than chunk_size, e.g. a base64 blob.
            size = self.cfg.chunk_size
            return [text[i : i + size] for i in range(0, len(text), size)]

        parts = [p for p in self.SEPARATORS[depth].split(text) if p.strip()]
        if len(parts) <= 1:
            return self._recurse(text, depth + 1)

        # Greedily repack parts up to chunk_size before descending further.
        out: list[str] = []
        buffer = ""
        for part in parts:
            candidate = f"{buffer}\n{part}" if buffer else part
            if len(candidate) <= self.cfg.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    out.append(buffer)
                if len(part) > self.cfg.chunk_size:
                    out.extend(self._recurse(part, depth + 1))
                    buffer = ""
                else:
                    buffer = part
        if buffer:
            out.append(buffer)
        return out

    def _apply_overlap(self, chunks: list[Chunk], text: str) -> list[Chunk]:
        """Prepend the tail of the previous chunk to each chunk.

        Structure-aware splitting still loses cross-reference context ("bu
        sürede", "yukarıdaki madde"). A small backward overlap restores the
        antecedent without the storage cost of fixed-window overlap.
        """
        if self.cfg.overlap <= 0 or len(chunks) < 2:
            return chunks

        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1].text[-self.cfg.overlap :]
            # Start the overlap at a word boundary so we do not paste a
            # half-word onto the front of the chunk.
            space = prev_tail.find(" ")
            if space > 0:
                prev_tail = prev_tail[space + 1 :]
            if prev_tail.strip():
                chunks[i].text = f"{prev_tail.strip()}\n{chunks[i].text}"
                chunks[i].char_start = max(0, chunks[i].char_start - len(prev_tail))
        return chunks


# ---------------------------------------------------------------------------
# 3. sentence window
# ---------------------------------------------------------------------------


class SentenceWindowChunker:
    """Index single sentences, but return them with their neighbours.

    This decouples two things that fixed chunking forces together: the unit you
    *match* against and the unit you *read*. A single sentence is a sharp
    embedding target - little dilution, high precision. But a sentence alone is
    usually too thin to answer from, so at retrieval time we hand the generator
    the sentence plus `window_size` neighbours on each side.

    The trade-off is index size: one vector per sentence rather than per
    paragraph, so the vector count grows several-fold. On a laptop corpus that
    is affordable; on millions of documents it would not be.
    """

    name = "sentence_window"

    def __init__(self, cfg: ChunkConfig) -> None:
        self.cfg = cfg

    def split(self, text: str, doc_id: str, source: str,
              meta: dict[str, Any] | None = None) -> list[Chunk]:
        text = clean_text(text)
        sentences = split_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        cursor = 0
        w = self.cfg.window_size

        for i, sentence in enumerate(sentences):
            idx = text.find(sentence, cursor)
            if idx == -1:
                idx = cursor
            cursor = idx + len(sentence)

            window = " ".join(sentences[max(0, i - w) : i + w + 1])
            payload = dict(meta or {})
            # `window_text` is what the generator reads; `text` is what we embed.
            payload["window_text"] = window
            chunks.append(
                _make(sentence, doc_id, source, i, idx, idx + len(sentence),
                      payload, self.name)
            )

        # Note: no _merge_short_chunks here. Merging would defeat the point -
        # short sentences are exactly what this strategy means to index sharply,
        # and the window supplies the missing context at read time.
        return chunks


# ---------------------------------------------------------------------------
# 4. semantic
# ---------------------------------------------------------------------------


class SemanticChunker:
    """Split where the topic actually changes, detected from embeddings.

    Method: embed every sentence, then walk the document computing the cosine
    *distance* between consecutive sentences. Adjacent sentences about the same
    topic sit close together; a topic shift shows up as a spike. We cut at the
    spikes.

    The threshold is a percentile of that distance distribution, not a fixed
    number - and that choice is the whole point. Absolute cosine values are not
    comparable across documents or languages (the concentration effect measured
    in this project puts them all in a narrow band), but the *shape* of a
    document's own distance distribution is meaningful relative to itself.
    Using the 90th percentile says "cut at this document's most abrupt 10% of
    transitions", which transfers across documents in a way that "cut when
    distance > 0.35" does not.

    Cost: one embedding call per sentence at index time, so this is by far the
    most expensive strategy. Whether it earns that cost is an empirical
    question - which is exactly what the benchmark answers.
    """

    name = "semantic"

    def __init__(self, cfg: ChunkConfig,
                 embed_fn: Callable[[Sequence[str]], np.ndarray]) -> None:
        self.cfg = cfg
        self.embed_fn = embed_fn

    def split(self, text: str, doc_id: str, source: str,
              meta: dict[str, Any] | None = None) -> list[Chunk]:
        text = clean_text(text)
        sentences = split_sentences(text)
        if len(sentences) < 3:
            # Too short for the statistics to mean anything; keep it whole.
            return [_make(text, doc_id, source, 0, 0, len(text), meta, self.name)] if text else []

        vectors = self.embed_fn(sentences)  # already L2-normalised
        # Cosine distance between consecutive sentences.
        similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)
        distances = 1.0 - similarities

        threshold = float(np.percentile(distances, self.cfg.breakpoint_percentile))
        breakpoints = [i + 1 for i, d in enumerate(distances) if d > threshold]

        bounds = [0, *breakpoints, len(sentences)]
        chunks: list[Chunk] = []
        cursor = 0
        ordinal = 0

        for lo, hi in zip(bounds[:-1], bounds[1:]):
            group = sentences[lo:hi]
            if not group:
                continue
            body = " ".join(group)
            idx = text.find(group[0], cursor)
            if idx == -1:
                idx = cursor
            cursor = idx + len(body)
            payload = dict(meta or {})
            payload["breakpoint_threshold"] = round(threshold, 4)
            chunks.append(
                _make(body, doc_id, source, ordinal, idx, idx + len(body),
                      payload, self.name)
            )
            ordinal += 1

        return _merge_short_chunks(chunks, self.cfg.min_chunk_chars)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def build_chunker(
    cfg: ChunkConfig,
    embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> Chunker:
    """Instantiate the configured strategy.

    `embed_fn` is only required by the semantic strategy; asking for it eagerly
    would force every caller to spin up a model they may not need.
    """
    if cfg.strategy == "fixed":
        return FixedChunker(cfg)
    if cfg.strategy == "recursive":
        return RecursiveChunker(cfg)
    if cfg.strategy == "sentence_window":
        return SentenceWindowChunker(cfg)
    if cfg.strategy == "semantic":
        if embed_fn is None:
            raise ValueError(
                "The 'semantic' chunking strategy needs an embed_fn "
                "(pass FoundryRuntime.embed)."
            )
        return SemanticChunker(cfg, embed_fn)
    raise ValueError(f"Unknown chunking strategy: {cfg.strategy!r}")
