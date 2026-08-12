"""Turning files on disk into text we can chunk.

Deliberately narrow: PDF, Markdown, HTML, and plain text. That covers the
regulation PDFs and course-catalogue pages this project targets, and every
extra format is another silent failure mode.

The PDF path carries the most risk. Text extraction is lossy in ways that are
invisible until retrieval quality drops - two-column layouts interleave, tables
lose their structure, headers repeat on every page. We detect and strip
repeated headers/footers because, left in, they become the highest-frequency
n-grams in the corpus and distort BM25's IDF weighting for every query.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .normalize import (
    build_vocabulary,
    clean_text,
    detect_language,
    repair_pdf_word_breaks,
)

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm"}


@dataclass
class Document:
    """A loaded source document, before chunking."""

    doc_id: str
    source: str          # display name used in citations
    text: str
    path: str
    language: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# format-specific readers
# ---------------------------------------------------------------------------


def _read_txt(path: Path) -> tuple[str, dict[str, Any]]:
    return path.read_text(encoding="utf-8", errors="replace"), {}


def _read_markdown(path: Path) -> tuple[str, dict[str, Any]]:
    """Keep Markdown mostly as-is.

    Heading markers are retained on purpose: RecursiveChunker uses them as split
    points, and they carry real structural signal that stripping would discard.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    title = ""
    for line in raw.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    return raw, {"title": title} if title else {}


def _read_html(path: Path) -> tuple[str, dict[str, Any]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(
        path.read_text(encoding="utf-8", errors="replace"), "lxml"
    )
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    # separator="\n" keeps block-level boundaries, which the chunker needs.
    return soup.get_text(separator="\n"), {"title": title} if title else {}


def _read_pdf(path: Path) -> tuple[str, dict[str, Any]]:
    """Extract text per page, tag page numbers, and drop running headers.

    Page numbers are preserved as inline markers so a citation can say "s.14"
    instead of pointing at a 60-page PDF and wishing the reader luck.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - a bad page should not kill the doc
            logger.warning("%s: page extraction failed (%s)", path.name, exc)
            pages.append("")

    pages = _strip_repeated_lines(pages)

    body = "\n\n".join(
        f"[[page:{i + 1}]]\n{text}" for i, text in enumerate(pages) if text.strip()
    )
    meta: dict[str, Any] = {"n_pages": len(pages)}
    if reader.metadata and reader.metadata.title:
        meta["title"] = str(reader.metadata.title).strip()
    return body, meta


def _strip_repeated_lines(pages: list[str], min_ratio: float = 0.6) -> list[str]:
    """Remove lines that appear on most pages - headers, footers, page numbers.

    Why bother: BM25 weights a term by inverse document frequency. A header
    repeated on 60 pages becomes 60 occurrences of a term that carries zero
    information, which drags down the IDF of every word it contains and lets
    boilerplate compete with real content. Removing it is a correctness fix for
    the lexical index, not cosmetics.

    The 0.6 threshold is conservative: a line must appear on most pages before
    we call it boilerplate, so a sentence that legitimately recurs a few times
    survives.
    """
    if len(pages) < 3:
        return pages

    counts: Counter[str] = Counter()
    for page in pages:
        # Only the first and last few lines can be headers/footers.
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        for line in lines[:3] + lines[-3:]:
            if 3 <= len(line) <= 120:
                counts[line] += 1

    threshold = max(3, int(len(pages) * min_ratio))
    boilerplate = {line for line, n in counts.items() if n >= threshold}
    if not boilerplate:
        return pages

    logger.debug("Stripping %d boilerplate lines", len(boilerplate))
    cleaned = []
    for page in pages:
        kept = [ln for ln in page.splitlines() if ln.strip() not in boilerplate]
        cleaned.append("\n".join(kept))
    return cleaned


_READERS = {
    ".pdf": _read_pdf,
    ".md": _read_markdown,
    ".markdown": _read_markdown,
    ".txt": _read_txt,
    ".html": _read_html,
    ".htm": _read_html,
}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def load_document(path: str | Path, doc_id: str | None = None) -> Document | None:
    """Load one file. Returns None if it is unsupported or yields no text."""
    path = Path(path)
    suffix = path.suffix.lower()

    reader = _READERS.get(suffix)
    if reader is None:
        logger.debug("Skipping unsupported file: %s", path.name)
        return None

    try:
        raw, meta = reader(path)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop ingestion
        logger.error("Could not read %s: %s", path.name, exc)
        return None

    text = clean_text(raw)
    if len(text) < 50:
        logger.warning("%s yielded almost no text (%d chars) - skipping", path.name, len(text))
        return None

    return Document(
        doc_id=doc_id or path.stem,
        source=meta.get("title") or path.stem,
        text=text,
        path=str(path),
        language=detect_language(text),
        meta=meta,
    )


def load_corpus(
    directory: str | Path, recursive: bool = True, repair_word_breaks: bool = True
) -> list[Document]:
    """Load every supported file in a directory.

    Sorted for determinism: the same corpus must produce the same doc ordering
    on every run, or two benchmark runs are not comparable.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Corpus directory not found: {directory}")

    pattern = "**/*" if recursive else "*"
    paths = sorted(
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )

    docs: list[Document] = []
    for path in paths:
        # Use the path relative to the corpus root as the id, so two files with
        # the same stem in different folders do not collide.
        rel = path.relative_to(directory)
        doc = load_document(path, doc_id=str(rel.with_suffix("")))
        if doc is not None:
            docs.append(doc)

    # Repair PDF word-splitting as a second pass, because the evidence for a
    # repair is the rest of the corpus: a word only counts as real if it occurs
    # elsewhere. That means we cannot do this while loading a single file.
    if repair_word_breaks and docs:
        vocabulary = build_vocabulary(d.text for d in docs)
        repaired_total = 0
        for doc in docs:
            before = doc.text
            doc.text = repair_pdf_word_breaks(before, vocabulary)
            # Each repair joins two tokens into one, so the drop in token count
            # is the exact number of repairs. Comparing tokens pairwise would
            # not work: one join shifts every later token and inflates the count.
            repaired_total += len(before.split()) - len(doc.text.split())
        if repaired_total:
            logger.info("Repaired %d PDF word breaks", repaired_total)

    logger.info(
        "Loaded %d documents (%d chars) from %s",
        len(docs), sum(len(d) for d in docs), directory,
    )
    return docs


def corpus_stats(docs: Iterable[Document]) -> dict[str, Any]:
    """Summary used in the report and shown on the dashboard."""
    docs = list(docs)
    if not docs:
        return {"n_docs": 0}

    languages: Counter[str] = Counter(d.language for d in docs)
    lengths = [len(d) for d in docs]
    return {
        "n_docs": len(docs),
        "total_chars": sum(lengths),
        "median_chars": sorted(lengths)[len(lengths) // 2],
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "languages": dict(languages),
    }
