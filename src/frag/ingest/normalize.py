"""Text normalisation and sentence segmentation, with Turkish handled properly.

Two Turkish-specific traps are handled here, and both silently corrupt a RAG
pipeline if ignored.

**1. The dotted/dotless i.** Turkish has four i-letters: i I ı İ. The pair
(i, İ) and (ı, I) are the real case partners, which is the opposite of what
`str.lower()` does under a non-Turkish locale: Python maps "I" -> "i", but in
Turkish "I" must map to "ı". Get this wrong and "IŞIK" lowercases to "işik"
instead of "ışık", so the BM25 index stores a token that no query will ever
match. Python's `str.lower()` has no locale awareness, so we do the two
Turkish-specific substitutions by hand before folding the rest.

**2. Ordinals look like sentence ends.** Legal and academic Turkish is full of
"1. madde", "2. fıkra", "3. bent". A naive split on "." shatters exactly the
documents we care about most. The segmenter below refuses to split when the
period follows a bare number, or a known abbreviation.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable

# Abbreviations whose trailing period never ends a sentence.
TURKISH_ABBREVIATIONS = {
    "dr", "prof", "doç", "yrd", "öğr", "gör", "arş", "av", "sy", "sn",
    "md", "bkz", "vb", "vs", "örn", "yy", "no", "nu", "sf", "s", "c",
    "bkn", "age", "agm", "çev", "haz", "ed", "yay", "bşk", "gen", "alb",
    "tbmm", "yök", "ösym", "mad", "fık", "krş", "yrd.doç",
}

# Repeated whitespace, including the non-breaking spaces PDFs love to emit.
_WS_RE = re.compile(r"[ \t  -​]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
# Hyphenation across a line break: "yönet-\nmelik" -> "yönetmelik"
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


def turkish_lower(text: str) -> str:
    """Lowercase with Turkish i-rules applied first.

    Order matters: İ -> i and I -> ı must happen *before* the generic lower(),
    otherwise Python's default mapping wins and the dotted/dotless distinction
    is destroyed.
    """
    return text.replace("İ", "i").replace("I", "ı").lower()


def turkish_upper(text: str) -> str:
    """Uppercase with Turkish i-rules applied first."""
    return text.replace("i", "İ").replace("ı", "I").upper()


# A lowercase run, a single space, then a short lowercase run that continues the
# same word. Restricted to Turkish letters so we never join across digits or
# punctuation.
_BROKEN_WORD_RE = re.compile(
    r"\b([a-zçğıöşü]{3,})\s([a-zçğıöşü]{1,3})\b", flags=re.IGNORECASE
)


def repair_pdf_word_breaks(text: str, vocabulary: set[str] | None = None) -> str:
    """Rejoin words that PDF extraction split with a stray space.

    Kerned PDFs routinely emit "Yönetm eliği" and "yararl andığı": the glyph
    positions are correct but the extractor guesses a word boundary from
    spacing. Measured on this corpus, that happens 28-256 times per document,
    and it damages *both* retrieval arms - the embedding model sees
    out-of-vocabulary fragments, and BM25 indexes tokens no query will produce.

    The repair is only safe with evidence, because Turkish has many legitimate
    short words ("de", "bir", "ile") that follow a longer one. So we join a pair
    only when the concatenation appears elsewhere in the corpus as a real word,
    and the fragment does not. `vocabulary` supplies that evidence; without it
    the function is a no-op rather than a guess.
    """
    if not vocabulary:
        return text

    def _maybe_join(match: re.Match[str]) -> str:
        head, tail = match.group(1), match.group(2)
        joined = f"{head}{tail}"
        lowered_joined = turkish_lower(joined)
        lowered_tail = turkish_lower(tail)
        # Join only if the merged form is a known word and the fragment is not a
        # word in its own right. That second condition is what protects
        # "olarak de" from becoming "olarakde".
        if lowered_joined in vocabulary and lowered_tail not in vocabulary:
            return joined
        return match.group(0)

    return _BROKEN_WORD_RE.sub(_maybe_join, text)


def build_vocabulary(texts: Iterable[str], min_count: int = 2) -> set[str]:
    """Collect words that appear at least `min_count` times across the corpus.

    Used as the evidence base for `repair_pdf_word_breaks`. Requiring two
    occurrences keeps a word that is itself the product of a break from voting
    for its own correctness.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(re.findall(r"[a-zçğıöşü]{2,}", turkish_lower(text)))
    return {word for word, n in counts.items() if n >= min_count}


def clean_text(text: str) -> str:
    """Normalise whitespace and PDF extraction artefacts.

    Unicode NFC composition comes first so that "ş" written as s + combining
    cedilla compares equal to the precomposed character. Without it, two
    visually identical Turkish words can hash differently and never match.
    """
    text = unicodedata.normalize("NFC", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    # Strip trailing spaces on each line without collapsing paragraph breaks.
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def fold_for_lexical(text: str) -> str:
    """Aggressive fold used only for the lexical (BM25) index.

    Diacritics are removed so that a query typed without Turkish characters -
    "kayit dondurma", which is how people actually type on a phone - still
    matches "kayıt dondurma" in the corpus. This is applied symmetrically to
    documents and queries, so no information is lost at match time.

    It is *not* used for embeddings: the embedding model was trained on properly
    accented text and folding would move the input off-distribution.
    """
    lowered = turkish_lower(text)
    decomposed = unicodedata.normalize("NFD", lowered)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    # NFD does not decompose these, so map them explicitly.
    return (
        stripped.replace("ı", "i")
        .replace("ğ", "g")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ç", "c")
    )


_SENTENCE_END_RE = re.compile(r"([.!?…])(\s+|$)")


def split_sentences(text: str) -> list[str]:
    """Segment Turkish text into sentences.

    Deliberately rule-based rather than model-based: it must run on every chunk
    of every document during indexing, and a neural segmenter would dominate
    the ingestion budget for a gain that does not show up in retrieval metrics.

    A period does *not* end a sentence when it follows:
      * a known abbreviation ("vb.", "Dr.", "md.")
      * a bare number ("1. madde", "2019. yılda")
      * a single capital letter (initials: "M. Kemal")
    """
    if not text.strip():
        return []

    sentences: list[str] = []
    start = 0

    for match in _SENTENCE_END_RE.finditer(text):
        end = match.end(1)
        candidate = text[start:end].strip()
        if not candidate:
            continue

        if match.group(1) == "." and _is_false_sentence_end(text, match.start(1)):
            continue

        sentences.append(candidate)
        start = match.end()

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)

    return sentences


def _is_false_sentence_end(text: str, period_index: int) -> bool:
    """True when the period at `period_index` is not a sentence terminator."""
    preceding = text[:period_index]
    # Last run of word characters before the period.
    token_match = re.search(r"([\w’']+)$", preceding, flags=re.UNICODE)
    if not token_match:
        return False
    token = token_match.group(1)

    # "1. madde", "2019. yıl" - ordinals and years.
    if token.isdigit():
        return True

    # Single capital letter: an initial, e.g. "M. Kemal".
    if len(token) == 1 and token.isupper():
        return True

    if turkish_lower(token) in TURKISH_ABBREVIATIONS:
        return True

    return False


# Inflectional suffixes, longest first so that "larindan" is tried before "lar".
# Written in folded form (see fold_for_lexical) because that is the alphabet the
# lexical index actually stores.
# Every entry is already folded (a-z only): fold_for_lexical maps ı->i, ü->u,
# ö->o, ş->s, ç->c, ğ->g, so the vowel-harmony variants of one suffix collapse
# onto a single spelling here. That is why "ların" and "lerin" both appear as
# "larin"/"lerin" but "nın" does not appear separately from "nin".
_TURKISH_SUFFIXES = [
    "larindan", "lerinden", "lariyla", "leriyle", "larimiz", "lerimiz",
    "lardan", "lerden", "siniz", "sunuz",
    "larin", "lerin", "lara", "lere", "lari", "leri",
    "imiz", "iniz", "ndan", "nden",
    "lar", "ler", "dan", "den", "tan", "ten", "nin", "nun",
    "lik", "luk", "cik", "cuk",
    "in", "un", "im", "um", "si", "su", "ya", "ye",
    "da", "de", "ta", "te", "na", "ne", "yi", "yu", "ci", "cu", "li", "lu",
    "i", "u", "e", "a",
]

# Below this length, stripping a suffix destroys the word instead of stemming it
# ("de" -> "" , "kar" -> "k"). Turkish roots are rarely shorter than 3 letters.
_MIN_STEM_LENGTH = 4


def turkish_stem(word: str, max_strips: int = 2) -> str:
    """Strip inflectional suffixes from an already-folded Turkish word.

    This is a heuristic, not a morphological analyser. Turkish is agglutinative:
    a single root can carry a chain of suffixes ("ev-ler-imiz-den"), so exact
    matching in a lexical index fails constantly - a user searching "kayıt
    dondurma" gets no BM25 signal from a document saying "kayıtların
    dondurulması", even though it is the passage they want.

    A real analyser (Zemberek and friends) would do better, but it is a heavy
    Java dependency and this project's whole premise is that everything runs
    locally with no extra services. So we take the cheap 80% and then *measure*
    whether it helped, rather than assuming. See docs/03.

    Known failure mode: over-stemming collapses distinct words ("kar" snow vs
    "kara" land). `_MIN_STEM_LENGTH` and the two-strip cap bound the damage.
    Stemming is applied only to the lexical index; the dense index sees
    untouched text, so a mistake here degrades one retrieval arm, never both.
    """
    if len(word) <= _MIN_STEM_LENGTH:
        return word

    stem = word
    for _ in range(max_strips):
        for suffix in _TURKISH_SUFFIXES:
            if stem.endswith(suffix) and len(stem) - len(suffix) >= _MIN_STEM_LENGTH:
                stem = stem[: -len(suffix)]
                break
        else:
            break
    return stem


def lexical_tokens(
    text: str, stem: bool = True, truncate: int | None = None
) -> list[str]:
    """Fold, tokenise, optionally stem - the exact text the BM25 index sees.

    Applied identically to documents at index time and to queries at search
    time. That symmetry is what makes the transformation safe: both sides land
    in the same normalised alphabet, so nothing is lost at match time.

    `truncate` keeps only the first N characters of each stem. It exists to
    reach the suffix-stripping blind spot measured on this corpus: Turkish
    *derivational* morphology inserts material mid-word, so "dondurulması" and
    "dondurma" stem to "dondurulm" and "dondurm" and never match, even though
    they are the same verb. Truncating both to 6 characters yields "dondur" for
    each.

    The cost is precision: shorter keys collide more, so unrelated words start
    matching. Whether that trade is worth it is a corpus-specific empirical
    question, which is why this is a parameter the benchmark sweeps rather than
    a decision baked into the code.
    """
    folded = fold_for_lexical(text)
    words = re.findall(r"[a-z0-9]+", folded)
    if stem:
        words = [turkish_stem(w) for w in words]
    if truncate:
        words = [w[:truncate] if len(w) > truncate else w for w in words]
    return words


def detect_language(text: str) -> str:
    """Cheap Turkish-vs-English detector: 'tr', 'en', or 'unknown'.

    Uses two orthogonal signals - characters that only exist in Turkish, and
    high-frequency function words. Function words alone are unreliable on short
    strings; the character check alone misses Turkish text typed without
    diacritics. Requiring either signal keeps it robust enough for the routing
    decision it feeds (which stopword list and prompt language to use).
    """
    if not text.strip():
        return "unknown"

    sample = text[:2000]
    turkish_chars = sum(sample.count(c) for c in "ğışĞİŞ")
    if turkish_chars / max(len(sample), 1) > 0.004:
        return "tr"

    words = set(re.findall(r"[\w’']+", turkish_lower(sample)))
    tr_markers = {"ve", "bir", "için", "ile", "bu", "olarak", "veya", "da", "de", "en"}
    en_markers = {"the", "and", "for", "with", "this", "that", "are", "from", "which"}
    tr_hits = len(words & tr_markers)
    en_hits = len(words & en_markers)

    if tr_hits > en_hits:
        return "tr"
    if en_hits > tr_hits:
        return "en"
    return "unknown"
