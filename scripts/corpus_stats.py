"""Corpus and IDF statistics, on the unit BM25 actually indexes.

Written after a sunum-öncesi kontrol caught a real error in this project's own
write-up. The README claimed:

    "MADDE 9 dokümanın hepsinde, 195 kez → IDF = log(N/df) = 0"

Both halves were wrong. `MADDE` occurs in eight of the nine documents (the
ninth is in English), and - the part that matters - **the BM25 index is built
over chunks, not documents**. `index/store.py` writes one FTS5 row per chunk,
so in the IDF term N is 347, not 9, and n_t counts chunks. Computing IDF at the
document level described a system that does not exist.

The corrected reading is less dramatic and more useful: IDF never reaches zero,
but the boilerplate vocabulary of regulations carries roughly a quarter to an
eighth of the weight of topical terms. A student's natural question is written
mostly in boilerplate, which is exactly when the lexical arm has least to work
with - and exactly why the dense arm has to be there.

This script exists so the number is reproducible rather than remembered.

Usage:
    PYTHONPATH=src python scripts/corpus_stats.py
    PYTHONPATH=src python scripts/corpus_stats.py --terms burs tez yurt
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys

sys.path.insert(0, "src")

from frag.ingest.loaders import corpus_stats, load_corpus  # noqa: E402
from frag.ingest.normalize import detect_language, lexical_tokens  # noqa: E402

DEFAULT_INDEX = "data/index/bogazici.db"
DEFAULT_CORPUS = "data/corpus"

# Chosen to span the range: the first two are regulation boilerplate, the rest
# are topical. The contrast is the finding; a single term would not show it.
DEFAULT_TERMS = ["öğrenci", "madde", "kayıt", "yönetmelik",
                 "tez", "disiplin", "yurt", "burs"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--terms", nargs="+", default=DEFAULT_TERMS)
    parser.add_argument("--skip-corpus", action="store_true",
                        help="only read the index, do not re-parse the PDFs")
    args = parser.parse_args()

    if not args.skip_corpus:
        docs = load_corpus(args.corpus)
        stats = corpus_stats(docs)
        print("=== korpus ===")
        print(f"  belge          {stats['n_docs']}")
        print(f"  toplam karakter {stats['total_chars']:,}".replace(",", "."))
        print(f"  medyan / min / maks  {stats['median_chars']} / "
              f"{stats['min_chars']} / {stats['max_chars']}")
        print("\n  belge başına:")
        for d in sorted(docs, key=lambda d: -len(d.text)):
            print(f"    {detect_language(d.text):>3}  {len(d.text):>7}  {d.doc_id}")

    conn = sqlite3.connect(args.index)
    rows = conn.execute("SELECT text FROM chunks").fetchall()
    n_chunks = len(rows)
    # Tokenised exactly as the lexical arm does, so df counts what BM25 counts.
    tokenised = [set(lexical_tokens(text)) for (text,) in rows]

    print(f"\n=== IDF, {n_chunks} chunk üzerinde ===")
    print("  BM25 indeksi chunk başına bir satır tutuyor; N burada belge sayısı")
    print("  değil chunk sayısıdır. Belge seviyesinde hesaplanan bir IDF,")
    print("  indeksin yaptığı işi tarif etmez.\n")
    print(f"  {'terim':<14}{'df':>6}{'df/N':>9}{'IDF':>9}")
    print("  " + "-" * 38)

    measured = []
    for term in args.terms:
        tokens = lexical_tokens(term)
        if not tokens:
            continue
        stem = tokens[0]
        df = sum(1 for toks in tokenised if stem in toks)
        idf = math.log(n_chunks / df) if df else float("inf")
        measured.append((term, df, idf))
        print(f"  {term:<14}{df:>6}{df / n_chunks:>9.3f}{idf:>9.3f}")

    if len(measured) >= 2:
        measured.sort(key=lambda t: t[2])
        low, high = measured[0], measured[-1]
        print(
            f"\n  En yaygın terim ({low[0]}, IDF {low[2]:.3f}) ile en nadir terim "
            f"({high[0]}, IDF {high[2]:.3f})\n  arasında {high[2] / low[2]:.1f} kat fark var. "
            f"Sorgu kalıp sözcüklerden oluştuğunda\n  BM25'in elinde ayırt edici bilgi çok azalıyor."
        )

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
