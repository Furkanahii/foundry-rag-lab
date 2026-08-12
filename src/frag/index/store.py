"""Hybrid index: dense vectors and BM25 in a single SQLite file.

The project plan suggests storing embeddings as JSON text and scanning them in
Python. That works for eight documents and stops working somewhere around a few
thousand, for a reason worth stating precisely: the scan is O(N x D) *in Python
object space*, so every comparison pays interpreter overhead rather than
hitting BLAS.

Here both retrieval arms live in the same file:

  * `chunk_vec`   - a `sqlite-vec` virtual table (dense, ANN/brute-force in C)
  * `chunks_fts`  - an FTS5 virtual table (lexical, BM25 in C)

One file, no server, no Docker. It copies to a USB stick and still works, which
is the honest version of "local-first" that the offline premise deserves.

## Why both arms

Dense retrieval matches *meaning* and misses exact tokens - course codes
("CMPE 150"), dates, proper nouns. BM25 matches *tokens* and misses paraphrase.
The failure modes are close to independent, which is exactly the condition under
which fusing two rankers beats either one. docs/03 works through the
probabilistic argument.

## Why L2 distance is safe for cosine ranking

Vectors are L2-normalised before insertion, so for unit vectors

    |a - b|^2 = 2 - 2(a . b)

which is monotone decreasing in the cosine. Ranking by ascending L2 distance
therefore produces exactly the cosine ranking - we get to use sqlite-vec's
native distance without giving up the similarity we actually reason about.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..ingest.chunkers import Chunk
from ..ingest.loaders import Document
from ..ingest.normalize import lexical_tokens

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """One retrieved chunk with the score from whichever arm produced it."""

    chunk_id: str
    rowid: int
    text: str
    source: str
    doc_id: str
    ordinal: int
    score: float           # arm-specific: cosine for dense, BM25 for lexical
    rank: int              # 1-based rank within its own arm
    arm: str               # "dense" | "lexical"
    window_text: str | None = None
    meta: dict[str, Any] | None = None

    @property
    def read_text(self) -> str:
        """What the generator should read.

        For sentence-window chunking the embedded sentence is deliberately
        narrow; the surrounding window is the readable unit.
        """
        return self.window_text or self.text

    def page(self) -> int | None:
        """Page number, recovered from the `[[page:N]]` markers the PDF loader
        embeds in the text.

        The marker travels with the text through chunking, so a chunk knows
        which page it came from without any extra bookkeeping. When a chunk
        straddles a page break it carries several markers; we report the first,
        which is where the passage begins.
        """
        match = re.search(r"\[\[page:(\d+)\]\]", self.text)
        return int(match.group(1)) if match else None

    def citation_label(self) -> str:
        """Human-readable provenance shown next to an answer.

        This is what makes an answer checkable: the reader must be able to open
        the source document and find the claim. A citation that only says
        "burs_yonergesi" sends them through nine pages; one that says
        "burs_yonergesi · s.3" does not.
        """
        parts = [self.source or self.doc_id]
        page = self.page()
        if page is not None:
            parts.append(f"s.{page}")
        return " · ".join(parts)

    def display_text(self) -> str:
        """Text with internal page markers stripped, for showing to a human."""
        return re.sub(r"\[\[page:\d+\]\]\s*", "", self.read_text).strip()


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS documents (
    doc_id    TEXT PRIMARY KEY,
    source    TEXT NOT NULL,
    path      TEXT,
    language  TEXT,
    n_chars   INTEGER,
    meta      TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    rowid       INTEGER PRIMARY KEY,
    chunk_id    TEXT UNIQUE NOT NULL,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    window_text TEXT,
    char_start  INTEGER,
    char_end    INTEGER,
    source      TEXT,
    language    TEXT,
    meta        TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# Contentless FTS5 table: we already store the text in `chunks`, so letting FTS5
# keep a second copy would double the corpus on disk for no benefit. `content=''`
# stores only the inverted index, and we join back on rowid.
FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    tokens,
    content='',
    tokenize='unicode61'
);
"""


class HybridStore:
    """SQLite-backed store holding chunks, their vectors, and a BM25 index."""

    def __init__(self, path: str | Path, embedding_dim: int | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self._conn: sqlite3.Connection | None = None

    # ---- connection --------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            # isolation_level=None disables Python's implicit transaction
            # handling. Without it, the driver silently opens a transaction
            # before our own BEGIN and the explicit one fails. We want explicit
            # control because add_chunks must be genuinely all-or-nothing across
            # three tables.
            self._conn = sqlite3.connect(str(self.path), isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            self._load_vec_extension(self._conn)
            self._conn.executescript(SCHEMA)
            self._conn.executescript(FTS_SCHEMA)
        return self._conn

    @staticmethod
    def _load_vec_extension(conn: sqlite3.Connection) -> None:
        """Load sqlite-vec into this connection.

        Extension loading is disabled again immediately afterwards. Leaving it
        on would let any later SQL string load arbitrary native code, which is a
        gratuitous risk in a program that ingests files from disk.
        """
        try:
            import sqlite_vec
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sqlite-vec is required. Run: pip install sqlite-vec"
            ) from exc

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

    def _ensure_vec_table(self, dim: int) -> None:
        """Create the vector table once the embedding dimension is known.

        The dimension is a property of the model, not of our config, so we learn
        it from the first batch of vectors rather than hardcoding 1024 and
        breaking the moment someone swaps the embedding model.
        """
        if self.embedding_dim is None:
            self.embedding_dim = dim
        elif self.embedding_dim != dim:
            raise ValueError(
                f"Index was built with dim={self.embedding_dim} but received "
                f"dim={dim}. Rebuild the index after changing embedding models."
            )
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0("
            f"  chunk_rowid INTEGER PRIMARY KEY,"
            f"  embedding float[{dim}]"
            f")"
        )
        self.set_meta("embedding_dim", str(dim))

    # ---- metadata ----------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO index_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM index_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    # ---- writing -----------------------------------------------------------

    def reset(self) -> None:
        """Drop all content. Used when chunking or the embedding model changes."""
        conn = self.conn
        for stmt in (
            "DROP TABLE IF EXISTS chunk_vec",
            "DROP TABLE IF EXISTS chunks_fts",
            "DELETE FROM chunks",
            "DELETE FROM documents",
            "DELETE FROM index_meta",
        ):
            conn.execute(stmt)
        conn.executescript(FTS_SCHEMA)
        conn.commit()
        self.embedding_dim = None

    def add_documents(self, docs: Iterable[Document]) -> None:
        rows = [
            (d.doc_id, d.source, d.path, d.language, len(d.text),
             json.dumps(d.meta, ensure_ascii=False))
            for d in docs
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO documents"
            "(doc_id, source, path, language, n_chars, meta) VALUES (?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray,
        language: str = "tr",
        stem_lexical: bool = True,
    ) -> None:
        """Insert chunks into all three tables in one transaction.

        All-or-nothing on purpose: a crash halfway through would otherwise leave
        the dense and lexical arms holding different chunk sets, and every
        subsequent fusion would silently mis-rank.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"{len(chunks)} chunks but {len(embeddings)} embeddings"
            )
        if not chunks:
            return

        self._ensure_vec_table(int(embeddings.shape[1]))
        conn = self.conn

        try:
            conn.execute("BEGIN")
            for chunk, vector in zip(chunks, embeddings):
                cursor = conn.execute(
                    "INSERT OR REPLACE INTO chunks"
                    "(chunk_id, doc_id, ordinal, text, window_text, char_start,"
                    " char_end, source, language, meta)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        chunk.chunk_id, chunk.doc_id, chunk.ordinal, chunk.text,
                        chunk.meta.get("window_text"), chunk.char_start,
                        chunk.char_end, chunk.source, language,
                        json.dumps(chunk.meta, ensure_ascii=False),
                    ),
                )
                rowid = cursor.lastrowid

                # Lexical arm: store the folded+stemmed token stream, which is
                # exactly what queries will be transformed into.
                tokens = " ".join(lexical_tokens(chunk.text, stem=stem_lexical))
                conn.execute(
                    "INSERT INTO chunks_fts(rowid, tokens) VALUES (?, ?)",
                    (rowid, tokens),
                )

                # Dense arm.
                conn.execute(
                    "INSERT OR REPLACE INTO chunk_vec(chunk_rowid, embedding) "
                    "VALUES (?, ?)",
                    (rowid, np.asarray(vector, dtype=np.float32).tobytes()),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        logger.info("Indexed %d chunks", len(chunks))

    # ---- reading -----------------------------------------------------------

    def count_chunks(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])

    def count_documents(self) -> int:
        return int(
            self.conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        )

    def search_dense(self, query_vector: np.ndarray, top_k: int = 30) -> list[SearchHit]:
        """k-nearest neighbours by L2 distance over unit vectors.

        The returned `score` is converted back to cosine similarity via
        cos = 1 - d^2/2, so both arms report numbers on interpretable scales and
        downstream fusion never has to remember which direction "better" is.
        """
        if self.embedding_dim is None:
            stored = self.get_meta("embedding_dim")
            if stored is None:
                return []
            self.embedding_dim = int(stored)

        vector = np.asarray(query_vector, dtype=np.float32).ravel()
        rows = self.conn.execute(
            """
            SELECT v.chunk_rowid AS rowid, v.distance AS distance,
                   c.chunk_id, c.text, c.window_text, c.source, c.doc_id,
                   c.ordinal, c.meta
            FROM chunk_vec v
            JOIN chunks c ON c.rowid = v.chunk_rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (vector.tobytes(), top_k),
        ).fetchall()

        hits: list[SearchHit] = []
        for rank, row in enumerate(rows, start=1):
            distance = float(row["distance"])
            cosine = 1.0 - (distance**2) / 2.0
            hits.append(
                SearchHit(
                    chunk_id=row["chunk_id"], rowid=int(row["rowid"]),
                    text=row["text"], source=row["source"], doc_id=row["doc_id"],
                    ordinal=int(row["ordinal"]), score=cosine, rank=rank,
                    arm="dense", window_text=row["window_text"],
                    meta=json.loads(row["meta"] or "{}"),
                )
            )
        return hits

    def search_lexical(
        self, query: str, top_k: int = 30, stem: bool = True
    ) -> list[SearchHit]:
        """BM25 search over the folded/stemmed token stream.

        SQLite returns bm25() as a *negative* number (more negative = better) so
        that plain ascending ORDER BY sorts best-first. We flip the sign so
        `score` means "higher is better" in both arms, matching the dense side.
        """
        tokens = lexical_tokens(query, stem=stem)
        if not tokens:
            return []

        # Quote each token: an unquoted token containing FTS5 syntax would be
        # parsed as an operator, and user queries are not FTS5 expressions.
        match_expr = " OR ".join(f'"{t}"' for t in tokens)

        try:
            rows = self.conn.execute(
                """
                SELECT f.rowid AS rowid, bm25(chunks_fts) AS score,
                       c.chunk_id, c.text, c.window_text, c.source, c.doc_id,
                       c.ordinal, c.meta
                FROM chunks_fts f
                JOIN chunks c ON c.rowid = f.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (match_expr, top_k),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("Lexical search failed for %r: %s", query, exc)
            return []

        return [
            SearchHit(
                chunk_id=row["chunk_id"], rowid=int(row["rowid"]), text=row["text"],
                source=row["source"], doc_id=row["doc_id"],
                ordinal=int(row["ordinal"]), score=-float(row["score"]), rank=rank,
                arm="lexical", window_text=row["window_text"],
                meta=json.loads(row["meta"] or "{}"),
            )
            for rank, row in enumerate(rows, start=1)
        ]

    def get_chunk(self, chunk_id: str) -> SearchHit | None:
        row = self.conn.execute(
            "SELECT rowid, chunk_id, text, window_text, source, doc_id, ordinal, meta"
            " FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return SearchHit(
            chunk_id=row["chunk_id"], rowid=int(row["rowid"]), text=row["text"],
            source=row["source"], doc_id=row["doc_id"], ordinal=int(row["ordinal"]),
            score=0.0, rank=0, arm="lookup", window_text=row["window_text"],
            meta=json.loads(row["meta"] or "{}"),
        )

    def all_chunk_ids(self) -> list[str]:
        return [
            r["chunk_id"]
            for r in self.conn.execute("SELECT chunk_id FROM chunks ORDER BY rowid")
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "n_documents": self.count_documents(),
            "n_chunks": self.count_chunks(),
            "embedding_dim": self.embedding_dim,
            "size_mb": round(self.path.stat().st_size / 1_048_576, 2)
            if self.path.exists() else 0.0,
            "index_fingerprint": self.get_meta("index_fingerprint"),
            "chunking_strategy": self.get_meta("chunking_strategy"),
            "embedding_model": self.get_meta("embedding_model"),
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> HybridStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
