"""Document-branch search: FTS5 MATCH + weighted BM25."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

# FTS5 treats these as query syntax; a user's plain-language question must not
# be parsed as operators.
_FTS_SPECIAL = re.compile(r'[":^*()\-]')


@dataclass
class Hit:
    item_id: str
    score: float
    branch: str = "document"
    snippet: str = ""


# Function words and query-framing verbs carry no content but, under OR
# semantics, let an unrelated short document match on one of them. BM25's
# length normalisation then ranks that short document first.
STOPWORDS = frozenset("""
a an the of for to in on at by with from and or is are was were be been
my me mine i you your our we us this that these those it its
give show find get me please need want looking look search
""".split())


def to_match_query(text: str) -> str:
    """Turn a natural-language question into a safe FTS5 OR-query.

    Each term is quoted so punctuation cannot leak into FTS5's query grammar,
    and terms are OR-ed: an AND over a whole question almost always returns
    nothing once OCR has dropped a word.

    Stopwords are dropped. If every term is a stopword the original terms are
    kept, so a query never silently becomes empty.
    """
    cleaned = _FTS_SPECIAL.sub(" ", text)
    terms = [t for t in cleaned.split() if t.strip()]
    if not terms:
        return ""
    content = [t for t in terms if t.lower() not in STOPWORDS]
    return " OR ".join(f'"{t}"' for t in (content or terms))


def search(
    conn: sqlite3.Connection,
    query_text: str,
    top_k: int = 10,
    weights: tuple[float, float] = (1.0, 5.0),
) -> list[Hit]:
    """Rank documents by BM25.

    SQLite's bm25() returns *negative* scores, most-negative being the best
    match, so results are ordered ascending and the sign flipped for display.
    The leading 0.0 weight is for the UNINDEXED item_id column: bm25() needs
    one weight per declared column.
    """
    match = to_match_query(query_text)
    if not match:
        return []
    w_text, w_type = weights
    sql = (
        "SELECT item_id, bm25(doc_fts, 0.0, ?, ?) AS score, "
        "       snippet(doc_fts, 1, '[', ']', ' ... ', 12) AS snip "
        "FROM doc_fts WHERE doc_fts MATCH ? ORDER BY score LIMIT ?"
    )
    try:
        rows = conn.execute(sql, (w_text, w_type, match, top_k)).fetchall()
    except sqlite3.OperationalError as e:
        # A malformed MATCH expression should yield no hits, not a crash.
        print(f"  ! FTS query failed ({e}) for {match!r}")
        return []
    return [Hit(item_id=r["item_id"], score=-float(r["score"]), snippet=r["snip"] or "")
            for r in rows]
