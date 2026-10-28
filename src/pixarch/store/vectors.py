"""Vector storage and exact cosine search.

At baseline corpus size a brute-force matmul is exact and sub-millisecond, so
there is no approximate index here. The `VectorStore` protocol is the seam that
lets FAISS / usearch / ObjectBox drop in later without touching call sites.
"""
from __future__ import annotations

import sqlite3
from typing import Protocol, Sequence

import numpy as np

DTYPE = np.float32


def to_blob(vec: np.ndarray) -> bytes:
    """L2-normalise and serialise a single embedding."""
    v = np.asarray(vec, dtype=DTYPE).ravel()
    norm = float(np.linalg.norm(v))
    if norm > 0:
        v = v / norm
    return v.astype(DTYPE).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=DTYPE)


class VectorStore(Protocol):
    def add(self, item_id: str, vec: np.ndarray, model: str) -> None: ...
    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]: ...


class SqliteVectorStore:
    """Embeddings live in the `embeddings` table; search is an exact matmul.

    The matrix is loaded once and cached. Call `invalidate()` after writes if a
    search follows in the same process.
    """

    def __init__(self, conn: sqlite3.Connection, *, restrict_to: Sequence[str] | None = None):
        self.conn = conn
        self._restrict_to = set(restrict_to) if restrict_to is not None else None
        self._ids: list[str] | None = None
        self._matrix: np.ndarray | None = None

    def add(self, item_id: str, vec: np.ndarray, model: str) -> None:
        blob = to_blob(vec)
        with self.conn:
            self.conn.execute(
                "INSERT INTO embeddings (item_id, model, dim, vec) VALUES (?,?,?,?) "
                "ON CONFLICT(item_id) DO UPDATE SET model=excluded.model, "
                "dim=excluded.dim, vec=excluded.vec",
                (item_id, model, len(blob) // 4, blob),
            )
        self.invalidate()

    def add_many(self, rows: Sequence[tuple[str, np.ndarray]], model: str) -> int:
        payload = [(i, model, len(to_blob(v)) // 4, to_blob(v)) for i, v in rows]
        with self.conn:
            self.conn.executemany(
                "INSERT INTO embeddings (item_id, model, dim, vec) VALUES (?,?,?,?) "
                "ON CONFLICT(item_id) DO UPDATE SET model=excluded.model, "
                "dim=excluded.dim, vec=excluded.vec",
                payload,
            )
        self.invalidate()
        return len(payload)

    def invalidate(self) -> None:
        self._ids = None
        self._matrix = None

    def _load(self) -> tuple[list[str], np.ndarray]:
        if self._ids is None or self._matrix is None:
            rows = self.conn.execute(
                "SELECT item_id, vec FROM embeddings ORDER BY item_id"
            ).fetchall()
            if self._restrict_to is not None:
                rows = [r for r in rows if r["item_id"] in self._restrict_to]
            self._ids = [r["item_id"] for r in rows]
            self._matrix = (
                np.vstack([from_blob(r["vec"]) for r in rows])
                if rows
                else np.zeros((0, 0), dtype=DTYPE)
            )
        return self._ids, self._matrix

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        ids, matrix = self._load()
        if not ids:
            return []
        q = np.asarray(query, dtype=DTYPE).ravel()
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm
        # Both sides are L2-normalised, so the dot product is cosine similarity.
        scores = matrix @ q
        k = min(top_k, len(ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(ids[i], float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self._load()[0])
