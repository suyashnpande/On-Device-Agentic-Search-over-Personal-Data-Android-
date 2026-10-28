"""Single entry point for a baseline query: route, then search one branch."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field

from ..clip_model import ClipEncoder
from ..store.vectors import SqliteVectorStore
from . import bm25_search, clip_search
from .bm25_search import Hit
from .router import DOCUMENT, SCENE, QueryRouter, Route


@dataclass
class QueryResult:
    query: str
    route: Route
    hits: list[Hit] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def ranked_ids(self) -> list[str]:
        return [h.item_id for h in self.hits]


class SearchEngine:
    """Routes to exactly one branch. No fusion, no retry -- that is the point.

    Fusing the two signals is the proposal's own cross-cutting contribution and
    must not be pre-empted by the baseline it will be measured against.
    """

    def __init__(self, conn: sqlite3.Connection, encoder: ClipEncoder, cfg: dict):
        self.conn = conn
        self.encoder = encoder
        self.cfg = cfg
        self.router = QueryRouter(encoder, cfg["classify"]["prompts"])
        # The scene index covers only scene-routed items (including anything
        # sent to both branches by the low-confidence rule).
        scene_ids = [
            r["item_id"]
            for r in conn.execute(
                "SELECT item_id FROM items WHERE bucket_pred=? OR routed_both=1", (SCENE,)
            )
        ]
        self.vectors = SqliteVectorStore(conn, restrict_to=scene_ids)

    def search(self, query_text: str, top_k: int | None = None, force_branch: str | None = None):
        top_k = top_k or self.cfg["search"]["top_k"]
        start = time.perf_counter()
        route = (
            Route(force_branch, 1.0, "forced") if force_branch else self.router.route(query_text)
        )
        if route.branch == DOCUMENT:
            hits = bm25_search.search(
                self.conn, query_text, top_k, tuple(self.cfg["search"]["bm25_weights"])
            )
        else:
            hits = clip_search.search(self.encoder, self.vectors, query_text, top_k)
        return QueryResult(
            query=query_text, route=route, hits=hits,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )
