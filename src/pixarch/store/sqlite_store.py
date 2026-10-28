"""SQLite-backed index: schema bootstrap, item upserts, timing records."""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

ITEM_COLUMNS = (
    "item_id", "path", "sha256", "bytes", "width", "height", "capture_date",
    "is_screenshot", "gold_bucket", "gold_subtype", "origin", "source_url",
    "license", "attribution", "notes",
)


def connect(
    db_path: str | Path, *, create: bool = True, check_same_thread: bool = True
) -> sqlite3.Connection:
    """Open the index, applying the schema when it is missing.

    `check_same_thread=False` is for read-only servers (the Gradio demo), where
    the handler runs on a worker thread while the connection was opened on the
    main thread. Callers that do this must serialise their own access -- a
    single connection shared across threads is not safe for concurrent writes.
    """
    db_path = Path(db_path)
    if not create and not db_path.exists():
        raise FileNotFoundError(f"index not found: {db_path} (run build_index.py first)")
    conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    if create:
        conn.executescript(SCHEMA_PATH.read_text())
        conn.commit()
    return conn


def upsert_items(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    """Insert or replace ingest-stage item rows. Leaves Stage-1 columns alone."""
    cols = ", ".join(ITEM_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in ITEM_COLUMNS)
    updates = ", ".join(f"{c}=excluded.{c}" for c in ITEM_COLUMNS if c != "item_id")
    sql = (
        f"INSERT INTO items ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT(item_id) DO UPDATE SET {updates}"
    )
    # is_screenshot is NOT NULL; an explicit None would bypass the column
    # default, so coerce it here rather than relying on callers.
    payload = [
        {**{c: r.get(c) for c in ITEM_COLUMNS},
         "is_screenshot": int(bool(r.get("is_screenshot")))}
        for r in rows
    ]
    with conn:
        conn.executemany(sql, payload)
    return len(payload)


def set_bucket_prediction(
    conn: sqlite3.Connection, item_id: str, bucket: str, conf: float, routed_both: bool
) -> None:
    with conn:
        conn.execute(
            "UPDATE items SET bucket_pred=?, bucket_conf=?, routed_both=? WHERE item_id=?",
            (bucket, conf, int(routed_both), item_id),
        )


def items_for_branch(conn: sqlite3.Connection, branch: str) -> list[sqlite3.Row]:
    """Items routed to `branch` -- including low-confidence items sent to both."""
    return conn.execute(
        "SELECT * FROM items WHERE bucket_pred = ? OR routed_both = 1 ORDER BY item_id",
        (branch,),
    ).fetchall()


def record_timing(
    conn: sqlite3.Connection, run_id: str, stage: str, n_items: int, total_ms: float
) -> None:
    with conn:
        conn.execute(
            "INSERT INTO timings (run_id, stage, n_items, total_ms) VALUES (?,?,?,?)",
            (run_id, stage, n_items, total_ms),
        )


@contextmanager
def timed(conn: sqlite3.Connection, run_id: str, stage: str) -> Iterator[dict[str, int]]:
    """Time a pipeline stage and persist the result.

    Yields a dict whose ``n`` key the caller sets to the item count.
    """
    counter = {"n": 0}
    start = time.perf_counter()
    try:
        yield counter
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        record_timing(conn, run_id, stage, counter["n"], elapsed_ms)
