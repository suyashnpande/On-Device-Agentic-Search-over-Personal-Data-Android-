"""Guards that ranking depends only on image content.

Filenames, paths, item ids and provenance metadata must never influence a
result. On a device the gallery holds opaque names like IMG_20231205_123456.jpg,
so any ranking that improves because a file is called `doc_aadhaar_000.jpg` is
measuring our own naming convention rather than the system.

An earlier version indexed `filename` as an FTS column, which was worth
+0.125 p@1 on document lookup -- a leak that flattered the headline number.
"""
import inspect

import numpy as np
import pytest

from pixarch.query import bm25_search, clip_search, router
from pixarch.query.engine import SearchEngine
from pixarch.store import sqlite_store as store
from pixarch.store.vectors import SqliteVectorStore

FORBIDDEN = ("gold_bucket", "gold_subtype", "filename", "source_url", "attribution")


@pytest.mark.parametrize("module", [bm25_search, clip_search, router])
def test_query_modules_never_read_labels_or_paths(module):
    src = inspect.getsource(module)
    code = "\n".join(
        line for line in src.splitlines()
        if not line.lstrip().startswith("#")
    )
    for token in FORBIDDEN:
        assert token not in code, f"{module.__name__} references {token!r}"


def test_fts_indexes_only_content_derived_columns():
    conn = store.connect(":memory:")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(doc_fts)")]
    conn.close()
    # ocr_text comes from pixels; doc_type is derived from ocr_text.
    assert set(cols) == {"item_id", "ocr_text", "doc_type"}, cols


def _build(conn, item_id, text):
    store.upsert_items(conn, [{"item_id": item_id, "path": f"documents/{item_id}.jpg",
                               "sha256": item_id, "gold_bucket": "document",
                               "gold_subtype": "aadhaar"}])
    store.set_bucket_prediction(conn, item_id, "document", 0.5, False)
    with conn:
        conn.execute("INSERT INTO doc_fts (item_id, ocr_text, doc_type) VALUES (?,?,?)",
                     (item_id, text, "unknown"))


def test_a_term_present_only_in_the_filename_is_unfindable():
    """The decisive check: 'aadhaar' in the path but not in the OCR text."""
    conn = store.connect(":memory:")
    _build(conn, "doc_aadhaar_000", "some unrelated printed words")
    hits = bm25_search.search(conn, "aadhaar", top_k=5, weights=(1.0, 5.0))
    conn.close()
    assert hits == [], "a filename token reached the ranking"


def test_scene_ranking_is_unchanged_by_renaming():
    """Scrambling ids and paths must not alter scene scores."""
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(4, 8)).astype(np.float32)

    def run(names):
        conn = store.connect(":memory:")
        store.upsert_items(conn, [
            {"item_id": n, "path": f"{n}.jpg", "sha256": n, "gold_bucket": "scene"}
            for n in names])
        vs = SqliteVectorStore(conn)
        vs.add_many(list(zip(names, vecs)), model="t")
        out = vs.search(vecs[2], top_k=4)
        conn.close()
        return [(names.index(i), round(s, 6)) for i, s in out]

    descriptive = ["scn_beach_000", "scn_beach_001", "scn_mountain_000", "scn_food_000"]
    opaque = ["IMG_00001", "IMG_00002", "IMG_00003", "IMG_00004"]
    assert run(descriptive) == run(opaque)
