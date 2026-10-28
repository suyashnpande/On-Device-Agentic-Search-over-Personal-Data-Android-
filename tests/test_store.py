"""SQLite store and vector round-trips."""
import numpy as np
import pytest

from pixarch.store import sqlite_store as store
from pixarch.store.vectors import SqliteVectorStore, from_blob, to_blob


@pytest.fixture()
def conn():
    c = store.connect(":memory:")
    yield c
    c.close()


def test_blob_round_trip_normalises():
    vec = np.array([3.0, 4.0], dtype=np.float32)       # norm 5
    got = from_blob(to_blob(vec))
    assert np.allclose(got, [0.6, 0.8])
    assert np.isclose(np.linalg.norm(got), 1.0)


def test_zero_vector_does_not_divide_by_zero():
    got = from_blob(to_blob(np.zeros(4, dtype=np.float32)))
    assert np.allclose(got, 0.0)


def test_upsert_is_idempotent(conn):
    row = {"item_id": "a", "path": "p.jpg", "sha256": "x", "gold_bucket": "scene"}
    store.upsert_items(conn, [row])
    store.upsert_items(conn, [{**row, "gold_subtype": "beach"}])
    got = conn.execute("SELECT COUNT(*) c FROM items").fetchone()["c"]
    assert got == 1
    assert conn.execute("SELECT gold_subtype FROM items").fetchone()[0] == "beach"


def test_items_for_branch_includes_dual_routed(conn):
    store.upsert_items(conn, [
        {"item_id": "doc", "path": "d", "sha256": "1"},
        {"item_id": "scn", "path": "s", "sha256": "2"},
        {"item_id": "both", "path": "b", "sha256": "3"},
    ])
    store.set_bucket_prediction(conn, "doc", "document", 0.5, False)
    store.set_bucket_prediction(conn, "scn", "scene", 0.5, False)
    store.set_bucket_prediction(conn, "both", "scene", 0.01, True)
    ids = {r["item_id"] for r in store.items_for_branch(conn, "document")}
    # A low-confidence item must appear in BOTH branches, not just its argmax.
    assert ids == {"doc", "both"}


def test_vector_search_ranks_exact_match_first(conn):
    store.upsert_items(conn, [{"item_id": f"i{n}", "path": "p", "sha256": str(n)} for n in range(4)])
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(4, 16))
    vs = SqliteVectorStore(conn)
    vs.add_many([(f"i{n}", vecs[n]) for n in range(4)], model="t")
    top = vs.search(vecs[2], top_k=2)
    assert top[0][0] == "i2"
    assert top[0][1] == pytest.approx(1.0, abs=1e-5)


def test_vector_search_respects_restriction(conn):
    store.upsert_items(conn, [{"item_id": f"i{n}", "path": "p", "sha256": str(n)} for n in range(3)])
    vecs = np.eye(3, dtype=np.float32)
    SqliteVectorStore(conn).add_many([(f"i{n}", vecs[n]) for n in range(3)], model="t")
    restricted = SqliteVectorStore(conn, restrict_to=["i0", "i1"])
    assert {i for i, _ in restricted.search(vecs[2], top_k=3)} == {"i0", "i1"}


def test_empty_store_returns_no_hits(conn):
    assert SqliteVectorStore(conn).search(np.ones(4), top_k=5) == []


def test_reindexing_an_item_leaves_no_stale_fts_rows(conn):
    """Re-running build_index must replace FTS content, not accumulate it.

    doc_fts carries item_id as UNINDEXED, so the delete is a scan rather than
    an indexed lookup. If it silently failed, every re-index would duplicate
    rows and corrupt BM25's term statistics.
    """
    for text in ("first version aadhaar", "second version passport"):
        with conn:
            conn.execute("DELETE FROM doc_fts WHERE item_id=?", ("x",))
            conn.execute(
                "INSERT INTO doc_fts (item_id, ocr_text, doc_type) VALUES (?,?,?)",
                ("x", text, "t"))

    assert conn.execute("SELECT COUNT(*) FROM doc_fts").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM doc_fts WHERE doc_fts MATCH 'aadhaar'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM doc_fts WHERE doc_fts MATCH 'passport'").fetchone()[0] == 1


def test_fts_index_has_no_filename_column(conn):
    """Filename must not be a ranking signal.

    A device gallery holds opaque names (IMG_20231205_123456.jpg), so matching
    a filename is a signal the deployed system would never have. Indexing our
    generated names leaked the class label into the ranking.
    """
    cols = [r[1] for r in conn.execute("PRAGMA table_info(doc_fts)")]
    assert "filename" not in cols, f"filename is indexed again: {cols}"
    assert "ocr_text" in cols and "doc_type" in cols


def test_fts_stems_english_plurals(conn):
    """A plural query must reach a singular document.

    Without the porter tokenizer, "receipts" matched zero of 18 receipts and
    the query returned nothing -- a total miss from one character.
    """
    with conn:
        conn.execute("INSERT INTO doc_fts (item_id, ocr_text, doc_type) VALUES (?,?,?)",
                     ("r1", "official receipt total 450", "receipt"))
    for term in ("receipt", "receipts"):
        n = conn.execute("SELECT COUNT(*) FROM doc_fts WHERE doc_fts MATCH ?", (f'"{term}"',)).fetchone()[0]
        assert n == 1, f"query {term!r} matched {n} docs"
