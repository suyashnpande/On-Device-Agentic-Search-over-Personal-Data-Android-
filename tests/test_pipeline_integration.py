"""End-to-end wiring: index -> route -> search -> evaluate.

Uses a stub encoder so the whole path is exercised without CLIP weights, a
GPU or the network. This is what catches wiring regressions (a wrong bm25()
weight count, a branch that silently returns nothing) that unit tests on the
individual pieces cannot see.
"""
import numpy as np
import pytest

from pixarch.doctype import classify_text
from pixarch.eval.metrics import evaluate_run
from pixarch.eval.runner import filename_search
from pixarch.query import bm25_search
from pixarch.query.engine import SearchEngine
from pixarch.store import sqlite_store as store
from pixarch.store.vectors import SqliteVectorStore

CFG = {
    "classify": {"prompts": {"document": ["a document"], "scene": ["a photo"]}},
    "search": {"top_k": 5, "bm25_weights": [1.0, 5.0]},
}

DOCS = {
    "doc_aadhaar_000": "government of india unique identification authority 1234 5678 9012",
    "doc_pan_000": "income tax department permanent account number abcde1234f",
    "doc_receipt_000": "tax invoice gst no 29abc total 450 00 cash",
}
SCENES = {"scn_beach_000": 0, "scn_mountain_000": 1}


class StubEncoder:
    """Deterministic one-hot embeddings; no model, no network."""

    id = "stub/one-hot"
    device = "cpu"
    _VOCAB = {"beach": 0, "mountain": 1}

    def encode_texts(self, texts):
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            for word, dim in self._VOCAB.items():
                if word in t.lower():
                    out[i, dim] = 1.0
            if not out[i].any():
                out[i, 3] = 1.0
        return out


@pytest.fixture()
def indexed():
    conn = store.connect(":memory:")
    rows = [{"item_id": i, "path": f"documents/{i}.jpg", "sha256": i,
             "gold_bucket": "document", "gold_subtype": i.split("_")[1]} for i in DOCS]
    rows += [{"item_id": i, "path": f"scenes/{i}.jpg", "sha256": i,
              "gold_bucket": "scene", "gold_subtype": i.split("_")[1]} for i in SCENES]
    store.upsert_items(conn, rows)

    for item_id, text in DOCS.items():
        store.set_bucket_prediction(conn, item_id, "document", 0.5, False)
        doc_type, rule, _ = classify_text(text)
        with conn:
            conn.execute(
                "INSERT INTO doc_fields (item_id, full_text, doc_type, doc_type_rule, engine) "
                "VALUES (?,?,?,?,?)", (item_id, text, doc_type, rule, "stub"))
            conn.execute(
                "INSERT INTO doc_fts (item_id, ocr_text, doc_type) VALUES (?,?,?)",
                (item_id, text, doc_type))

    vectors = SqliteVectorStore(conn)
    for item_id, dim in SCENES.items():
        store.set_bucket_prediction(conn, item_id, "scene", 0.5, False)
        vec = np.zeros(4, dtype=np.float32)
        vec[dim] = 1.0
        vectors.add(item_id, vec, "stub")

    yield conn
    conn.close()


@pytest.fixture()
def engine(indexed):
    return SearchEngine(indexed, StubEncoder(), CFG)


def test_document_query_routes_to_document_and_ranks_correctly(engine):
    result = engine.search("give me my aadhaar card")
    assert result.route.branch == "document"
    assert result.ranked_ids[0] == "doc_aadhaar_000"


def test_scene_query_routes_to_scene_and_ranks_correctly(engine):
    result = engine.search("find a photo of a beach")
    assert result.route.branch == "scene"
    assert result.ranked_ids[0] == "scn_beach_000"


def test_doc_type_field_boost_beats_body_text(indexed):
    """'pan' appears in the pan doc's doc_type field, weighted 5x over body."""
    hits = bm25_search.search(indexed, "pan", top_k=5, weights=(1.0, 5.0))
    assert hits and hits[0].item_id == "doc_pan_000"


def test_only_one_branch_fires_per_query(engine):
    """The baseline must not fuse: scene hits never appear in a document run."""
    doc_ids = set(engine.search("give me my aadhaar card").ranked_ids)
    assert not (doc_ids & set(SCENES))


def test_forced_branch_bypasses_the_router(engine):
    # The comparator systems depend on this: bm25_only must never route away.
    result = engine.search("find a photo of a beach", force_branch="document")
    assert result.route.branch == "document"
    assert result.route.reason == "forced"


def test_empty_query_returns_no_hits_without_raising(engine):
    assert engine.search("   ").hits == []


def test_filename_comparator_finds_by_subtype(indexed):
    hits = filename_search(indexed, "aadhaar", top_k=5)
    assert [h.item_id for h in hits] == ["doc_aadhaar_000"]


def test_end_to_end_metrics_are_computed(engine):
    queries = {"q1": "give me my aadhaar card", "q2": "find a photo of a beach"}
    qrels = {"q1": {"doc_aadhaar_000": 2}, "q2": {"scn_beach_000": 2}}
    run = {qid: engine.search(text).ranked_ids for qid, text in queries.items()}
    got = evaluate_run(run, qrels, k_values=(1, 5))
    assert got["recall@1"] == 1.0
    assert got["ndcg@5"] == pytest.approx(1.0)


def test_filename_comparator_does_not_read_gold_labels(indexed):
    """The trivial floor must not touch gold_subtype.

    Matching the ground-truth label column turns the floor into an oracle and
    understates how much routing is worth.
    """
    import inspect

    from pixarch.eval import runner
    source = inspect.getsource(runner.filename_search)
    assert "gold_subtype" not in source.split('"""')[2], "filename floor reads the gold label"


def test_filename_comparator_still_matches_the_path(indexed):
    assert [h.item_id for h in filename_search(indexed, "aadhaar", top_k=5)] == ["doc_aadhaar_000"]
