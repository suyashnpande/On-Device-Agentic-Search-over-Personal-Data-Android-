"""Evaluation: the baseline system plus the comparators the proposal requires."""
from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Callable

from ..query.bm25_search import Hit
from ..query.engine import SearchEngine
from ..query.router import DOCUMENT, SCENE
from .metrics import evaluate_run


def load_queries(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            qrels[row["query_id"]][row["item_id"]] = int(row["grade"])
    return dict(qrels)


def filename_search(conn: sqlite3.Connection, query: str, top_k: int) -> list[Hit]:
    """The trivial floor: substring match on the file path only.

    This is what a phone's file manager already gives you, and is the bar any
    real system has to clear.

    It must NOT touch `gold_subtype` -- that is the ground-truth label, and
    matching against it turns the floor into an oracle. An earlier version did,
    which inflated this row and made routing look less valuable than it is.
    """
    terms = [t for t in query.lower().split() if len(t) > 2]
    if not terms:
        return []
    clause = " OR ".join(["LOWER(path) LIKE ?"] * len(terms))
    params: list[str] = [f"%{t}%" for t in terms]
    rows = conn.execute(
        f"SELECT item_id FROM items WHERE {clause} LIMIT ?", (*params, top_k)
    ).fetchall()
    return [Hit(item_id=r["item_id"], score=1.0, branch="filename") for r in rows]


def build_systems(engine: SearchEngine, conn: sqlite3.Connection, top_k: int) -> dict[str, Callable]:
    """The baseline plus its three required comparators."""
    return {
        # The baseline: router picks exactly one branch.
        "routed": lambda q: engine.search(q, top_k).ranked_ids,
        # Single-model comparators -- do these make routing look worthwhile?
        "clip_only": lambda q: engine.search(q, top_k, force_branch=SCENE).ranked_ids,
        "bm25_only": lambda q: engine.search(q, top_k, force_branch=DOCUMENT).ranked_ids,
        # The trivial floor.
        "filename": lambda q: [h.item_id for h in filename_search(conn, q, top_k)],
    }


def run_eval(engine: SearchEngine, conn: sqlite3.Connection, queries, qrels, k_values, top_k):
    """Return per-system metrics overall and split by query kind."""
    systems = build_systems(engine, conn, top_k)
    results: dict[str, dict] = {}
    runs: dict[str, dict[str, list[str]]] = {}

    for name, fn in systems.items():
        run = {q["query_id"]: fn(q["text"]) for q in queries}
        runs[name] = run
        results[name] = {"overall": evaluate_run(run, qrels, k_values)}

        by_kind: dict[str, list[str]] = defaultdict(list)
        for q in queries:
            by_kind[q["kind"]].append(q["query_id"])
        for kind, qids in by_kind.items():
            subset = {qid: run[qid] for qid in qids}
            sub_qrels = {qid: qrels[qid] for qid in qids if qid in qrels}
            if sub_qrels:
                results[name][kind] = evaluate_run(subset, sub_qrels, k_values)

    return results, runs


def routing_accuracy(engine: SearchEngine, queries) -> tuple[float, list[tuple[str, str, str]]]:
    """Router accuracy against each query's gold_route, with the misroutes."""
    wrong: list[tuple[str, str, str]] = []
    correct = 0
    for q in queries:
        got = engine.router.route(q["text"]).branch
        if got == q["gold_route"]:
            correct += 1
        else:
            wrong.append((q["query_id"], q["gold_route"], got))
    return (correct / len(queries) if queries else 0.0), wrong


def classifier_confusion(conn: sqlite3.Connection) -> dict[tuple[str, str], int]:
    rows = conn.execute(
        "SELECT gold_bucket, bucket_pred, COUNT(*) c FROM items "
        "WHERE bucket_pred IS NOT NULL GROUP BY 1,2"
    ).fetchall()
    return {(r["gold_bucket"], r["bucket_pred"]): r["c"] for r in rows}
