#!/usr/bin/env python3
"""Expand data/queries.jsonl into data/qrels.csv against the manifest.

Queries declare relevance by *subtype* rather than by item id, so the judgement
set stays valid as the corpus grows -- adding ten more beach photos needs no
edit here. Per-item curated judgements can be layered on later by appending
rows with a `curated` source; those take precedence.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text())
    manifest_path = ROOT / cfg["paths"]["manifest"]
    queries_path = ROOT / cfg["paths"]["queries"]
    qrels_path = ROOT / cfg["paths"]["qrels"]

    with manifest_path.open(newline="", encoding="utf-8") as fh:
        items = list(csv.DictReader(fh))
    by_subtype: dict[str, list[str]] = {}
    for row in items:
        by_subtype.setdefault(row["subtype"], []).append(row["item_id"])

    rows: list[tuple[str, str, int]] = []
    unmatched: list[tuple[str, str]] = []
    queries = [json.loads(l) for l in queries_path.read_text().splitlines() if l.strip()]

    for q in queries:
        for subtype, grade in (q.get("relevant") or {}).items():
            ids = by_subtype.get(subtype, [])
            if not ids:
                unmatched.append((q["query_id"], subtype))
                continue
            rows.extend((q["query_id"], item_id, int(grade)) for item_id in ids)

    with qrels_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", "item_id", "grade"])
        w.writerows(rows)

    judged = {r[0] for r in rows}
    print(f"qrels -> {qrels_path}  ({len(rows)} judgements over {len(judged)} queries)")
    no_judgements = [q["query_id"] for q in queries if q["query_id"] not in judged]
    if no_judgements:
        print(f"  queries with no relevant items (expected for 'ambiguous'): {no_judgements}")
    if unmatched:
        print("  ! subtypes absent from the corpus (deferred to curation):")
        for qid, st in unmatched:
            print(f"      {qid}: {st}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
