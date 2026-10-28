#!/usr/bin/env python3
"""Build index.db: ingest -> classify -> document branch + scene branch.

    python scripts/build_index.py --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from pixarch import ingest as ingest_mod  # noqa: E402
from pixarch.classify import BucketClassifier  # noqa: E402
from pixarch.clip_model import ClipEncoder  # noqa: E402
from pixarch.doctype import classify_text  # noqa: E402
from pixarch.ocr import build_engine, run_ocr  # noqa: E402
from pixarch.store import sqlite_store as store  # noqa: E402
from pixarch.store.vectors import SqliteVectorStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/baseline.yaml"))
    ap.add_argument("--skip-ocr", action="store_true", help="scene branch only")
    ap.add_argument("--limit", type=int, default=0, help="cap items, for smoke tests")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    db_path = ROOT / cfg["paths"]["index_db"]
    run_id = uuid.uuid4().hex[:8]

    conn = store.connect(db_path)
    print(f"index: {db_path}  (run {run_id})")

    # -- Stage 0: ingest ----------------------------------------------------
    print("\n[0] ingest")
    with store.timed(conn, run_id, "ingest") as t:
        rows = list(ingest_mod.ingest(ROOT / cfg["paths"]["manifest"], raw_dir))
        if args.limit:
            rows = rows[: args.limit]
        t["n"] = store.upsert_items(conn, rows)
    print(f"    {t['n']} items")
    if not rows:
        print("no items ingested -- run scripts/fetch_dataset.py first", file=sys.stderr)
        return 1

    # -- Stage 1: CLIP encode once, then classify ---------------------------
    # The same embeddings serve classification and the scene index; encoding
    # twice would double the most expensive part of indexing.
    print("\n[1] CLIP encode + classify")
    encoder = ClipEncoder(
        model_name=cfg["clip"]["model"], pretrained=cfg["clip"]["pretrained"],
        device=cfg["clip"]["device"], batch_size=cfg["clip"]["batch_size"],
    )
    print(f"    model {encoder.id} on {encoder.device}")

    paths = [raw_dir / r["path"] for r in rows]
    id_by_path = {raw_dir / r["path"]: r["item_id"] for r in rows}

    with store.timed(conn, run_id, "clip_encode") as t:
        kept_paths, embeddings = encoder.encode_images(paths)
        t["n"] = len(kept_paths)
    print(f"    encoded {len(kept_paths)}/{len(paths)}")

    classifier = BucketClassifier(
        encoder, cfg["classify"]["prompts"], cfg["classify"]["confidence_threshold"]
    )
    with store.timed(conn, run_id, "classify") as t:
        verdicts = classifier.classify(embeddings)
        for path, (bucket, margin, both) in zip(kept_paths, verdicts):
            store.set_bucket_prediction(conn, id_by_path[path], bucket, margin, both)
        t["n"] = len(verdicts)

    counts: dict[str, int] = {}
    for bucket, _, both in verdicts:
        counts[bucket] = counts.get(bucket, 0) + 1
    n_both = sum(1 for _, _, b in verdicts if b)
    print(f"    {counts} | low-confidence routed to both: {n_both}")

    # -- Stage 2b: scene branch --------------------------------------------
    print("\n[2b] scene branch (embeddings)")
    vectors = SqliteVectorStore(conn)
    with store.timed(conn, run_id, "embed_store") as t:
        t["n"] = vectors.add_many(
            [(id_by_path[p], embeddings[i]) for i, p in enumerate(kept_paths)],
            model=encoder.id,
        )
    print(f"    stored {t['n']} embeddings")

    # -- Stage 2a: document branch -----------------------------------------
    if args.skip_ocr:
        print("\n[2a] document branch SKIPPED (--skip-ocr)")
        conn.close()
        return 0

    doc_items = store.items_for_branch(conn, "document")
    print(f"\n[2a] document branch: OCR over {len(doc_items)} items")
    engine = build_engine(
        cfg["ocr"]["engine"], cfg["ocr"]["languages"], gpu=(encoder.device == "cuda")
    )
    print(f"    engine {engine.name}")

    min_conf = cfg["ocr"]["min_confidence"]
    with store.timed(conn, run_id, "ocr") as t:
        for n, item in enumerate(doc_items, start=1):
            path = raw_dir / item["path"]
            try:
                blocks, raw_text, norm_text, ms = run_ocr(engine, path, min_conf)
            except Exception as e:  # noqa: BLE001 - one bad scan must not halt the run
                print(f"    ! OCR failed on {item['item_id']}: {type(e).__name__}")
                continue
            doc_type, rule, _ = classify_text(norm_text)
            with conn:
                conn.execute("DELETE FROM ocr_blocks WHERE item_id=?", (item["item_id"],))
                conn.executemany(
                    "INSERT INTO ocr_blocks (item_id, block_idx, text, conf, x0,y0,x1,y1) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    [(item["item_id"], i, b.text, b.conf, *b.box) for i, b in enumerate(blocks)],
                )
                conn.execute(
                    "INSERT INTO doc_fields (item_id, full_text, raw_text, doc_type, "
                    "doc_type_rule, engine, ocr_ms) VALUES (?,?,?,?,?,?,?) "
                    "ON CONFLICT(item_id) DO UPDATE SET full_text=excluded.full_text, "
                    "raw_text=excluded.raw_text, doc_type=excluded.doc_type, "
                    "doc_type_rule=excluded.doc_type_rule, engine=excluded.engine, "
                    "ocr_ms=excluded.ocr_ms",
                    (item["item_id"], norm_text, raw_text, doc_type, rule, engine.name, ms),
                )
                conn.execute("DELETE FROM doc_fts WHERE item_id=?", (item["item_id"],))
                conn.execute(
                    "INSERT INTO doc_fts (item_id, ocr_text, doc_type) VALUES (?,?,?)",
                    (item["item_id"], norm_text, doc_type),
                )
            if n % 10 == 0 or n == len(doc_items):
                print(f"    {n}/{len(doc_items)}")
            t["n"] = n

    types = conn.execute(
        "SELECT doc_type, COUNT(*) c FROM doc_fields GROUP BY 1 ORDER BY c DESC"
    ).fetchall()
    print("    doc_type:", {r["doc_type"]: r["c"] for r in types})

    print("\ntimings (ms):")
    for r in conn.execute(
        "SELECT stage, n_items, ROUND(total_ms,1) ms FROM timings WHERE run_id=?", (run_id,)
    ):
        per = f"{r['ms']/r['n_items']:.1f}/item" if r["n_items"] else "-"
        print(f"    {r['stage']:14} {r['n_items']:4}  {r['ms']:10.1f}  {per}")

    conn.close()
    print(f"\nindex size: {db_path.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
