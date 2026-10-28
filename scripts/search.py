#!/usr/bin/env python3
"""Query the baseline index from the command line.

    python scripts/search.py "give me my aadhaar card"
    python scripts/search.py "a beach" --branch scene --top-k 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from pixarch.clip_model import ClipEncoder  # noqa: E402
from pixarch.query.engine import SearchEngine  # noqa: E402
from pixarch.store import sqlite_store as store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", nargs="+")
    ap.add_argument("--config", default=str(ROOT / "configs/baseline.yaml"))
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--branch", choices=["document", "scene"], default=None,
                    help="bypass the router (for the single-model comparators)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    query = " ".join(args.query)

    conn = store.connect(ROOT / cfg["paths"]["index_db"], create=False)
    encoder = ClipEncoder(
        model_name=cfg["clip"]["model"], pretrained=cfg["clip"]["pretrained"],
        device=cfg["clip"]["device"], batch_size=cfg["clip"]["batch_size"],
    )
    engine = SearchEngine(conn, encoder, cfg)
    result = engine.search(query, top_k=args.top_k, force_branch=args.branch)

    print(f'\nquery: "{result.query}"')
    print(f"route: {result.route.branch}  (conf {result.route.confidence:.3f}, "
          f"{result.route.reason})   {result.elapsed_ms:.1f} ms\n")
    if not result.hits:
        print("  no hits")
        return 0

    for rank, hit in enumerate(result.hits, start=1):
        row = conn.execute(
            "SELECT path, gold_bucket, gold_subtype FROM items WHERE item_id=?", (hit.item_id,)
        ).fetchone()
        gold = f"{row['gold_bucket']}/{row['gold_subtype']}" if row else "?"
        print(f"  {rank:2}. {hit.score:7.3f}  {hit.item_id:22} [{gold}]")
        if hit.snippet:
            print(f"       {hit.snippet[:100]}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
