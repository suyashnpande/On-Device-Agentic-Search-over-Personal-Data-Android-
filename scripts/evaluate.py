#!/usr/bin/env python3
"""Evaluate the baseline and its comparators; write a markdown results table.

    python scripts/evaluate.py
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from pixarch.clip_model import ClipEncoder  # noqa: E402
from pixarch.eval import ocr_quality, runner  # noqa: E402
from pixarch.query.engine import SearchEngine  # noqa: E402
from pixarch.store import sqlite_store as store  # noqa: E402

# Precision leads: recall saturates at k/|relevant|, so recall@1 on a scene
# query with 10 relevant items tops out at 0.10 and misreads as a failure.
METRIC_ORDER = ["p@1", "p@5", "mrr@10", "ndcg@10", "recall@5", "recall@10"]


def fmt_table(results: dict, section: str) -> list[str]:
    lines = ["| system | " + " | ".join(METRIC_ORDER) + " | n |",
             "|---|" + "---|" * (len(METRIC_ORDER) + 1)]
    for name, sections in results.items():
        block = sections.get(section)
        if not block:
            continue
        cells = [f"{block.get(m, float('nan')):.3f}" for m in METRIC_ORDER]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {int(block.get('n_queries', 0))} |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/baseline.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    conn = store.connect(ROOT / cfg["paths"]["index_db"], create=False)
    queries = runner.load_queries(ROOT / cfg["paths"]["queries"])
    qrels = runner.load_qrels(ROOT / cfg["paths"]["qrels"])

    encoder = ClipEncoder(
        model_name=cfg["clip"]["model"], pretrained=cfg["clip"]["pretrained"],
        device=cfg["clip"]["device"], batch_size=cfg["clip"]["batch_size"],
    )
    engine = SearchEngine(conn, encoder, cfg)

    k_values = cfg["eval"]["k_values"]
    top_k = cfg["search"]["top_k"]
    results, _ = runner.run_eval(engine, conn, queries, qrels, k_values, top_k)
    acc, misroutes = runner.routing_accuracy(engine, queries)
    confusion = runner.classifier_confusion(conn)

    n_items = conn.execute("SELECT COUNT(*) c FROM items").fetchone()["c"]
    db_mb = (ROOT / cfg["paths"]["index_db"]).stat().st_size / 1e6

    out: list[str] = [
        f"# Baseline results -- {date.today().isoformat()}",
        "",
        f"Corpus: **{n_items} items** | index **{db_mb:.1f} MB** | "
        f"CLIP `{encoder.id}` on `{encoder.device}` | OCR `{cfg['ocr']['engine']}`",
        "",
        "> Indicative, not publishable: the corpus is small and auto-fetched, and",
        "> relevance is single-judge. Enough to catch regressions and prove the",
        "> plumbing; not enough to support a claim.",
        "",
        "## Retrieval -- overall",
        "",
        *fmt_table(results, "overall"),
        "",
        "**Reading the `filename` row:** it is inflated and should be treated as a",
        "loose upper bound, not the real floor. The fetcher names files after their",
        "class (`scenes/beach/scn_beach_000.jpg`), so substring matching on the path",
        "succeeds far more often than it would on a phone, where names look like",
        "`IMG_20231205_123456.jpg`. Preserving original filenames in the curated",
        "milestone will make this row meaningful.",
        "",
    ]
    for kind in ("doc_lookup", "scene_search", "ambiguous"):
        rows = fmt_table(results, kind)
        if len(rows) > 2:
            out += [f"## Retrieval -- {kind}", "", *rows, ""]

    out += [
        "## Routing", "",
        f"Router accuracy vs `gold_route`: **{acc:.1%}** ({len(misroutes)} misrouted)", "",
    ]
    if misroutes:
        out += ["| query | gold | predicted |", "|---|---|---|"]
        out += [f"| {q} | {g} | {p} |" for q, g, p in misroutes] + [""]

    out += ["## Bucket classification (gold -> predicted)", "",
            "| gold | predicted | n |", "|---|---|---|"]
    out += [f"| {g} | {p} | {c} |" for (g, p), c in sorted(confusion.items())] + [""]

    ocr_stats = ocr_quality.evaluate_ocr(conn, ROOT / "data/ocr_groundtruth.jsonl")
    if ocr_stats:
        out += [
            "## OCR accuracy (SROIE ground truth)", "",
            f"Engine `{ocr_stats['engine']}` over {int(ocr_stats['n_items'])} receipts "
            f"with upstream transcriptions.", "",
            "| metric | value |", "|---|---|",
            f"| CER | {ocr_stats['cer']:.3f} |",
            f"| WER | {ocr_stats['wer']:.3f} |", "",
        ]

    timings = conn.execute(
        "SELECT stage, SUM(n_items) n, ROUND(SUM(total_ms),1) ms FROM timings "
        "GROUP BY stage ORDER BY ms DESC"
    ).fetchall()
    if timings:
        out += ["## Indexing cost", "", "| stage | items | total ms | ms/item |", "|---|---|---|---|"]
        for r in timings:
            per = f"{r['ms']/r['n']:.1f}" if r["n"] else "-"
            out += [f"| {r['stage']} | {r['n']} | {r['ms']:.1f} | {per} |"]
        out += [""]

    text = "\n".join(out)
    print(text)

    out_path = Path(args.out) if args.out else (
        ROOT / cfg["paths"]["results_dir"] / f"baseline_{date.today().isoformat()}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"\nwritten -> {out_path}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
