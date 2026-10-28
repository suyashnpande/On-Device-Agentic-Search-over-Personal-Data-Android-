#!/usr/bin/env python3
"""Build the minimum baseline corpus and write data/manifest.csv.

Sources: Openverse (scenes), SROIE via HuggingFace (receipts), Wikimedia
Commons (identity-document specimens). Deliberately small and fast -- the
curated expansion appends rows to the same manifest later.

    python scripts/fetch_dataset.py --per-class 10 --receipts 20
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from pixarch import fetchers  # noqa: E402

MANIFEST_COLUMNS = [
    "item_id", "path", "bucket", "subtype", "is_screenshot",
    "source_url", "license", "attribution", "capture_date", "origin", "notes",
]


def write_manifest(records, manifest_path: Path) -> None:
    """Write the manifest, preserving any pre-existing curated rows.

    Curated items (origin != 'fetched') are hand-added and must survive a
    re-fetch -- that is what makes the later curated milestone cheap.
    """
    # Merge rather than overwrite. Rows this run did not produce are kept, so a
    # partial run (--skip, or one source failing) never destroys rows an
    # earlier run fetched, and hand-added curated rows always survive.
    preserved: list[dict] = []
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as fh:
            preserved = list(csv.DictReader(fh))

    fetched_ids = {r.item_id for r in records}
    preserved = [r for r in preserved if r["item_id"] not in fetched_ids]
    if preserved:
        n_curated = sum(1 for r in preserved if r.get("origin") != "fetched")
        print(f"  preserving {len(preserved)} existing row(s) ({n_curated} curated)")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.as_row().get(k, "") for k in MANIFEST_COLUMNS})
        for row in preserved:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_COLUMNS})
    print(f"\nmanifest -> {manifest_path}  ({len(records) + len(preserved)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/baseline.yaml"))
    ap.add_argument("--per-class", type=int, default=10, help="scene images per class")
    ap.add_argument("--receipts", type=int, default=20)
    ap.add_argument("--skip", nargs="*", default=[], choices=["scenes", "receipts", "docs"])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    manifest = ROOT / cfg["paths"]["manifest"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    records = []

    if "docs" not in args.skip:
        print("Wikimedia Commons (identity-document specimens)")
        try:
            records += fetchers.fetch_commons_docs(raw_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  ! commons failed: {e}")

    if "receipts" not in args.skip:
        print("\nSROIE receipts (HuggingFace)")
        try:
            records += fetchers.fetch_receipts_hf(
                raw_dir, n=args.receipts, gt_path=ROOT / "data/ocr_groundtruth.jsonl"
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ! SROIE unavailable ({type(e).__name__}); falling back to Openverse")
            try:
                records += fetchers.fetch_receipts_openverse(raw_dir, n=args.receipts)
            except Exception as e2:  # noqa: BLE001
                print(f"  ! receipt fallback also failed: {e2}")

    if "scenes" not in args.skip:
        print("\nScenes (SUN397 via HuggingFace datasets-server)")
        scene_records = []
        try:
            scene_records = fetchers.fetch_scenes_sun397(
                raw_dir, per_class=args.per_class,
                cache_path=ROOT / 'data/.sun397_offsets.json',
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ! SUN397 failed ({type(e).__name__}: {e})")
        if not scene_records:
            # Openverse sits behind a Cloudflare bot-challenge that 429s under
            # sustained use, so it is a fallback only -- never the primary.
            print("  falling back to Openverse")
            try:
                scene_records = fetchers.fetch_scenes(raw_dir, per_class=args.per_class)
            except Exception as e:  # noqa: BLE001
                print(f"  ! Openverse fallback also failed: {e}")
        records += scene_records

    if not records:
        print("\nNo records fetched -- nothing written.", file=sys.stderr)
        return 1

    write_manifest(records, manifest)

    by_bucket: dict[str, int] = {}
    for r in records:
        by_bucket[r.bucket] = by_bucket.get(r.bucket, 0) + 1
    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(by_bucket.items())))
    return 0


if __name__ == "__main__":
    _rc = main()
    # `datasets` streaming can abort the interpreter during finalization
    # (PyGILState_Release), which would turn a completed run into exit 134.
    # Everything is already written and flushed by here, so bypass teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(_rc)
