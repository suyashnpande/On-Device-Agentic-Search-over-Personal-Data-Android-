"""Stage 0 -- ingest: hash, dimensions, EXIF, screenshot heuristic.

Reads data/manifest.csv (the gold labels + provenance) and writes one `items`
row per file. Nothing here looks at image content beyond metadata.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Iterator

from PIL import Image, ExifTags

# EXIF tag ids, resolved once.
_TAG = {v: k for k, v in ExifTags.TAGS.items()}
_DATETIME_ORIGINAL = _TAG.get("DateTimeOriginal")
_MAKE = _TAG.get("Make")
_MODEL = _TAG.get("Model")
_FOCAL = _TAG.get("FocalLength")
_GPS = _TAG.get("GPSInfo")

# Common phone/desktop screen resolutions, both orientations. A screenshot is
# pixel-exact to the screen it came from; a photo almost never is.
_SCREEN_SIZES: set[tuple[int, int]] = set()
for _w, _h in [
    (1080, 1920), (1080, 2340), (1080, 2400), (1080, 2160), (1440, 3200),
    (1440, 2560), (1170, 2532), (1284, 2778), (828, 1792), (750, 1334),
    (720, 1280), (1125, 2436), (1242, 2688), (1179, 2556), (1290, 2796),
    (1920, 1080), (2560, 1440), (3840, 2160), (1366, 768), (1536, 864),
]:
    _SCREEN_SIZES.add((_w, _h))
    _SCREEN_SIZES.add((_h, _w))


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_exif(img: Image.Image) -> dict[str, Any]:
    """Pull the few EXIF fields the pipeline uses. Absent EXIF is normal."""
    out: dict[str, Any] = {"capture_date": None, "has_camera": False, "has_gps": False}
    try:
        exif = img.getexif()
    except Exception:  # noqa: BLE001 - malformed EXIF is common in web images
        return out
    if not exif:
        return out

    raw_dt = exif.get(_DATETIME_ORIGINAL) or exif.get(_TAG.get("DateTime"))
    if isinstance(raw_dt, str) and len(raw_dt) >= 10:
        # EXIF format is "YYYY:MM:DD HH:MM:SS"
        date_part = raw_dt[:10].replace(":", "-")
        out["capture_date"] = f"{date_part}{raw_dt[10:]}".strip()

    out["has_camera"] = bool(exif.get(_MAKE) or exif.get(_MODEL) or exif.get(_FOCAL))
    out["has_gps"] = bool(exif.get(_GPS))
    return out


def looks_like_screenshot(width: int, height: int, exif: dict[str, Any]) -> bool:
    """Heuristic, recorded as a feature -- never used to route in the baseline.

    A screenshot has no camera EXIF and is pixel-exact to a known screen size.
    Requiring both keeps false positives low on cropped photos.
    """
    if exif["has_camera"]:
        return False
    return (width, height) in _SCREEN_SIZES


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ingest(manifest_path: Path, raw_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield an `items` row per manifest entry whose file is present."""
    seen_hashes: dict[str, str] = {}
    for row in read_manifest(manifest_path):
        path = raw_dir / row["path"]
        if not path.exists():
            print(f"  ! missing file, skipping: {row['path']}")
            continue
        try:
            with Image.open(path) as img:
                width, height = img.size
                exif = read_exif(img)
        except Exception as e:  # noqa: BLE001 - a corrupt download must not halt ingest
            print(f"  ! unreadable image, skipping: {row['path']} ({type(e).__name__})")
            continue

        digest = sha256_of(path)
        if digest in seen_hashes:
            print(f"  ! duplicate of {seen_hashes[digest]}, skipping: {row['item_id']}")
            continue
        seen_hashes[digest] = row["item_id"]

        # A manifest row may assert is_screenshot; trust it over the heuristic.
        declared = str(row.get("is_screenshot", "")).strip().lower() in {"1", "true", "yes"}

        yield {
            "item_id": row["item_id"],
            "path": row["path"],
            "sha256": digest,
            "bytes": path.stat().st_size,
            "width": width,
            "height": height,
            "capture_date": row.get("capture_date") or exif["capture_date"],
            "is_screenshot": declared or looks_like_screenshot(width, height, exif),
            "gold_bucket": row.get("bucket"),
            "gold_subtype": row.get("subtype"),
            "origin": row.get("origin") or "fetched",
            "source_url": row.get("source_url"),
            "license": row.get("license"),
            "attribution": row.get("attribution"),
            "notes": row.get("notes"),
        }
