"""Corpus fetchers for the minimum baseline dataset.

Each fetcher downloads images into ``data/raw/<bucket>/<subtype>/`` and returns
manifest records. Every record carries source_url/license/attribution so the
corpus is reproducible and redistributable-by-reference rather than by copying
images into the repo.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

UA = "pixarch-research/0.1 (academic on-device retrieval research)"

SCENE_CLASSES = [
    "beach", "mountain", "trees", "vehicle", "party", "wedding",
    "college", "food", "travel", "fitness", "mall", "selfie",
]

# Openverse query strings tuned per class -- the bare class name is a poor
# query for several of these ("vehicle" returns diagrams, "travel" returns maps).
SCENE_QUERIES = {
    "beach": "beach sea sand",
    "mountain": "mountain landscape",
    "trees": "forest trees",
    "vehicle": "car parked street",
    "party": "birthday party people celebration",
    "wedding": "wedding ceremony",
    "college": "university campus students",
    "food": "food plate meal",
    "travel": "park outdoor path",
    "fitness": "gym fitness workout",
    "mall": "shopping mall interior",
    "selfie": "selfie portrait person",
}

# Wikimedia Commons identity-document specimens. Small and hand-picked: these
# are the only usable ones that exist (passport *covers* and SVG mock-ups are
# excluded -- they carry no OCR-able text, so they are dead weight to a BM25
# branch). PAN and driving licence stay thin until the curated milestone.
COMMONS_DOCS = [
    ("aadhaar",  "File:A sample of Aadhaar card.jpg"),
    ("aadhaar",  "File:Specimen of an Aadhaar Card 2024.png"),
    ("aadhaar",  "File:Aadhar PVC Front.jpg"),
    ("aadhaar",  "File:Sample PVC Aadhar Card back.jpg"),
    ("pan",      "File:A sample of Permanent Account Number (PAN) Card.jpg"),
    ("passport", "File:Indian Passport Bio Page 2021.jpg"),
    ("passport", "File:Indian Passport 01.jpg"),
    ("dl",       "File:Driving License issued by the Transport department of Karnataka State.jpg"),
    ("dl",       "File:International Driving Permit (India).jpg"),
]


@dataclass
class Record:
    item_id: str
    path: str
    bucket: str
    subtype: str
    is_screenshot: int = 0
    source_url: str = ""
    license: str = ""
    attribution: str = ""
    capture_date: str = ""
    origin: str = "fetched"
    notes: str = ""

    def as_row(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------- transport --
# Both upstreams rate-limit aggressively (datasets-server 429s, Openverse sits
# behind a Cloudflare bot-challenge), so every request goes through one global
# pacer. Without it a tight fetch loop gets the whole session throttled.
_MIN_INTERVAL = 1.2
_last_request_at = 0.0


def _pace(min_interval: float = _MIN_INTERVAL) -> None:
    global _last_request_at
    wait = min_interval - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _get(url: str, *, tries: int = 4, timeout: int = 30, min_interval: float = _MIN_INTERVAL) -> bytes:
    """GET with pacing and 429-aware backoff."""
    last: Exception | None = None
    for attempt in range(tries):
        _pace(min_interval)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:
                # Honour Retry-After when offered; otherwise back off hard.
                retry_after = e.headers.get("Retry-After")
                delay = float(retry_after) if (retry_after or "").isdigit() else 10.0 * (attempt + 1)
                time.sleep(min(delay, 60.0))
            elif attempt < tries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:  # noqa: BLE001 - retry on anything transport-ish
            last = e
            if attempt < tries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"GET failed after {tries} tries: {url}") from last


def _get_json(url: str, **kw) -> dict:
    return json.loads(_get(url, **kw))


def _save(data: bytes, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


# ----------------------------------------------------------------- scenes --
def fetch_scenes(raw_dir: Path, per_class: int = 10, classes=None) -> list[Record]:
    """Scene photos from the Openverse API (free, no key, CC-licensed)."""
    classes = classes or SCENE_CLASSES
    records: list[Record] = []
    for cls in classes:
        query = SCENE_QUERIES.get(cls, cls)
        url = "https://api.openverse.org/v1/images/?" + urllib.parse.urlencode({
            "q": query,
            # over-fetch: some results 404 or are too small once downloaded
            "page_size": per_class * 3,
            "license_type": "all-cc",
            "mature": "false",
            "aspect_ratio": "wide,square",
        })
        try:
            payload = _get_json(url, tries=2)
        except RuntimeError:
            # Openverse sits behind a Cloudflare bot-challenge. Once it starts
            # 429ing it stays that way for a while, so bail on the first
            # failure instead of burning minutes retrying every class.
            print(f"  ! {cls}: Openverse unavailable (rate-limited); abandoning Openverse")
            break

        kept = 0
        for res in payload.get("results", []):
            if kept >= per_class:
                break
            img_url = res.get("url")
            if not img_url:
                continue
            ext = Path(urllib.parse.urlparse(img_url).path).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                ext = ".jpg"
            item_id = f"scn_{cls}_{kept:03d}"
            dest = raw_dir / "scenes" / cls / f"{item_id}{ext}"
            try:
                blob = _get(img_url, tries=2, timeout=25)
            except RuntimeError:
                continue
            if len(blob) < 5_000:          # too small to be a usable photo
                continue
            _save(blob, dest)
            records.append(Record(
                item_id=item_id,
                path=str(dest.relative_to(raw_dir)),
                bucket="scene",
                subtype=cls,
                source_url=res.get("foreign_landing_url") or img_url,
                license=f"{res.get('license','')} {res.get('license_version','')}".strip(),
                attribution=(res.get("creator") or "")[:120],
            ))
            kept += 1
        print(f"  scenes/{cls:9} {kept:3d}/{per_class}")
    return records


# ------------------------------------------------------- identity documents --
def fetch_commons_docs(raw_dir: Path) -> list[Record]:
    """Identity-document specimens from Wikimedia Commons."""
    titles = [t for _, t in COMMONS_DOCS]
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "format": "json", "prop": "imageinfo",
        "iiprop": "url|size|extmetadata", "titles": "|".join(titles),
    })
    payload = _get_json(url)
    by_title = {
        pg["title"]: pg["imageinfo"][0]
        for pg in payload["query"]["pages"].values() if "imageinfo" in pg
    }

    records: list[Record] = []
    counters: dict[str, int] = {}
    for subtype, title in COMMONS_DOCS:
        info = by_title.get(title)
        if info is None:
            print(f"  ! commons: {title} unavailable")
            continue
        idx = counters.get(subtype, 0)
        counters[subtype] = idx + 1
        ext = Path(urllib.parse.urlparse(info["url"]).path).suffix.lower() or ".jpg"
        item_id = f"doc_{subtype}_{idx:03d}"
        dest = raw_dir / "documents" / subtype / f"{item_id}{ext}"
        _save(_get(info["url"]), dest)
        meta = info.get("extmetadata", {})
        records.append(Record(
            item_id=item_id,
            path=str(dest.relative_to(raw_dir)),
            bucket="document",
            subtype=subtype,
            source_url=info.get("descriptionurl", ""),
            license=meta.get("LicenseShortName", {}).get("value", ""),
            attribution=meta.get("Artist", {}).get("value", "")[:120],
            notes="public specimen; verify no real PII before publishing",
        ))
    for subtype, n in sorted(counters.items()):
        print(f"  documents/{subtype:9} {n:3d}")
    return records


# ---------------------------------------------------------------- receipts --
def fetch_receipts_hf(raw_dir: Path, n: int = 20, gt_path: Path | None = None) -> list[Record]:
    """Scanned receipts from SROIE (ICDAR 2019) via HuggingFace, streamed.

    Also writes the upstream OCR ground truth to `gt_path` as JSONL. That is
    what makes the PaddleOCR-vs-Tesseract CER/WER comparison free: SROIE ships
    per-receipt transcribed words, so no annotation is needed.

    Raises if the hub is unreachable -- the caller falls back to Openverse.
    """
    from datasets import load_dataset

    ds = load_dataset("jsdnrs/ICDAR2019-SROIE", split="train", streaming=True)
    records: list[Record] = []
    gt_lines: list[str] = []
    for idx, row in enumerate(ds):
        if idx >= n:
            break
        image = row.get("image")
        if image is None:
            continue
        item_id = f"doc_receipt_{idx:03d}"
        dest = raw_dir / "documents" / "receipt" / f"{item_id}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(dest, quality=92)
        words = row.get("words") or []
        gt_lines.append(json.dumps({
            "item_id": item_id,
            "sroie_key": row.get("key"),
            "words": words,
            "text": " ".join(words),
        }, ensure_ascii=False))
        records.append(Record(
            item_id=item_id,
            path=str(dest.relative_to(raw_dir)),
            bucket="document",
            subtype="receipt",
            source_url="https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE",
            license="CC-BY-4.0",
            attribution="ICDAR 2019 SROIE",
            notes="has OCR ground truth upstream",
        ))
    if gt_path is not None and gt_lines:
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
        print(f"  ocr ground truth -> {gt_path} ({len(gt_lines)} receipts)")
    print(f"  documents/receipt  {len(records):3d}  (SROIE)")
    return records


def fetch_receipts_openverse(raw_dir: Path, n: int = 20) -> list[Record]:
    """Fallback when the HuggingFace hub is unreachable.

    Lower quality than SROIE and with no OCR ground truth, so it cannot feed
    the CER/WER comparison -- it only keeps the pipeline unblocked.
    """
    recs = fetch_scenes(raw_dir, per_class=n, classes=["receipt"])
    out: list[Record] = []
    for i, r in enumerate(recs):
        src = raw_dir / r.path
        dest = raw_dir / "documents" / "receipt" / f"doc_receipt_{i:03d}{src.suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dest)
        r.item_id = f"doc_receipt_{i:03d}"
        r.path = str(dest.relative_to(raw_dir))
        r.bucket, r.subtype = "document", "receipt"
        r.notes = "Openverse fallback; NO OCR ground truth"
        out.append(r)
    return out


# ----------------------------------------------------- scenes via SUN397 ----
# Primary scene source. Openverse is kept as a fallback but is behind a
# Cloudflare bot-challenge that 429s under any sustained use, so it cannot be
# relied on. SUN397 is the standard scene-recognition dataset; we read it
# through the HuggingFace datasets-server, which serves individual rows as
# image URLs -- no 37GB shard download.
#
# `wedding` and `selfie` have no SUN397 equivalent (it is scene-centric, with
# no person-centric categories). They are deferred to the curated milestone.
SUN397_LABELS = {
    "beach": 46,
    "mountain": 244,
    "trees": 159,          # forest broadleaf
    "vehicle": 268,        # parking lot
    "party": 134,          # discotheque
    "college": 73,         # campus
    "food": 158,           # food court
    "travel": 265,         # park
    "fitness": 177,        # gymnasium indoor
    "mall": 319,           # shopping mall indoor
}

_DS_SERVER = "https://datasets-server.huggingface.co"
_SUN397_ARGS = "dataset=tanganke/sun397&config=default&split=train"


def _sun397_rows(offset: int, length: int) -> dict:
    return _get_json(f"{_DS_SERVER}/rows?{_SUN397_ARGS}&offset={offset}&length={length}", timeout=60)


def _sun397_scan_index(total: int, page: int = 100, cache_path: Path | None = None) -> dict[str, int]:
    """Map each label to the offset of its first row, by scanning the label column.

    The dataset groups each label into a contiguous run but shuffles the order
    of those runs (offset 2300 holds label 14 while offset 19800 holds label
    99), so neither arithmetic nor binary search can locate a label -- an
    earlier binary-search implementation silently returned the wrong class.
    Scanning is ~200 paged requests, so the result is cached to disk and the
    cost is paid once.
    """
    if cache_path and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("_total") == total:
                print(f"  using cached label index ({len(cached) - 1} labels)")
                return cached
        except Exception:  # noqa: BLE001 - a corrupt cache is not fatal
            pass

    print(f"  scanning label index ({total} rows, ~{total // page + 1} requests, one-off)...")
    first: dict[str, int] = {}
    for offset in range(0, total, page):
        try:
            payload = _get_json(
                f"{_DS_SERVER}/rows?{_SUN397_ARGS}&offset={offset}&length={page}&columns=label",
                tries=3, timeout=60,
            )
        except RuntimeError:
            print(f"    ! page at {offset} failed; continuing")
            continue
        for row in payload["rows"]:
            key = str(row["row"]["label"])
            if key not in first:
                first[key] = row["row_idx"]
        if offset and offset % 2000 == 0:
            print(f"    {offset}/{total}  ({len(first)} labels seen)")

    first["_total"] = total
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(first))
    print(f"  label index built: {len(first) - 1} labels")
    return first


def fetch_scenes_sun397(
    raw_dir: Path, per_class: int = 10, classes=None, cache_path: Path | None = None
) -> list[Record]:
    """Fetch scene images, caching the per-label offsets.

    Each binary search costs ~15 paced requests, so the offsets are cached to
    disk: a re-run or a resumed run skips straight to downloading.
    """
    classes = classes or list(SUN397_LABELS)
    total = _sun397_rows(0, 1)["num_rows_total"]
    records: list[Record] = []

    offsets = _sun397_scan_index(total, cache_path=cache_path)

    for cls in classes:
        label = SUN397_LABELS.get(cls)
        if label is None:
            print(f"  scenes/{cls:9}   -   (no SUN397 equivalent; deferred to curation)")
            continue
        if str(label) not in offsets:
            print(f"  ! {cls}: label {label} absent from the index; skipping")
            continue
        try:
            batch = _sun397_rows(offsets[str(label)], per_class * 2)
        except RuntimeError as e:
            print(f"  ! {cls}: datasets-server failed ({e}); skipping")
            continue

        kept = 0
        for row in batch["rows"]:
            if kept >= per_class:
                break
            payload = row["row"]
            if payload["label"] != label:        # ran past the class boundary
                break
            src = (payload.get("image") or {}).get("src")
            if not src:
                continue
            item_id = f"scn_{cls}_{kept:03d}"
            dest = raw_dir / "scenes" / cls / f"{item_id}.jpg"
            try:
                blob = _get(src, tries=3, timeout=40)
            except RuntimeError:
                continue
            if len(blob) < 3_000:
                continue
            _save(blob, dest)
            records.append(Record(
                item_id=item_id,
                path=str(dest.relative_to(raw_dir)),
                bucket="scene",
                subtype=cls,
                source_url="https://huggingface.co/datasets/tanganke/sun397",
                license="SUN397 research use",
                attribution="SUN397 (Xiao et al., CVPR 2010)",
            ))
            kept += 1
        print(f"  scenes/{cls:9} {kept:3d}/{per_class}")
    return records
