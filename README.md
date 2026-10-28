# Pixarch

On-device agentic retrieval over personal data — documents, scans, screenshots
and photos — answered in plain language, with nothing leaving the device.

This repository currently holds the **baseline**: the non-agentic, single-shot
system that every later contribution is measured against.

```
images ──► ① classify ──┬──► ② document bucket ──► OCR ──► FTS5 inverted index ──► BM25
                        │
                        └──► ③ scene bucket ────► CLIP embed ──► vector store ──► cosine
```

No agentic loop, no score fusion, no face branch — deliberately. The baseline's
job is to be a clean floor with honest numbers.

## Quickstart

```bash
pip install -r requirements.txt

python scripts/fetch_dataset.py     # build the corpus + data/manifest.csv
python scripts/make_qrels.py        # expand queries.jsonl -> qrels.csv
python scripts/build_index.py       # ingest -> classify -> OCR + embeddings
python scripts/search.py "give me my aadhaar card"
python scripts/evaluate.py          # metrics + comparators -> docs/results/
python app/demo.py                  # Gradio UI
pytest                              # unit tests
```

## Layout

| path | what |
|---|---|
| `src/pixarch/ingest.py` | hashing, EXIF, screenshot heuristic |
| `src/pixarch/classify.py` | CLIP zero-shot bucket routing |
| `src/pixarch/ocr.py` | `OCREngine` protocol + EasyOCR / Tesseract |
| `src/pixarch/doctype.py` | keyword rules for document subtype |
| `src/pixarch/store/` | SQLite schema, FTS5, vector store |
| `src/pixarch/query/` | router, BM25 search, CLIP search |
| `src/pixarch/eval/` | metrics and comparator systems |
| `configs/baseline.yaml` | every tunable; no magic numbers in code |
| `docs/design-decisions.md` | why each choice was made, and the traps hit |

Everything lands in a single `index.db`: one artifact to copy, inspect or ship.

## Data

`data/raw/` is gitignored; `data/manifest.csv` is committed and
`scripts/fetch_dataset.py` reconstructs the corpus from it, so the dataset is
reproducible without redistributing images. Every row carries
`source_url`/`license`/`attribution`.

Relevance is declared **by subtype** in `data/queries.jsonl` and expanded
against the manifest, so the query set stays valid as the corpus grows: adding
ten more beach photos needs no edit.

Growth is meant to be cheap — drop images in, append manifest rows, re-run
`build_index.py` and `evaluate.py`. No code changes.

> The current corpus is small and auto-fetched and relevance is single-judge,
> so results are **indicative, not publishable**: enough to catch regressions
> and prove the plumbing, not to support a claim. The benchmark-grade curated
> set is a separate milestone.
