-- Pixarch baseline index.
-- One SQLite file holds the corpus metadata, OCR text, the inverted index
-- (FTS5) and the CLIP embeddings: a single artifact to copy, inspect or ship.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- items ----
-- Gold columns (gold_*) come from data/manifest.csv and are never written by
-- the pipeline. Predicted columns (bucket_pred/bucket_conf) are written by
-- Stage 1. Keeping both here means evaluation is a single SQL join.
CREATE TABLE IF NOT EXISTS items (
    item_id       TEXT PRIMARY KEY,
    path          TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    bytes         INTEGER,
    width         INTEGER,
    height        INTEGER,
    capture_date  TEXT,              -- ISO-8601 from EXIF, NULL when absent
    is_screenshot INTEGER NOT NULL DEFAULT 0,

    -- from manifest.csv (ground truth + provenance)
    gold_bucket   TEXT,              -- document | scene | face
    gold_subtype  TEXT,
    origin        TEXT,              -- fetched | curated
    source_url    TEXT,
    license       TEXT,
    attribution   TEXT,
    notes         TEXT,

    -- from Stage 1
    bucket_pred   TEXT,
    bucket_conf   REAL,
    routed_both   INTEGER NOT NULL DEFAULT 0,   -- 1 when conf < threshold

    ingested_at   TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_items_sha      ON items(sha256);
CREATE INDEX IF NOT EXISTS idx_items_gold     ON items(gold_bucket, gold_subtype);
CREATE INDEX IF NOT EXISTS idx_items_pred     ON items(bucket_pred);
CREATE INDEX IF NOT EXISTS idx_items_origin   ON items(origin);

-- ----------------------------------------------------- document branch ----
CREATE TABLE IF NOT EXISTS ocr_blocks (
    item_id   TEXT NOT NULL REFERENCES items(item_id) ON DELETE CASCADE,
    block_idx INTEGER NOT NULL,
    text      TEXT NOT NULL,
    conf      REAL,
    x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER,
    PRIMARY KEY (item_id, block_idx)
);

CREATE TABLE IF NOT EXISTS doc_fields (
    item_id       TEXT PRIMARY KEY REFERENCES items(item_id) ON DELETE CASCADE,
    full_text     TEXT NOT NULL,     -- normalised, concatenated OCR
    raw_text      TEXT,              -- pre-normalisation, for debugging/CER
    doc_type      TEXT,              -- aadhaar | pan | dl | passport | receipt | bill | book | unknown
    doc_type_rule TEXT,              -- which rule fired, for explainability
    engine        TEXT,              -- tesseract | paddle
    ocr_ms        REAL
);

-- The inverted index + BM25 ranking, both free from FTS5. item_id is carried
-- UNINDEXED so hits map back to items without a rowid side-table.
--
-- There is deliberately NO filename column. On a device the gallery holds
-- opaque names like IMG_20231205_123456.jpg, so a filename match is not a
-- signal the deployed system would ever have. Indexing our own generated
-- names (doc_aadhaar_000.jpg) leaked the class label straight into the
-- ranking and was worth +0.125 p@1 on document lookup -- measured, then
-- removed. Zero-weighting was not enough: the tokens still count toward
-- document length and skew BM25's normalisation.
CREATE VIRTUAL TABLE IF NOT EXISTS doc_fts USING fts5(
    item_id UNINDEXED,
    ocr_text,
    doc_type,
    -- `porter` stems English terms, so a query for "receipts" reaches a
    -- document containing "receipt". Without it the plural matched zero of 18
    -- receipts and the query returned nothing at all. Porter is English-only;
    -- Devanagari passes through untouched.
    tokenize = "porter unicode61 remove_diacritics 2"
);

-- -------------------------------------------------------- scene branch ----
CREATE TABLE IF NOT EXISTS embeddings (
    item_id TEXT PRIMARY KEY REFERENCES items(item_id) ON DELETE CASCADE,
    model   TEXT NOT NULL,
    dim     INTEGER NOT NULL,
    vec     BLOB NOT NULL            -- float32, L2-normalised
);

-- -------------------------------------------------------------- timing ----
CREATE TABLE IF NOT EXISTS timings (
    run_id     TEXT NOT NULL,
    stage      TEXT NOT NULL,
    n_items    INTEGER,
    total_ms   REAL,
    created_at TEXT DEFAULT (datetime('now'))
);
