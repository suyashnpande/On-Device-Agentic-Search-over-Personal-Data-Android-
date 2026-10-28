# Design decisions — baseline

Decisions taken while building the baseline, with the reasoning, so they can be
revisited deliberately rather than rediscovered. Thread 1 of the proposal asks
for exactly this document.

## Storage: SQLite FTS5 for the inverted index and BM25

The architecture note calls for "inverted indexing → BM25". FTS5 provides both,
ships with Python's stdlib `sqlite3` *and* with Android, and exposes a `bm25()`
ranking function with per-column weights.

Rejected: hand-rolling an index (work, no upside), and `rank_bm25`/Whoosh
(neither survives the Android port). FTS5 is the only option that is identical
on desktop and device, which matters because the port is the whole point.

Two details worth remembering:
- `bm25()` returns **negative** scores, most-negative being best. Results are
  ordered ascending and the sign is flipped for display.
- `bm25()` takes one weight per declared column **including `UNINDEXED` ones**,
  so `item_id` needs a leading `0.0`.

## Classification: CLIP zero-shot, not a trained classifier

Zero-shot needs no training labels, so all manifest labels stay a clean
evaluation set, and it reuses the encoder the scene branch already loads — one
model doing two jobs, which halves what the on-device build must budget for.

Confidence is the **margin** between the best and second-best bucket, not the
raw similarity: CLIP's absolute scores are not comparable across images, but
the gap between two buckets for one image is usable. Below the threshold an
item is indexed into **both** branches rather than guessed.

## OCR: EasyOCR by default, behind an `OCREngine` protocol

The note names Google ML Kit Text Recognition v2, which is Android-only. On
desktop the engine must provide Latin **and** Devanagari.

Tesseract needs the `tesseract-ocr` system package and this environment has no
passwordless sudo, so EasyOCR is the default: pip-installable, supports `en`
and `hi`, and runs on the GPU already present. Tesseract remains implemented as
the comparator for whoever can install it.

All engines sit behind `OCREngine.extract(path) -> list[TextBlock]`, so the
Android build swaps in ML Kit by changing `ocr.engine` in the config. The
protocol earned its keep on day one, when the intended engine turned out to be
uninstallable.

## Document subtype: keyword rules, indexed as a separate field

Raw BM25 over OCR text is brittle — on a noisy scan the one distinguishing
phrase is often the text that comes back mangled. Tagging `doc_type` from
keyword rules and indexing it as its own FTS5 column (weighted 5× against the
body's 1×) is what makes "give me my aadhaar card" reliable.

Short Latin terms (`pan`, `gst`, `dl`) are matched **whole-word**; as
substrings they fire on "company" and "august" and silently tag ordinary
photos as identity documents. Longer terms and Devanagari use substring
matching, since `\b` is not meaningful for Devanagari.

## Vector search: exact brute force, no FAISS

At baseline corpus size a `numpy` matmul over L2-normalised float32 is exact
and sub-millisecond. An approximate index would add a dependency, a tuning
surface and a recall loss, for nothing. `VectorStore` is the seam where
FAISS/usearch/ObjectBox goes when the corpus justifies it.

## Routing: rules first, CLIP only when rules abstain

The baseline is the "agentic loop switched off" ablation, so it must stay
single-shot: no rewriting, no retrying, no clarifying questions, and **no
fusion** of the two branches. Fusing incomparable signals is the proposal's own
cross-cutting contribution and must not be pre-empted by the thing it will be
measured against.

## Dataset sourcing

| Need | Source | Outcome |
|---|---|---|
| Receipts | SROIE (ICDAR 2019) via HuggingFace | ✅ 20, **with OCR ground truth** |
| ID documents | Wikimedia Commons specimens | ⚠️ 9 total; PAN and DL are 1–2 each |
| Scenes | SUN397 via HF datasets-server | ✅ 10 classes |
| Scenes (first attempt) | Openverse API | ❌ Cloudflare bot-challenge, 429s under any sustained use |
| Screenshots | — | deferred: team-captured, no public substitute |

Rejected: **RVL-CDIP**, despite having `invoice`/`form` classes — it is 1990s
grayscale fax-quality litigation scans, and including it would make the
benchmark unrealistic in exactly the way the proposal criticises.

### SUN397 row ordering — a trap worth recording

SUN397's rows group each label into a contiguous run but **shuffle the order of
those runs**: offset 2300 holds label 14 while offset 19800 holds label 99.
Sparse probing (offsets 0/500/2000/8000) returns increasing labels and looks
sorted, which it is not.

A binary search built on that false assumption returned the wrong class
*silently* — `beach` resolved to an offset holding label 120, and the fetcher
reported `0/10` rather than erroring. The fix is a one-off scan of the label
column (`columns=label`, ~199 paged requests), cached to
`data/.sun397_offsets.json`.

Also: the `/filter` endpoint returns HTTP 500 for this dataset, and bulk
download is not an option (7.3 GB of train shards at ~404 MB each).

### Rate limiting

Both upstreams throttle hard: the HF datasets-server 429s under burst load and
Openverse sits behind a Cloudflare bot-challenge. All requests go through one
global pacer (1.2 s minimum interval) with 429-aware backoff that honours
`Retry-After`. Without it, a tight loop gets the whole session throttled — which
is how the first fetch attempt failed.

## Calibrating the classifier threshold (measured)

The confidence threshold started at 0.10 and was wrong by a wide margin. On the
127-item corpus:

| bucket | min | median | max |
|---|---|---|---|
| scene | 0.014 | 0.068 | 0.118 |
| document | 0.049 | 0.110 | 0.159 |

The two distributions overlap heavily, and 0.10 sits *above* the scene median,
so it hedged 100 of 127 items into both branches — OCR over the entire corpus
for no gain. Argmax is already **127/127 correct**, so the threshold's only job
is a narrow hedge. At 0.05 it hedges 14 items (11%).

Two things to carry forward:
- The margin is **not comparable across buckets** (document margins run
  systematically higher), so a single global threshold is the wrong shape. A
  per-bucket threshold is the obvious next step.
- τ is fitted on the same corpus we evaluate on. Re-derive it on a held-out
  split once the curated set exists.

## Precision leads the results table, not recall

Each scene query has 10 relevant items, so `recall@k` is capped at `k/10`:
a *perfect* ranking scores recall@1 = 0.10 and recall@5 = 0.50. Reported as the
headline that reads as near-total failure when it is in fact the ceiling.

The results table therefore leads with `p@1` / `p@5` and MRR, with recall kept
further right for completeness. Worth remembering when the curated set changes
the number of relevant items per query — the ceiling moves with it.

## Why the document side fails (measured)

Scene search is effectively solved on this corpus (p@1 = 1.000). Document
lookup sits at p@1 = 0.625, and the causes are specific and separable.

### First: two leaks of our own making, now fixed

`filename` was an indexed FTS column weighted 0.5. Our files are named
`doc_aadhaar_000.jpg`, so the ranker was matching the class label out of the
path -- a signal no deployed system has, since a real gallery holds
`IMG_20231205_123456.jpg`. Removing it cost **p@1 0.625 -> 0.500**, which is
the size of the leak. Zero-weighting is insufficient: the tokens still count
toward document length and skew BM25 normalisation, so the column is gone.

There was also no **stemming**. `"receipts"` matched *zero* of 18 receipts
while `"receipt"` matched all 18, so one plural turned a query into a total
miss. Enabling FTS5's `porter` tokenizer restored **p@1 0.500 -> 0.625** --
legitimately this time, recovering exactly what the leak had been masking.

### What remains, by cause

OCR output for the 9 identity documents, by source type:

| source type | items | OCR result |
|---|---|---|
| annotated security-feature diagram | aadhaar_002, aadhaar_003 | reads the *annotation labels* ("guilloche pattern", "invisible logo"), not card content |
| passport cover (closed book) | passport_001 | **0 characters** |
| blank template, field labels only | pan_000 | 44 chars, no issuer name |
| low-resolution thumbnail | dl_000, aadhaar_000 | 20-89 chars, partial |
| clean document photo | aadhaar_001, passport_000, dl_001 | 262-483 chars, correctly typed |

Only **3 of 9** identity documents OCR cleanly. All **20 receipts** do
(286-815 chars, 20/20 correctly typed). Resolution is *not* the driver -- the
largest image (12.5 MP passport cover) yields nothing while the smallest
(0.10 MP) yields 89 characters. What matters is whether the image shows a
document face bearing text.

The three surviving failures each have a distinct cause:

1. **The target is unfindable (q002 "pan card").** The only PAN specimen is a
   blank template; no "PAN" or "Income Tax Department" text survives OCR, so
   the item contains nothing to match. No ranking change can fix this -- it is
   a data defect.
2. **BM25 length normalisation (q006 "tax invoice with gst number").** The
   44-character PAN fragment scores 5.10 on the single incidental term
   "number", beating 449-character receipts that match "tax", "invoice" *and*
   "gst". Short documents are over-rewarded, and SQLite's `bm25()` exposes no
   `k1`/`b` to retune. Twelve hedged scene images with ~0 characters also sit
   in the document index and drag the mean document length down, sharpening
   the same bias.
3. **Cross-language vocabulary mismatch (q007 "government identity
   document").** The Aadhaar card states its purpose in Devanagari
   ("आधार पहचान का प्रमाण है" -- Aadhaar is proof of identity) while the query
   is in English; meanwhile a SROIE receipt literally contains the token
   "document". Lexical matching cannot bridge that, and it is a recurring
   problem for Indian documents specifically.

### Why this is left unfixed

(1) is a dataset problem for the curated milestone. (2) and (3) are exactly
"the simple approach returns a confident wrong answer" -- the trigger condition
the agentic loop exists to detect and recover from, and the case for query
rewriting and cross-lingual expansion. Engineering them away inside the
baseline would delete the very thing the next phase is meant to beat.

What *was* fixed is only what made the baseline unfairly weak or dishonest:
stopwords, stemming, and the filename leak.

## Known limitations

- **Geography skew.** SROIE receipts are Malaysian/Singaporean and SUN397 is
  Western-weighted, while the motivation is an Indian phone gallery. The
  team-captured screenshots are the corrective.
- **Thin ID classes.** 1 PAN, 2 DL, 4 Aadhaar. Report per-class numbers with
  their n; do not quote them as stable.
- **No screenshots yet**, so the expected CLIP-on-screenshots weakness is not
  yet measurable.
- **No `wedding` or `selfie` scene class** — SUN397 is scene-centric and has no
  person-centric categories.
- **Single-judge relevance** over 20 queries. Indicative, not publishable.
- **The `filename` comparator is inflated.** The fetcher names files after
  their class, so path matching succeeds where a real gallery
  (`IMG_20231205_123456.jpg`) would return nothing. An earlier version was
  worse still -- it matched against `gold_subtype`, the ground-truth label,
  which made the "trivial floor" an oracle. The leak is fixed and guarded by a
  test; the naming inflation needs original filenames preserved during
  curation.
