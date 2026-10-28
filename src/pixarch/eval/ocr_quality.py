"""OCR accuracy against SROIE ground truth.

SROIE ships a transcription per receipt, so character and word error rates cost
no annotation. This is the number that later justifies (or kills) the ML Kit
swap on Android, and feeds the ColPali-vs-OCR decision in Thread 1.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    """Edit distance over characters or tokens, O(min(len)) memory."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,          # deletion
                current[j - 1] + 1,       # insertion
                previous[j - 1] + (ca != cb),  # substitution
            ))
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, *, by_word: bool) -> float:
    """CER or WER. An empty reference yields 0.0 when the hypothesis is empty."""
    ref = reference.split() if by_word else reference
    hyp = hypothesis.split() if by_word else hypothesis
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def load_ground_truth(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["item_id"]] = row.get("text", "")
    return out


def evaluate_ocr(conn: sqlite3.Connection, gt_path: Path) -> dict[str, float]:
    """Mean CER/WER over items that have both OCR output and ground truth.

    Both sides are normalised the same way, so this measures recognition rather
    than punctuation and casing conventions.
    """
    from ..ocr import normalise

    truth = load_ground_truth(gt_path)
    if not truth:
        return {}

    rows = conn.execute("SELECT item_id, full_text, engine FROM doc_fields").fetchall()
    cers: list[float] = []
    wers: list[float] = []
    engine = ""
    for row in rows:
        reference = truth.get(row["item_id"])
        if reference is None:
            continue
        engine = row["engine"] or engine
        ref_norm = normalise(reference)
        cers.append(error_rate(ref_norm, row["full_text"], by_word=False))
        wers.append(error_rate(ref_norm, row["full_text"], by_word=True))

    if not cers:
        return {}
    return {
        "n_items": float(len(cers)),
        "cer": sum(cers) / len(cers),
        "wer": sum(wers) / len(wers),
        "engine": engine,
    }
