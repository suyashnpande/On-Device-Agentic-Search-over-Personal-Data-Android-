"""Stage 2a -- document subtype tagging from OCR text.

Rule-based on purpose. The tag is indexed as its own FTS5 column and weighted
above the body text, which is what makes "give me my aadhaar card" reliable:
raw BM25 over OCR text alone is brittle when a scan is noisy and the one
distinguishing phrase comes back mangled.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

UNKNOWN = "unknown"


@dataclass(frozen=True)
class Rule:
    doc_type: str
    name: str
    # Any of `any_of` matching is enough; every term in `all_of` must match.
    any_of: tuple[str, ...] = ()
    all_of: tuple[str, ...] = ()
    weight: int = 1


# Ordered by specificity: the first highest-weight match wins.
RULES: tuple[Rule, ...] = (
    Rule("aadhaar", "aadhaar-authority", weight=3,
         any_of=("unique identification authority", "uidai",
                 "आधार", "भारतीय विशिष्ट पहचान प्राधिकरण")),
    Rule("aadhaar", "aadhaar-keyword", weight=2, any_of=("aadhaar", "aadhar")),
    Rule("pan", "pan-incometax", weight=3,
         any_of=("permanent account number", "income tax department",
                 "आयकर विभाग")),
    Rule("pan", "pan-keyword", weight=1, any_of=("pan card",)),
    Rule("passport", "passport-republic", weight=3,
         all_of=("passport",), any_of=("republic of india", "भारत गणराज्य")),
    Rule("passport", "passport-mrz", weight=3,
         any_of=("p<ind", "type passport", "passport no")),
    Rule("passport", "passport-keyword", weight=1, any_of=("passport",)),
    Rule("dl", "dl-transport", weight=3,
         any_of=("driving licence", "driving license",
                 "transport department", "international driving permit")),
    Rule("dl", "dl-cov", weight=2, any_of=("cov issue date", "lmv", "mcwg")),
    Rule("receipt", "receipt-total", weight=2,
         any_of=("total", "subtotal", "grand total", "cash", "change"),
         all_of=()),
    Rule("receipt", "receipt-tax", weight=3,
         any_of=("gst", "tax invoice", "invoice no", "receipt no", "bill no")),
    # "account number" and "due date" are deliberately NOT here. PAN stands for
    # "Permanent Account Number", so a blank PAN template whose only legible
    # text is its field labels was being tagged `bill`; "due date" appears on
    # ordinary invoices too. A utility bill needs a utility-specific signal.
    Rule("bill", "bill-utility", weight=3,
         any_of=("electricity bill", "utility bill", "units consumed",
                 "meter reading", "consumer no", "billing period",
                 "previous reading", "sanctioned load")),
    Rule("book", "book-isbn", weight=3, any_of=("isbn", "chapter", "copyright ©")),
)


def _present(term: str, text: str) -> bool:
    """Whole-word match for short Latin terms, substring otherwise.

    Short terms like "pan" or "gst" produce constant false positives as
    substrings ("company", "august"), while Devanagari has no \\b word
    boundaries in the regex sense, so it falls back to substring.
    """
    if len(term) <= 4 and term.isascii():
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    return term in text


def classify_text(normalised_text: str) -> tuple[str, str, int]:
    """Return (doc_type, rule_name, weight). Unmatched text yields UNKNOWN."""
    if not normalised_text:
        return UNKNOWN, "", 0

    best: tuple[str, str, int] = (UNKNOWN, "", 0)
    for rule in RULES:
        if rule.all_of and not all(_present(t, normalised_text) for t in rule.all_of):
            continue
        if rule.any_of and not any(_present(t, normalised_text) for t in rule.any_of):
            continue
        if not rule.any_of and not rule.all_of:
            continue
        if rule.weight > best[2]:
            best = (rule.doc_type, rule.name, rule.weight)
    return best


DOC_TYPES = sorted({r.doc_type for r in RULES} | {UNKNOWN})
