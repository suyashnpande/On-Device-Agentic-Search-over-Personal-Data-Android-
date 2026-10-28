"""Stage 3 -- query routing.

Deliberately rule-first and single-shot. This is the ablation the agentic loop
must beat, so it must not quietly become clever: no rewriting, no retrying, no
asking the user, no fusing the two branches.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..clip_model import ClipEncoder

DOCUMENT = "document"
SCENE = "scene"

# Vocabulary that is decisive on its own: if the question names a document
# type or a document-ish action, it is a document lookup.
_DOC_TERMS = {
    "aadhaar", "aadhar", "pan", "passport", "licence", "license", "dl",
    "receipt", "receipts", "bill", "bills", "invoice", "statement",
    "document", "documents", "card", "id", "identity", "certificate",
    "ticket", "voucher", "payment", "tax", "gst", "book", "page", "scan",
}
_DOC_PHRASES = ("find my", "give me my", "show me my", "where is my")

_SCENE_TERMS = {
    "photo", "photos", "picture", "pictures", "pic", "pics", "selfie",
    "beach", "mountain", "forest", "trees", "party", "wedding", "food",
    "gym", "mall", "campus", "college", "park", "travel", "trip", "sunset",
}

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class Route:
    branch: str
    confidence: float
    reason: str


class QueryRouter:
    """Keyword rules first; CLIP prototype similarity only when rules abstain."""

    def __init__(self, encoder: ClipEncoder | None = None, prompts: dict | None = None):
        self.encoder = encoder
        self.prototypes: dict[str, np.ndarray] = {}
        if encoder is not None and prompts:
            for bucket, texts in prompts.items():
                emb = encoder.encode_texts(texts).mean(axis=0)
                self.prototypes[bucket] = emb / max(float(np.linalg.norm(emb)), 1e-12)

    def route(self, query_text: str) -> Route:
        lowered = query_text.lower()
        tokens = set(_WORD.findall(lowered))

        doc_hits = tokens & _DOC_TERMS
        scene_hits = tokens & _SCENE_TERMS
        phrase_hit = any(p in lowered for p in _DOC_PHRASES)

        # A possessive phrase plus a document noun is the clearest signal there
        # is ("give me my aadhaar card"), so it outranks a bare scene word.
        if phrase_hit and doc_hits:
            return Route(DOCUMENT, 1.0, f"phrase+terms:{','.join(sorted(doc_hits))}")
        if doc_hits and not scene_hits:
            return Route(DOCUMENT, 0.9, f"terms:{','.join(sorted(doc_hits))}")
        if scene_hits and not doc_hits:
            return Route(SCENE, 0.9, f"terms:{','.join(sorted(scene_hits))}")
        if phrase_hit:
            return Route(DOCUMENT, 0.7, "phrase")

        # Rules abstained (no signal, or both fired). Fall back to CLIP.
        if self.prototypes:
            emb = self.encoder.encode_texts([query_text])[0]
            scored = sorted(
                ((b, float(emb @ p)) for b, p in self.prototypes.items()),
                key=lambda kv: -kv[1],
            )
            best, best_score = scored[0]
            margin = best_score - scored[1][1] if len(scored) > 1 else 1.0
            return Route(best, float(margin), f"clip:{best_score:.3f}")

        # No encoder available: default to scene, the larger bucket.
        return Route(SCENE, 0.0, "default")
