"""Stage 1 -- zero-shot bucket classification with CLIP.

Chosen over a trained classifier so that every manifest label stays available
as a clean evaluation set, and so the scene branch's encoder does double duty.
"""
from __future__ import annotations

import numpy as np

from .clip_model import ClipEncoder


class BucketClassifier:
    """Zero-shot classifier over bucket prompt ensembles.

    Each bucket is represented by the mean of its prompt embeddings. Confidence
    is the *margin* between the best and second-best bucket, not the raw score:
    CLIP's absolute similarities are not comparable across images, but the gap
    between two buckets for the same image is a usable signal.
    """

    def __init__(self, encoder: ClipEncoder, prompts: dict[str, list[str]], threshold: float):
        self.encoder = encoder
        self.buckets = sorted(prompts)
        self.threshold = threshold
        self.prototypes = np.vstack([self._prototype(prompts[b]) for b in self.buckets])

    def _prototype(self, prompts: list[str]) -> np.ndarray:
        embeddings = self.encoder.encode_texts(prompts)
        proto = embeddings.mean(axis=0)
        return proto / max(float(np.linalg.norm(proto)), 1e-12)

    def classify(self, image_embeddings: np.ndarray) -> list[tuple[str, float, bool]]:
        """Return (bucket, margin, route_to_both) per image embedding."""
        if image_embeddings.size == 0:
            return []
        scores = image_embeddings @ self.prototypes.T          # (n_images, n_buckets)
        order = np.argsort(-scores, axis=1)
        out: list[tuple[str, float, bool]] = []
        for row, ranking in zip(scores, order):
            best = self.buckets[ranking[0]]
            margin = (
                float(row[ranking[0]] - row[ranking[1]]) if len(ranking) > 1 else 1.0
            )
            out.append((best, margin, margin < self.threshold))
        return out
