"""Scene-branch search: CLIP text embedding -> exact cosine over the corpus."""
from __future__ import annotations

from ..clip_model import ClipEncoder
from ..store.vectors import VectorStore
from .bm25_search import Hit


def search(encoder: ClipEncoder, store: VectorStore, query_text: str, top_k: int = 10) -> list[Hit]:
    vec = encoder.encode_texts([query_text])[0]
    return [
        Hit(item_id=item_id, score=score, branch="scene")
        for item_id, score in store.search(vec, top_k)
    ]
