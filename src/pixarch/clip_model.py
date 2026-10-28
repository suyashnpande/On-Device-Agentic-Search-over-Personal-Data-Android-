"""CLIP encoder shared by Stage 1 (classification) and Stage 2b (scene search).

Loading one model for both jobs is deliberate: it halves the model memory the
eventual on-device build has to budget for.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image


def resolve_device(requested: str = "auto") -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


class ClipEncoder:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "auto",
        batch_size: int = 32,
    ):
        import open_clip

        self.device = resolve_device(device)
        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @property
    def id(self) -> str:
        return f"{self.model_name}/{self.pretrained}"

    @staticmethod
    def _normalise(t: torch.Tensor) -> torch.Tensor:
        return t / t.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        tokens = self.tokenizer(list(texts)).to(self.device)
        feats = self._normalise(self.model.encode_text(tokens))
        return feats.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_images(self, paths: Sequence[Path]) -> tuple[list[Path], np.ndarray]:
        """Encode images in batches.

        Returns the paths that decoded successfully alongside their embeddings,
        so a corrupt file drops out rather than misaligning the whole batch.
        """
        kept: list[Path] = []
        chunks: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            tensors, ok_paths = [], []
            for p in batch_paths:
                try:
                    with Image.open(p) as im:
                        tensors.append(self.preprocess(im.convert("RGB")))
                    ok_paths.append(p)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! cannot encode {p.name}: {type(e).__name__}")
            if not tensors:
                continue
            stack = torch.stack(tensors).to(self.device)
            feats = self._normalise(self.model.encode_image(stack))
            chunks.append(feats.cpu().numpy().astype(np.float32))
            kept.extend(ok_paths)

        if not chunks:
            return [], np.zeros((0, 512), dtype=np.float32)
        return kept, np.vstack(chunks)
