"""Stage 2a -- OCR.

Engines sit behind the `OCREngine` protocol so the desktop baseline and the
eventual Android build differ by one line of config. On Android the engine
becomes Google ML Kit Text Recognition v2; nothing at the call sites changes.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass
class TextBlock:
    text: str
    conf: float                       # 0-100
    box: tuple[int, int, int, int]    # x0, y0, x1, y1


class OCREngine(Protocol):
    name: str

    def extract(self, image_path: Path) -> list[TextBlock]: ...


# --------------------------------------------------------------- EasyOCR ----
class EasyOCREngine:
    """Default desktop engine.

    Chosen over Tesseract because it installs from pip with no system package
    (no sudo available here) and supports Devanagari alongside Latin, which the
    architecture note requires for Hindi/Hinglish documents.
    """

    name = "easyocr"

    def __init__(self, languages: Sequence[str] = ("en", "hi"), gpu: bool = True):
        import easyocr

        # EasyOCR uses 2-letter codes; map from the config's ISO-639-2 names.
        lang_map = {"eng": "en", "hin": "hi", "en": "en", "hi": "hi"}
        langs = [lang_map.get(l, l) for l in languages]
        self.languages = langs
        self.reader = easyocr.Reader(langs, gpu=gpu, verbose=False)

    def extract(self, image_path: Path) -> list[TextBlock]:
        results = self.reader.readtext(str(image_path))
        blocks: list[TextBlock] = []
        for poly, text, conf in results:
            xs = [int(p[0]) for p in poly]
            ys = [int(p[1]) for p in poly]
            blocks.append(
                TextBlock(text=text, conf=float(conf) * 100.0,
                          box=(min(xs), min(ys), max(xs), max(ys)))
            )
        return blocks


# ------------------------------------------------------------- Tesseract ----
class TesseractEngine:
    """Comparator engine. Needs the `tesseract-ocr` system package."""

    name = "tesseract"

    def __init__(self, languages: Sequence[str] = ("eng", "hin")):
        import pytesseract  # noqa: F401

        self.lang = "+".join(languages)

    def extract(self, image_path: Path) -> list[TextBlock]:
        import pytesseract
        from PIL import Image

        with Image.open(image_path) as img:
            data = pytesseract.image_to_data(
                img, lang=self.lang, output_type=pytesseract.Output.DICT
            )
        blocks: list[TextBlock] = []
        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            x, y, w, h = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            blocks.append(TextBlock(text=text, conf=conf, box=(x, y, x + w, y + h)))
        return blocks


def build_engine(name: str, languages: Sequence[str], gpu: bool = True) -> OCREngine:
    if name == "easyocr":
        return EasyOCREngine(languages, gpu=gpu)
    if name == "tesseract":
        return TesseractEngine(languages)
    raise ValueError(f"unknown OCR engine: {name!r} (expected 'easyocr' or 'tesseract')")


# ----------------------------------------------------------- normalisation --
_WS = re.compile(r"\s+")
# Keep word characters across scripts (Latin + Devanagari) and digits; drop the
# punctuation that OCR invents on noisy scans.
_PUNCT = re.compile(r"[^\w\sऀ-ॿ]", re.UNICODE)


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Digit groups are preserved as written -- an Aadhaar number is printed as
    three 4-digit groups and splitting or joining them would break matching.
    Devanagari is left as-is.
    """
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text)
    return text.strip().lower()


def run_ocr(engine: OCREngine, image_path: Path, min_conf: float = 0.0):
    """Run OCR and return (blocks, raw_text, normalised_text, elapsed_ms)."""
    start = time.perf_counter()
    blocks = engine.extract(image_path)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    kept = [b for b in blocks if b.conf < 0 or b.conf >= min_conf]
    raw_text = " ".join(b.text for b in kept)
    return kept, raw_text, normalise(raw_text), elapsed_ms
