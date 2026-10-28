"""Ingest helpers: the screenshot heuristic and OCR text normalisation."""
import pytest

from pixarch.ingest import looks_like_screenshot
from pixarch.ocr import normalise

NO_CAMERA = {"has_camera": False, "has_gps": False, "capture_date": None}
WITH_CAMERA = {"has_camera": True, "has_gps": False, "capture_date": None}


@pytest.mark.parametrize("size", [(1080, 2400), (2400, 1080), (1170, 2532), (1920, 1080)])
def test_known_screen_sizes_without_camera_exif_are_screenshots(size):
    assert looks_like_screenshot(*size, NO_CAMERA)


def test_camera_exif_rules_out_a_screenshot():
    """Both conditions are required, to keep false positives low.

    A photo cropped to exactly 1080x2400 would otherwise be misread.
    """
    assert not looks_like_screenshot(1080, 2400, WITH_CAMERA)


@pytest.mark.parametrize("size", [(4080, 3072), (573, 411), (1000, 1000)])
def test_non_screen_sizes_are_not_screenshots(size):
    assert not looks_like_screenshot(*size, NO_CAMERA)


def test_normalise_lowercases_and_strips_punctuation():
    assert normalise("Govt. of India -- AADHAAR!!") == "govt of india aadhaar"


def test_normalise_preserves_digit_grouping():
    """Aadhaar is printed as three 4-digit groups; regrouping breaks matching."""
    assert normalise("1234 5678 9012") == "1234 5678 9012"


def test_normalise_keeps_devanagari():
    assert "भारत" in normalise("भारत सरकार Income-Tax")


def test_normalise_collapses_whitespace():
    assert normalise("  a \n\t b  ") == "a b"
