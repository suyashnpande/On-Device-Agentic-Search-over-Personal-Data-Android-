"""Edit distance and error rates against hand-computed values."""
import pytest

from pixarch.eval.ocr_quality import error_rate, levenshtein


@pytest.mark.parametrize("a,b,expected", [
    ("kitten", "sitting", 3),
    ("", "abc", 3),
    ("abc", "", 3),
    ("same", "same", 0),
])
def test_levenshtein(a, b, expected):
    assert levenshtein(a, b) == expected


def test_levenshtein_is_symmetric():
    assert levenshtein("flaw", "lawn") == levenshtein("lawn", "flaw")


def test_cer_is_distance_over_reference_length():
    # one substitution over 6 reference characters
    assert error_rate("abcdef", "abcdeX", by_word=False) == pytest.approx(1 / 6)


def test_wer_counts_whole_tokens():
    assert error_rate("the quick brown fox", "the slow brown fox", by_word=True) == pytest.approx(0.25)


def test_perfect_match_is_zero():
    assert error_rate("total 450", "total 450", by_word=True) == 0.0


def test_empty_reference_with_empty_hypothesis_is_zero():
    assert error_rate("", "", by_word=False) == 0.0


def test_empty_reference_with_output_is_full_error():
    assert error_rate("", "spurious", by_word=False) == 1.0
