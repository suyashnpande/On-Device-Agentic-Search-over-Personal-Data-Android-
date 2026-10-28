"""Doc-type rules against fixture OCR strings, including near-miss negatives."""
import pytest

from pixarch.doctype import UNKNOWN, classify_text

CASES = [
    ("government of india unique identification authority 1234 5678 9012", "aadhaar"),
    ("आधार भारत सरकार", "aadhaar"),
    ("income tax department permanent account number abcde1234f", "pan"),
    ("republic of india passport no z1234567", "passport"),
    ("transport department driving licence mh01 lmv", "dl"),
    ("international driving permit", "dl"),
    ("tax invoice gst no 29abc total 450 00", "receipt"),
    ("electricity bill units consumed 320 meter reading", "bill"),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_rules_fire(text, expected):
    assert classify_text(text)[0] == expected


def test_empty_text_is_unknown():
    assert classify_text("")[0] == UNKNOWN


@pytest.mark.parametrize("text", [
    "photos from the company picnic in august",   # 'pan' and 'gst' as substrings
    "a panoramic view of the mountains",          # 'pan' inside 'panoramic'
    "we spent the day at the park",
])
def test_short_terms_do_not_match_as_substrings(text):
    """Short Latin terms use whole-word matching.

    Without it, 'company' matches 'pan' and 'august' matches 'gst', which
    silently tags ordinary photos as identity documents.
    """
    assert classify_text(text)[0] == UNKNOWN


def test_more_specific_rule_wins_over_weaker_one():
    # 'passport-republic' (weight 3) must beat 'passport-keyword' (weight 1).
    doc_type, rule, weight = classify_text("republic of india passport")
    assert doc_type == "passport"
    assert weight == 3


@pytest.mark.parametrize("text", [
    "permanent account number middle surname dd mmyyyy",
    "middle surname ith dd mmiyyyy account number",   # real OCR of a blank PAN template
])
def test_account_number_alone_is_not_a_utility_bill(text):
    """'account number' must not imply a bill.

    PAN stands for 'Permanent Account Number', so a blank PAN template whose
    only legible text is its field labels was being tagged `bill`. A utility
    bill now needs a utility-specific term.
    """
    assert classify_text(text)[0] != "bill"


def test_due_date_alone_is_not_a_utility_bill():
    """Ordinary invoices carry a due date; it is not utility-specific."""
    assert classify_text("please pay by the due date shown above")[0] != "bill"


def test_genuine_utility_bill_still_matches():
    assert classify_text("electricity bill units consumed 320 meter reading")[0] == "bill"
    assert classify_text("consumer no 4412 billing period march")[0] == "bill"
