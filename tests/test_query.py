"""Router rules and FTS5 query construction."""
import pytest

from pixarch.query.bm25_search import to_match_query
from pixarch.query.router import DOCUMENT, SCENE, QueryRouter


@pytest.fixture()
def router():
    return QueryRouter()          # rules only; no CLIP fallback


@pytest.mark.parametrize("query,expected", [
    ("give me my aadhaar card", DOCUMENT),
    ("show me my passport", DOCUMENT),
    ("that electricity bill", DOCUMENT),
    ("tax invoice with gst number", DOCUMENT),
    ("find a photo of a beach", SCENE),
    ("pictures of food", SCENE),
    ("gym workout", SCENE),
])
def test_routing(router, query, expected):
    assert router.route(query).branch == expected


def test_possessive_phrase_outranks_a_scene_word(router):
    # "photo" is a scene term but "give me my ... card" is decisive.
    route = router.route("give me my aadhaar card photo")
    assert route.branch == DOCUMENT
    assert "phrase" in route.reason


def test_router_without_encoder_falls_back_to_default(router):
    route = router.route("xyzzy")
    assert route.branch == SCENE
    assert route.reason == "default"


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_yields_no_match_expression(query):
    assert to_match_query(query) == ""


def test_match_query_quotes_terms_and_strips_fts_operators():
    # Unquoted '-' and '"' are FTS5 operators and would be a syntax error.
    # 'from' is dropped as a stopword.
    got = to_match_query('receipt from 12-05 "total"')
    assert got == '"receipt" OR "12" OR "05" OR "total"'


def test_match_query_uses_or_not_and():
    """OR, because OCR reliably drops words.

    An AND over a full question returns nothing the moment one term is
    mis-recognised, which on noisy scans is most of the time.
    """
    assert " OR " in to_match_query("aadhaar card")
    assert " AND " not in to_match_query("aadhaar card")


def test_stopwords_are_dropped():
    assert to_match_query("give me my aadhaar card") == '"aadhaar" OR "card"'


def test_all_stopword_query_keeps_its_terms():
    """A query must never silently become an empty MATCH expression."""
    got = to_match_query("show me my")
    assert got and "show" in got


def test_stopwords_do_not_strip_content_words():
    got = to_match_query("total amount paid at the store")
    for word in ("total", "amount", "paid", "store"):
        assert f'"{word}"' in got
    assert '"at"' not in got and '"the"' not in got
