"""Metrics checked against hand-computed values."""
import math

from pixarch.eval.metrics import (
    dcg, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank, evaluate_run,
)

QRELS = {"a": 2, "b": 1, "c": 0}


def test_recall_counts_only_positive_grades():
    assert recall_at_k(["a", "c"], QRELS, 2) == 0.5      # a of {a,b}
    assert recall_at_k(["a", "b"], QRELS, 2) == 1.0
    assert recall_at_k(["c"], QRELS, 1) == 0.0


def test_recall_with_no_positives_is_zero():
    assert recall_at_k(["a"], {"a": 0}, 1) == 0.0


def test_precision_at_k():
    assert precision_at_k(["a", "c"], QRELS, 2) == 0.5
    assert precision_at_k(["a", "b"], QRELS, 2) == 1.0


def test_reciprocal_rank_is_first_hit():
    assert reciprocal_rank(["c", "a"], QRELS, 10) == 0.5
    assert reciprocal_rank(["a", "c"], QRELS, 10) == 1.0
    assert reciprocal_rank(["c"], QRELS, 10) == 0.0


def test_dcg_matches_hand_calculation():
    # gains 3, 1 at ranks 1, 2 -> 3/log2(2) + 1/log2(3)
    assert math.isclose(dcg([3, 1]), 3 / 1.0 + 1 / math.log2(3))


def test_ndcg_perfect_ordering_is_one():
    assert math.isclose(ndcg_at_k(["a", "b", "c"], QRELS, 3), 1.0)


def test_ndcg_penalises_inverted_ordering():
    # ideal = [3,1]; actual = [1,3] -> (1 + 3/log2(3)) / (3 + 1/log2(3))
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert math.isclose(ndcg_at_k(["b", "a"], QRELS, 2), expected)


def test_ndcg_is_zero_when_nothing_relevant():
    assert ndcg_at_k(["c"], {"c": 0}, 1) == 0.0


def test_evaluate_run_skips_unjudged_queries():
    run = {"q1": ["a", "b"], "q2": ["x"]}
    qrels = {"q1": QRELS, "q2": {}}
    got = evaluate_run(run, qrels, k_values=(1, 2))
    assert got["n_queries"] == 1.0
    assert got["recall@2"] == 1.0


def test_precision_is_reported_and_recall_saturates():
    """With many relevant items, recall@k is capped at k/|relevant|.

    This is why precision leads the results table: a scene query with 10
    relevant items can never score above recall@1 = 0.1, which reads as a
    failure when it is in fact a perfect result.
    """
    qrels = {f"i{n}": 2 for n in range(10)}
    run = {"q": [f"i{n}" for n in range(10)]}        # a perfect ranking
    got = evaluate_run(run, {"q": qrels}, k_values=(1, 5))
    assert got["p@1"] == 1.0          # perfect, and reads as perfect
    assert got["p@5"] == 1.0
    assert got["recall@1"] == 0.1     # also perfect, but reads as failure
    assert got["recall@5"] == 0.5
