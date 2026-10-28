"""Retrieval metrics over graded relevance judgements (grades 0/1/2)."""
from __future__ import annotations

import math
from typing import Mapping, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Fraction of all relevant items (grade > 0) retrieved in the top k."""
    positives = {i for i, g in relevant.items() if g > 0}
    if not positives:
        return 0.0
    return len(set(ranked[:k]) & positives) / len(positives)


def precision_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for i in ranked[:k] if relevant.get(i, 0) > 0) / k


def reciprocal_rank(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """1 / rank of the first relevant hit, 0 if none inside k."""
    for idx, item in enumerate(ranked[:k], start=1):
        if relevant.get(item, 0) > 0:
            return 1.0 / idx
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Normalised DCG with the standard exponential gain (2^rel - 1)."""
    gains = [(2 ** relevant.get(i, 0)) - 1 for i in ranked[:k]]
    ideal = sorted(((2 ** g) - 1 for g in relevant.values()), reverse=True)[:k]
    denom = dcg(ideal)
    return dcg(gains) / denom if denom > 0 else 0.0


def evaluate_run(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int]],
    k_values: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Mean metrics across queries. Queries with no judgements are skipped."""
    judged = [q for q in run if qrels.get(q)]
    if not judged:
        return {}
    out: dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = sum(recall_at_k(run[q], qrels[q], k) for q in judged) / len(judged)
        # Precision matters here because recall saturates: a scene query has 10
        # relevant items, so recall@k can never exceed k/10 and recall@1=0.10
        # is the ceiling, not a failure.
        out[f"p@{k}"] = sum(precision_at_k(run[q], qrels[q], k) for q in judged) / len(judged)
        out[f"ndcg@{k}"] = sum(ndcg_at_k(run[q], qrels[q], k) for q in judged) / len(judged)
    max_k = max(k_values)
    out[f"mrr@{max_k}"] = sum(reciprocal_rank(run[q], qrels[q], max_k) for q in judged) / len(judged)
    out["n_queries"] = float(len(judged))
    return out
