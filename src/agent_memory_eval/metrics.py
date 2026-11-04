"""Ranking metrics for evaluating memory retrieval."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite, log2
from typing import Literal


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def _unique(items: Iterable[str]) -> list[str]:
    """Preserve ranking order while removing duplicate identifiers."""
    return list(dict.fromkeys(items))


def recall_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Fraction of relevant identifiers found in the first k results."""
    _validate_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(relevant_set.intersection(_unique(retrieved)[:k])) / len(relevant_set)


def precision_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Fraction of the first k ranking slots that are relevant."""
    _validate_k(k)
    relevant_set = set(relevant)
    ranked = _unique(retrieved)[:k]
    return sum(item in relevant_set for item in ranked) / k


def reciprocal_rank(relevant: Iterable[str], retrieved: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for rank, item in enumerate(_unique(retrieved), start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: Iterable[str], retrieved: Sequence[str], k: int) -> float:
    """Binary-relevance normalized discounted cumulative gain."""
    _validate_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    ranked = _unique(retrieved)[:k]
    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, item in enumerate(ranked, start=1)
        if item in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def graded_ndcg_at_k(
    relevance: Mapping[str, int | float],
    retrieved: Sequence[str],
    k: int,
    gain: Literal["linear", "exponential"] = "exponential",
) -> float:
    """NDCG@k for graded relevance scores.

    Relevance scores must be finite, non-negative numbers. The default
    exponential gain is 2**score - 1; linear gain uses the score unchanged.
    Duplicate retrieved IDs count only at their first occurrence.
    """
    _validate_k(k)
    if gain not in ("linear", "exponential"):
        raise ValueError("gain must be 'linear' or 'exponential'")

    scores: dict[str, float] = {}
    for item, score in relevance.items():
        if (
            not isinstance(item, str)
            or not item.strip()
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not isfinite(score)
            or score < 0
        ):
            raise ValueError("relevance must map non-empty IDs to finite non-negative numbers")
        if gain == "exponential" and score > 1023:
            raise ValueError("exponential gain scores must be <= 1023")
        scores[item] = float(score)

    def to_gain(score: float) -> float:
        return score if gain == "linear" else 2.0**score - 1.0

    ideal_gains = sorted((to_gain(score) for score in scores.values()), reverse=True)[:k]
    ideal_dcg = sum(value / log2(rank + 1) for rank, value in enumerate(ideal_gains, start=1))
    if ideal_dcg == 0.0:
        return 0.0

    ranked = _unique(retrieved)[:k]
    dcg = sum(
        to_gain(scores[item]) / log2(rank + 1)
        for rank, item in enumerate(ranked, start=1)
        if item in scores
    )
    return dcg / ideal_dcg


@dataclass(frozen=True)
class EvaluationResult:
    """Metric values for one query."""

    query_id: str
    recall: float
    precision: float
    reciprocal_rank: float
    ndcg: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


def evaluate(
    query_id: str,
    relevant: Iterable[str],
    retrieved: Sequence[str],
    k: int = 5,
) -> EvaluationResult:
    """Calculate the standard metric bundle for one query."""
    relevant_items = list(relevant)
    return EvaluationResult(
        query_id=query_id,
        recall=recall_at_k(relevant_items, retrieved, k),
        precision=precision_at_k(relevant_items, retrieved, k),
        reciprocal_rank=reciprocal_rank(relevant_items, retrieved),
        ndcg=ndcg_at_k(relevant_items, retrieved, k),
    )
