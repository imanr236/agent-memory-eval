"""Ranking metrics for evaluating memory retrieval."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from math import log2


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
