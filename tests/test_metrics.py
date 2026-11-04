import math

import pytest

from agent_memory_eval.metrics import (
    evaluate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["x", "a", "b"], 2) == pytest.approx(1 / 3)


def test_precision_at_k_uses_k_as_denominator():
    assert precision_at_k(["a"], ["a"], 3) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_first_relevant_result():
    assert reciprocal_rank(["b", "c"], ["x", "c", "b"]) == pytest.approx(1 / 2)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], ["a", "b", "x"], 3) == pytest.approx(1.0)


def test_ndcg_penalizes_lower_rank():
    assert ndcg_at_k(["a"], ["x", "a"], 2) == pytest.approx(1 / math.log2(3))


def test_empty_relevant_set_returns_zero():
    assert recall_at_k([], ["a"], 5) == 0.0
    assert precision_at_k([], ["a"], 5) == 0.0
    assert reciprocal_rank([], ["a"]) == 0.0
    assert ndcg_at_k([], ["a"], 5) == 0.0


def test_duplicate_retrievals_are_not_double_counted():
    assert recall_at_k(["a", "b"], ["a", "a", "b"], 3) == 1.0
    assert reciprocal_rank(["a"], ["x", "a", "a"]) == 0.5


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k_raises(k):
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], k)


def test_evaluate_returns_metric_bundle():
    result = evaluate("q1", ["m1"], ["m1", "m2"], k=2)
    assert result.query_id == "q1"
    assert result.recall == 1.0
    assert result.precision == 0.5
    assert result.reciprocal_rank == 1.0
    assert result.ndcg == 1.0
