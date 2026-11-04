import math

import pytest

from agent_memory_eval.metrics import (
    evaluate,
    graded_ndcg_at_k,
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
    assert recall_at_k(["b"], ["a", "a", "b"], 2) == 1.0
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


def test_graded_ndcg_perfect_ranking_is_one():
    assert graded_ndcg_at_k(
        {"high": 3, "medium": 2, "low": 1}, ["high", "medium", "low"], 3
    ) == pytest.approx(1.0)


def test_graded_ndcg_penalizes_high_relevance_at_lower_rank():
    score = graded_ndcg_at_k({"high": 3, "low": 1}, ["low", "high"], 2)
    assert 0.0 < score < 1.0


def test_graded_ndcg_supports_linear_gain():
    score = graded_ndcg_at_k({"high": 3, "low": 1}, ["high", "low"], 2, gain="linear")
    assert score == pytest.approx(1.0)


def test_graded_ndcg_ignores_duplicate_retrieved_ids():
    score = graded_ndcg_at_k({"a": 2, "b": 1}, ["a", "a", "b"], 2)
    assert score == pytest.approx(1.0)


def test_graded_ndcg_returns_zero_for_zero_gain():
    assert graded_ndcg_at_k({"a": 0, "b": 0}, ["a", "b"], 2) == 0.0


@pytest.mark.parametrize(
    "relevance",
    [
        {"a": -1},
        {"a": float("nan")},
        {"a": float("inf")},
        {"a": True},
        {" ": 1},
    ],
)
def test_graded_ndcg_rejects_invalid_relevance(relevance):
    with pytest.raises(ValueError):
        graded_ndcg_at_k(relevance, ["a"], 1)


def test_graded_ndcg_rejects_unknown_gain():
    with pytest.raises(ValueError, match="gain"):
        graded_ndcg_at_k({"a": 1}, ["a"], 1, gain="unknown")


def test_graded_ndcg_rejects_excessive_exponential_score():
    with pytest.raises(ValueError, match="<= 1023"):
        graded_ndcg_at_k({"a": 1024}, ["a"], 1)
