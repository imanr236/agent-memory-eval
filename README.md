# agent-memory-eval

A lightweight Python toolkit for evaluating ranked retrieval from AI agent memory systems.

> **Status:** early development (v0.1.0). The project provides deterministic retrieval metrics and a simple JSONL workflow.

## Why this project?

Agent memory systems can store useful information and still retrieve the wrong memories at the wrong time. This project provides a small, reproducible evaluation layer so developers can measure retrieval quality before changing memory strategies.

## Features

- Recall@k
- Precision@k
- Reciprocal rank
- Binary Normalized Discounted Cumulative Gain (NDCG@k)
- Graded NDCG@k with linear or exponential gain
- JSONL dataset validation
- Command-line evaluation with aggregate and per-query JSON output
- Unit tests and GitHub Actions CI

The JSONL CLI workflow currently uses binary relevance: each memory identifier is either relevant or not relevant for a query. Graded relevance is available through the Python API.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
```

## Quick start

Each JSONL line represents one query. It must include:

- `query_id`: a non-empty string
- `relevant_ids`: list of memory IDs judged relevant to the query
- `retrieved_ids`: list of memory IDs in the order returned by the system

Example:

```json
{"query_id":"q1","relevant_ids":["memory-a","memory-c"],"retrieved_ids":["memory-a","memory-b","memory-c"]}
```

Run the included example:

```bash
agent-memory-eval examples/sample.jsonl -k 5
```

Or write the report to a file:

```bash
agent-memory-eval examples/sample.jsonl -k 3 --output report.json
```

## Python API

### Binary relevance

```python
from agent_memory_eval import evaluate, recall_at_k

relevant = ["memory-a", "memory-c"]
retrieved = ["memory-a", "memory-b", "memory-c"]

print(recall_at_k(relevant, retrieved, k=2))  # 0.5

result = evaluate("q1", relevant, retrieved, k=3)
print(result.to_dict())
```

### Graded relevance

Use a score to express how useful each memory is for a query. Higher scores mean stronger relevance.

```python
from agent_memory_eval import graded_ndcg_at_k

relevance = {
    "memory-a": 3,  # highly relevant
    "memory-b": 2,  # partially relevant
    "memory-c": 1,  # weakly relevant
}
retrieved = ["memory-b", "memory-a", "memory-c"]

score = graded_ndcg_at_k(relevance, retrieved, k=3)
print(score)
```

The default `gain="exponential"` uses `2**relevance - 1`, giving stronger preference to highly relevant memories. Use `gain="linear"` to use raw relevance scores instead. Scores must be finite and non-negative. Exponential scores are limited to 1023 to avoid numeric overflow.

## Metric definitions

| Metric | Meaning |
| --- | --- |
| Recall@k | Relevant memories found in the first k results divided by all relevant memories |
| Precision@k | Relevant results in the first k slots divided by k |
| Reciprocal rank | 1 / rank of the first relevant result, or 0 if none is found |
| NDCG@k | Discounted binary gain normalized by the ideal ranking |
| Graded NDCG@k | Discounted relevance gain normalized by the ideal graded ranking |

Duplicate retrieved IDs are ignored after their first occurrence. For Precision@k, unfilled slots count as non-relevant. Empty ground-truth relevance sets return 0 for binary metrics; an all-zero graded relevance map returns 0 for graded NDCG.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Roadmap

- [x] Add graded relevance and configurable gain functions
- [ ] Add dataset-level confidence intervals and bootstrap comparisons
- [ ] Add memory update and temporal consistency evaluation tasks
- [ ] Add adapters for common agent memory stores
- [ ] Publish package documentation and a versioned release

## Contributing

Issues and pull requests are welcome. Please include tests for behavior changes and describe metric assumptions in the pull request.

## License

This project is licensed under the [MIT License](LICENSE).
