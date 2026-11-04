# agent-memory-eval

A lightweight Python toolkit for evaluating ranked retrieval from AI agent memory systems.

> **Status:** early development (v0.1.0). The first release focuses on transparent, deterministic retrieval metrics and a simple JSONL workflow.

## Why this project?

Agent memory systems can store useful information and still retrieve the wrong memories at the wrong time. This project provides a small, reproducible evaluation layer so developers can measure retrieval quality before changing memory strategies.

## Features

- Recall@k
- Precision@k
- Reciprocal rank
- Normalized Discounted Cumulative Gain (NDCG@k)
- JSONL dataset validation
- Command-line evaluation with aggregate and per-query JSON output
- Unit tests and GitHub Actions CI

The initial release uses binary relevance: each memory identifier is either relevant or not relevant for a query.

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
agent-memory-eval examples/sample.jsonl --k 5
```

Or write the report to a file:

```bash
agent-memory-eval examples/sample.jsonl --k 3 --output report.json
```

## Python API

```python
from agent_memory_eval import evaluate, recall_at_k

relevant = ["memory-a", "memory-c"]
retrieved = ["memory-a", "memory-b", "memory-c"]

print(recall_at_k(relevant, retrieved, k=2))  # 0.5

result = evaluate("q1", relevant, retrieved, k=3)
print(result.to_dict())
```

## Metric definitions

| Metric | Meaning |
| --- | --- |
| Recall@k | Relevant memories found in the first k results divided by all relevant memories |
| Precision@k | Relevant results in the first k slots divided by k |
| Reciprocal rank | 1 / rank of the first relevant result, or 0 if none is found |
| NDCG@k | Discounted gain of the ranking normalized by the ideal ranking |

Duplicate retrieved IDs are ignored after their first occurrence. For Precision@k, unfilled slots count as non-relevant. Empty ground-truth relevance sets return 0 for all metrics; datasets should ideally avoid such rows unless they are intentional no-answer cases.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Roadmap

- [ ] Add graded relevance and configurable gain functions
- [ ] Add dataset-level confidence intervals and bootstrap comparisons
- [ ] Add memory update and temporal consistency evaluation tasks
- [ ] Add adapters for common agent memory stores
- [ ] Publish package documentation and a versioned release

## Contributing

Issues and pull requests are welcome. Please include tests for behavior changes and describe metric assumptions in the pull request.

## License

No license has been selected yet. Until a license is added, default copyright rules apply and reuse permissions should not be assumed.
