"""JSONL dataset loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievalExample:
    """One query and its expected and retrieved memory identifiers."""

    query_id: str
    relevant_ids: list[str]
    retrieved_ids: list[str]


def _string_list(value: Any, field: str, line_number: int) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"line {line_number}: '{field}' must be a list of strings")
    return value


def load_jsonl(path: str | Path) -> list[RetrievalExample]:
    """Load query examples from a JSON Lines file."""
    examples: list[RetrievalExample] = []
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: each record must be a JSON object")
            query_id = record.get("query_id")
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(f"line {line_number}: 'query_id' must be a non-empty string")
            relevant = _string_list(record.get("relevant_ids"), "relevant_ids", line_number)
            retrieved = _string_list(record.get("retrieved_ids"), "retrieved_ids", line_number)
            examples.append(RetrievalExample(query_id, relevant, retrieved))
    if not examples:
        raise ValueError(f"dataset is empty: {source}")
    return examples
