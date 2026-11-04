import json

import pytest

from agent_memory_eval.dataset import load_jsonl


def test_load_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"query_id": "q1", "relevant_ids": ["m1"], "retrieved_ids": ["m1", "m2"]})
        + "\n",
        encoding="utf-8",
    )
    examples = load_jsonl(path)
    assert len(examples) == 1
    assert examples[0].query_id == "q1"
    assert examples[0].relevant_ids == ["m1"]


@pytest.mark.parametrize(
    "line, message",
    [
        ("{bad json}", "invalid JSON"),
        ('{"query_id":"q1","relevant_ids":"m1","retrieved_ids":[]}', "relevant_ids"),
        ('{"query_id":"","relevant_ids":[],"retrieved_ids":[]}', "query_id"),
        ("[]", "JSON object"),
    ],
)
def test_invalid_records_raise_clear_error(tmp_path, line, message):
    path = tmp_path / "bad.jsonl"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_jsonl(path)


def test_empty_dataset_raises(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_jsonl(path)
