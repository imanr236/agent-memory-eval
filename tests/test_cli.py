import json

from agent_memory_eval.cli import main


def test_cli_prints_aggregate_report(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"query_id":"q1","relevant_ids":["m1"],"retrieved_ids":["m1","m2"]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["agent-memory-eval", str(dataset), "--k", "2"])
    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["num_queries"] == 1
    assert report["aggregate"]["recall"] == 1.0
    assert report["aggregate"]["precision"] == 0.5


def test_cli_writes_report_to_file(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "report.json"
    dataset.write_text(
        '{"query_id":"q1","relevant_ids":["m1"],"retrieved_ids":["m1"]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["agent-memory-eval", str(dataset), "--output", str(output)],
    )
    assert main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["num_queries"] == 1
