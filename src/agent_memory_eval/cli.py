"""Command-line interface for agent-memory-eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_memory_eval.dataset import load_jsonl
from agent_memory_eval.metrics import evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-memory-eval",
        description="Evaluate ranked memory retrieval results from a JSONL dataset.",
    )
    parser.add_argument("dataset", type=Path, help="Path to a JSONL evaluation dataset")
    parser.add_argument("-k", type=int, default=5, help="Cutoff for @k metrics (default: 5)")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.k < 1:
        print("error: -k must be a positive integer", file=sys.stderr)
        return 2
    try:
        examples = load_jsonl(args.dataset)
        results = [
            evaluate(example.query_id, example.relevant_ids, example.retrieved_ids, args.k)
            for example in examples
        ]
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    names = ("recall", "precision", "reciprocal_rank", "ndcg")
    report = {
        "dataset": str(args.dataset),
        "k": args.k,
        "num_queries": len(results),
        "aggregate": {
            name: round(sum(getattr(result, name) for result in results) / len(results), 6)
            for name in names
        },
        "per_query": [result.to_dict() for result in results],
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        try:
            args.output.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            print(f"error writing report: {exc}", file=sys.stderr)
            return 2
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
