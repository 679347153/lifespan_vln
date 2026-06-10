#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from benchmark.metrics import EpisodeResult, aggregate_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate benchmark results")
    p.add_argument("--input", required=True, help="Path to episode_results.jsonl")
    p.add_argument("--output", default="benchmark/eval/summary.json")
    return p.parse_args()


def parse_line(data: dict) -> EpisodeResult:
    return EpisodeResult(
        success=bool(data.get("success", False)),
        path_length=float(data.get("path_length", 0.0)),
        shortest_path_length=float(data.get("shortest_path_length", 0.0)),
        elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
        dynamic_memory_correct=int(data.get("dynamic_memory_correct", 0)),
        dynamic_memory_total=int(data.get("dynamic_memory_total", 0)),
        fixed_memory_correct=int(data.get("fixed_memory_correct", 0)),
        fixed_memory_total=int(data.get("fixed_memory_total", 0)),
        steps=int(data.get("steps", 0)),
    )


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    results = []
    with in_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(parse_line(json.loads(line)))

    summary = aggregate_metrics(results)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[Done] Evaluation summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
