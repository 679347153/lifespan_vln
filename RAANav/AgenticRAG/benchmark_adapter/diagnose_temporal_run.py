from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .common import as_path, write_json


def _read_json_optional(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(item, dict):
                records.append(item)
    return records


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _count_by(records: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        value = record.get(key)
        if value is not None:
            counter[str(value)] += 1
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 6)


def _top_labels(records: Iterable[Dict[str, Any]], *, limit: int = 20) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    clip_dims: Counter[str] = Counter()
    positioned = 0
    for record in records:
        label = str(record.get("label") or "").strip()
        if not label:
            continue
        counter[label] += 1
        if record.get("pos_2d") is not None or record.get("pos_3d") is not None:
            positioned += 1
        dim = record.get("clip_embedding_dim")
        if dim is not None:
            clip_dims[str(dim)] += 1
    return [
        {"label": label, "count": count}
        for label, count in counter.most_common(limit)
    ], dict(sorted(clip_dims.items())), positioned


def _candidate_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    rounds = len(records)
    rounds_with_candidates = 0
    selected_backend: Counter[str] = Counter()
    selected_labels: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    selected_scores: List[Optional[float]] = []
    candidate_counts: List[int] = []

    for record in records:
        top = record.get("top_candidates")
        if not isinstance(top, list):
            top = []
        candidate_counts.append(len(top))
        if top:
            rounds_with_candidates += 1
        selected = record.get("selected_candidate")
        if isinstance(selected, dict):
            selected_backend[str(selected.get("sim_backend") or "unknown")] += 1
            selected_labels[str(selected.get("label") or "unknown")] += 1
            selected_scores.append(_safe_float(selected.get("S_final")))
        reason = record.get("fallback_reason")
        if reason:
            fallback_reasons[str(reason)] += 1

    return {
        "candidate_rounds": rounds,
        "rounds_with_candidates": rounds_with_candidates,
        "rounds_without_candidates": rounds - rounds_with_candidates,
        "candidate_recall_rate": round(rounds_with_candidates / rounds, 6) if rounds else None,
        "avg_candidates_per_round": _mean([float(x) for x in candidate_counts]),
        "avg_selected_score": _mean(selected_scores),
        "fallback_reasons": dict(fallback_reasons),
        "selected_backend": dict(selected_backend),
        "selected_labels": dict(selected_labels.most_common(20)),
    }


def _diagnostic_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    final_dists = [_safe_float(r.get("final_dist")) for r in records]
    min_dists = [_safe_float(r.get("min_dist")) for r in records]
    fallback_counts = [_safe_float(r.get("fallback_count"), 0.0) for r in records]
    return {
        "subtasks": len(records),
        "found": sum(1 for r in records if bool(r.get("found"))),
        "perception_found": sum(1 for r in records if bool(r.get("perception_found"))),
        "mra": sum(1 for r in records if bool(r.get("mra"))),
        "ghr3": sum(1 for r in records if bool(r.get("ghr3"))),
        "ghr5": sum(1 for r in records if bool(r.get("ghr5"))),
        "avg_final_dist": _mean(final_dists),
        "best_final_dist": min([v for v in final_dists if v is not None], default=None),
        "avg_candidate_min_dist": _mean(min_dists),
        "best_candidate_min_dist": min([v for v in min_dists if v is not None], default=None),
        "avg_fallback_count": _mean(fallback_counts),
        "by_task_type": _count_by(records, "task_type"),
        "by_query_modality": _count_by(records, "query_modality"),
        "by_seen_layout_count_before": _count_by(records, "seen_layout_count_before"),
        "targets": _count_by(records, "target_object"),
    }


def _memory_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_event = _count_by(records, "event")
    by_label = _count_by(records, "label")
    return {
        "memory_events": len(records),
        "by_event": by_event,
        "top_labels": dict(list(by_label.items())[:20]),
    }


def _failure_bucket(
    summary: Dict[str, Any],
    diagnostics: Dict[str, Any],
    candidates: Dict[str, Any],
    observations: Dict[str, Any],
    memory: Dict[str, Any],
    feedback_count: int,
) -> str:
    if int(observations.get("detections", 0) or 0) <= 0:
        return "perception_empty"
    if int(memory.get("memory_events", 0) or 0) <= 0:
        return "memory_not_updated"
    if int(candidates.get("candidate_rounds", 0) or 0) > 0 and int(candidates.get("rounds_with_candidates", 0) or 0) <= 0:
        return "retrieval_no_candidates"
    if int(diagnostics.get("ghr5", 0) or 0) <= 0:
        return "candidates_far_from_goal"
    if int(diagnostics.get("ghr5", 0) or 0) > 0 and int(diagnostics.get("found", 0) or 0) <= 0:
        return "candidate_near_but_navigation_or_success_radius_failed"
    if float(summary.get("success_rate", 0.0) or 0.0) <= 0.0 and feedback_count > 0:
        return "negative_feedback_after_failed_candidates"
    if float(summary.get("success_rate", 0.0) or 0.0) <= 0.0:
        return "unsuccessful_unknown"
    return "healthy"


def diagnose(output_dir: Path) -> Dict[str, Any]:
    output_dir = as_path(output_dir)
    summary = _read_json_optional(output_dir / "summary.json") or {}
    temporal_summary = _read_json_optional(output_dir / "temporal_summary.json") or {}
    perception_summary = _read_json_optional(output_dir / "perception_summary.json") or {}
    diagnostics_records = _read_jsonl(output_dir / "memory_diagnostics.jsonl")
    candidate_records = _read_jsonl(output_dir / "candidate_traces.jsonl")
    observation_records = _read_jsonl(output_dir / "observation_traces.jsonl")
    memory_records = _read_jsonl(output_dir / "memory_updates.jsonl")
    feedback_records = _read_jsonl(output_dir / "negative_feedback.jsonl")
    transition_records = _read_jsonl(output_dir / "layout_transitions.jsonl")
    episode_results = _read_jsonl(output_dir / "episode_results.jsonl")

    top_labels, clip_dims, positioned = _top_labels(observation_records)
    observation_stats = {
        "detections": len(observation_records),
        "positioned_detections": positioned,
        "clip_embedding_dims": clip_dims,
        "top_labels": top_labels,
        "perception_summary": perception_summary,
    }
    diagnostic_stats = _diagnostic_stats(diagnostics_records)
    candidate_stats = _candidate_stats(candidate_records)
    memory_stats = _memory_stats(memory_records)

    result = {
        "output_dir": str(output_dir),
        "benchmark_summary": summary,
        "temporal_summary": temporal_summary,
        "episodes": len(episode_results) or summary.get("evaluated_episodes"),
        "layout_transitions": len(transition_records),
        "observations": observation_stats,
        "diagnostics": diagnostic_stats,
        "candidates": candidate_stats,
        "memory": memory_stats,
        "negative_feedback": {
            "events": len(feedback_records),
            "by_label": _count_by(feedback_records, "label"),
        },
        "failure_bucket": _failure_bucket(
            summary,
            diagnostic_stats,
            candidate_stats,
            observation_stats,
            memory_stats,
            len(feedback_records),
        ),
    }
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnose run_temporal_habitat_loop output files.")
    parser.add_argument("--output-dir", required=True, help="Output directory produced by run_temporal_habitat_loop.")
    parser.add_argument("--write-json", default="", help="Optional path for a diagnosis JSON report.")
    args = parser.parse_args(argv)
    report = diagnose(Path(args.output_dir))
    if args.write_json:
        write_json(args.write_json, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
