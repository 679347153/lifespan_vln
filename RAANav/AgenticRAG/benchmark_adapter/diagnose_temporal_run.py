from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .common import as_path, is_noise_detection_label, sanitize_detection_label, write_json


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
        if key == "label":
            value = sanitize_detection_label(value)
            if is_noise_detection_label(value):
                continue
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
        label = sanitize_detection_label(record.get("label"))
        if is_noise_detection_label(label):
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
            selected_label = sanitize_detection_label(selected.get("label"))
            if selected_label and not is_noise_detection_label(selected_label):
                selected_labels[selected_label] += 1
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
    subtasks = len(records)
    found = sum(1 for r in records if bool(r.get("found")))
    perception_found = sum(1 for r in records if bool(r.get("perception_found")))
    mra = sum(1 for r in records if bool(r.get("mra")))
    ghr3 = sum(1 for r in records if bool(r.get("ghr3")))
    ghr5 = sum(1 for r in records if bool(r.get("ghr5")))
    avg_final_dist = _mean(final_dists)
    avg_candidate_min_dist = _mean(min_dists)
    return {
        "subtasks": subtasks,
        "found": found,
        "found_rate": round(found / subtasks, 6) if subtasks else None,
        "perception_found": perception_found,
        "perception_found_rate": round(perception_found / subtasks, 6) if subtasks else None,
        "perception_to_benchmark_found_gap": perception_found - found,
        "mra": mra,
        "mra_rate": round(mra / subtasks, 6) if subtasks else None,
        "ghr3": ghr3,
        "ghr3_rate": round(ghr3 / subtasks, 6) if subtasks else None,
        "ghr5": ghr5,
        "ghr5_rate": round(ghr5 / subtasks, 6) if subtasks else None,
        "ghr5_to_found_gap": ghr5 - found,
        "avg_final_dist": avg_final_dist,
        "best_final_dist": min([v for v in final_dists if v is not None], default=None),
        "avg_candidate_min_dist": avg_candidate_min_dist,
        "best_candidate_min_dist": min([v for v in min_dists if v is not None], default=None),
        "avg_final_minus_candidate_min_dist": (
            round(float(avg_final_dist) - float(avg_candidate_min_dist), 6)
            if avg_final_dist is not None and avg_candidate_min_dist is not None
            else None
        ),
        "avg_fallback_count": _mean(fallback_counts),
        "by_task_type": _count_by(records, "task_type"),
        "by_query_modality": _count_by(records, "query_modality"),
        "by_seen_layout_count_before": _count_by(records, "seen_layout_count_before"),
        "targets": _count_by(records, "target_object"),
    }


def _memory_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_event = _count_by(records, "event")
    by_label = _count_by(records, "label")
    migrated_room_events = 0
    negative_room_feedback_events = 0
    room_event_counter: Counter[str] = Counter()
    for record in records:
        details = record.get("details")
        if not isinstance(details, dict):
            continue
        room_id = details.get("room_id") or details.get("current_room_id")
        if room_id:
            room_event_counter[str(room_id)] += 1
        if record.get("event") == "migrated" and details.get("previous_room_id") != details.get("current_room_id"):
            migrated_room_events += 1
        if record.get("event") == "negative_feedback" and details.get("room_id"):
            negative_room_feedback_events += 1
    return {
        "memory_events": len(records),
        "by_event": by_event,
        "top_labels": dict(list(by_label.items())[:20]),
        "migrated_room_events": migrated_room_events,
        "negative_room_feedback_events": negative_room_feedback_events,
        "top_rooms_by_event_count": dict(room_event_counter.most_common(20)),
    }


def _scene_memory_stats(output_dir: Path) -> Dict[str, Any]:
    scene_root = output_dir / "scene_memory"
    room_count = 0
    objects_with_room_belief = 0
    top_rooms: Counter[str] = Counter()
    scene_count = 0
    if not scene_root.exists():
        return {
            "scene_memory_files": 0,
            "room_count": 0,
            "objects_with_room_belief": 0,
            "top_rooms_by_track_count": {},
        }
    for path in scene_root.rglob("scene_memory_final.json"):
        payload = _read_json_optional(path) or {}
        scene_count += 1
        rooms = payload.get("rooms")
        if isinstance(rooms, dict):
            room_count += len(rooms)
            for room_id, room in rooms.items():
                if not isinstance(room, dict):
                    continue
                refs = room.get("object_refs")
                if isinstance(refs, dict):
                    top_rooms[str(room_id)] += len(refs)
        for track in payload.get("tracks", []) or []:
            if not isinstance(track, dict):
                continue
            stats = track.get("cooccur_stats")
            if isinstance(stats, dict) and stats.get("room_belief"):
                objects_with_room_belief += 1
    return {
        "scene_memory_files": scene_count,
        "room_count": room_count,
        "objects_with_room_belief": objects_with_room_belief,
        "top_rooms_by_track_count": dict(top_rooms.most_common(20)),
    }


def _rate_value(record: Dict[str, Any], key: str) -> Optional[float]:
    value = _safe_float(record.get(key))
    return value if value is not None else None


def _temporal_trend(temporal_summary: Dict[str, Any]) -> Dict[str, Any]:
    by_seen = temporal_summary.get("by_seen_layout_count_before")
    if not isinstance(by_seen, dict) or not by_seen:
        return {}
    rows: List[Dict[str, Any]] = []
    for key, value in by_seen.items():
        if not isinstance(value, dict):
            continue
        try:
            seen = int(key)
        except Exception:
            continue
        rows.append({"seen_layout_count_before": seen, **value})
    rows.sort(key=lambda item: int(item["seen_layout_count_before"]))
    if not rows:
        return {}
    first = rows[0]
    last = rows[-1]
    best_ghr5 = max(rows, key=lambda item: float(item.get("ghr5", 0.0) or 0.0))
    best_min_dist = min(rows, key=lambda item: float(item.get("avg_min_dist", float("inf")) or float("inf")))
    first_found = _rate_value(first, "found_rate")
    last_found = _rate_value(last, "found_rate")
    first_ghr5 = _rate_value(first, "ghr5")
    last_ghr5 = _rate_value(last, "ghr5")
    first_min = _rate_value(first, "avg_min_dist")
    last_min = _rate_value(last, "avg_min_dist")
    return {
        "first_seen": first,
        "last_seen": last,
        "best_ghr5_seen": best_ghr5,
        "best_avg_min_dist_seen": best_min_dist,
        "delta_found_rate_last_minus_first": (
            round(float(last_found) - float(first_found), 6)
            if first_found is not None and last_found is not None
            else None
        ),
        "delta_ghr5_last_minus_first": (
            round(float(last_ghr5) - float(first_ghr5), 6)
            if first_ghr5 is not None and last_ghr5 is not None
            else None
        ),
        "delta_avg_min_dist_last_minus_first": (
            round(float(last_min) - float(first_min), 6)
            if first_min is not None and last_min is not None
            else None
        ),
        "late_memory_improvement_signal": bool(
            last_found is not None
            and first_found is not None
            and last_ghr5 is not None
            and first_ghr5 is not None
            and last_min is not None
            and first_min is not None
            and (last_found > first_found or last_ghr5 > first_ghr5)
            and last_min < first_min
        ),
    }


def _failure_flags(
    summary: Dict[str, Any],
    diagnostics: Dict[str, Any],
    candidates: Dict[str, Any],
    observations: Dict[str, Any],
    memory: Dict[str, Any],
    feedback_count: int,
    temporal_trend: Dict[str, Any],
) -> List[str]:
    flags: List[str] = []
    if int(observations.get("detections", 0) or 0) <= 0:
        flags.append("perception_empty")
    if int(memory.get("memory_events", 0) or 0) <= 0:
        flags.append("memory_not_updated")
    if int(candidates.get("candidate_rounds", 0) or 0) > 0 and int(candidates.get("rounds_with_candidates", 0) or 0) <= 0:
        flags.append("retrieval_no_candidates")
    if int(diagnostics.get("ghr5", 0) or 0) <= 0:
        flags.append("candidates_far_from_goal")
    if int(diagnostics.get("ghr5_to_found_gap", 0) or 0) > 0:
        flags.append("candidate_near_but_not_converted_to_success")
    if int(diagnostics.get("perception_to_benchmark_found_gap", 0) or 0) >= max(5, int(diagnostics.get("found", 0) or 0) * 3):
        flags.append("perception_goal_mismatch_or_false_positive_confirmation")
    if float(summary.get("success_rate", 0.0) or 0.0) <= 0.0 and feedback_count > 0:
        flags.append("negative_feedback_after_failed_candidates")
    if temporal_trend.get("late_memory_improvement_signal"):
        flags.append("late_memory_improvement_signal")
    if float(summary.get("success_rate", 0.0) or 0.0) > 0.0 and not flags:
        flags.append("healthy")
    return flags


def _failure_bucket(
    summary: Dict[str, Any],
    diagnostics: Dict[str, Any],
    candidates: Dict[str, Any],
    observations: Dict[str, Any],
    memory: Dict[str, Any],
    feedback_count: int,
    temporal_trend: Dict[str, Any],
) -> str:
    flags = _failure_flags(summary, diagnostics, candidates, observations, memory, feedback_count, temporal_trend)
    if int(observations.get("detections", 0) or 0) <= 0:
        return "perception_empty"
    if int(memory.get("memory_events", 0) or 0) <= 0:
        return "memory_not_updated"
    if int(candidates.get("candidate_rounds", 0) or 0) > 0 and int(candidates.get("rounds_with_candidates", 0) or 0) <= 0:
        return "retrieval_no_candidates"
    if int(diagnostics.get("ghr5", 0) or 0) <= 0:
        return "candidates_far_from_goal"
    if "perception_goal_mismatch_or_false_positive_confirmation" in flags:
        return "perception_goal_mismatch_or_false_positive_confirmation"
    if "candidate_near_but_not_converted_to_success" in flags:
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
    memory_stats.update(_scene_memory_stats(output_dir))
    temporal_trend = _temporal_trend(temporal_summary)
    failure_flags = _failure_flags(
        summary,
        diagnostic_stats,
        candidate_stats,
        observation_stats,
        memory_stats,
        len(feedback_records),
        temporal_trend,
    )

    result = {
        "output_dir": str(output_dir),
        "benchmark_summary": summary,
        "temporal_summary": temporal_summary,
        "temporal_trend": temporal_trend,
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
        "failure_flags": failure_flags,
        "failure_bucket": _failure_bucket(
            summary,
            diagnostic_stats,
            candidate_stats,
            observation_stats,
            memory_stats,
            len(feedback_records),
            temporal_trend,
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
