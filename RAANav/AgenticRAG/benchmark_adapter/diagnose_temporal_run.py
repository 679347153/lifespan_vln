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
    selected_match_strength: Counter[str] = Counter()
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
            selected_match_strength[str(selected.get("label_alias_match_strength") or "unknown")] += 1
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
        "selected_match_strength": dict(selected_match_strength.most_common()),
    }


def _subtask_key(record: Dict[str, Any]) -> str:
    episode_id = record.get("episode_id")
    subtask_id = record.get("subtask_id")
    if episode_id is None and subtask_id is None:
        return ""
    return f"{episode_id}::{subtask_id}"


def _candidate_list(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    top = record.get("top_candidates")
    if isinstance(top, list):
        return [item for item in top if isinstance(item, dict)]
    selected = record.get("selected_candidate")
    if isinstance(selected, dict):
        return [selected]
    candidate = record.get("candidate")
    if isinstance(candidate, dict):
        return [candidate]
    return []


def _selected_candidate(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    selected = record.get("selected_candidate")
    if isinstance(selected, dict):
        return selected
    top = _candidate_list(record)
    if top:
        return top[0]
    candidate = record.get("candidate")
    if isinstance(candidate, dict):
        return candidate
    return None


def _success_gap_decision(diagnostics: Dict[str, Any], candidates: Dict[str, Any]) -> Dict[str, Any]:
    subtasks = int(diagnostics.get("subtasks", 0) or 0)
    found_rate = _safe_float(diagnostics.get("found_rate"), 0.0) or 0.0
    ghr5_rate = _safe_float(diagnostics.get("ghr5_rate"), 0.0) or 0.0
    perception_found_rate = _safe_float(diagnostics.get("perception_found_rate"), 0.0) or 0.0
    candidate_recall_rate = _safe_float(candidates.get("candidate_recall_rate"), 0.0) or 0.0
    ghr5_to_found_gap = int(diagnostics.get("ghr5_to_found_gap", 0) or 0)
    ghr5_gap_rate = round(ghr5_to_found_gap / subtasks, 6) if subtasks else None
    perception_gap_rate = round(perception_found_rate - found_rate, 6)

    thresholds = {
        "topk_rerank": {
            "ghr5_rate_min": 0.45,
            "found_rate_max": 0.25,
            "ghr5_to_found_gap_rate_min": 0.20,
        },
        "fix_retrieval_first": {
            "ghr5_rate_max": 0.35,
            "candidate_recall_rate_max": 0.50,
        },
        "inspect_spatial_confirmation": {
            "perception_found_minus_found_min": 0.20,
        },
    }

    if ghr5_rate >= 0.45 and found_rate <= 0.25 and (ghr5_gap_rate or 0.0) >= 0.20:
        recommendation = "implement_topk_nav_rerank"
        reason = "GHR@5 已有召回但 Found 明显偏低，优先检查导航选点、路径代价、可见性和成功半径转换。"
    elif ghr5_rate < 0.35 or candidate_recall_rate < 0.50:
        recommendation = "fix_perception_retrieval_first"
        reason = "候选召回或 GHR@5 偏低，优先继续修 perception、target name map、CLIP 阈值和 label sanitizer。"
    elif perception_gap_rate >= 0.20:
        recommendation = "inspect_spatial_confirmation_depth"
        reason = "perception_found 明显高于 benchmark found，优先检查 spatial confirmation、深度反投影偏差和 success radius。"
    else:
        recommendation = "continue_diagnosis_before_strategy_change"
        reason = "当前指标未满足明确触发条件，建议先结合 HTML 查看候选距离、fallback 和目标确认失败样例。"

    return {
        "recommendation": recommendation,
        "reason": reason,
        "thresholds": thresholds,
        "observed": {
            "subtasks": subtasks,
            "found_rate": round(found_rate, 6),
            "ghr5_rate": round(ghr5_rate, 6),
            "candidate_recall_rate": round(candidate_recall_rate, 6),
            "perception_found_rate": round(perception_found_rate, 6),
            "ghr5_to_found_gap": ghr5_to_found_gap,
            "ghr5_to_found_gap_rate": ghr5_gap_rate,
            "perception_found_minus_found": perception_gap_rate,
        },
    }


def _success_gap_bucket(record: Dict[str, Any], candidate_info: Dict[str, Any]) -> Dict[str, Any]:
    flags: List[str] = []
    found = bool(record.get("found"))
    perception_found = bool(record.get("perception_found"))
    ghr5 = bool(record.get("ghr5"))
    rounds_with_candidates = int(candidate_info.get("rounds_with_candidates", 0) or 0)
    candidate_rounds = int(candidate_info.get("candidate_rounds", 0) or 0)
    fallback_count = int(record.get("fallback_count", 0) or 0)
    success_radius = _safe_float(record.get("success_radius") or record.get("target_success_radius"))
    min_dist = _safe_float(record.get("min_dist"))
    final_dist = _safe_float(record.get("final_dist"))

    if found:
        return {"bucket": "found", "flags": ["found"]}
    if not perception_found:
        flags.append("perception_goal_not_confirmed")
    if candidate_rounds <= 0 or rounds_with_candidates <= 0:
        flags.append("retrieval_no_candidates")
    if not ghr5:
        flags.append("candidates_far_from_goal")
    if ghr5:
        flags.append("candidate_near_but_not_converted_to_success")
    if perception_found:
        flags.append("perception_goal_mismatch_or_false_positive_confirmation")
    if fallback_count > 0:
        flags.append("fallback_used")
    if (
        success_radius is not None
        and min_dist is not None
        and final_dist is not None
        and min_dist <= success_radius
        and final_dist > success_radius
    ):
        flags.append("navigation_or_success_radius_failed")

    if "perception_goal_mismatch_or_false_positive_confirmation" in flags:
        bucket = "perception_goal_mismatch_or_false_positive_confirmation"
    elif "retrieval_no_candidates" in flags:
        bucket = "retrieval_no_candidates"
    elif "candidates_far_from_goal" in flags:
        bucket = "candidates_far_from_goal"
    elif "navigation_or_success_radius_failed" in flags:
        bucket = "navigation_or_success_radius_failed"
    elif "candidate_near_but_not_converted_to_success" in flags:
        bucket = "candidate_near_but_not_converted_to_success"
    else:
        bucket = "unsuccessful_unknown"
    return {"bucket": bucket, "flags": flags}


def build_success_gap_report(output_dir: Path) -> Dict[str, Any]:
    output_dir = as_path(output_dir)
    diagnostics_records = _read_jsonl(output_dir / "memory_diagnostics.jsonl")
    candidate_records = _read_jsonl(output_dir / "candidate_traces.jsonl")
    diagnostic_stats = _diagnostic_stats(diagnostics_records)
    candidate_stats = _candidate_stats(candidate_records)

    candidates_by_subtask: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in candidate_records:
        key = _subtask_key(record)
        if key:
            candidates_by_subtask[key].append(record)

    rows: List[Dict[str, Any]] = []
    bucket_counter: Counter[str] = Counter()
    flag_counter: Counter[str] = Counter()
    for record in diagnostics_records:
        key = _subtask_key(record)
        cand_records = candidates_by_subtask.get(key, [])
        candidate_rounds = len(cand_records)
        rounds_with_candidates = sum(1 for item in cand_records if _candidate_list(item))
        selected_labels: Counter[str] = Counter()
        selected_backends: Counter[str] = Counter()
        selected_strengths: Counter[str] = Counter()
        selected_scores: List[Optional[float]] = []
        top_candidate_count = 0
        for item in cand_records:
            top = _candidate_list(item)
            top_candidate_count += len(top)
            selected = _selected_candidate(item)
            if not selected:
                continue
            label = sanitize_detection_label(selected.get("label"))
            if label and not is_noise_detection_label(label):
                selected_labels[label] += 1
            backend = selected.get("sim_backend") or selected.get("backend")
            if backend:
                selected_backends[str(backend)] += 1
            selected_strengths[str(selected.get("label_alias_match_strength") or "unknown")] += 1
            selected_scores.append(_safe_float(selected.get("S_final") or selected.get("score")))

        candidate_info = {
            "candidate_rounds": candidate_rounds,
            "rounds_with_candidates": rounds_with_candidates,
        }
        bucket_info = _success_gap_bucket(record, candidate_info)
        bucket_counter[bucket_info["bucket"]] += 1
        for flag in bucket_info["flags"]:
            flag_counter[flag] += 1

        rows.append(
            {
                "episode_id": record.get("episode_id"),
                "subtask_id": record.get("subtask_id"),
                "scene_name": record.get("scene_name"),
                "layout_id": record.get("layout_id"),
                "state_index": record.get("state_index"),
                "seen_layout_count_before": record.get("seen_layout_count_before"),
                "task_type": record.get("task_type"),
                "query_modality": record.get("query_modality"),
                "target_object": record.get("target_object"),
                "query_label": record.get("query_label"),
                "alias_labels": record.get("alias_labels"),
                "strong_alias_labels": record.get("strong_alias_labels"),
                "weak_alias_labels": record.get("weak_alias_labels"),
                "found": bool(record.get("found")),
                "perception_found": bool(record.get("perception_found")),
                "mra": bool(record.get("mra")),
                "ghr3": bool(record.get("ghr3")),
                "ghr5": bool(record.get("ghr5")),
                "ghr10": bool(record.get("ghr10")),
                "min_dist": _safe_float(record.get("min_dist")),
                "top1_dist": _safe_float(record.get("top1_dist")),
                "top3_min_dist": _safe_float(record.get("top3_min_dist")),
                "top5_min_dist": _safe_float(record.get("top5_min_dist")),
                "top10_min_dist": _safe_float(record.get("top10_min_dist")),
                "final_dist": _safe_float(record.get("final_dist")),
                "final_minus_best_candidate_dist": _safe_float(record.get("final_minus_best_candidate_dist")),
                "success_radius": _safe_float(record.get("success_radius") or record.get("target_success_radius")),
                "target_confirm_threshold": _safe_float(record.get("target_confirm_threshold")),
                "target_semantic_match_count": int(record.get("target_semantic_match_count", 0) or 0),
                "target_positioned_match_count": int(record.get("target_positioned_match_count", 0) or 0),
                "closest_target_detection_distance": _safe_float(record.get("closest_target_detection_distance")),
                "closest_target_detection_label": record.get("closest_target_detection_label"),
                "closest_target_detection_pos": record.get("closest_target_detection_pos"),
                "target_detection_distance_minus_threshold": _safe_float(
                    record.get("target_detection_distance_minus_threshold")
                ),
                "fallback_count": int(record.get("fallback_count", 0) or 0),
                "candidate_rounds": candidate_rounds,
                "rounds_with_candidates": rounds_with_candidates,
                "top_candidate_count": top_candidate_count,
                "selected_labels": dict(selected_labels.most_common(8)),
                "selected_backend": dict(selected_backends.most_common(8)),
                "selected_match_strength": dict(selected_strengths.most_common()),
                "avg_selected_score": _mean(selected_scores),
                "failure_bucket": bucket_info["bucket"],
                "failure_flags": bucket_info["flags"],
            }
        )

    summary = {
        **diagnostic_stats,
        "candidate_recall_rate": candidate_stats.get("candidate_recall_rate"),
        "rounds_with_candidates": candidate_stats.get("rounds_with_candidates"),
        "rounds_without_candidates": candidate_stats.get("rounds_without_candidates"),
        "avg_candidates_per_round": candidate_stats.get("avg_candidates_per_round"),
        "bucket_counts": dict(bucket_counter.most_common()),
        "flag_counts": dict(flag_counter.most_common()),
    }
    decision = _success_gap_decision(diagnostic_stats, candidate_stats)
    return {
        "output_dir": str(output_dir),
        "summary": summary,
        "decision": decision,
        "subtasks": rows,
    }


def _diagnostic_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    final_dists = [_safe_float(r.get("final_dist")) for r in records]
    min_dists = [_safe_float(r.get("min_dist")) for r in records]
    top1_dists = [_safe_float(r.get("top1_dist")) for r in records]
    top3_min_dists = [_safe_float(r.get("top3_min_dist")) for r in records]
    top5_min_dists = [_safe_float(r.get("top5_min_dist")) for r in records]
    top10_min_dists = [_safe_float(r.get("top10_min_dist")) for r in records]
    final_minus_best = [_safe_float(r.get("final_minus_best_candidate_dist")) for r in records]
    closest_detection_dists = [_safe_float(r.get("closest_target_detection_distance")) for r in records]
    detection_margin_deltas = [_safe_float(r.get("target_detection_distance_minus_threshold")) for r in records]
    fallback_counts = [_safe_float(r.get("fallback_count"), 0.0) for r in records]
    subtasks = len(records)
    found = sum(1 for r in records if bool(r.get("found")))
    perception_found = sum(1 for r in records if bool(r.get("perception_found")))
    mra = sum(1 for r in records if bool(r.get("mra")))
    ghr3 = sum(1 for r in records if bool(r.get("ghr3")))
    ghr5 = sum(1 for r in records if bool(r.get("ghr5")))
    ghr10 = sum(1 for r in records if bool(r.get("ghr10")))
    semantic_match_subtasks = sum(1 for r in records if int(r.get("target_semantic_match_count", 0) or 0) > 0)
    positioned_match_subtasks = sum(1 for r in records if int(r.get("target_positioned_match_count", 0) or 0) > 0)
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
        "ghr10": ghr10,
        "ghr10_rate": round(ghr10 / subtasks, 6) if subtasks else None,
        "ghr5_to_found_gap": ghr5 - found,
        "avg_final_dist": avg_final_dist,
        "best_final_dist": min([v for v in final_dists if v is not None], default=None),
        "avg_candidate_min_dist": avg_candidate_min_dist,
        "best_candidate_min_dist": min([v for v in min_dists if v is not None], default=None),
        "avg_top1_dist": _mean(top1_dists),
        "avg_top3_min_dist": _mean(top3_min_dists),
        "avg_top5_min_dist": _mean(top5_min_dists),
        "avg_top10_min_dist": _mean(top10_min_dists),
        "avg_final_minus_candidate_min_dist": (
            round(float(avg_final_dist) - float(avg_candidate_min_dist), 6)
            if avg_final_dist is not None and avg_candidate_min_dist is not None
            else None
        ),
        "avg_final_minus_best_candidate_dist": _mean(final_minus_best),
        "avg_fallback_count": _mean(fallback_counts),
        "target_semantic_match_subtasks": semantic_match_subtasks,
        "target_semantic_match_rate": round(semantic_match_subtasks / subtasks, 6) if subtasks else None,
        "target_positioned_match_subtasks": positioned_match_subtasks,
        "target_positioned_match_rate": round(positioned_match_subtasks / subtasks, 6) if subtasks else None,
        "avg_closest_target_detection_distance": _mean(closest_detection_dists),
        "avg_target_detection_distance_minus_threshold": _mean(detection_margin_deltas),
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
    success_gap_summary = _success_gap_decision(diagnostic_stats, candidate_stats)
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
        "success_gap_summary": success_gap_summary,
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
    parser.add_argument(
        "--write-success-gap-report",
        default="",
        help=(
            "Optional path for a per-subtask success gap report. "
            "If omitted and --write-json is set, writes success_gap_report.json next to the diagnosis JSON."
        ),
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    report = diagnose(output_dir)
    if args.write_json:
        write_json(args.write_json, report)
    success_gap_path = args.write_success_gap_report
    if not success_gap_path and args.write_json:
        success_gap_path = str(Path(args.write_json).with_name("success_gap_report.json"))
    if success_gap_path:
        write_json(success_gap_path, build_success_gap_report(output_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
