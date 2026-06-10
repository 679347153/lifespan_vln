from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmark.evaluate import evaluate as benchmark_evaluate
from benchmark.habitat_adapter import HabitatLayoutAdapter, euclidean_distance
from benchmark.schemas import (
    Episode,
    Pose,
    SubtaskTrace,
    Trajectory,
    TrajectoryStep,
    read_episode,
    to_json_dict,
)

from .common import as_path, euclidean_2d, write_json
from .dataset_index import DatasetIndex
from .episode_to_queries import query_from_subtask
from .formal_scoring import (
    FusionParams,
    apply_negative_feedback,
    candidates_to_dict,
    rank_candidates,
)
from .image_goal_index import ImageGoalIndex
from .memory_builder import build_memory_from_seen_layouts
from .offline_eval import _aggregate_group, _iter_episode_paths, _safe_json_value, load_episodes_for_args


def _pose_from_candidate(candidate: Dict[str, Any], target_pose: Pose) -> Pose:
    return Pose(
        x=float(candidate["world_x"]),
        y=float(target_pose.y),
        z=float(candidate["world_z"]),
        yaw=float(target_pose.yaw),
    )


def _try_adapter(episode: Episode, args: argparse.Namespace) -> Optional[HabitatLayoutAdapter]:
    try:
        adapter = HabitatLayoutAdapter(
            episode.scene_name,
            layout_path=Path(episode.scene_state.layout_path) if episode.scene_state.layout_path else None,
            data_dir=Path(args.data_dir),
            objects_dir=Path(args.objects_dir),
            enable_physics=False,
            load_layout_objects=bool(args.load_layout_objects),
            require_habitat=not bool(args.euclidean_fallback),
        )
        return adapter
    except Exception:
        if not bool(args.euclidean_fallback):
            raise
        return None


def _snap_candidate(adapter: Optional[HabitatLayoutAdapter], candidate_pose: Pose) -> Pose:
    if adapter is None or adapter.sim is None:
        return candidate_pose
    try:
        import numpy as np

        point = np.asarray([candidate_pose.x, candidate_pose.y, candidate_pose.z], dtype=np.float32)
        snapped = adapter.pathfinder.snap_point(point)
        if snapped is None:
            return candidate_pose
        return Pose(float(snapped[0]), float(snapped[1]), float(snapped[2]), candidate_pose.yaw)
    except Exception:
        return candidate_pose


def _segment_distance(adapter: Optional[HabitatLayoutAdapter], start: Pose, goal: Pose) -> float:
    if adapter is not None and adapter.sim is not None:
        try:
            dist = adapter.geodesic_distance(start, goal)
            if math.isfinite(dist):
                return float(dist)
        except Exception:
            pass
    return euclidean_distance(start, goal)


def _empty_trace(current_pose: Pose, subtask_id: str) -> SubtaskTrace:
    return SubtaskTrace(
        subtask_id=subtask_id,
        final_pose=current_pose,
        path_length=0.0,
        steps=1,
        elapsed_seconds=0.0,
        reported_success=False,
        metadata={"reason": "no_candidates"},
    )


def run_episode_closed_loop(
    episode: Episode,
    index: DatasetIndex,
    args: argparse.Namespace,
    *,
    memory_cache: Optional[Dict[Tuple[str, int, int], Any]] = None,
) -> Tuple[Trajectory, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    start_time = time.time()
    cache = memory_cache if memory_cache is not None else {}
    cache_key = (episode.scene_name, episode.state_index, episode.seen_layout_count_before)
    if cache_key not in cache:
        cache[cache_key] = build_memory_from_seen_layouts(
            index,
            episode.scene_name,
            episode.state_index,
            episode.seen_layout_count_before,
        )
    memory_floors = cache[cache_key]
    image_index = ImageGoalIndex(images_dir=args.images_dir)
    params = FusionParams()

    adapter = _try_adapter(episode, args)
    current_pose = episode.start_pose
    total_steps = 0
    traces: List[SubtaskTrace] = []
    steps: List[TrajectoryStep] = [TrajectoryStep(t=0, position=current_pose, action="RESET", completed_subtask_ids=[])]
    diagnostics: List[Dict[str, Any]] = []
    candidate_traces: List[Dict[str, Any]] = []
    failure_feedback: List[Dict[str, Any]] = []
    found_count = 0

    try:
        for subtask in episode.subtasks:
            query = query_from_subtask(episode, subtask)
            visited_points: List[List[float]] = []
            subtask_found = False
            final_dist = float("inf")
            path_length = 0.0
            last_candidates: List[Dict[str, Any]] = []
            final_pose = current_pose
            miss_feedback: List[Dict[str, Any]] = []

            for round_idx in range(max(1, int(args.max_rounds))):
                candidates = rank_candidates(
                    memory_floors,
                    query,
                    episode.state_index,
                    exploration_round=round_idx,
                    max_candidates=args.max_candidates,
                    image_index=image_index,
                    params=params,
                )
                last_candidates = candidates_to_dict(candidates)
                if not candidates:
                    break
                candidate = candidates[0]
                candidate_pose = _snap_candidate(adapter, _pose_from_candidate(last_candidates[0], subtask.target_position))
                dist_to_goal = euclidean_2d([candidate_pose.x, candidate_pose.z], [subtask.target_position.x, subtask.target_position.z])
                segment_len = _segment_distance(adapter, current_pose, candidate_pose)
                path_length += segment_len
                visited_points.append([candidate_pose.x, candidate_pose.z])
                candidate_record = _safe_json_value(
                    {
                        "episode_id": episode.episode_id,
                        "subtask_id": subtask.subtask_id,
                        "round": round_idx,
                        "candidate": last_candidates[0],
                        "snapped_pose": to_json_dict(candidate_pose),
                        "distance_to_goal": dist_to_goal,
                        "segment_path_length": segment_len,
                        "habitat_active": bool(adapter is not None and adapter.sim is not None),
                    }
                )
                candidate_traces.append(candidate_record)
                current_pose = candidate_pose
                final_pose = candidate_pose
                final_dist = dist_to_goal
                if dist_to_goal <= float(subtask.success_radius):
                    subtask_found = True
                    break
                feedback = apply_negative_feedback(memory_floors, candidate.obj_id, params=params)
                feedback.update(
                    {
                        "episode_id": episode.episode_id,
                        "subtask_id": subtask.subtask_id,
                        "round": round_idx,
                        "candidate_point": [candidate_pose.x, candidate_pose.z],
                        "distance_to_goal": dist_to_goal,
                    }
                )
                miss_feedback.append(_safe_json_value(feedback))
                failure_feedback.append(_safe_json_value(feedback))

            if subtask_found:
                found_count += 1
            if not visited_points:
                trace = _empty_trace(current_pose, subtask.subtask_id)
            else:
                trace = SubtaskTrace(
                    subtask_id=subtask.subtask_id,
                    final_pose=final_pose,
                    path_length=round(path_length, 6),
                    steps=max(1, len(visited_points)),
                    elapsed_seconds=round(time.time() - start_time, 6),
                    reported_success=subtask_found,
                    metadata={
                        "raanav_found": subtask_found,
                        "raanav_sss": len(visited_points),
                        "raanav_min_dist": final_dist if math.isfinite(final_dist) else None,
                    },
                )
            traces.append(trace)
            total_steps += trace.steps
            steps.append(
                TrajectoryStep(
                    t=total_steps,
                    position=current_pose,
                    action="RAANAV_CLOSED_LOOP_SEARCH",
                    completed_subtask_ids=[subtask.subtask_id],
                )
            )
            dists = [
                euclidean_2d([c["world_x"], c["world_z"]], [subtask.target_position.x, subtask.target_position.z])
                for c in last_candidates
            ]
            diagnostics.append(
                _safe_json_value(
                    {
                        "episode_id": episode.episode_id,
                        "subtask_id": subtask.subtask_id,
                        "scene_name": episode.scene_name,
                        "layout_id": episode.layout_id,
                        "state_index": episode.state_index,
                        "seen_layout_count_before": episode.seen_layout_count_before,
                        "task_type": subtask.task_type,
                        "target_object": subtask.target_object,
                        "query_label": query.query_label,
                        "query_modality": image_index.query_modality(query),
                        "target_object_id": str(subtask.target_object_id),
                        "target_position": [subtask.target_position.x, subtask.target_position.z],
                        "success_radius": subtask.success_radius,
                        "found": subtask_found,
                        "sss": len(visited_points),
                        "mra": bool(dists and dists[0] <= float(subtask.success_radius)),
                        "ghr3": any(d <= float(subtask.success_radius) for d in dists[:3]),
                        "ghr5": any(d <= float(subtask.success_radius) for d in dists[:5]),
                        "min_dist": min(dists) if dists else None,
                        "final_dist": final_dist,
                        "visited_points": visited_points,
                        "peaks": last_candidates,
                        "failure_feedback": miss_feedback,
                    }
                )
            )
    finally:
        if adapter is not None:
            adapter.close()

    trajectory = Trajectory(
        episode_id=episode.episode_id,
        steps=steps,
        finished=True,
        finish_reason="all_subtasks_processed",
        elapsed_seconds=round(time.time() - start_time, 6),
        agent_id="raanav_closed_loop",
        scene_name=episode.scene_name,
        layout_id=episode.layout_id,
        seen_layout_count_before=episode.seen_layout_count_before,
        subtask_traces=traces,
        metadata={
            "runner": "RAANav.benchmark_adapter.run_closed_loop",
            "habitat_active": False,  # set per-candidate in candidate_traces; adapter is closed now
            "memory_metrics": {
                "dynamic_memory_correct": found_count,
                "dynamic_memory_total": len(episode.subtasks),
                "fixed_memory_correct": 0,
                "fixed_memory_total": 0,
            },
        },
    )
    return trajectory, diagnostics, candidate_traces, failure_feedback


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _aggregate_group(records)


def _summaries(records: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    summary = _aggregate(records)
    summary["episodes"] = len({r["episode_id"] for r in records})
    by_type: Dict[str, Any] = {}
    grouped_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped_type[str(r.get("task_type", "unknown"))].append(r)
        grouped_type["all"].append(r)
    for key, items in sorted(grouped_type.items()):
        by_type[key] = _aggregate(items)
    grouped_seen: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped_seen[int(r.get("seen_layout_count_before", 0))].append(r)
    curve: List[Dict[str, Any]] = []
    baseline_found: Optional[float] = None
    baseline_mra: Optional[float] = None
    for seen, items in sorted(grouped_seen.items()):
        metrics = _aggregate(items)
        found = float(metrics["found_rate"])
        mra = float(metrics["mra"])
        if baseline_found is None:
            baseline_found = found
            baseline_mra = mra
        curve.append(
            {
                "seen_layout_count_before": seen,
                "subtasks": metrics["subtasks"],
                "found_rate": found,
                "mra": mra,
                "ghr3": metrics["ghr3"],
                "ghr5": metrics["ghr5"],
                "avg_sss": metrics["avg_sss"],
                "found_delta": found - float(baseline_found),
                "mra_delta": mra - float(baseline_mra),
            }
        )
    return summary, by_type, curve


def run(args: argparse.Namespace) -> Dict[str, Any]:
    index = DatasetIndex(args.split_manifest)
    episodes_root, episode_paths = load_episodes_for_args(args, index)
    output_dir = as_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_path = output_dir / "run.jsonl"
    diag_path = output_dir / "memory_diagnostics.jsonl"
    candidate_path = output_dir / "candidate_traces.jsonl"
    feedback_path = output_dir / "failure_feedback.jsonl"

    memory_cache: Dict[Tuple[str, int, int], Any] = {}
    all_diagnostics: List[Dict[str, Any]] = []
    warnings: List[str] = []
    feedback_count = 0

    with run_path.open("w", encoding="utf-8") as run_f, diag_path.open("w", encoding="utf-8") as diag_f, candidate_path.open("w", encoding="utf-8") as cand_f, feedback_path.open("w", encoding="utf-8") as fb_f:
        for ep_path in episode_paths:
            episode = read_episode(ep_path)
            warnings.extend(index.validate_episode_layout(episode))
            trajectory, diagnostics, candidate_traces, feedback = run_episode_closed_loop(
                episode,
                index,
                args,
                memory_cache=memory_cache,
            )
            run_f.write(json.dumps(to_json_dict(trajectory), ensure_ascii=False) + "\n")
            for record in diagnostics:
                all_diagnostics.append(record)
                diag_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            for record in candidate_traces:
                cand_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            for record in feedback:
                feedback_count += 1
                fb_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    memory_summary, memory_by_type, memory_curve = _summaries(all_diagnostics)
    write_json(output_dir / "memory_summary.json", memory_summary)
    write_json(output_dir / "memory_by_task_type.json", memory_by_type)
    write_json(output_dir / "memory_exploration_curve.json", memory_curve)
    write_json(output_dir / "adapter_warnings.json", {"warnings": warnings, "warning_count": len(warnings)})

    bench_args = argparse.Namespace(
        episodes=str(episodes_root),
        trajectories=str(run_path),
        output_dir=str(output_dir),
        data_dir=args.data_dir,
        objects_dir=args.objects_dir,
        euclidean_fallback=bool(args.euclidean_fallback),
    )
    benchmark_result = benchmark_evaluate(bench_args)
    result = {
        "version": index.version,
        "episodes": len(episode_paths),
        "output_dir": str(output_dir),
        "run_jsonl": str(run_path),
        "memory_diagnostics": str(diag_path),
        "candidate_traces": str(candidate_path),
        "failure_feedback": str(feedback_path),
        "failure_feedback_count": feedback_count,
        "memory_summary": memory_summary,
        "benchmark_summary": benchmark_result.get("summary", {}),
        "warnings": len(warnings),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAANav Agentic-RAG closed-loop benchmark adapter.")
    parser.add_argument("--split-manifest", default="benchmark/splits/benchmark_split_longterm_v1.json")
    parser.add_argument("--episodes", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default="benchmark/eval/raanav_closed_loop_longterm_v1")
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--images-dir", default="objects_images")
    parser.add_argument("--data-dir", default="hm3d")
    parser.add_argument("--objects-dir", default="objects")
    parser.add_argument("--load-layout-objects", action="store_true")
    parser.add_argument("--euclidean-fallback", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

