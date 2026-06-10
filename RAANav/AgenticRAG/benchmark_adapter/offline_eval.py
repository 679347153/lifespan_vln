from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from benchmark.evaluate import evaluate as benchmark_evaluate
from benchmark.schemas import (
    Episode,
    Pose,
    SubtaskTrace,
    Trajectory,
    TrajectoryStep,
    read_episode,
    to_json_dict,
)

from .common import as_path, euclidean_pose, repo_root, write_json
from .dataset_index import DatasetIndex
from .episode_to_queries import query_from_subtask
from .lightweight_search import SearchResult, search_memory
from .memory_builder import build_memory_from_seen_layouts


def _iter_episode_paths(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*.json") if p.is_file() and p.name != "manifest.json")


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json_value(v) for v in value]
    return value


def _pose_from_xz(point: List[float], y: float = 0.0, yaw: float = 0.0) -> Pose:
    return Pose(x=float(point[0]), y=float(y), z=float(point[1]), yaw=float(yaw))


def _trace_from_search(current_pose: Pose, target_pose: Pose, subtask_id: str, result: SearchResult) -> Tuple[SubtaskTrace, Pose]:
    path_length = 0.0
    last_pose = current_pose
    final_pose = current_pose
    if result.visited_points:
        for point in result.visited_points:
            next_pose = _pose_from_xz(point, y=target_pose.y, yaw=target_pose.yaw)
            path_length += euclidean_pose(last_pose, next_pose)
            last_pose = next_pose
        final_pose = last_pose
    if result.found and result.visited_points:
        final_pose = last_pose
    steps = max(1, int(result.sss or len(result.visited_points) or 1))
    trace = SubtaskTrace(
        subtask_id=subtask_id,
        final_pose=final_pose,
        path_length=round(float(path_length), 6),
        steps=steps,
        elapsed_seconds=0.0,
        reported_success=result.found,
        metadata={
            "raanav_found": result.found,
            "raanav_sss": result.sss,
            "raanav_min_dist": None if not math.isfinite(result.min_dist) else result.min_dist,
        },
    )
    return trace, final_pose


def _diagnostic_record(
    episode: Episode,
    subtask: Any,
    query: Any,
    result: SearchResult,
) -> Dict[str, Any]:
    return _safe_json_value(
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
            "target_object_id": str(subtask.target_object_id),
            "target_position": [subtask.target_position.x, subtask.target_position.z],
            "success_radius": subtask.success_radius,
            "found": result.found,
            "sss": result.sss,
            "mra": result.mra,
            "ghr3": result.ghr3,
            "ghr5": result.ghr5,
            "min_dist": result.min_dist,
            "final_dist": result.final_dist,
            "visited_points": result.visited_points,
            "peaks": result.peaks,
        }
    )


def run_episode_offline(
    episode: Episode,
    index: DatasetIndex,
    *,
    max_candidates: int = 10,
    memory_cache: Optional[Dict[Tuple[str, int, int], Any]] = None,
) -> Tuple[Trajectory, List[Dict[str, Any]]]:
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

    current_pose = episode.start_pose
    total_steps = 0
    traces: List[SubtaskTrace] = []
    steps: List[TrajectoryStep] = [
        TrajectoryStep(t=0, position=current_pose, action="RESET", completed_subtask_ids=[])
    ]
    diagnostics: List[Dict[str, Any]] = []
    found_count = 0

    for subtask in episode.subtasks:
        query = query_from_subtask(episode, subtask)
        gt_xz = [float(subtask.target_position.x), float(subtask.target_position.z)]
        result = search_memory(
            memory_floors,
            query,
            gt_xz,
            success_radius=float(subtask.success_radius),
            max_candidates=max_candidates,
        )
        if result.found:
            found_count += 1
        trace, current_pose = _trace_from_search(current_pose, subtask.target_position, subtask.subtask_id, result)
        traces.append(trace)
        total_steps += trace.steps
        steps.append(
            TrajectoryStep(
                t=total_steps,
                position=current_pose,
                action="RAANAV_MEMORY_SEARCH",
                completed_subtask_ids=[subtask.subtask_id],
            )
        )
        diagnostics.append(_diagnostic_record(episode, subtask, query, result))

    trajectory = Trajectory(
        episode_id=episode.episode_id,
        steps=steps,
        finished=True,
        finish_reason="all_subtasks_processed",
        elapsed_seconds=round(time.time() - start_time, 6),
        agent_id="raanav_offline_mvp",
        scene_name=episode.scene_name,
        layout_id=episode.layout_id,
        seen_layout_count_before=episode.seen_layout_count_before,
        subtask_traces=traces,
        metadata={
            "runner": "RAANav.benchmark_adapter.offline_eval",
            "memory_metrics": {
                "dynamic_memory_correct": found_count,
                "dynamic_memory_total": len(episode.subtasks),
                "fixed_memory_correct": 0,
                "fixed_memory_total": 0,
            },
        },
    )
    return trajectory, diagnostics


def _aggregate_group(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {
            "subtasks": 0,
            "found_rate": 0.0,
            "mra": 0.0,
            "ghr3": 0.0,
            "ghr5": 0.0,
            "avg_sss": 0.0,
            "avg_min_dist": None,
        }
    finite_dists = [float(r["min_dist"]) for r in records if r.get("min_dist") is not None]
    return {
        "subtasks": n,
        "found_rate": sum(1 for r in records if r.get("found")) / n,
        "mra": sum(1 for r in records if r.get("mra")) / n,
        "ghr3": sum(1 for r in records if r.get("ghr3")) / n,
        "ghr5": sum(1 for r in records if r.get("ghr5")) / n,
        "avg_sss": sum(float(r.get("sss") or 0) for r in records) / n,
        "avg_min_dist": (sum(finite_dists) / len(finite_dists)) if finite_dists else None,
    }


def summarize_diagnostics(records: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    summary = _aggregate_group(records)
    summary["episodes"] = len({r["episode_id"] for r in records})

    by_type: Dict[str, Any] = {}
    grouped_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped_type[str(r.get("task_type", "unknown"))].append(r)
        grouped_type["all"].append(r)
    for task_type, items in sorted(grouped_type.items()):
        by_type[task_type] = _aggregate_group(items)

    grouped_seen: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped_seen[int(r.get("seen_layout_count_before", 0))].append(r)
    curve: List[Dict[str, Any]] = []
    baseline_found: Optional[float] = None
    baseline_mra: Optional[float] = None
    for seen_count in sorted(grouped_seen):
        metrics = _aggregate_group(grouped_seen[seen_count])
        found_rate = float(metrics["found_rate"])
        mra = float(metrics["mra"])
        if baseline_found is None:
            baseline_found = found_rate
            baseline_mra = mra
        curve.append(
            {
                "seen_layout_count_before": seen_count,
                "subtasks": metrics["subtasks"],
                "found_rate": found_rate,
                "mra": mra,
                "ghr3": metrics["ghr3"],
                "ghr5": metrics["ghr5"],
                "avg_sss": metrics["avg_sss"],
                "found_delta": found_rate - float(baseline_found),
                "mra_delta": mra - float(baseline_mra),
            }
        )
    return summary, by_type, curve


def load_episodes_for_args(args: argparse.Namespace, index: DatasetIndex) -> Tuple[Path, List[Path]]:
    if args.episodes:
        root = as_path(args.episodes)
    else:
        root = index.episodes_root / args.split
    paths = _iter_episode_paths(root)
    if args.scene:
        paths = [p for p in paths if f"{args.scene}" in p.parts]
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No episode JSON files found for root={root}, scene={args.scene!r}")
    return root, paths


def run(args: argparse.Namespace) -> Dict[str, Any]:
    index = DatasetIndex(args.split_manifest)
    episodes_root, episode_paths = load_episodes_for_args(args, index)
    output_dir = as_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.jsonl"
    diag_path = output_dir / "memory_diagnostics.jsonl"

    memory_cache: Dict[Tuple[str, int, int], Any] = {}
    all_diagnostics: List[Dict[str, Any]] = []
    warnings: List[str] = []

    with run_path.open("w", encoding="utf-8") as run_f, diag_path.open("w", encoding="utf-8") as diag_f:
        for ep_path in episode_paths:
            episode = read_episode(ep_path)
            warnings.extend(index.validate_episode_layout(episode))
            trajectory, diagnostics = run_episode_offline(
                episode,
                index,
                max_candidates=args.max_candidates,
                memory_cache=memory_cache,
            )
            run_f.write(json.dumps(to_json_dict(trajectory), ensure_ascii=False) + "\n")
            for record in diagnostics:
                all_diagnostics.append(record)
                diag_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    memory_summary, memory_by_type, memory_curve = summarize_diagnostics(all_diagnostics)
    write_json(output_dir / "memory_summary.json", memory_summary)
    write_json(output_dir / "memory_by_task_type.json", memory_by_type)
    write_json(output_dir / "memory_exploration_curve.json", memory_curve)
    write_json(output_dir / "adapter_warnings.json", {"warnings": warnings, "warning_count": len(warnings)})

    bench_args = argparse.Namespace(
        episodes=str(episodes_root),
        trajectories=str(run_path),
        output_dir=str(output_dir),
        data_dir="hm3d",
        objects_dir="objects",
        euclidean_fallback=True,
    )
    benchmark_result = benchmark_evaluate(bench_args)

    result = {
        "version": index.version,
        "episodes": len(episode_paths),
        "output_dir": str(output_dir),
        "run_jsonl": str(run_path),
        "memory_diagnostics": str(diag_path),
        "memory_summary": memory_summary,
        "benchmark_summary": benchmark_result.get("summary", {}),
        "warnings": len(warnings),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAANav offline MVP on dynamic layout benchmark episodes.")
    parser.add_argument("--split-manifest", default="benchmark/splits/benchmark_split_longterm_v1.json")
    parser.add_argument("--episodes", default=None, help="Episode JSON file or directory. Defaults to split metadata root + split.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--scene", default=None, help="Optional scene filter.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of episodes for smoke tests.")
    parser.add_argument("--output-dir", default="benchmark/eval/raanav_longterm_v1")
    parser.add_argument("--max-candidates", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()

