from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .habitat_adapter import HabitatLayoutAdapter, euclidean_distance, path_length_from_poses
from .metrics import (
    EpisodeResult,
    SubtaskResult,
    aggregate_metrics,
    aggregate_subtask_metrics,
    exploration_curve,
)
from .schemas import Episode, Pose, Subtask, SubtaskTrace, Trajectory, read_episode, read_trajectory_jsonl


def _iter_episode_paths(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*.json") if p.is_file() and p.name != "manifest.json")


def load_episode_map(path: Path) -> Dict[str, Episode]:
    episodes = [read_episode(p) for p in _iter_episode_paths(path)]
    if not episodes:
        raise FileNotFoundError(f"No episode JSON files found under {path}")
    return {ep.episode_id: ep for ep in episodes}


def _last_pose(trajectory: Trajectory, fallback: Pose) -> Pose:
    if trajectory.steps:
        return trajectory.steps[-1].position
    if trajectory.subtask_traces:
        return trajectory.subtask_traces[-1].final_pose
    return fallback


def _start_pose(episode: Episode, trajectory: Trajectory) -> Pose:
    if trajectory.steps:
        return trajectory.steps[0].position
    metadata = trajectory.metadata if isinstance(trajectory.metadata, dict) else {}
    if "start_pose" in metadata:
        return Pose.from_any(metadata["start_pose"])
    return episode.start_pose


def _trace_by_id(trajectory: Trajectory) -> Dict[str, SubtaskTrace]:
    return {trace.subtask_id: trace for trace in trajectory.subtask_traces}


def _segment_shortest_distance(current: Pose, subtask: Subtask, adapter: Optional[HabitatLayoutAdapter]) -> float:
    if adapter is None or adapter.sim is None:
        return euclidean_distance(current, subtask.target_position)
    distance = adapter.geodesic_distance(current, subtask.target_position)
    if math.isfinite(distance):
        return float(distance)
    return euclidean_distance(current, subtask.target_position)


def _trajectory_path_length(trajectory: Trajectory) -> float:
    if trajectory.subtask_traces:
        return sum(trace.path_length for trace in trajectory.subtask_traces)
    return path_length_from_poses([step.position for step in trajectory.steps])


def evaluate_trajectory(
    episode: Episode,
    trajectory: Trajectory,
    args: argparse.Namespace,
) -> EpisodeResult:
    trace_map = _trace_by_id(trajectory)
    actual_start_pose = _start_pose(episode, trajectory)
    current_shortest_pose = actual_start_pose
    last_completion_pose = actual_start_pose
    subtask_results: List[SubtaskResult] = []

    adapter: Optional[HabitatLayoutAdapter] = None
    try:
        adapter = HabitatLayoutAdapter(
            episode.scene_name,
            layout_path=Path(episode.scene_state.layout_path) if episode.scene_state.layout_path else None,
            data_dir=Path(args.data_dir),
            objects_dir=Path(args.objects_dir),
            enable_physics=False,
            load_layout_objects=False,
            require_habitat=not bool(args.euclidean_fallback),
        )
    except Exception:
        if not bool(args.euclidean_fallback):
            raise
        adapter = None

    try:
        for subtask in episode.subtasks:
            trace = trace_map.get(subtask.subtask_id)
            if trace is None:
                final_pose = _last_pose(trajectory, last_completion_pose)
                path_length = 0.0
                steps = 0
            else:
                final_pose = trace.final_pose
                path_length = float(trace.path_length)
                steps = int(trace.steps)
            distance_to_goal = euclidean_distance(final_pose, subtask.target_position)
            success = distance_to_goal <= float(subtask.success_radius)
            shortest = _segment_shortest_distance(current_shortest_pose, subtask, adapter)
            subtask_results.append(
                SubtaskResult(
                    subtask_id=subtask.subtask_id,
                    task_type=subtask.task_type,
                    success=success,
                    distance_to_goal=distance_to_goal,
                    path_length=path_length,
                    shortest_path_length=shortest,
                    steps=steps,
                )
            )
            current_shortest_pose = subtask.target_position
            last_completion_pose = final_pose
    finally:
        if adapter is not None:
            adapter.close()

    path_length = _trajectory_path_length(trajectory)
    shortest_path_length = sum(x.shortest_path_length for x in subtask_results)
    success = all(x.success for x in subtask_results) and len(subtask_results) == len(episode.subtasks)
    memory = trajectory.metadata.get("memory_metrics", {}) if isinstance(trajectory.metadata, dict) else {}
    return EpisodeResult(
        episode_id=episode.episode_id,
        scene_name=episode.scene_name,
        layout_id=episode.layout_id,
        seen_layout_count_before=episode.seen_layout_count_before,
        success=success,
        path_length=path_length,
        shortest_path_length=shortest_path_length,
        elapsed_seconds=trajectory.elapsed_seconds,
        steps=sum(x.steps for x in subtask_results) or len(trajectory.steps),
        subtask_results=subtask_results,
        agent_id=trajectory.agent_id,
        dynamic_memory_correct=int(memory.get("dynamic_memory_correct", 0)),
        dynamic_memory_total=int(memory.get("dynamic_memory_total", 0)),
        fixed_memory_correct=int(memory.get("fixed_memory_correct", 0)),
        fixed_memory_total=int(memory.get("fixed_memory_total", 0)),
    )


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    episodes = load_episode_map(Path(args.episodes))
    trajectories = read_trajectory_jsonl(Path(args.trajectories))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[EpisodeResult] = []
    missing: List[str] = []
    for traj in trajectories:
        episode = episodes.get(traj.episode_id)
        if episode is None:
            missing.append(traj.episode_id)
            continue
        results.append(evaluate_trajectory(episode, traj, args))

    episode_results_path = output_dir / "episode_results.jsonl"
    with episode_results_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    summary = aggregate_metrics(results)
    summary.update(
        {
            "evaluated_episodes": len(results),
            "trajectory_count": len(trajectories),
            "missing_episode_count": len(missing),
            "missing_episode_ids": missing[:20],
        }
    )
    by_task_type = aggregate_subtask_metrics(results)
    curve = exploration_curve(results)

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "by_task_type.json").write_text(json.dumps(by_task_type, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "exploration_curve.json").write_text(json.dumps(curve, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "summary": summary,
        "episode_results": str(episode_results_path),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dynamic household object navigation trajectories.")
    parser.add_argument("--episodes", required=True, help="Episode JSON file or directory")
    parser.add_argument("--trajectories", required=True, help="Trajectory JSONL produced by benchmark.runner or compatible agent")
    parser.add_argument("--output-dir", default="benchmark/eval/latest")
    parser.add_argument("--data-dir", default="hm3d")
    parser.add_argument("--objects-dir", default="objects")
    parser.add_argument("--euclidean-fallback", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    summary = evaluate(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
