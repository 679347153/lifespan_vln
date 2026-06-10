from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence

TASK_TYPES = ("open_vocab", "image_goal", "language_goal")


@dataclass
class SubtaskResult:
    subtask_id: str
    task_type: str
    success: bool
    distance_to_goal: float
    path_length: float
    shortest_path_length: float
    steps: int


@dataclass
class EpisodeResult:
    episode_id: str
    scene_name: str = ""
    layout_id: str = ""
    seen_layout_count_before: int = 0
    success: bool = False
    path_length: float = 0.0
    shortest_path_length: float = 0.0
    elapsed_seconds: float = 0.0
    steps: int = 0
    subtask_results: List[SubtaskResult] = field(default_factory=list)
    agent_id: str = "unknown"
    dynamic_memory_correct: int = 0
    dynamic_memory_total: int = 0
    fixed_memory_correct: int = 0
    fixed_memory_total: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeResult":
        return cls(
            episode_id=str(data.get("episode_id", "")),
            scene_name=str(data.get("scene_name", "")),
            layout_id=str(data.get("layout_id", "")),
            seen_layout_count_before=int(data.get("seen_layout_count_before", 0)),
            success=bool(data.get("success", False)),
            path_length=float(data.get("path_length", 0.0)),
            shortest_path_length=float(data.get("shortest_path_length", 0.0)),
            elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
            steps=int(data.get("steps", 0)),
            subtask_results=[
                SubtaskResult(
                    subtask_id=str(x.get("subtask_id", "")),
                    task_type=str(x.get("task_type", "")),
                    success=bool(x.get("success", False)),
                    distance_to_goal=float(x.get("distance_to_goal", 0.0)),
                    path_length=float(x.get("path_length", 0.0)),
                    shortest_path_length=float(x.get("shortest_path_length", 0.0)),
                    steps=int(x.get("steps", 0)),
                )
                for x in data.get("subtask_results", [])
            ],
            agent_id=str(data.get("agent_id", "unknown")),
            dynamic_memory_correct=int(data.get("dynamic_memory_correct", 0)),
            dynamic_memory_total=int(data.get("dynamic_memory_total", 0)),
            fixed_memory_correct=int(data.get("fixed_memory_correct", 0)),
            fixed_memory_total=int(data.get("fixed_memory_total", 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def spl(success: bool, path_length: float, shortest_path_length: float) -> float:
    if not success or shortest_path_length <= 0:
        return 0.0
    return shortest_path_length / max(path_length, shortest_path_length)


def soft_spl(
    path_length: float,
    shortest_path_length: float,
    goal_distance: float,
    threshold: float = 1.0,
) -> float:
    soft_success = max(0.0, 1.0 - safe_div(goal_distance, threshold))
    if shortest_path_length <= 0:
        return 0.0
    return soft_success * shortest_path_length / max(path_length, shortest_path_length)


def aggregate_episode_metrics(results: Sequence[EpisodeResult]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "spl": 0.0,
            "avg_steps": 0.0,
            "avg_elapsed_seconds": 0.0,
            "avg_path_length": 0.0,
            "avg_shortest_path_length": 0.0,
        }
    return {
        "episodes": n,
        "success_rate": sum(1 for r in results if r.success) / n,
        "spl": sum(spl(r.success, r.path_length, r.shortest_path_length) for r in results) / n,
        "avg_steps": sum(r.steps for r in results) / n,
        "avg_elapsed_seconds": sum(r.elapsed_seconds for r in results) / n,
        "avg_path_length": sum(r.path_length for r in results) / n,
        "avg_shortest_path_length": sum(r.shortest_path_length for r in results) / n,
    }


def aggregate_subtask_metrics(results: Sequence[EpisodeResult]) -> Dict[str, Any]:
    by_type: Dict[str, List[SubtaskResult]] = {k: [] for k in TASK_TYPES}
    by_type["all"] = []
    for episode in results:
        for subtask in episode.subtask_results:
            by_type.setdefault(subtask.task_type, []).append(subtask)
            by_type["all"].append(subtask)

    out: Dict[str, Any] = {}
    for task_type, items in sorted(by_type.items()):
        n = len(items)
        if n == 0:
            out[task_type] = {"subtasks": 0, "success_rate": 0.0, "spl": 0.0}
            continue
        out[task_type] = {
            "subtasks": n,
            "success_rate": sum(1 for x in items if x.success) / n,
            "spl": sum(spl(x.success, x.path_length, x.shortest_path_length) for x in items) / n,
            "avg_distance_to_goal": sum(x.distance_to_goal for x in items) / n,
            "avg_steps": sum(x.steps for x in items) / n,
        }
    return out


def exploration_curve(results: Sequence[EpisodeResult]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault(int(result.seen_layout_count_before), []).append(result)

    curve: List[Dict[str, Any]] = []
    baseline_sr = None
    baseline_spl = None
    for seen_count in sorted(grouped):
        metrics = aggregate_episode_metrics(grouped[seen_count])
        sr = float(metrics["success_rate"])
        spl_value = float(metrics["spl"])
        if baseline_sr is None:
            baseline_sr = sr
            baseline_spl = spl_value
        curve.append(
            {
                "seen_layout_count_before": seen_count,
                "episodes": metrics["episodes"],
                "success_rate": sr,
                "spl": spl_value,
                "sr_delta": sr - baseline_sr,
                "spl_delta": spl_value - baseline_spl,
                "sr_gain_pct": safe_div(sr - baseline_sr, max(baseline_sr, 1e-6)) * 100.0,
                "spl_gain_pct": safe_div(spl_value - baseline_spl, max(baseline_spl, 1e-6)) * 100.0,
            }
        )
    return curve


def aggregate_metrics(results: Sequence[EpisodeResult]) -> Dict[str, Any]:
    episode_metrics = aggregate_episode_metrics(results)
    dyn_correct = sum(r.dynamic_memory_correct for r in results)
    dyn_total = sum(r.dynamic_memory_total for r in results)
    fixed_correct = sum(r.fixed_memory_correct for r in results)
    fixed_total = sum(r.fixed_memory_total for r in results)
    episode_metrics.update(
        {
            "dynamic_memory_accuracy": safe_div(dyn_correct, dyn_total),
            "fixed_memory_accuracy": safe_div(fixed_correct, fixed_total),
        }
    )
    return episode_metrics
