from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from .habitat_adapter import HabitatLayoutAdapter, euclidean_distance
from .schemas import Episode, Pose, Subtask, SubtaskTrace, Trajectory, TrajectoryStep, read_episode, to_json_dict


class AgentAdapter(Protocol):
    agent_id: str

    def reset(self, episode: Episode) -> None:
        ...

    def act(self, observation: Dict[str, Any], subtask: Subtask) -> str:
        ...

    def on_subtask_done(self, episode: Episode, subtask: Subtask, trace: SubtaskTrace) -> None:
        ...


class NoopAgent:
    def __init__(self, agent_id: str = "noop") -> None:
        self.agent_id = agent_id

    def reset(self, episode: Episode) -> None:
        return None

    def act(self, observation: Dict[str, Any], subtask: Subtask) -> str:
        return "STOP"

    def on_subtask_done(self, episode: Episode, subtask: Subtask, trace: SubtaskTrace) -> None:
        return None


class OracleAgent(NoopAgent):
    def __init__(self, agent_id: str = "oracle") -> None:
        super().__init__(agent_id)

    def act(self, observation: Dict[str, Any], subtask: Subtask) -> str:
        return "ORACLE_NAVIGATE"


def _iter_episode_paths(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*.json") if p.is_file() and p.name != "manifest.json")


def load_episodes(path: Path) -> List[Episode]:
    episodes = [read_episode(p) for p in _iter_episode_paths(path)]
    if not episodes:
        raise FileNotFoundError(f"No episode JSON files found under {path}")
    return episodes


def _load_agent(spec: Optional[str], mode: str, agent_id: str) -> AgentAdapter:
    if spec:
        module_name, sep, attr = spec.partition(":")
        module = importlib.import_module(module_name)
        factory = getattr(module, attr or "create_agent")
        agent = factory()
        if not hasattr(agent, "agent_id"):
            agent.agent_id = agent_id
        return agent
    if mode == "noop":
        return NoopAgent(agent_id)
    return OracleAgent(agent_id)


def _sample_start_pose(episode: Episode, args: argparse.Namespace) -> Pose:
    if not bool(args.sample_start_pose):
        return episode.start_pose
    import random

    rng = random.Random(episode.seed)
    try:
        with HabitatLayoutAdapter(
            episode.scene_name,
            layout_path=Path(episode.scene_state.layout_path) if episode.scene_state.layout_path else None,
            data_dir=Path(args.data_dir),
            objects_dir=Path(args.objects_dir),
            load_layout_objects=False,
            require_habitat=True,
        ) as adapter:
            return adapter.sample_start_pose(rng)
    except Exception:
        if not bool(args.euclidean_fallback):
            raise
        return episode.start_pose


def _segment_shortest_distance(current: Pose, target: Pose, adapter: Optional[HabitatLayoutAdapter]) -> float:
    if adapter is None or adapter.sim is None:
        return euclidean_distance(current, target)
    distance = adapter.geodesic_distance(current, target)
    if math.isfinite(distance):
        return distance
    return euclidean_distance(current, target)


def run_episode(
    episode: Episode,
    agent: AgentAdapter,
    args: argparse.Namespace,
) -> Trajectory:
    start_time = time.time()
    start_pose = _sample_start_pose(episode, args)
    current_pose = start_pose
    total_steps = 0
    step_records: List[TrajectoryStep] = [
        TrajectoryStep(t=0, position=current_pose, action="RESET", completed_subtask_ids=[])
    ]
    traces: List[SubtaskTrace] = []

    adapter: Optional[HabitatLayoutAdapter] = None
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
    except Exception:
        if not bool(args.euclidean_fallback):
            raise
        adapter = None

    agent.reset(episode)
    try:
        for subtask in episode.subtasks:
            action = agent.act({"pose": current_pose, "episode": episode}, subtask)
            if isinstance(agent, OracleAgent):
                final_pose = subtask.target_position
                path_length = _segment_shortest_distance(current_pose, final_pose, adapter)
                steps = max(1, int(math.ceil(path_length / float(args.step_size))))
            else:
                final_pose = current_pose
                path_length = 0.0
                steps = 1

            total_steps += steps
            trace = SubtaskTrace(
                subtask_id=subtask.subtask_id,
                final_pose=final_pose,
                path_length=float(path_length),
                steps=steps,
                elapsed_seconds=max(0.0, time.time() - start_time),
                reported_success=None,
                metadata={"last_action": action},
            )
            traces.append(trace)
            agent.on_subtask_done(episode, subtask, trace)
            current_pose = final_pose
            step_records.append(
                TrajectoryStep(
                    t=total_steps,
                    position=current_pose,
                    action=str(action),
                    completed_subtask_ids=[subtask.subtask_id],
                )
            )
    finally:
        if adapter is not None:
            adapter.close()

    return Trajectory(
        episode_id=episode.episode_id,
        agent_id=getattr(agent, "agent_id", "unknown"),
        scene_name=episode.scene_name,
        layout_id=episode.layout_id,
        seen_layout_count_before=episode.seen_layout_count_before,
        steps=step_records,
        subtask_traces=traces,
        finished=True,
        finish_reason="all_subtasks_processed",
        elapsed_seconds=max(0.0, time.time() - start_time),
        metadata={
            "runner": "benchmark.runner",
            "mode": args.mode,
            "start_pose": to_json_dict(start_pose),
        },
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    episodes = load_episodes(Path(args.episodes))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agent = _load_agent(args.agent_module, args.mode, args.agent_id)

    with output_path.open("w", encoding="utf-8") as f:
        for episode in episodes:
            traj = run_episode(episode, agent, args)
            f.write(json.dumps(to_json_dict(traj), ensure_ascii=False) + "\n")

    return {
        "episodes": len(episodes),
        "output": str(output_path),
        "agent_id": getattr(agent, "agent_id", args.agent_id),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dynamic household object navigation episodes.")
    parser.add_argument("--episodes", required=True, help="Episode JSON file or directory")
    parser.add_argument("--output", default="benchmark/eval/run.jsonl")
    parser.add_argument("--mode", choices=("oracle", "noop"), default="oracle")
    parser.add_argument("--agent-id", default="oracle")
    parser.add_argument("--agent-module", default=None, help="Optional module:factory returning an external AgentAdapter")
    parser.add_argument("--data-dir", default="hm3d")
    parser.add_argument("--objects-dir", default="objects")
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--sample-start-pose", action="store_true")
    parser.add_argument("--load-layout-objects", action="store_true")
    parser.add_argument("--euclidean-fallback", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    summary = run(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
