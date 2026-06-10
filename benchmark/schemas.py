from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

TaskType = Literal["open_vocab", "image_goal", "language_goal"]
SplitName = Literal["train", "val", "test"]


@dataclass
class Pose:
    x: float
    y: float
    z: float
    yaw: float = 0.0

    @classmethod
    def from_any(cls, value: Any) -> "Pose":
        if isinstance(value, Pose):
            return value
        if isinstance(value, dict):
            return cls(
                x=float(value.get("x", value.get("position", [0, 0, 0])[0] if isinstance(value.get("position"), list) else 0.0)),
                y=float(value.get("y", value.get("position", [0, 0, 0])[1] if isinstance(value.get("position"), list) else 0.0)),
                z=float(value.get("z", value.get("position", [0, 0, 0])[2] if isinstance(value.get("position"), list) else 0.0)),
                yaw=float(value.get("yaw", 0.0)),
            )
        if isinstance(value, list) and len(value) >= 3:
            yaw = float(value[3]) if len(value) >= 4 else 0.0
            return cls(float(value[0]), float(value[1]), float(value[2]), yaw)
        raise ValueError(f"Cannot parse Pose from {value!r}")

    def as_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.yaw]


@dataclass
class Subtask:
    subtask_id: str
    task_type: TaskType
    target_object: str
    prompt: str
    target_object_id: Any = None
    target_position: Pose = field(default_factory=lambda: Pose(0.0, 0.0, 0.0, 0.0))
    language_prompt: Optional[str] = None
    image_path: Optional[str] = None
    success_radius: float = 1.2
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Subtask":
        prompt = str(data.get("prompt", data.get("language_prompt", "")))
        return cls(
            subtask_id=str(data["subtask_id"]),
            task_type=data["task_type"],
            target_object=str(data.get("target_object", data.get("target_object_id", ""))),
            prompt=prompt,
            target_object_id=data.get("target_object_id", data.get("target_object")),
            target_position=Pose.from_any(data.get("target_position", data.get("target_pose", [0, 0, 0]))),
            language_prompt=data.get("language_prompt"),
            image_path=data.get("image_path"),
            success_radius=float(data.get("success_radius", 1.2)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SceneState:
    state_id: str
    time_index: int
    layout_id: str = ""
    layout_path: str = ""
    fixed_objects: List[str] = field(default_factory=list)
    movable_objects: List[str] = field(default_factory=list)
    transition_hint: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneState":
        return cls(
            state_id=str(data["state_id"]),
            time_index=int(data.get("time_index", data.get("state_index", 0))),
            layout_id=str(data.get("layout_id", data.get("state_id", ""))),
            layout_path=str(data.get("layout_path", "")),
            fixed_objects=list(data.get("fixed_objects", [])),
            movable_objects=list(data.get("movable_objects", [])),
            transition_hint=dict(data.get("transition_hint", {})),
        )


@dataclass
class Episode:
    episode_id: str
    split: SplitName
    scene_name: str
    scene_state: SceneState
    seed: int
    start_pose: Pose
    max_steps: int
    subtasks: List[Subtask]
    metadata: Dict[str, Any]
    layout_id: str = ""
    state_index: int = 0
    seen_layout_count_before: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        scene_state = SceneState.from_dict(data["scene_state"])
        return cls(
            episode_id=str(data["episode_id"]),
            split=data.get("split", "val"),
            scene_name=str(data["scene_name"]),
            scene_state=scene_state,
            seed=int(data.get("seed", 0)),
            start_pose=Pose.from_any(data.get("start_pose", [0, 0, 0])),
            max_steps=int(data.get("max_steps", 500)),
            subtasks=[Subtask.from_dict(x) for x in data.get("subtasks", [])],
            metadata=dict(data.get("metadata", {})),
            layout_id=str(data.get("layout_id", scene_state.layout_id)),
            state_index=int(data.get("state_index", scene_state.time_index)),
            seen_layout_count_before=int(data.get("seen_layout_count_before", scene_state.time_index)),
        )


@dataclass
class SubtaskTrace:
    subtask_id: str
    final_pose: Pose
    path_length: float
    steps: int
    elapsed_seconds: float = 0.0
    reported_success: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubtaskTrace":
        return cls(
            subtask_id=str(data["subtask_id"]),
            final_pose=Pose.from_any(data.get("final_pose", data.get("position", [0, 0, 0]))),
            path_length=float(data.get("path_length", 0.0)),
            steps=int(data.get("steps", 0)),
            elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
            reported_success=data.get("reported_success"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class TrajectoryStep:
    t: int
    position: Pose
    action: str
    completed_subtask_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryStep":
        return cls(
            t=int(data.get("t", data.get("step", 0))),
            position=Pose.from_any(data.get("position", [0, 0, 0])),
            action=str(data.get("action", "")),
            completed_subtask_ids=list(data.get("completed_subtask_ids", [])),
        )


@dataclass
class Trajectory:
    episode_id: str
    steps: List[TrajectoryStep]
    finished: bool
    finish_reason: str
    elapsed_seconds: float
    agent_id: str = "unknown"
    scene_name: str = ""
    layout_id: str = ""
    seen_layout_count_before: int = 0
    subtask_traces: List[SubtaskTrace] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        return cls(
            episode_id=str(data["episode_id"]),
            steps=[TrajectoryStep.from_dict(x) for x in data.get("steps", [])],
            finished=bool(data.get("finished", False)),
            finish_reason=str(data.get("finish_reason", "")),
            elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
            agent_id=str(data.get("agent_id", "unknown")),
            scene_name=str(data.get("scene_name", "")),
            layout_id=str(data.get("layout_id", "")),
            seen_layout_count_before=int(data.get("seen_layout_count_before", 0)),
            subtask_traces=[SubtaskTrace.from_dict(x) for x in data.get("subtask_traces", [])],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SplitManifest:
    version: str
    seed: int
    train_scenes: List[str]
    val_scenes: List[str]
    test_scenes: List[str] = field(default_factory=list)
    episode_count: int = 0
    layout_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def _assert_subtask_balance(subtasks: List[Subtask]) -> None:
    counts = {"open_vocab": 0, "image_goal": 0, "language_goal": 0}
    for item in subtasks:
        counts[item.task_type] += 1
    if max(counts.values()) - min(counts.values()) > 1:
        raise ValueError(f"Subtask type distribution is unbalanced: {counts}")


def validate_episode(ep: Episode) -> None:
    if not (5 <= len(ep.subtasks) <= 10):
        raise ValueError("Each episode must contain 5 to 10 subtasks")
    if ep.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if ep.seen_layout_count_before < 0:
        raise ValueError("seen_layout_count_before must be non-negative")
    _assert_subtask_balance(ep.subtasks)
    for subtask in ep.subtasks:
        if subtask.task_type == "image_goal" and not subtask.image_path:
            raise ValueError(f"image_goal subtask missing image_path: {subtask.subtask_id}")


def to_json_dict(obj: object) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Cannot serialize object of type {type(obj).__name__}")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_json_dict(obj), f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def read_episode(path: Path) -> Episode:
    return Episode.from_dict(read_json(path))


def read_trajectory_jsonl(path: Path) -> List[Trajectory]:
    trajectories: List[Trajectory] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"Trajectory JSONL line {line_no} is not an object")
            trajectories.append(Trajectory.from_dict(data))
    return trajectories


def read_split_manifest(path: Path) -> SplitManifest:
    data = read_json(path)
    return SplitManifest(
        version=str(data["version"]),
        seed=int(data["seed"]),
        train_scenes=list(data.get("train_scenes", [])),
        val_scenes=list(data.get("val_scenes", [])),
        test_scenes=list(data.get("test_scenes", [])),
        episode_count=int(data.get("episode_count", 0)),
        layout_count=int(data.get("layout_count", 0)),
        metadata=dict(data.get("metadata", {})),
    )
