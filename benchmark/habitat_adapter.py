from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from hm3d_paths import resolve_scene_paths

try:
    import habitat_sim  # type: ignore[import-not-found]
    import magnum as mn  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only outside Habitat envs.
    habitat_sim = None
    mn = None

from .schemas import Episode, Pose, Subtask


def pose_to_array(pose: Pose) -> np.ndarray:
    return np.asarray([pose.x, pose.y, pose.z], dtype=np.float32)


def euclidean_distance(a: Pose | Sequence[float], b: Pose | Sequence[float]) -> float:
    pa = pose_to_array(a) if isinstance(a, Pose) else np.asarray(a[:3], dtype=np.float32)
    pb = pose_to_array(b) if isinstance(b, Pose) else np.asarray(b[:3], dtype=np.float32)
    return float(np.linalg.norm(pa - pb))


def path_length_from_poses(poses: Sequence[Pose]) -> float:
    if len(poses) < 2:
        return 0.0
    return sum(euclidean_distance(poses[i - 1], poses[i]) for i in range(1, len(poses)))


def _yaw_to_quat(yaw_deg: float) -> Any:
    if mn is None:
        return None
    return mn.Quaternion.rotation(mn.Rad(math.radians(float(yaw_deg))), mn.Vector3(0.0, 1.0, 0.0))


def _resolve_template_handle(template_mgr: Any, model_id: str) -> Optional[str]:
    candidates = [model_id]
    if not model_id.endswith(".object_config.json"):
        candidates.append(f"{model_id}.object_config.json")
    if not model_id.endswith("_4k") and not model_id.endswith("_4k.object_config.json"):
        candidates.extend([f"{model_id}_4k", f"{model_id}_4k.object_config.json"])
    for key in candidates:
        try:
            handles = template_mgr.get_template_handles(key)
            if handles:
                return handles[0]
        except Exception:
            continue
    try:
        needle = model_id.lower().replace(".object_config.json", "")
        for handle in template_mgr.get_template_handles():
            name = os.path.basename(str(handle)).lower().replace(".object_config.json", "")
            if name == needle or name == f"{needle}_4k":
                return handle
    except Exception:
        pass
    return None


class HabitatLayoutAdapter:
    """Small Habitat-Sim helper for benchmark generation, runner, and evaluation."""

    def __init__(
        self,
        scene_name: str,
        layout_path: Optional[Path] = None,
        data_dir: Path = Path("hm3d"),
        objects_dir: Path = Path("objects"),
        enable_physics: bool = False,
        load_layout_objects: bool = True,
        require_habitat: bool = True,
    ) -> None:
        self.scene_name = scene_name
        self.layout_path = layout_path
        self.data_dir = Path(data_dir)
        self.objects_dir = Path(objects_dir)
        self.enable_physics = bool(enable_physics)
        self.sim = None
        self.scene_paths = resolve_scene_paths(scene_name, require_semantic=False, root=self.data_dir)
        if habitat_sim is None:
            if require_habitat:
                raise ImportError("habitat_sim is required for HabitatLayoutAdapter")
            return
        if self.scene_paths is None:
            raise FileNotFoundError(f"Cannot resolve HM3D scene: {scene_name}")
        self.sim = self._make_simulator()
        self._load_templates()
        if layout_path is not None and load_layout_objects:
            self.load_layout(layout_path)

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def __enter__(self) -> "HabitatLayoutAdapter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _make_simulator(self) -> Any:
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = str(self.scene_paths.dataset_config)
        sim_cfg.scene_id = str(self.scene_paths.stage_glb)
        sim_cfg.enable_physics = self.enable_physics
        sim_cfg.gpu_device_id = 0

        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = "color"
        sensor.sensor_type = habitat_sim.SensorType.COLOR
        sensor.resolution = [64, 64]
        sensor.position = [0.0, 1.35, 0.0]

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [sensor]
        return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

    def _load_templates(self) -> None:
        if self.sim is None or not self.objects_dir.is_dir():
            return
        mgr = self.sim.get_object_template_manager()
        try:
            if hasattr(mgr, "load_configs"):
                mgr.load_configs(str(self.objects_dir.resolve()))
            elif hasattr(mgr, "add_template_search_path"):
                mgr.add_template_search_path(str(self.objects_dir.resolve()))
        except Exception:
            return

    def load_layout(self, layout_path: Path) -> int:
        if self.sim is None:
            return 0
        import json

        payload = json.loads(Path(layout_path).read_text(encoding="utf-8"))
        objects = payload.get("objects", [])
        if not isinstance(objects, list):
            return 0

        rom = self.sim.get_rigid_object_manager()
        mgr = self.sim.get_object_template_manager()
        loaded = 0
        for obj_cfg in objects:
            model_id = str(obj_cfg.get("model_id", "")).strip()
            pos = obj_cfg.get("position")
            if not model_id or not isinstance(pos, list) or len(pos) < 3:
                continue
            handle = _resolve_template_handle(mgr, model_id)
            if handle is None:
                continue
            try:
                obj = rom.add_object_by_template_handle(handle)
                obj.translation = np.asarray(pos[:3], dtype=np.float32)
                rot = obj_cfg.get("rotation", [0.0, 0.0, 0.0])
                yaw = float(rot[1]) if isinstance(rot, list) and len(rot) >= 2 else 0.0
                quat = _yaw_to_quat(yaw)
                if quat is not None:
                    obj.rotation = quat
                if hasattr(obj, "motion_type") and hasattr(habitat_sim, "physics"):
                    obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                loaded += 1
            except Exception:
                continue
        return loaded

    @property
    def pathfinder(self) -> Any:
        if self.sim is None:
            raise RuntimeError("Habitat simulator is not available")
        return self.sim.pathfinder

    def sample_start_pose(self, rng: random.Random, max_tries: int = 100) -> Pose:
        pf = self.pathfinder
        for _ in range(max_tries):
            point = pf.get_random_navigable_point()
            if point is None or np.any(np.isnan(point)):
                continue
            yaw = rng.uniform(-math.pi, math.pi)
            return Pose(float(point[0]), float(point[1]), float(point[2]), float(yaw))
        raise RuntimeError(f"Could not sample a navigable start pose for scene {self.scene_name}")

    def geodesic_distance(self, start: Pose | Sequence[float], goal: Pose | Sequence[float]) -> float:
        pf = self.pathfinder
        s = pose_to_array(start) if isinstance(start, Pose) else np.asarray(start[:3], dtype=np.float32)
        g = pose_to_array(goal) if isinstance(goal, Pose) else np.asarray(goal[:3], dtype=np.float32)
        try:
            distance = pf.geodesic_distance(s, g)
        except TypeError:
            distance = pf.geodesic_distance(s, [g])
        if distance is None or not np.isfinite(distance):
            return float("inf")
        return float(distance)

    def shortest_path_sum(self, start_pose: Pose, subtasks: Iterable[Subtask]) -> float:
        total = 0.0
        current = start_pose
        for subtask in subtasks:
            dist = self.geodesic_distance(current, subtask.target_position)
            if not np.isfinite(dist):
                return float("inf")
            total += dist
            current = subtask.target_position
        return float(total)


def shortest_path_sum_euclidean(start_pose: Pose, subtasks: Iterable[Subtask]) -> float:
    total = 0.0
    current = start_pose
    for subtask in subtasks:
        total += euclidean_distance(current, subtask.target_position)
        current = subtask.target_position
    return float(total)


def episode_shortest_path_sum(
    episode: Episode,
    data_dir: Path = Path("hm3d"),
    objects_dir: Path = Path("objects"),
    use_euclidean_fallback: bool = True,
) -> float:
    try:
        with HabitatLayoutAdapter(
            episode.scene_name,
            layout_path=Path(episode.scene_state.layout_path) if episode.scene_state.layout_path else None,
            data_dir=data_dir,
            objects_dir=objects_dir,
            enable_physics=False,
            load_layout_objects=False,
            require_habitat=not use_euclidean_fallback,
        ) as adapter:
            if adapter.sim is None:
                return shortest_path_sum_euclidean(episode.start_pose, episode.subtasks)
            dist = adapter.shortest_path_sum(episode.start_pose, episode.subtasks)
            if np.isfinite(dist):
                return dist
    except Exception:
        if not use_euclidean_fallback:
            raise
    return shortest_path_sum_euclidean(episode.start_pose, episode.subtasks)
