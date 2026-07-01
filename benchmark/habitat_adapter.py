from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


def horizontal_distance(a: Pose | Sequence[float], b: Pose | Sequence[float]) -> float:
    pa = pose_to_array(a) if isinstance(a, Pose) else np.asarray(a[:3], dtype=np.float32)
    pb = pose_to_array(b) if isinstance(b, Pose) else np.asarray(b[:3], dtype=np.float32)
    return float(np.linalg.norm(pa[[0, 2]] - pb[[0, 2]]))


def path_length_from_poses(poses: Sequence[Pose]) -> float:
    if len(poses) < 2:
        return 0.0
    return sum(euclidean_distance(poses[i - 1], poses[i]) for i in range(1, len(poses)))


def _round_point(point: Sequence[float], ndigits: int = 4) -> List[float]:
    return [round(float(v), ndigits) for v in point]


def _find_scene_info_path(data_dir: Path, scene_name: str) -> Optional[Path]:
    candidates = [
        data_dir / "scene_info_export" / f"{scene_name}_scene_info.json",
        data_dir / "minival" / "scene_info_export" / f"{scene_name}_scene_info.json",
        data_dir / "val" / "scene_info_export" / f"{scene_name}_scene_info.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    if data_dir.exists():
        matches = sorted(data_dir.rglob(f"{scene_name}_scene_info.json"))
        if matches:
            return matches[0]
    return None


def _semantic_regions_from_scene_info(data_dir: Path, scene_name: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    path = _find_scene_info_path(data_dir, scene_name)
    if path is None:
        return [], None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [], str(path)
    regions: List[Dict[str, Any]] = []
    for room in payload.get("rooms", []) if isinstance(payload, dict) else []:
        if not isinstance(room, dict):
            continue
        bbox = room.get("bounding_box")
        mn = bbox.get("min") if isinstance(bbox, dict) else None
        mx = bbox.get("max") if isinstance(bbox, dict) else None
        center = room.get("room_center")
        if not (isinstance(mn, list) and len(mn) >= 3 and isinstance(mx, list) and len(mx) >= 3):
            continue
        region_id = room.get("region_id")
        rid = f"region_{region_id}"
        x0, z0 = float(mn[0]), float(mn[2])
        x1, z1 = float(mx[0]), float(mx[2])
        polygon = [
            [round(x0, 4), round(z0, 4)],
            [round(x1, 4), round(z0, 4)],
            [round(x1, 4), round(z1, 4)],
            [round(x0, 4), round(z1, 4)],
        ]
        top_categories = []
        cats = room.get("categories")
        if isinstance(cats, dict):
            top_categories = [
                {"label": str(k), "count": int(v)}
                for k, v in sorted(cats.items(), key=lambda item: int(item[1]), reverse=True)[:8]
            ]
        regions.append(
            {
                "region_id": region_id,
                "room_id": rid,
                "label": rid,
                "source": "semantic_region_bbox",
                "bbox": {"min": _round_point(mn[:3]), "max": _round_point(mx[:3])},
                "center": _round_point(center[:3]) if isinstance(center, list) and len(center) >= 3 else None,
                "polygon": polygon,
                "object_count": int(room.get("object_count", 0) or 0),
                "top_categories": top_categories,
            }
        )
    return regions, str(path)


def _layout_objects_for_geometry(layout_path: Optional[Path]) -> List[Dict[str, Any]]:
    if layout_path is None or not Path(layout_path).exists():
        return []
    try:
        payload = json.loads(Path(layout_path).read_text(encoding="utf-8"))
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for obj in payload.get("objects", []) if isinstance(payload, dict) else []:
        if not isinstance(obj, dict):
            continue
        pos = obj.get("position")
        if not (isinstance(pos, list) and len(pos) >= 3):
            continue
        out.append(
            {
                "id": obj.get("id"),
                "name": obj.get("name") or obj.get("model_id") or obj.get("id"),
                "model_id": obj.get("model_id"),
                "position": _round_point(pos[:3]),
                "sampled_region_id": obj.get("sampled_region_id"),
                "target_instance_id": obj.get("target_instance_id"),
            }
        )
    return out


def _bounds_from_points(points: Sequence[Sequence[float]], padding: float = 1.0) -> Optional[Dict[str, float]]:
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    zs = [float(p[2]) for p in points]
    return {
        "min_x": round(min(xs) - padding, 4),
        "max_x": round(max(xs) + padding, 4),
        "min_y": round(min(ys), 4),
        "max_y": round(max(ys), 4),
        "min_z": round(min(zs) - padding, 4),
        "max_z": round(max(zs) + padding, 4),
    }


def _merge_bounds(primary: Optional[Dict[str, float]], points: Sequence[Sequence[float]], padding: float = 1.0) -> Dict[str, float]:
    extra = _bounds_from_points(points, padding=padding)
    if primary is None:
        return extra or {"min_x": -5.0, "max_x": 5.0, "min_y": 0.0, "max_y": 0.0, "min_z": -5.0, "max_z": 5.0}
    if extra is None:
        return primary
    return {
        "min_x": min(primary["min_x"], extra["min_x"]),
        "max_x": max(primary["max_x"], extra["max_x"]),
        "min_y": min(primary["min_y"], extra["min_y"]),
        "max_y": max(primary["max_y"], extra["max_y"]),
        "min_z": min(primary["min_z"], extra["min_z"]),
        "max_z": max(primary["max_z"], extra["max_z"]),
    }


def _contour_segments_from_grid(grid: np.ndarray, bounds: Dict[str, float], resolution: float, max_segments: int = 12000) -> List[List[List[float]]]:
    if grid.size == 0:
        return []
    h, w = grid.shape
    segments: List[List[List[float]]] = []

    def x_at(col: int) -> float:
        return float(bounds["min_x"]) + col * resolution

    def z_at(row: int) -> float:
        return float(bounds["min_z"]) + row * resolution

    for r in range(h):
        for c in range(w):
            if not bool(grid[r, c]):
                continue
            if r == 0 or not bool(grid[r - 1, c]):
                segments.append([_round_point([x_at(c), z_at(r)], 3), _round_point([x_at(c + 1), z_at(r)], 3)])
            if r == h - 1 or not bool(grid[r + 1, c]):
                segments.append([_round_point([x_at(c), z_at(r + 1)], 3), _round_point([x_at(c + 1), z_at(r + 1)], 3)])
            if c == 0 or not bool(grid[r, c - 1]):
                segments.append([_round_point([x_at(c), z_at(r)], 3), _round_point([x_at(c), z_at(r + 1)], 3)])
            if c == w - 1 or not bool(grid[r, c + 1]):
                segments.append([_round_point([x_at(c + 1), z_at(r)], 3), _round_point([x_at(c + 1), z_at(r + 1)], 3)])
            if len(segments) >= max_segments:
                return segments
    return segments


def _yaw_to_quat(yaw_deg: float) -> Any:
    if mn is None:
        return None
    return mn.Quaternion.rotation(mn.Rad(math.radians(float(yaw_deg))), mn.Vector3(0.0, 1.0, 0.0))


def _yaw_to_agent_quat_coeffs(yaw_deg: float) -> List[float]:
    """Habitat AgentState expects quaternion coeffs [x, y, z, w]."""
    half = math.radians(float(yaw_deg)) / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def _yaw_to_degrees(yaw: float) -> float:
    """Accept either radians-like or degrees-like yaw values."""
    value = float(yaw)
    if abs(value) <= (2.0 * math.pi + 1e-6):
        return math.degrees(value)
    return value


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
        sensor_width: int = 64,
        sensor_height: int = 64,
        enable_depth_sensor: bool = False,
    ) -> None:
        self.scene_name = scene_name
        self.layout_path = layout_path
        self.data_dir = Path(data_dir)
        self.objects_dir = Path(objects_dir)
        self.enable_physics = bool(enable_physics)
        self.sensor_width = int(sensor_width)
        self.sensor_height = int(sensor_height)
        self.enable_depth_sensor = bool(enable_depth_sensor)
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
        sensor.resolution = [self.sensor_height, self.sensor_width]
        sensor.position = [0.0, 1.35, 0.0]

        sensors = [sensor]
        if self.enable_depth_sensor:
            depth_sensor = habitat_sim.CameraSensorSpec()
            depth_sensor.uuid = "depth"
            depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor.resolution = [self.sensor_height, self.sensor_width]
            depth_sensor.position = [0.0, 1.35, 0.0]
            sensors.append(depth_sensor)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensors
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

    def set_agent_pose(self, pose: Pose) -> None:
        if self.sim is None or habitat_sim is None:
            return
        state = habitat_sim.AgentState()
        state.position = np.asarray([pose.x, pose.y, pose.z], dtype=np.float32)
        state.rotation = _yaw_to_agent_quat_coeffs(_yaw_to_degrees(pose.yaw))
        self.sim.get_agent(0).set_state(state)

    def get_agent_pose(self) -> Pose:
        if self.sim is None:
            raise RuntimeError("Habitat simulator is not available")
        state = self.sim.get_agent(0).get_state()
        pos = state.position
        return Pose(float(pos[0]), float(pos[1]), float(pos[2]), 0.0)

    def observe(self, pose: Optional[Pose] = None) -> Dict[str, Any]:
        if self.sim is None:
            raise RuntimeError("Habitat simulator is not available")
        if pose is not None:
            self.set_agent_pose(pose)
        return dict(self.sim.get_sensor_observations())

    def snap_pose(self, pose: Pose) -> Pose:
        if self.sim is None:
            return pose
        point = np.asarray([pose.x, pose.y, pose.z], dtype=np.float32)
        snapped = self.pathfinder.snap_point(point)
        if snapped is None or np.any(np.isnan(snapped)):
            return pose
        return Pose(float(snapped[0]), float(snapped[1]), float(snapped[2]), pose.yaw)

    def shortest_path_points(self, start: Pose, goal: Pose) -> List[Pose]:
        if self.sim is None or habitat_sim is None or not hasattr(habitat_sim, "ShortestPath"):
            return [goal]
        path = habitat_sim.ShortestPath()
        path.requested_start = pose_to_array(start)
        path.requested_end = pose_to_array(goal)
        try:
            found = self.pathfinder.find_path(path)
        except Exception:
            found = False
        if not found:
            return [goal]
        points = getattr(path, "points", None) or []
        out: List[Pose] = []
        for point in points[1:]:
            out.append(Pose(float(point[0]), float(point[1]), float(point[2]), goal.yaw))
        return out or [goal]

    def geodesic_distance(self, start: Pose | Sequence[float], goal: Pose | Sequence[float]) -> float:
        pf = self.pathfinder
        s = pose_to_array(start) if isinstance(start, Pose) else np.asarray(start[:3], dtype=np.float32)
        g = pose_to_array(goal) if isinstance(goal, Pose) else np.asarray(goal[:3], dtype=np.float32)
        if hasattr(pf, "geodesic_distance"):
            try:
                distance = pf.geodesic_distance(s, g)
            except TypeError:
                distance = pf.geodesic_distance(s, [g])
            if distance is None or not np.isfinite(distance):
                return float("inf")
            return float(distance)

        if habitat_sim is None or not hasattr(habitat_sim, "ShortestPath"):
            return float("inf")
        try:
            path = habitat_sim.ShortestPath()
            path.requested_start = s
            path.requested_end = g
            found = pf.find_path(path)
            distance = getattr(path, "geodesic_distance", float("inf"))
            if not found or distance is None or not np.isfinite(distance):
                return float("inf")
            return float(distance)
        except Exception:
            return float("inf")

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

    def export_scene_geometry(
        self,
        *,
        grid_resolution: float = 0.25,
        max_random_points: int = 6000,
        max_navmesh_points: int = 2000,
    ) -> Dict[str, Any]:
        """Export lightweight scene geometry for offline debug visualization."""
        semantic_regions, scene_info_path = _semantic_regions_from_scene_info(self.data_dir, self.scene_name)
        layout_objects = _layout_objects_for_geometry(self.layout_path)
        semantic_points: List[List[float]] = []
        for region in semantic_regions:
            bbox = region.get("bbox") if isinstance(region, dict) else None
            if isinstance(bbox, dict):
                mn = bbox.get("min")
                mx = bbox.get("max")
                if isinstance(mn, list) and len(mn) >= 3:
                    semantic_points.append([float(mn[0]), float(mn[1]), float(mn[2])])
                if isinstance(mx, list) and len(mx) >= 3:
                    semantic_points.append([float(mx[0]), float(mx[1]), float(mx[2])])

        navigable_points: List[List[float]] = []
        navmesh_contours: List[List[List[float]]] = []
        bounds: Optional[Dict[str, float]] = None
        navmesh_available = self.sim is not None
        if self.sim is not None:
            pf = self.pathfinder
            random_points: List[List[float]] = []
            for _ in range(max(0, int(max_random_points))):
                try:
                    point = pf.get_random_navigable_point()
                except Exception:
                    break
                if point is None or np.any(np.isnan(point)):
                    continue
                random_points.append([float(point[0]), float(point[1]), float(point[2])])
            bounds = _bounds_from_points(random_points, padding=1.0)
            if len(random_points) > max_navmesh_points:
                stride = max(1, len(random_points) // max_navmesh_points)
                navigable_points = [_round_point(p, 4) for p in random_points[::stride][:max_navmesh_points]]
            else:
                navigable_points = [_round_point(p, 4) for p in random_points]
            bounds = _merge_bounds(bounds, semantic_points + [obj["position"] for obj in layout_objects], padding=1.0)
            res = max(0.05, float(grid_resolution))
            width = max(1, int(math.ceil((bounds["max_x"] - bounds["min_x"]) / res)))
            height = max(1, int(math.ceil((bounds["max_z"] - bounds["min_z"]) / res)))
            if width * height <= 350000:
                grid = np.zeros((height, width), dtype=bool)
                y = float(np.median([p[1] for p in random_points])) if random_points else float(bounds.get("min_y", 0.0))
                for r in range(height):
                    z = bounds["min_z"] + (r + 0.5) * res
                    for c in range(width):
                        x = bounds["min_x"] + (c + 0.5) * res
                        try:
                            grid[r, c] = bool(pf.is_navigable(np.asarray([x, y, z], dtype=np.float32)))
                        except Exception:
                            grid[r, c] = False
                navmesh_contours = _contour_segments_from_grid(grid, bounds, res)
        else:
            bounds = _merge_bounds(None, semantic_points + [obj["position"] for obj in layout_objects], padding=1.0)

        room_overlays = [
            {
                "room_id": region.get("room_id"),
                "label": region.get("label"),
                "source": region.get("source"),
                "polygon": region.get("polygon"),
                "center": region.get("center"),
                "object_count": region.get("object_count"),
            }
            for region in semantic_regions
        ]
        return {
            "schema_version": 1,
            "scene_name": self.scene_name,
            "source": {
                "mode": "semantic_region_first_navmesh_fallback",
                "scene_info_path": scene_info_path,
                "stage_glb": str(self.scene_paths.stage_glb) if self.scene_paths else None,
                "navmesh": str(self.scene_paths.navmesh) if self.scene_paths and self.scene_paths.navmesh else None,
                "layout_path": str(self.layout_path) if self.layout_path else None,
                "navmesh_available": bool(navmesh_available),
            },
            "bounds": bounds,
            "grid_resolution": float(grid_resolution),
            "navmesh_contours": navmesh_contours,
            "navigable_points": navigable_points,
            "semantic_regions": semantic_regions,
            "layout_objects": layout_objects,
            "room_overlays": room_overlays,
            "notes": [
                "semantic_regions are approximate AABB overlays when source=semantic_region_bbox",
                "navmesh_contours represent sampled navigable footprint boundaries",
            ],
        }


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
