from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from benchmark.habitat_adapter import HabitatLayoutAdapter, euclidean_distance
from benchmark.schemas import Pose
from remote_vision_server.client import RemoteOpenVocabDetector


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class ObservationBatch:
    detections: List[Dict[str, Any]]
    records: List[Dict[str, Any]]
    elapsed_seconds: float


@dataclass
class StepResult:
    final_pose: Pose
    path_length: float
    steps: int
    path: List[Dict[str, float]]


def _camera_to_world(points_cam: np.ndarray, agent_pos: np.ndarray, agent_heading_deg: float) -> np.ndarray:
    if len(points_cam) == 0:
        return np.empty((0, 3), dtype=np.float32)
    theta = math.radians(agent_heading_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy, cz = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    wx = cx * cos_t - cz * sin_t + agent_pos[0]
    wy = agent_pos[1] - cy
    wz = -cx * sin_t - cz * cos_t + agent_pos[2]
    return np.stack([wx, wy, wz], axis=-1).astype(np.float32)


def _jsonable_detection(det: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in det.items():
        if key in {"mask", "_rgb"}:
            continue
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        else:
            out[key] = value
    emb = out.get("clip_embedding")
    if isinstance(emb, list) and len(emb) > 16:
        out["clip_embedding_dim"] = len(emb)
        out["clip_embedding"] = emb[:8]
    return out


class HabitatVisionLoop:
    """Real Habitat RGB-D loop with remote open-vocabulary perception."""

    def __init__(
        self,
        *,
        data_dir: str = "hm3d",
        objects_dir: str = "objects",
        load_layout_objects: bool = True,
        require_habitat: bool = True,
        sensor_width: int = 640,
        sensor_height: int = 480,
        max_depth: float = 5.0,
        step_size: float = 0.5,
        remote_vision_base_url: Optional[str] = None,
        remote_vision_use_ssh_tunnel: bool = False,
        remote_vision_ssh_host: str = "7.216.187.6",
        remote_vision_ssh_port: int = 30180,
        remote_vision_ssh_user: str = "root",
        remote_vision_ssh_password: Optional[str] = None,
        remote_vision_remote_port: int = 8010,
        remote_vision_local_port: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.objects_dir = Path(objects_dir)
        self.load_layout_objects = bool(load_layout_objects)
        self.require_habitat = bool(require_habitat)
        self.sensor_width = int(sensor_width)
        self.sensor_height = int(sensor_height)
        self.max_depth = float(max_depth)
        self.step_size = float(step_size)
        self.intrinsics = CameraIntrinsics(
            fx=self.sensor_width / 2.0,
            fy=self.sensor_width / 2.0,
            cx=(self.sensor_width - 1.0) / 2.0,
            cy=(self.sensor_height - 1.0) / 2.0,
        )
        self.detector = RemoteOpenVocabDetector(
            base_url=remote_vision_base_url,
            use_ssh_tunnel=remote_vision_use_ssh_tunnel,
            ssh_host=remote_vision_ssh_host,
            ssh_port=remote_vision_ssh_port,
            ssh_user=remote_vision_ssh_user,
            ssh_password=remote_vision_ssh_password,
            remote_port=remote_vision_remote_port,
            local_port=remote_vision_local_port,
            return_clip_embedding=True,
        )
        self.adapter: Optional[HabitatLayoutAdapter] = None
        self.current_pose: Optional[Pose] = None
        self.scene_name = ""
        self.layout_path: Optional[Path] = None
        self.perception_calls = 0
        self.perception_failures = 0
        self.perception_elapsed = 0.0
        self.detection_count = 0

    def close(self) -> None:
        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None
        self.detector.close()

    def reset_layout(
        self,
        scene_name: str,
        layout_path: Path,
        start_pose: Pose,
        *,
        sample_start_seed: Optional[int] = None,
    ) -> None:
        if self.adapter is not None:
            self.adapter.close()
        self.scene_name = scene_name
        self.layout_path = Path(layout_path)
        self.adapter = HabitatLayoutAdapter(
            scene_name,
            layout_path=self.layout_path,
            data_dir=self.data_dir,
            objects_dir=self.objects_dir,
            enable_physics=False,
            load_layout_objects=self.load_layout_objects,
            require_habitat=self.require_habitat,
            sensor_width=self.sensor_width,
            sensor_height=self.sensor_height,
            enable_depth_sensor=True,
        )
        if sample_start_seed is None:
            pose = self.adapter.snap_pose(start_pose)
        else:
            pose = self.adapter.sample_start_pose(random.Random(int(sample_start_seed)))
        self.adapter.set_agent_pose(pose)
        self.current_pose = pose

    def set_pose(self, pose: Pose, *, sample_start_seed: Optional[int] = None) -> Pose:
        if self.adapter is None:
            raise RuntimeError("HabitatVisionLoop.reset_layout must be called before set_pose().")
        if sample_start_seed is None:
            next_pose = self.adapter.snap_pose(pose)
        else:
            next_pose = self.adapter.sample_start_pose(random.Random(int(sample_start_seed)))
        self.adapter.set_agent_pose(next_pose)
        self.current_pose = next_pose
        return next_pose

    def observe(self, *, n_views: int, text_prompt: str, step: int, state_index: int, layout_id: str) -> ObservationBatch:
        if self.adapter is None or self.current_pose is None:
            raise RuntimeError("HabitatVisionLoop.reset_layout must be called before observe().")
        start = time.time()
        detections: List[Dict[str, Any]] = []
        records: List[Dict[str, Any]] = []
        base = self.current_pose
        view_count = max(1, int(n_views))
        for view_idx in range(view_count):
            yaw = float(base.yaw) + (360.0 * view_idx / view_count)
            view_pose = Pose(base.x, base.y, base.z, yaw)
            obs = self.adapter.observe(view_pose)
            rgb = obs.get("color")
            depth = obs.get("depth")
            if rgb is None or depth is None:
                continue
            rgb_np = np.asarray(rgb)
            if rgb_np.ndim == 3 and rgb_np.shape[2] >= 3:
                rgb_np = rgb_np[:, :, :3].copy()
            depth_np = np.asarray(depth).astype(np.float32)
            call_start = time.time()
            try:
                view_dets = self.detector.detect(
                    rgb_np,
                    text_prompt,
                    box_threshold=0.35,
                    text_threshold=0.35,
                    return_masks=True,
                    return_clip_embedding=True,
                )
                self.perception_calls += 1
                self.perception_elapsed += time.time() - call_start
            except Exception:
                self.perception_calls += 1
                self.perception_failures += 1
                raise
            agent_pos = np.asarray([view_pose.x, view_pose.y, view_pose.z], dtype=np.float32)
            for det in view_dets:
                self._attach_depth_position(det, depth_np, agent_pos, yaw)
                det["_view_index"] = view_idx
                det["_state_index"] = int(state_index)
                det["_layout_id"] = layout_id
                detections.append(det)
                self.detection_count += 1
                record = _jsonable_detection(det)
                record.update(
                    {
                        "scene_name": self.scene_name,
                        "layout_id": layout_id,
                        "state_index": int(state_index),
                        "step": int(step),
                        "view_index": view_idx,
                        "agent_pose": {"x": view_pose.x, "y": view_pose.y, "z": view_pose.z, "yaw": view_pose.yaw},
                    }
                )
                records.append(record)
        self.adapter.set_agent_pose(base)
        self.current_pose = base
        return ObservationBatch(detections=detections, records=records, elapsed_seconds=time.time() - start)

    def _attach_depth_position(
        self,
        det: Dict[str, Any],
        depth: np.ndarray,
        agent_pos: np.ndarray,
        heading_deg: float,
    ) -> None:
        bbox = det.get("bbox_xyxy") or [0, 0, 0, 0]
        mask = det.get("mask")
        valid_depths = np.asarray([], dtype=np.float32)
        if isinstance(mask, np.ndarray) and mask.shape[:2] == depth.shape[:2]:
            mask_depths = depth[mask.astype(bool)]
            valid_depths = mask_depths[(mask_depths > 0) & (mask_depths < self.max_depth)]
        if len(valid_depths) == 0:
            cx = int((float(bbox[0]) + float(bbox[2])) / 2.0)
            cy = int((float(bbox[1]) + float(bbox[3])) / 2.0)
            cx = min(max(cx, 0), depth.shape[1] - 1)
            cy = min(max(cy, 0), depth.shape[0] - 1)
            d = float(depth[cy, cx])
            if d <= 0 or d >= self.max_depth:
                det["pos_3d"] = None
                det["pos_2d"] = None
                det["depth_median"] = None
                return
            valid_depths = np.asarray([d], dtype=np.float32)
        median_d = float(np.median(valid_depths))
        if isinstance(mask, np.ndarray) and np.any(mask):
            ys, xs = np.where(mask.astype(bool))
            u = float(np.mean(xs))
            v = float(np.mean(ys))
        else:
            u = float((float(bbox[0]) + float(bbox[2])) / 2.0)
            v = float((float(bbox[1]) + float(bbox[3])) / 2.0)
        cam_x = (u - self.intrinsics.cx) * median_d / self.intrinsics.fx
        cam_y = (v - self.intrinsics.cy) * median_d / self.intrinsics.fy
        cam_z = median_d
        world = _camera_to_world(np.asarray([[cam_x, cam_y, cam_z]], dtype=np.float32), agent_pos, heading_deg)
        wx, wy, wz = float(world[0, 0]), float(world[0, 1]), float(world[0, 2])
        det["pos_3d"] = [wx, wy, wz]
        det["pos_2d"] = [wx, wz]
        det["depth_median"] = median_d

    def step_to(self, candidate_pose: Pose, *, max_micro_steps: int) -> StepResult:
        if self.adapter is None or self.current_pose is None:
            raise RuntimeError("HabitatVisionLoop.reset_layout must be called before step_to().")
        goal = self.adapter.snap_pose(candidate_pose)
        raw_points = self.adapter.shortest_path_points(self.current_pose, goal)
        points = raw_points[: max(1, int(max_micro_steps))]
        if not points:
            points = [goal]
        path_length = 0.0
        last = self.current_pose
        for pose in points:
            path_length += euclidean_distance(last, pose)
            self.adapter.set_agent_pose(pose)
            last = pose
        self.current_pose = last
        return StepResult(
            final_pose=last,
            path_length=round(float(path_length), 6),
            steps=len(points),
            path=[{"x": p.x, "y": p.y, "z": p.z, "yaw": p.yaw} for p in points],
        )

    def perception_summary(self) -> Dict[str, Any]:
        return {
            "calls": self.perception_calls,
            "failures": self.perception_failures,
            "detections": self.detection_count,
            "avg_elapsed_seconds": self.perception_elapsed / self.perception_calls if self.perception_calls else 0.0,
        }
