from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from benchmark.schemas import Pose


try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised only on systems without OpenCV.
    cv2 = None  # type: ignore


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _candidate_xz(candidate: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    if "world_x" in candidate and "world_z" in candidate:
        return _safe_float(candidate.get("world_x")), _safe_float(candidate.get("world_z"))
    pos = candidate.get("pos_2d")
    if isinstance(pos, list) and len(pos) >= 2:
        return _safe_float(pos[0]), _safe_float(pos[1])
    return None


def _event_xz(event: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    for key in ("smoothed_pos_2d", "raw_pos_2d", "pos_2d"):
        pos = details.get(key)
        if isinstance(pos, list) and len(pos) >= 2:
            return _safe_float(pos[0]), _safe_float(pos[1])
    return None


def _short(value: Any, max_len: int = 36) -> str:
    text = str(value if value is not None else "")
    return text if len(text) <= max_len else text[: max_len - 1] + "~"


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return _short(value, 20)


@dataclass
class VideoSelector:
    index_pairs: set[Tuple[int, int]]
    subtask_ids: set[str]
    raw_tokens: set[str]
    matched_tokens: set[str]

    @classmethod
    def parse(cls, spec: str) -> "VideoSelector":
        index_pairs: set[Tuple[int, int]] = set()
        subtask_ids: set[str] = set()
        raw_tokens: set[str] = set()
        for token in (item.strip() for item in str(spec or "").split(",")):
            if not token:
                continue
            raw_tokens.add(token)
            parts = token.split(":")
            if len(parts) == 2:
                try:
                    index_pairs.add((int(parts[0]), int(parts[1])))
                    continue
                except Exception:
                    pass
            subtask_ids.add(token)
        return cls(index_pairs=index_pairs, subtask_ids=subtask_ids, raw_tokens=raw_tokens, matched_tokens=set())

    def should_record(self, episode_index: int, subtask_index: int, subtask_id: str) -> bool:
        matched: List[str] = []
        if (int(episode_index), int(subtask_index)) in self.index_pairs:
            matched.append(f"{episode_index}:{subtask_index}")
        if str(subtask_id) in self.subtask_ids:
            matched.append(str(subtask_id))
        for token in matched:
            self.matched_tokens.add(token)
        return bool(matched)

    def unmatched_tokens(self) -> List[str]:
        return sorted(self.raw_tokens - self.matched_tokens)


class SearchVideoRecorder:
    """Compose a per-subtask MP4 dashboard during a live Habitat search."""

    def __init__(
        self,
        *,
        output_dir: Path,
        context: Dict[str, Any],
        fps: float = 8.0,
        width: int = 1920,
        height: int = 1080,
        max_candidate_k: int = 5,
        include_observation_views: bool = False,
        expected_observation_views: int = 4,
        scene_geometry_path: Optional[Path] = None,
        scene_geometry: Optional[Dict[str, Any]] = None,
        video_layout: str = "dashboard",
        save_frames: bool = False,
        map_history: str = "recent",
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for --record-search-video, but cv2 could not be imported.")
        self.output_dir = Path(output_dir)
        self.context = dict(context)
        self.fps = float(fps)
        self.width = max(1280, int(width))
        self.height = max(720, int(height))
        self.max_candidate_k = int(max_candidate_k)
        self.include_observation_views = bool(include_observation_views)
        self.expected_observation_views = max(1, int(expected_observation_views))
        self.video_layout = str(video_layout or "dashboard")
        self.save_frames = bool(save_frames)
        self.map_history = str(map_history or "recent")
        self.video_path = self.output_dir / "search.mp4"
        self.frames_dir = self.output_dir / "frames"
        self.manifest_path = self.output_dir / "video_manifest.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.scene_geometry_path = Path(scene_geometry_path) if scene_geometry_path else None
        self.scene_geometry = dict(scene_geometry) if isinstance(scene_geometry, dict) else self._load_geometry(self.scene_geometry_path)
        self.geometry_loaded = bool(self.scene_geometry)
        self.frame_count = 0
        self.phase = "INIT"
        self.round_index: Optional[int] = None
        self.last_rgb: Optional[np.ndarray] = None
        self.current_pose: Optional[Pose] = None
        self.trajectory: List[Tuple[float, float]] = []
        self.candidates: List[Dict[str, Any]] = []
        self.selected_candidate: Optional[Dict[str, Any]] = None
        self.planned_pose: Optional[Pose] = None
        self.planning_info: Dict[str, Any] = {}
        self.memory_events: List[Dict[str, Any]] = []
        self.feedback_events: List[Dict[str, Any]] = []
        self.detections: List[Dict[str, Any]] = []
        self.observation_views: Dict[Tuple[str, int], Dict[int, Dict[str, Any]]] = {}
        self.selected_rounds: List[int] = []
        self.recorded_phases: List[str] = []
        self.recorded_views_per_round: Dict[str, List[int]] = {}
        self._writer = cv2.VideoWriter(
            str(self.video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"cannot create video writer: {self.video_path}")

    def record_observation_frame(
        self,
        *,
        rgb: np.ndarray,
        detections: Sequence[Dict[str, Any]],
        pose: Pose,
        view_index: int,
        phase: str,
        round_index: Optional[int],
    ) -> None:
        view_idx = int(view_index)
        round_idx = int(round_index or 0)
        phase_name = str(phase or "OBSERVE")
        bgr = self._rgb_to_bgr(rgb)
        self.last_rgb = bgr
        self.current_pose = pose
        self._add_pose(pose)
        self.observation_views.setdefault((phase_name, round_idx), {})[view_idx] = {
            "rgb": bgr,
            "detections": [dict(item) for item in detections[:30]],
            "pose": pose,
            "view_index": view_idx,
        }
        view_key = f"{round_idx}:{phase_name}"
        views = set(self.recorded_views_per_round.get(view_key, []))
        views.add(view_idx)
        self.recorded_views_per_round[view_key] = sorted(views)
        if not self.include_observation_views and view_idx != 0:
            return
        self.phase = phase_name
        self.round_index = round_idx
        self.detections = [dict(item) for item in detections[:20]]
        if not self.include_observation_views:
            self._write_frame()

    def record_observation_summary(self, *, phase: str, round_index: int, repeat: int = 2) -> None:
        phase_name = str(phase or "OBSERVE")
        self.phase = f"{phase_name}_SUMMARY"
        self.round_index = int(round_index)
        views = self.observation_views.get((phase_name, int(round_index)), {})
        self.detections = []
        for item in views.values():
            self.detections.extend(dict(det) for det in item.get("detections", [])[:20])
        self._write_frame(repeat=repeat)

    def record_planning_state(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        selected_candidate: Optional[Dict[str, Any]],
        planned_pose: Pose,
        planning_info: Dict[str, Any],
        memory_delta: Sequence[Dict[str, Any]],
        round_index: int,
    ) -> None:
        self.phase = "PLAN"
        self.round_index = int(round_index)
        if int(round_index) not in self.selected_rounds:
            self.selected_rounds.append(int(round_index))
        self.candidates = [dict(item) for item in candidates[: max(1, self.max_candidate_k)]]
        self.selected_candidate = dict(selected_candidate) if selected_candidate else None
        self.planned_pose = planned_pose
        self.planning_info = dict(planning_info)
        self.memory_events.extend(dict(item) for item in memory_delta[-40:])
        self._write_frame(repeat=2)

    def record_navigation_frame(self, *, rgb: np.ndarray, pose: Pose, path_so_far: Sequence[Dict[str, float]]) -> None:
        self.phase = "MOVE"
        self.last_rgb = self._rgb_to_bgr(rgb)
        self.current_pose = pose
        for item in path_so_far:
            if isinstance(item, dict):
                self._add_xz(_safe_float(item.get("x")), _safe_float(item.get("z")))
        self._write_frame()

    def record_feedback_state(
        self,
        *,
        feedback: Sequence[Dict[str, Any]],
        memory_events: Sequence[Dict[str, Any]],
        phase: str = "FEEDBACK",
    ) -> None:
        self.phase = str(phase)
        self.feedback_events.extend(dict(item) for item in feedback[-20:])
        self.memory_events.extend(dict(item) for item in memory_events[-40:])
        self._write_frame(repeat=2)

    def finish_subtask(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        self.phase = "SUMMARY"
        self.context.update(summary)
        self._write_frame(repeat=max(1, int(round(self.fps))))
        self._writer.release()
        manifest = {
            **self.context,
            "video_path": str(self.video_path),
            "frame_count": int(self.frame_count),
            "fps": float(self.fps),
            "selected_rounds": sorted(set(self.selected_rounds)),
            "recorded_phases": list(self.recorded_phases),
            "recorded_views_per_round": self.recorded_views_per_round,
            "geometry_loaded": bool(self.geometry_loaded),
            "geometry_path": str(self.scene_geometry_path) if self.scene_geometry_path else None,
            "layout_mode": self.video_layout,
            "map_history": self.map_history,
        }
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def _load_geometry(self, path: Optional[Path]) -> Dict[str, Any]:
        if path is None or not Path(path).exists():
            return {}
        try:
            with Path(path).open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _add_pose(self, pose: Pose) -> None:
        self._add_xz(float(pose.x), float(pose.z))

    def _add_xz(self, x: float, z: float) -> None:
        xz = (float(x), float(z))
        if not self.trajectory or math.hypot(self.trajectory[-1][0] - xz[0], self.trajectory[-1][1] - xz[1]) > 1e-4:
            self.trajectory.append(xz)
            self.trajectory = self.trajectory[-600:]

    def _rgb_to_bgr(self, rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb)
        if image.ndim == 2:
            if np.issubdtype(image.dtype, np.floating) and float(np.nanmax(image)) <= 1.5:
                image = image * 255.0
            image = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] >= 3:
            image = image[:, :, :3]
            if np.issubdtype(image.dtype, np.floating) and float(np.nanmax(image)) <= 1.5:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def _fit(self, image: Optional[np.ndarray], width: int, height: int, background: Tuple[int, int, int] = (18, 20, 24)) -> np.ndarray:
        output = np.full((height, width, 3), background, dtype=np.uint8)
        if image is None or image.size == 0:
            return output
        scale = min(width / image.shape[1], height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
        )
        y = (height - resized.shape[0]) // 2
        x = (width - resized.shape[1]) // 2
        output[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        return output

    def _put_lines(
        self,
        canvas: np.ndarray,
        lines: Iterable[str],
        origin: Tuple[int, int],
        *,
        scale: float = 0.44,
        color: Tuple[int, int, int] = (230, 235, 240),
        step: int = 24,
        max_chars: int = 78,
    ) -> None:
        x, y = origin
        for line in lines:
            cv2.putText(canvas, str(line)[:max_chars], (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += step

    def _project(self, x: float, z: float, bounds: Tuple[float, float, float, float], rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
        min_x, max_x, min_z, max_z = bounds
        rx, ry, rw, rh = rect
        px = rx + int(round((float(x) - min_x) / max(0.001, max_x - min_x) * rw))
        py = ry + rh - int(round((float(z) - min_z) / max(0.001, max_z - min_z) * rh))
        return px, py

    def _map_bounds(self) -> Tuple[float, float, float, float]:
        bounds = self.scene_geometry.get("bounds") if isinstance(self.scene_geometry, dict) else None
        if isinstance(bounds, dict) and all(k in bounds for k in ("min_x", "max_x", "min_z", "max_z")):
            return (
                _safe_float(bounds.get("min_x")) - 0.5,
                _safe_float(bounds.get("max_x")) + 0.5,
                _safe_float(bounds.get("min_z")) - 0.5,
                _safe_float(bounds.get("max_z")) + 0.5,
            )
        points: List[Tuple[float, float]] = self._history_points()
        target = self.context.get("target_position")
        if isinstance(target, list) and len(target) >= 2:
            points.append((_safe_float(target[0]), _safe_float(target[1])))
        if self.planned_pose is not None:
            points.append((float(self.planned_pose.x), float(self.planned_pose.z)))
        for cand in self.candidates:
            xz = _candidate_xz(cand)
            if xz:
                points.append(xz)
        for event in self._memory_events_for_map():
            xz = _event_xz(event)
            if xz:
                points.append(xz)
        if not points:
            return -5.0, 5.0, -5.0, 5.0
        xs = [p[0] for p in points]
        zs = [p[1] for p in points]
        return min(xs) - 1.0, max(xs) + 1.0, min(zs) - 1.0, max(zs) + 1.0

    def _history_points(self) -> List[Tuple[float, float]]:
        if self.map_history == "full":
            return list(self.trajectory)
        if self.map_history == "round":
            return list(self.trajectory[-80:])
        return list(self.trajectory[-200:])

    def _memory_events_for_map(self) -> List[Dict[str, Any]]:
        if self.map_history == "full":
            return list(self.memory_events[-400:])
        if self.map_history == "round":
            current_round = self.round_index
            return [e for e in self.memory_events[-200:] if e.get("round") == current_round]
        return list(self.memory_events[-100:])

    def _draw_geometry(self, canvas: np.ndarray, rect: Tuple[int, int, int, int], bounds: Tuple[float, float, float, float]) -> None:
        if not self.scene_geometry:
            return
        contours = self.scene_geometry.get("navmesh_contours") or []
        nav_pts = self.scene_geometry.get("navigable_points") or []
        regions = self.scene_geometry.get("semantic_regions") or []
        layout_objects = self.scene_geometry.get("layout_objects") or []
        for point in nav_pts[:1600]:
            if isinstance(point, list) and len(point) >= 3:
                px, py = self._project(_safe_float(point[0]), _safe_float(point[2]), bounds, rect)
                cv2.circle(canvas, (px, py), 1, (205, 213, 224), -1)
        for seg in contours[:16000]:
            if isinstance(seg, list) and len(seg) >= 2 and len(seg[0]) >= 2 and len(seg[1]) >= 2:
                p0 = self._project(_safe_float(seg[0][0]), _safe_float(seg[0][1]), bounds, rect)
                p1 = self._project(_safe_float(seg[1][0]), _safe_float(seg[1][1]), bounds, rect)
                cv2.line(canvas, p0, p1, (150, 163, 182), 1, cv2.LINE_AA)
        for region in regions[:80]:
            polygon = region.get("polygon") if isinstance(region, dict) else None
            if not isinstance(polygon, list) or len(polygon) < 3:
                continue
            pts = np.asarray([self._project(_safe_float(p[0]), _safe_float(p[1]), bounds, rect) for p in polygon if isinstance(p, list) and len(p) >= 2], dtype=np.int32)
            if len(pts) >= 3:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [pts], (215, 232, 255))
                cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
                cv2.polylines(canvas, [pts], True, (80, 130, 210), 1, cv2.LINE_AA)
            center = region.get("center") if isinstance(region, dict) else None
            if isinstance(center, list) and len(center) >= 3:
                px, py = self._project(_safe_float(center[0]), _safe_float(center[2]), bounds, rect)
                cv2.putText(canvas, f"R{region.get('region_id', '')}", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (55, 95, 170), 1, cv2.LINE_AA)
        for obj in layout_objects[:400]:
            pos = obj.get("position") if isinstance(obj, dict) else None
            if isinstance(pos, list) and len(pos) >= 3:
                px, py = self._project(_safe_float(pos[0]), _safe_float(pos[2]), bounds, rect)
                cv2.rectangle(canvas, (px - 2, py - 2), (px + 2, py + 2), (120, 115, 30), -1)

    def _draw_map(self, canvas: np.ndarray, rect: Tuple[int, int, int, int]) -> None:
        rx, ry, rw, rh = rect
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (247, 250, 253), -1)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (210, 218, 228), 1)
        bounds = self._map_bounds()
        self._draw_geometry(canvas, rect, bounds)
        target = self.context.get("target_position")
        if isinstance(target, list) and len(target) >= 2:
            px, py = self._project(_safe_float(target[0]), _safe_float(target[1]), bounds, rect)
            cv2.circle(canvas, (px, py), 8, (40, 40, 230), -1, cv2.LINE_AA)
            cv2.putText(canvas, "target", (px + 8, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 180), 1, cv2.LINE_AA)
        traj = self._history_points()
        if len(traj) >= 2:
            pts = [self._project(x, z, bounds, rect) for x, z in traj]
            cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, (20, 24, 32), 2, cv2.LINE_AA)
            for p0, p1 in zip(pts[:-1], pts[1:]):
                if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 14:
                    continue
                angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
                tip = p1
                a1 = (int(tip[0] - 8 * math.cos(angle - 0.45)), int(tip[1] - 8 * math.sin(angle - 0.45)))
                a2 = (int(tip[0] - 8 * math.cos(angle + 0.45)), int(tip[1] - 8 * math.sin(angle + 0.45)))
                cv2.line(canvas, tip, a1, (20, 24, 32), 1, cv2.LINE_AA)
                cv2.line(canvas, tip, a2, (20, 24, 32), 1, cv2.LINE_AA)
            cv2.circle(canvas, pts[-1], 5, (20, 24, 32), -1, cv2.LINE_AA)
        for idx, cand in enumerate(self.candidates[: self.max_candidate_k]):
            xz = _candidate_xz(cand)
            if not xz:
                continue
            px, py = self._project(xz[0], xz[1], bounds, rect)
            color = (70, 165, 80) if idx else (255, 120, 30)
            cv2.circle(canvas, (px, py), 5, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, str(idx + 1), (px + 5, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
        if self.planned_pose is not None:
            px, py = self._project(float(self.planned_pose.x), float(self.planned_pose.z), bounds, rect)
            cv2.drawMarker(canvas, (px, py), (220, 80, 20), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            cv2.putText(canvas, "planned", (px + 8, py + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 70, 20), 1, cv2.LINE_AA)
        for event in self._memory_events_for_map():
            xz = _event_xz(event)
            if not xz:
                continue
            px, py = self._project(xz[0], xz[1], bounds, rect)
            color = (40, 80, 220) if event.get("event") == "negative_feedback" else (255, 160, 50)
            cv2.circle(canvas, (px, py), 3, color, -1, cv2.LINE_AA)
        status = "geometry loaded" if self.geometry_loaded else "geometry unavailable"
        cv2.putText(canvas, f"2D map | {status}", (rx + 10, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 82, 98), 1, cv2.LINE_AA)

    def _draw_detection_boxes(self, image: np.ndarray, detections: Sequence[Dict[str, Any]]) -> np.ndarray:
        if image is None or getattr(image, "size", 0) == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        out = image.copy()
        h, w = out.shape[:2]
        for det in detections[:20]:
            bbox = det.get("bbox_xyxy")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            x0 = int(max(0, min(w - 1, _safe_float(bbox[0]))))
            y0 = int(max(0, min(h - 1, _safe_float(bbox[1]))))
            x1 = int(max(0, min(w - 1, _safe_float(bbox[2]))))
            y1 = int(max(0, min(h - 1, _safe_float(bbox[3]))))
            cv2.rectangle(out, (x0, y0), (x1, y1), (70, 220, 255), 1)
            cv2.putText(out, _short(det.get("label", ""), 24), (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 220, 255), 1, cv2.LINE_AA)
        return out

    def _draw_phase_overview(self, canvas: np.ndarray, rect: Tuple[int, int, int, int]) -> None:
        rx, ry, rw, rh = rect
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (240, 244, 248), -1)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (190, 202, 216), 1)
        title = f"{self.phase} | round={self.round_index}"
        cv2.putText(canvas, title, (rx + 24, ry + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (35, 45, 60), 2, cv2.LINE_AA)
        lines: List[str] = []
        selected = self.selected_candidate or {}
        if self.phase == "PLAN":
            lines.extend(
                [
                    f"planning_mode: {self.planning_info.get('planning_mode', '')}",
                    f"selected_room: {self.planning_info.get('selected_room_id', '')}",
                    f"pose_source: {self.planning_info.get('selected_pose_source', '')}",
                    f"room_switch: {self.planning_info.get('room_switch_reason', '')}",
                    f"selected_candidate: {_short(selected.get('label', ''), 42)}",
                    f"selected_score: {_fmt(selected.get('S_final', selected.get('score', '')))}",
                ]
            )
            if self.planned_pose is not None:
                lines.append(f"planned_pose: x={_fmt(self.planned_pose.x)} z={_fmt(self.planned_pose.z)} yaw={_fmt(self.planned_pose.yaw, 1)}")
            lines.append("")
            lines.append("Top candidates:")
            for idx, cand in enumerate(self.candidates[: min(8, self.max_candidate_k)]):
                lines.append(
                    f"{idx+1}. {_short(cand.get('label', ''), 30)} "
                    f"S={_fmt(cand.get('S_final', cand.get('score', '')))} "
                    f"backend={_short(cand.get('sim_backend', ''), 24)}"
                )
        elif self.phase == "FEEDBACK":
            lines.extend(
                [
                    f"feedback_events: {len(self.feedback_events)}",
                    f"memory_events_total: {len(self.memory_events)}",
                    f"current_round: {self.round_index}",
                    "",
                    "Latest feedback:",
                ]
            )
            for event in self.feedback_events[-8:]:
                lines.append(f"- {_short(event.get('label', ''), 32)} | {_short(event.get('reason', ''), 46)}")
            if not self.feedback_events:
                lines.append("- no negative feedback in this phase")
            lines.append("")
            lines.append("Current memory delta:")
            current = [e for e in self.memory_events if e.get("round") == self.round_index]
            for event in current[-8:]:
                details = event.get("details") if isinstance(event.get("details"), dict) else {}
                lines.append(f"- {_short(event.get('event', ''), 18)} {_short(event.get('label', ''), 30)} room={_short(details.get('room_id', ''), 18)}")
        elif self.phase == "SUMMARY":
            lines.extend(
                [
                    f"target: {self.context.get('target_object', '')}",
                    f"query: {self.context.get('query_label', '')}",
                    f"success/found: {self.context.get('success', self.context.get('found', ''))}",
                    f"perception_found: {self.context.get('perception_found', '')}",
                    f"final_dist: {_fmt(self.context.get('final_dist'))}",
                    f"path_length: {_fmt(self.context.get('path_length'))}",
                    f"steps: {self.context.get('steps', '')}",
                    f"geometry_loaded: {self.geometry_loaded}",
                    f"recorded_phases: {', '.join(self.recorded_phases[-10:])}",
                ]
            )
        else:
            lines.extend(
                [
                    "No RGB frame is available for this phase.",
                    "The map, planning panel, memory panel and timeline remain valid.",
                    f"trajectory_points: {len(self.trajectory)}",
                    f"candidates: {len(self.candidates)}",
                    f"memory_events: {len(self.memory_events)}",
                ]
            )
        y = ry + 92
        for line in lines:
            cv2.putText(canvas, str(line)[:110], (rx + 28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (38, 52, 70), 1, cv2.LINE_AA)
            y += 30
            if y > ry + rh - 24:
                break
        if self.last_rgb is not None and self.phase in {"PLAN", "FEEDBACK", "SUMMARY"}:
            inset_w = min(320, max(180, rw // 4))
            inset_h = min(220, max(130, rh // 4))
            inset = self._fit(self.last_rgb, inset_w, inset_h, background=(22, 26, 32))
            ix = rx + rw - inset_w - 20
            iy = ry + 20
            canvas[iy : iy + inset_h, ix : ix + inset_w] = inset
            cv2.rectangle(canvas, (ix, iy), (ix + inset_w, iy + inset_h), (80, 92, 110), 1)
            cv2.putText(canvas, "last RGB", (ix + 8, iy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 250, 255), 1, cv2.LINE_AA)

    def _draw_visual_panel(self, canvas: np.ndarray, rect: Tuple[int, int, int, int]) -> None:
        rx, ry, rw, rh = rect
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (12, 15, 20), -1)
        if self.phase in {"PLAN", "FEEDBACK", "SUMMARY"}:
            self._draw_phase_overview(canvas, rect)
            return
        if "OBSERVE" in self.phase and self.include_observation_views:
            source_phase = self.phase.replace("_SUMMARY", "")
            views = self.observation_views.get((source_phase, int(self.round_index or 0)), {})
            items = [views[k] for k in sorted(views)]
            if items:
                cols = 2 if len(items) <= 4 else int(math.ceil(math.sqrt(len(items))))
                rows = int(math.ceil(len(items) / cols))
                gap = 8
                cell_w = max(1, (rw - gap * (cols - 1)) // cols)
                cell_h = max(1, (rh - gap * (rows - 1)) // rows)
                for idx, item in enumerate(items):
                    cx = rx + (idx % cols) * (cell_w + gap)
                    cy = ry + (idx // cols) * (cell_h + gap)
                    img = self._draw_detection_boxes(item.get("rgb"), item.get("detections", []))
                    canvas[cy : cy + cell_h, cx : cx + cell_w] = self._fit(img, cell_w, cell_h)
                    pose = item.get("pose")
                    yaw = getattr(pose, "yaw", "")
                    label = f"view={item.get('view_index')} yaw={_fmt(yaw, 1)} det={len(item.get('detections', []))}"
                    cv2.rectangle(canvas, (cx, cy), (cx + min(cell_w, 260), cy + 22), (0, 0, 0), -1)
                    cv2.putText(canvas, label, (cx + 6, cy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (245, 250, 255), 1, cv2.LINE_AA)
                return
            if "_SUMMARY" in self.phase:
                self._draw_phase_overview(canvas, rect)
                return
        img = self._draw_detection_boxes(self.last_rgb, self.detections) if self.last_rgb is not None and "OBSERVE" in self.phase else self.last_rgb
        if img is None or getattr(img, "size", 0) == 0:
            self._draw_phase_overview(canvas, rect)
            return
        canvas[ry : ry + rh, rx : rx + rw] = self._fit(img, rw, rh)

    def _draw_planning_panel(self, canvas: np.ndarray, rect: Tuple[int, int, int, int]) -> None:
        rx, ry, rw, rh = rect
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (26, 31, 39), -1)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (62, 74, 90), 1)
        selected = self.selected_candidate or {}
        lines = [
            "Planning / Candidates",
            f"mode={self.planning_info.get('planning_mode', '')} room={self.planning_info.get('selected_room_id', '')}",
            f"pose={self.planning_info.get('selected_pose_source', '')} switch={self.planning_info.get('room_switch_reason', '')}",
            f"selected={_short(selected.get('label', ''), 26)} S={_fmt(selected.get('S_final', selected.get('score', '')))} backend={_short(selected.get('sim_backend', ''), 22)}",
        ]
        for idx, cand in enumerate(self.candidates[: self.max_candidate_k]):
            lines.append(
                f"{idx+1}. {_short(cand.get('label',''),22):22s} S={_fmt(cand.get('S_final', cand.get('score','')))} "
                f"{_short(cand.get('sim_backend',''),18)} room={_short(cand.get('current_room_id',''),16)}"
            )
        self._put_lines(canvas, lines, (rx + 12, ry + 24), scale=0.43, step=23, max_chars=96)

    def _draw_memory_panel(self, canvas: np.ndarray, rect: Tuple[int, int, int, int]) -> None:
        rx, ry, rw, rh = rect
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (24, 28, 34), -1)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (62, 74, 90), 1)
        lines = [
            "Memory / Feedback",
            f"events={len(self.memory_events)} feedback={len(self.feedback_events)} history={self.map_history}",
        ]
        current_round = self.round_index
        current_events = [e for e in self.memory_events if e.get("round") == current_round]
        for event in current_events[-8:]:
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            lines.append(f"{_short(event.get('event'),18)} {_short(event.get('label'),24)} room={_short(details.get('room_id',''),16)}")
        if not current_events:
            for event in self.memory_events[-5:]:
                details = event.get("details") if isinstance(event.get("details"), dict) else {}
                lines.append(f"{_short(event.get('event'),18)} {_short(event.get('label'),24)} room={_short(details.get('room_id',''),16)}")
        for event in self.feedback_events[-5:]:
            lines.append(f"FB {_short(event.get('label',''),24)} {_short(event.get('reason',''),34)}")
        self._put_lines(canvas, lines, (rx + 12, ry + 24), scale=0.42, step=22, color=(225, 230, 235), max_chars=96)

    def _draw_timeline(self, canvas: np.ndarray, rect: Tuple[int, int, int, int]) -> None:
        rx, ry, rw, rh = rect
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (10, 12, 16), -1)
        phases = ["PRE_OBSERVE", "PLAN", "MOVE", "POST_OBSERVE", "FEEDBACK", "SUMMARY"]
        active = self.phase.replace("_SUMMARY", "")
        seg_w = rw // len(phases)
        for idx, phase in enumerate(phases):
            x0 = rx + idx * seg_w
            x1 = rx + (idx + 1) * seg_w - 8
            color = (70, 130, 220) if phase == active or self.phase.startswith(phase) else (54, 63, 75)
            cv2.rectangle(canvas, (x0, ry + 16), (x1, ry + rh - 14), color, -1)
            cv2.putText(canvas, phase, (x0 + 8, ry + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 248, 252), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"round={self.round_index} frame={self.frame_count}", (rx + 8, ry + rh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (165, 190, 220), 1, cv2.LINE_AA)

    def _write_frame(self, *, repeat: int = 1) -> None:
        if self.phase not in self.recorded_phases:
            self.recorded_phases.append(self.phase)
        canvas = np.full((self.height, self.width, 3), (18, 20, 24), dtype=np.uint8)
        header = (
            f"{self.context.get('episode_index', '')}:{self.context.get('subtask_index', '')} "
            f"{self.context.get('target_object', '')} | query={self.context.get('query_label', '')} "
            f"| phase={self.phase} round={self.round_index}"
        )
        cv2.rectangle(canvas, (0, 0), (self.width, 64), (10, 12, 16), -1)
        cv2.putText(canvas, header[:150], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (250, 250, 250), 2, cv2.LINE_AA)
        pad = 20
        timeline_h = 76
        body_top = 82
        body_bottom = self.height - timeline_h - 22
        right_w = max(420, int(self.width * 0.28)) if self.video_layout == "compact" else max(520, int(self.width * 0.34))
        left_w = self.width - right_w - pad * 3
        left_h = body_bottom - body_top
        right_x = pad * 2 + left_w
        map_h = int(left_h * 0.42)
        plan_h = int(left_h * 0.28)
        mem_h = left_h - map_h - plan_h - pad * 2
        self._draw_visual_panel(canvas, (pad, body_top, left_w, left_h))
        self._draw_map(canvas, (right_x, body_top, right_w, map_h))
        self._draw_planning_panel(canvas, (right_x, body_top + map_h + pad, right_w, plan_h))
        self._draw_memory_panel(canvas, (right_x, body_top + map_h + plan_h + pad * 2, right_w, mem_h))
        self._draw_timeline(canvas, (pad, self.height - timeline_h - 8, self.width - pad * 2, timeline_h))
        footer = (
            f"layout={self.context.get('layout_id','')} state={self.context.get('state_index','')} "
            f"found={self.context.get('found', '')} final_dist={self.context.get('final_dist', '')} "
            f"geometry={'yes' if self.geometry_loaded else 'no'}"
        )
        cv2.putText(canvas, footer[:160], (20, self.height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 185, 220), 1, cv2.LINE_AA)
        for _ in range(max(1, int(repeat))):
            self._writer.write(canvas)
            if self.save_frames:
                frame_path = self.frames_dir / f"frame_{self.frame_count:06d}_{self.phase}.png"
                cv2.imwrite(str(frame_path), canvas)
            self.frame_count += 1
