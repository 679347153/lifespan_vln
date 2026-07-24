from __future__ import annotations

import json
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


def _pose_dict(pose: Optional[Pose]) -> Dict[str, float]:
    if pose is None:
        return {}
    return {"x": float(pose.x), "y": float(pose.y), "z": float(pose.z), "yaw": float(pose.yaw)}


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
        width: int = 1600,
        height: int = 900,
        max_candidate_k: int = 5,
        include_observation_views: bool = False,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for --record-search-video, but cv2 could not be imported.")
        self.output_dir = Path(output_dir)
        self.context = dict(context)
        self.fps = float(fps)
        self.width = max(1600, int(width))
        self.height = max(900, int(height))
        self.max_candidate_k = int(max_candidate_k)
        self.include_observation_views = bool(include_observation_views)
        self.video_path = self.output_dir / "search.mp4"
        self.frames_dir = self.output_dir / "frames"
        self.manifest_path = self.output_dir / "video_manifest.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
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
        self.selected_rounds: List[int] = []
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
        if not self.include_observation_views and int(view_index) != 0:
            return
        self.phase = str(phase)
        self.round_index = round_index
        self.last_rgb = self._rgb_to_bgr(rgb)
        self.current_pose = pose
        self.detections = [dict(item) for item in detections[:20]]
        self._add_pose(pose)
        self._write_frame()

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
        self.trajectory = [(_safe_float(item.get("x")), _safe_float(item.get("z"))) for item in path_so_far if isinstance(item, dict)]
        if not self.trajectory:
            self._add_pose(pose)
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
        }
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def _add_pose(self, pose: Pose) -> None:
        xz = (float(pose.x), float(pose.z))
        if not self.trajectory or self.trajectory[-1] != xz:
            self.trajectory.append(xz)
            self.trajectory = self.trajectory[-400:]

    def _rgb_to_bgr(self, rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb)
        if image.ndim == 2:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] >= 3:
            image = image[:, :, :3].astype(np.uint8)
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
        scale: float = 0.45,
        color: Tuple[int, int, int] = (230, 235, 240),
        step: int = 24,
    ) -> None:
        x, y = origin
        for line in lines:
            cv2.putText(canvas, str(line)[:80], (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += step

    def _project(self, x: float, z: float, bounds: Tuple[float, float, float, float], rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
        min_x, max_x, min_z, max_z = bounds
        rx, ry, rw, rh = rect
        px = rx + int(round((float(x) - min_x) / max(0.001, max_x - min_x) * rw))
        py = ry + rh - int(round((float(z) - min_z) / max(0.001, max_z - min_z) * rh))
        return px, py

    def _map_bounds(self) -> Tuple[float, float, float, float]:
        points: List[Tuple[float, float]] = list(self.trajectory)
        target = self.context.get("target_position")
        if isinstance(target, list) and len(target) >= 2:
            points.append((_safe_float(target[0]), _safe_float(target[1])))
        if self.planned_pose is not None:
            points.append((float(self.planned_pose.x), float(self.planned_pose.z)))
        for cand in self.candidates:
            xz = _candidate_xz(cand)
            if xz:
                points.append(xz)
        for event in self.memory_events[-80:]:
            xz = _event_xz(event)
            if xz:
                points.append(xz)
        if not points:
            return -5.0, 5.0, -5.0, 5.0
        xs = [p[0] for p in points]
        zs = [p[1] for p in points]
        return min(xs) - 1.0, max(xs) + 1.0, min(zs) - 1.0, max(zs) + 1.0

    def _draw_map(self, canvas: np.ndarray, rect: Tuple[int, int, int, int]) -> None:
        rx, ry, rw, rh = rect
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (245, 248, 252), -1)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (210, 218, 228), 1)
        bounds = self._map_bounds()
        target = self.context.get("target_position")
        if isinstance(target, list) and len(target) >= 2:
            px, py = self._project(_safe_float(target[0]), _safe_float(target[1]), bounds, rect)
            cv2.circle(canvas, (px, py), 7, (40, 40, 230), -1, cv2.LINE_AA)
            cv2.putText(canvas, "target", (px + 8, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 180), 1, cv2.LINE_AA)
        if len(self.trajectory) >= 2:
            pts = np.asarray([self._project(x, z, bounds, rect) for x, z in self.trajectory], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (20, 24, 32), 2, cv2.LINE_AA)
            cv2.circle(canvas, tuple(pts[-1]), 5, (20, 24, 32), -1, cv2.LINE_AA)
        for idx, cand in enumerate(self.candidates[: self.max_candidate_k]):
            xz = _candidate_xz(cand)
            if not xz:
                continue
            px, py = self._project(xz[0], xz[1], bounds, rect)
            color = (80, 170, 70) if idx else (255, 120, 30)
            cv2.circle(canvas, (px, py), 5, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, str(idx + 1), (px + 5, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
        if self.planned_pose is not None:
            px, py = self._project(float(self.planned_pose.x), float(self.planned_pose.z), bounds, rect)
            cv2.drawMarker(canvas, (px, py), (220, 80, 20), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        for event in self.memory_events[-60:]:
            xz = _event_xz(event)
            if not xz:
                continue
            px, py = self._project(xz[0], xz[1], bounds, rect)
            color = (255, 160, 50) if event.get("observation_phase") in {"pre_move", "post_move"} else (220, 190, 70)
            cv2.circle(canvas, (px, py), 3, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, "2D search map", (rx + 10, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (75, 85, 100), 1, cv2.LINE_AA)

    def _draw_detection_boxes(self, image: np.ndarray) -> np.ndarray:
        out = image.copy()
        h, w = out.shape[:2]
        for det in self.detections[:20]:
            bbox = det.get("bbox_xyxy")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            x0 = int(max(0, min(w - 1, _safe_float(bbox[0]))))
            y0 = int(max(0, min(h - 1, _safe_float(bbox[1]))))
            x1 = int(max(0, min(w - 1, _safe_float(bbox[2]))))
            y1 = int(max(0, min(h - 1, _safe_float(bbox[3]))))
            cv2.rectangle(out, (x0, y0), (x1, y1), (70, 220, 255), 1)
            cv2.putText(out, str(det.get("label", ""))[:24], (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 220, 255), 1, cv2.LINE_AA)
        return out

    def _write_frame(self, *, repeat: int = 1) -> None:
        canvas = np.full((self.height, self.width, 3), (18, 20, 24), dtype=np.uint8)
        header = (
            f"{self.context.get('episode_index', '')}:{self.context.get('subtask_index', '')} "
            f"{self.context.get('target_object', '')} | phase={self.phase} round={self.round_index}"
        )
        cv2.rectangle(canvas, (0, 0), (self.width, 58), (10, 12, 16), -1)
        cv2.putText(canvas, header[:110], (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (250, 250, 250), 2, cv2.LINE_AA)
        rgb_panel = self._draw_detection_boxes(self.last_rgb) if self.last_rgb is not None and self.phase in {"PRE_OBSERVE", "POST_OBSERVE"} else self.last_rgb
        canvas[72:692, 20:1020] = self._fit(rgb_panel, 1000, 620)
        self._draw_map(canvas, (1040, 72, 540, 350))
        selected = self.selected_candidate or {}
        candidate_lines = [
            "Planning",
            f"mode: {self.planning_info.get('planning_mode', '')}",
            f"room: {self.planning_info.get('selected_room_id', '')}",
            f"pose: {self.planning_info.get('selected_pose_source', '')}",
            f"selected: {selected.get('label', '')} score={selected.get('S_final', selected.get('score', ''))}",
        ]
        for idx, cand in enumerate(self.candidates[: self.max_candidate_k]):
            candidate_lines.append(f"{idx+1}. {cand.get('label','')} S={cand.get('S_final', cand.get('score',''))} {cand.get('sim_backend','')}")
        self._put_lines(canvas, candidate_lines, (1048, 456), scale=0.45, step=24)
        memory_lines = [
            "Memory / Feedback",
            f"events: {len(self.memory_events)} feedback: {len(self.feedback_events)}",
        ]
        for event in self.memory_events[-8:]:
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            memory_lines.append(f"{event.get('event')} {event.get('label')} {details.get('room_id', '')}")
        for event in self.feedback_events[-4:]:
            memory_lines.append(f"FB {event.get('label','')} {event.get('reason','')}")
        self._put_lines(canvas, memory_lines, (1048, 660), scale=0.42, step=23, color=(225, 230, 235))
        footer = (
            f"layout={self.context.get('layout_id','')} state={self.context.get('state_index','')} "
            f"frames={self.frame_count} found={self.context.get('found', '')} final_dist={self.context.get('final_dist', '')}"
        )
        cv2.putText(canvas, footer[:140], (20, self.height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 185, 220), 1, cv2.LINE_AA)
        for _ in range(max(1, int(repeat))):
            self._writer.write(canvas)
            self.frame_count += 1
