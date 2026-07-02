from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from RAANav.AgenticRAG.semantic_map import Floor, Object, Room

from .common import euclidean_2d, is_noise_detection_label, normalize_label, sanitize_detection_label


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _pos2d_from_detection(det: Dict[str, Any]) -> Optional[List[float]]:
    pos = det.get("pos_2d")
    if isinstance(pos, dict) and "x" in pos:
        return [float(pos["x"]), float(pos.get("y", pos.get("z", 0.0)))]
    if isinstance(pos, list) and len(pos) >= 2:
        return [float(pos[0]), float(pos[1])]
    pos3 = det.get("pos_3d")
    if isinstance(pos3, list) and len(pos3) >= 3:
        return [float(pos3[0]), float(pos3[2])]
    return None


def _region_from_pos2d(pos2d: List[float], radius: float = 0.25) -> List[Dict[str, float]]:
    x, z = float(pos2d[0]), float(pos2d[1])
    return [
        {"x": x - radius, "y": z - radius},
        {"x": x + radius, "y": z - radius},
        {"x": x + radius, "y": z + radius},
        {"x": x - radius, "y": z + radius},
    ]


def _clip_cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class MemoryEvent:
    event: str
    track_id: str
    label: str
    state_index: int
    layout_id: str
    step: int
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "track_id": self.track_id,
            "label": self.label,
            "state_index": self.state_index,
            "layout_id": self.layout_id,
            "step": self.step,
            "details": self.details,
        }


@dataclass
class RoomObjectEvidence:
    obj_id: str
    seen_count: int = 0
    last_seen_state_index: int = 0
    last_seen_layout_id: str = ""
    room_exist_prob: float = 0.0
    negative_count: int = 0
    positions_2d: List[List[float]] = field(default_factory=list)
    first_seen_time: str = ""
    last_seen_time: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obj_id": self.obj_id,
            "seen_count": self.seen_count,
            "last_seen_state_index": self.last_seen_state_index,
            "last_seen_layout_id": self.last_seen_layout_id,
            "room_exist_prob": round(float(self.room_exist_prob), 6),
            "negative_count": self.negative_count,
            "positions_2d": self.positions_2d,
            "first_seen_time": self.first_seen_time,
            "last_seen_time": self.last_seen_time,
        }


@dataclass
class RoomMemory:
    room_id: str
    room_label: str
    object_refs: Dict[str, RoomObjectEvidence] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_id": self.room_id,
            "room_label": self.room_label,
            "object_refs": {
                obj_id: ref.to_dict()
                for obj_id, ref in sorted(self.object_refs.items(), key=lambda kv: kv[0])
            },
            "track_count": len(self.object_refs),
        }


class SceneMemory:
    """Scene-level long-term memory updated from real visual observations."""

    def __init__(
        self,
        scene_name: str,
        *,
        merge_distance: float = 0.75,
        migration_distance: float = 4.0,
        clip_migration_threshold: float = 0.92,
        room_grid_size_m: float = 4.0,
    ) -> None:
        self.scene_name = scene_name
        self.merge_distance = float(merge_distance)
        self.migration_distance = float(migration_distance)
        self.clip_migration_threshold = float(clip_migration_threshold)
        self.room_grid_size_m = float(room_grid_size_m)
        self.room_assignment_mode = "pseudo_grid"
        self._objects: Dict[str, Object] = {}
        self._counter = 0
        self.rooms: Dict[str, RoomMemory] = {}

    def __len__(self) -> int:
        return len(self._objects)

    def objects(self) -> List[Object]:
        return list(self._objects.values())

    def floors(self) -> List[Floor]:
        if self.rooms:
            room_objs: List[Room] = []
            for room_id, room_mem in sorted(self.rooms.items(), key=lambda kv: kv[0]):
                objects = [
                    self._objects[obj_id]
                    for obj_id in sorted(room_mem.object_refs)
                    if obj_id in self._objects
                ]
                room_objs.append(
                    Room(
                        room_id=room_id,
                        room_name={"1": room_mem.room_label},
                        objects=objects,
                        floor_id="F0",
                        N=len(objects),
                        description={"1": f"{self.room_assignment_mode} room"},
                    )
                )
            return [Floor(floor_id="F0", rooms=room_objs, description=f"scene_memory={self.scene_name}")]

        rooms: Dict[str, List[Object]] = {}
        for obj in self._objects.values():
            rooms.setdefault(obj.room_id or "unknown", []).append(obj)
        room_objs = [
            Room(
                room_id=room_id,
                room_name={"1": f"region_{room_id}"},
                objects=sorted(items, key=lambda item: item.obj_id),
                floor_id="F0",
                N=len(items),
            )
            for room_id, items in sorted(rooms.items(), key=lambda kv: kv[0])
        ]
        return [Floor(floor_id="F0", rooms=room_objs, description=f"scene_memory={self.scene_name}")]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_name": self.scene_name,
            "track_count": len(self._objects),
            "tracks": [obj.to_dict() for obj in sorted(self._objects.values(), key=lambda item: item.obj_id)],
            "rooms": {
                room_id: room.to_dict()
                for room_id, room in sorted(self.rooms.items(), key=lambda kv: kv[0])
            },
            "room_assignment": {
                "mode": self.room_assignment_mode,
                "grid_size_m": self.room_grid_size_m,
            },
        }

    def _new_track_id(self, label: str) -> str:
        self._counter += 1
        return f"{self.scene_name}:{normalize_label(label) or 'object'}:{self._counter:05d}"

    def assign_room_id(self, pos2d: List[float], explicit_room_id: str = "") -> str:
        if explicit_room_id and explicit_room_id != "unknown":
            return str(explicit_room_id)
        grid = self.room_grid_size_m if self.room_grid_size_m > 0 else 4.0
        gx = math.floor(float(pos2d[0]) / grid)
        gz = math.floor(float(pos2d[1]) / grid)
        return f"pseudo_room_{gx}_{gz}"

    def _room_memory(self, room_id: str) -> RoomMemory:
        if room_id not in self.rooms:
            self.rooms[room_id] = RoomMemory(room_id=room_id, room_label=room_id)
        return self.rooms[room_id]

    def _normalize_room_belief(self, belief: Dict[str, float]) -> Dict[str, float]:
        clipped = {str(k): max(0.0, float(v)) for k, v in belief.items() if str(k)}
        total = sum(clipped.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in sorted(clipped.items(), key=lambda kv: kv[0])}

    def _update_room_memory(
        self,
        obj: Object,
        *,
        room_id: str,
        pos2d: List[float],
        state_index: int,
        layout_id: str,
        step: int,
        event_type: str,
        now: str,
    ) -> Dict[str, Any]:
        stats = obj.cooccur_stats if isinstance(obj.cooccur_stats, dict) else {}
        previous_room_id = str(stats.get("current_room_id") or obj.room_id or "")
        belief = dict(stats.get("room_belief") or {})
        for key in list(belief):
            belief[key] = float(belief.get(key, 0.0) or 0.0) * 0.95
        belief[room_id] = float(belief.get(room_id, 0.0) or 0.0) + 0.25
        belief = self._normalize_room_belief(belief) or {room_id: 1.0}

        history = list(stats.get("room_history", []))
        history.append(
            {
                "room_id": room_id,
                "layout_id": layout_id,
                "state_index": int(state_index),
                "step": int(step),
                "pos_2d": [round(float(pos2d[0]), 4), round(float(pos2d[1]), 4)],
                "seen_time": now,
                "confidence": round(float(belief.get(room_id, 0.0)), 6),
                "event_type": event_type,
            }
        )
        stats["current_room_id"] = room_id
        stats["room_belief"] = belief
        stats["room_history"] = history[-100:]
        obj.cooccur_stats = stats
        obj.room_id = room_id

        room = self._room_memory(room_id)
        ref = room.object_refs.get(obj.obj_id)
        if ref is None:
            ref = RoomObjectEvidence(obj_id=obj.obj_id, first_seen_time=now)
            room.object_refs[obj.obj_id] = ref
        ref.seen_count += 1
        ref.last_seen_state_index = int(state_index)
        ref.last_seen_layout_id = layout_id
        ref.room_exist_prob = float(belief.get(room_id, 0.0))
        ref.last_seen_time = now
        ref.positions_2d.append([round(float(pos2d[0]), 4), round(float(pos2d[1]), 4)])
        ref.positions_2d = ref.positions_2d[-50:]

        return {
            "room_id": room_id,
            "previous_room_id": previous_room_id or None,
            "current_room_id": room_id,
            "room_exist_prob": round(float(belief.get(room_id, 0.0)), 6),
        }

    def _best_match(self, det: Dict[str, Any]) -> Tuple[Optional[Object], float, float]:
        label = normalize_label(det.get("label"))
        pos2d = _pos2d_from_detection(det)
        emb = det.get("clip_embedding") if isinstance(det.get("clip_embedding"), list) else []
        best: Optional[Object] = None
        best_dist = float("inf")
        best_clip = 0.0
        for obj in self._objects.values():
            same_label = normalize_label(obj.label) == label
            clip_score = _clip_cosine(emb, obj.clip_embedding)
            if pos2d and obj.pos_2d:
                try:
                    dist = euclidean_2d(pos2d, obj.pos_2d)
                except Exception:
                    dist = float("inf")
            else:
                dist = float("inf")
            if same_label and dist < best_dist:
                best, best_dist, best_clip = obj, dist, clip_score
            elif clip_score > best_clip and clip_score >= self.clip_migration_threshold:
                best, best_dist, best_clip = obj, dist, clip_score
        return best, best_dist, best_clip

    def update_from_detections(
        self,
        detections: Iterable[Dict[str, Any]],
        *,
        state_index: int,
        layout_id: str,
        step: int,
        room_id: str = "unknown",
    ) -> List[MemoryEvent]:
        events: List[MemoryEvent] = []
        for det in detections:
            label = sanitize_detection_label(det.get("label"))
            if is_noise_detection_label(label):
                continue
            if det.get("label") != label:
                det = dict(det)
                raw_label = str(det.get("raw_label") or det.get("label") or "").strip()
                if raw_label:
                    det["raw_label"] = raw_label
                det["label"] = label
            pos2d = _pos2d_from_detection(det)
            if pos2d is None:
                continue
            pos3d = det.get("pos_3d") if isinstance(det.get("pos_3d"), list) else [pos2d[0], 0.0, pos2d[1]]
            clip_embedding = det.get("clip_embedding") if isinstance(det.get("clip_embedding"), list) else []
            bbox = det.get("bbox_xyxy")
            now = utc_now_iso()
            observed_room_id = self.assign_room_id(pos2d, room_id)
            match, dist, clip_score = self._best_match(det)
            if match is None:
                track_id = self._new_track_id(label)
                obj = Object(
                    obj_id=track_id,
                    label=label,
                    region=_region_from_pos2d(pos2d),
                    stability=0.5,
                    clip_embedding=clip_embedding,
                    cfd=1.0,
                    room_id=observed_room_id,
                    imgs={},
                    N=1,
                    description={"1": f"visual_track label={label}"},
                    last_update_time=now,
                    cooccur_stats={
                        "scene_name": self.scene_name,
                        "first_seen_state_index": int(state_index),
                        "last_seen_state_index": int(state_index),
                        "seen_count": 1,
                        "miss_count": 0,
                        "source_observations": [
                            {"state_index": int(state_index), "layout_id": layout_id, "step": int(step)}
                        ],
                    },
                    exist_prob=1.0,
                    pos_3d=pos3d,
                    pos_2d=pos2d,
                    bbox_3d={"source_bbox_xyxy": bbox} if bbox is not None else None,
                )
                self._objects[track_id] = obj
                details = self._update_room_memory(
                    obj,
                    room_id=observed_room_id,
                    pos2d=pos2d,
                    state_index=state_index,
                    layout_id=layout_id,
                    step=step,
                    event_type="new",
                    now=now,
                )
                events.append(MemoryEvent("new", track_id, label, state_index, layout_id, step, details))
                continue

            event_type = "merged"
            if dist > self.merge_distance and (
                normalize_label(match.label) == label and dist <= self.migration_distance
                or clip_score >= self.clip_migration_threshold
            ):
                event_type = "migrated"
            elif dist > self.merge_distance and clip_score < self.clip_migration_threshold:
                track_id = self._new_track_id(label)
                obj = Object(
                    obj_id=track_id,
                    label=label,
                    region=_region_from_pos2d(pos2d),
                    stability=0.5,
                    clip_embedding=clip_embedding,
                    cfd=1.0,
                    room_id=observed_room_id,
                    imgs={},
                    N=1,
                    description={"1": f"visual_track label={label}"},
                    last_update_time=now,
                    cooccur_stats={
                        "scene_name": self.scene_name,
                        "first_seen_state_index": int(state_index),
                        "last_seen_state_index": int(state_index),
                        "seen_count": 1,
                        "miss_count": 0,
                        "source_observations": [
                            {"state_index": int(state_index), "layout_id": layout_id, "step": int(step)}
                        ],
                    },
                    exist_prob=1.0,
                    pos_3d=pos3d,
                    pos_2d=pos2d,
                    bbox_3d={"source_bbox_xyxy": bbox} if bbox is not None else None,
                )
                self._objects[track_id] = obj
                details = self._update_room_memory(
                    obj,
                    room_id=observed_room_id,
                    pos2d=pos2d,
                    state_index=state_index,
                    layout_id=layout_id,
                    step=step,
                    event_type="new",
                    now=now,
                )
                details["reason"] = "same_label_far"
                events.append(
                    MemoryEvent(
                        "new",
                        track_id,
                        label,
                        state_index,
                        layout_id,
                        step,
                        details,
                    )
                )
                continue

            stats = match.cooccur_stats or {}
            history = list(stats.get("source_observations", []))
            history.append({"state_index": int(state_index), "layout_id": layout_id, "step": int(step)})
            stats.update(
                {
                    "scene_name": self.scene_name,
                    "last_seen_state_index": int(state_index),
                    "seen_count": int(stats.get("seen_count", 0) or 0) + 1,
                    "source_observations": history[-50:],
                }
            )
            if "first_seen_state_index" not in stats:
                stats["first_seen_state_index"] = int(state_index)
            match.cooccur_stats = stats
            match.last_update_time = now
            match.pos_2d = pos2d
            match.pos_3d = pos3d
            match.region = _region_from_pos2d(pos2d)
            match.exist_prob = min(1.0, float(match.exist_prob or 1.0) + 0.15)
            if clip_embedding:
                match.clip_embedding = clip_embedding
            if bbox is not None:
                match.bbox_3d = {"source_bbox_xyxy": bbox}
            details = self._update_room_memory(
                match,
                room_id=observed_room_id,
                pos2d=pos2d,
                state_index=state_index,
                layout_id=layout_id,
                step=step,
                event_type=event_type,
                now=now,
            )
            details.update({"distance": None if not math.isfinite(dist) else dist, "clip_score": clip_score})
            events.append(
                MemoryEvent(
                    event_type,
                    match.obj_id,
                    label,
                    state_index,
                    layout_id,
                    step,
                    details,
                )
            )
        return events

    def record_negative_feedback(
        self,
        track_id: str,
        *,
        state_index: int,
        layout_id: str,
        step: int,
        reason: str,
    ) -> Optional[MemoryEvent]:
        obj = self._objects.get(track_id)
        if obj is None:
            return None
        stats = obj.cooccur_stats or {}
        stats["miss_count"] = int(stats.get("miss_count", 0) or 0) + 1
        stats["negative_feedback_count"] = int(stats.get("negative_feedback_count", 0) or 0) + 1
        room_id = str(stats.get("current_room_id") or obj.room_id or "")
        room_belief_before = None
        room_belief_after = None
        if room_id:
            belief = dict(stats.get("room_belief") or {})
            room_belief_before = float(belief.get(room_id, 0.0) or 0.0)
            belief[room_id] = max(0.0, room_belief_before * 0.35)
            room_belief_after = float(belief.get(room_id, 0.0) or 0.0)
            stats["room_belief"] = belief
            room = self.rooms.get(room_id)
            if room is not None and track_id in room.object_refs:
                ref = room.object_refs[track_id]
                ref.negative_count += 1
                ref.room_exist_prob = room_belief_after
        obj.cooccur_stats = stats
        obj.exist_prob = max(0.0, min(1.0, float(obj.exist_prob or 1.0) * 0.35))
        return MemoryEvent(
            "negative_feedback",
            track_id,
            obj.label,
            state_index,
            layout_id,
            step,
            {
                "reason": reason,
                "exist_prob": obj.exist_prob,
                "miss_count": stats["miss_count"],
                "room_id": room_id or None,
                "room_belief_before": room_belief_before,
                "room_belief_after": room_belief_after,
            },
        )
