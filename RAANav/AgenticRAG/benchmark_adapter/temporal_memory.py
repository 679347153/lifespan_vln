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


def _position_confidence(det: Dict[str, Any]) -> float:
    try:
        return max(0.0, min(1.0, float(det.get("position_confidence", 0.5))))
    except Exception:
        return 0.5


def _weighted_pos2d(detections: List[Dict[str, Any]]) -> Optional[List[float]]:
    weighted: List[Tuple[float, List[float]]] = []
    for det in detections:
        pos = _pos2d_from_detection(det)
        if pos is None:
            continue
        weighted.append((_position_confidence(det) + 1e-3, pos))
    if not weighted:
        return None
    total = sum(w for w, _ in weighted)
    return [
        sum(w * float(pos[0]) for w, pos in weighted) / total,
        sum(w * float(pos[1]) for w, pos in weighted) / total,
    ]


def _best_conf_detection(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(detections, key=lambda det: (_position_confidence(det), float(det.get("score", 0.0) or 0.0)))


def _consolidate_detection_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = dict(_best_conf_detection(group))
    pos2d = _weighted_pos2d(group)
    if pos2d is not None:
        best["pos_2d"] = [float(pos2d[0]), float(pos2d[1])]
        old_pos3 = best.get("pos_3d") if isinstance(best.get("pos_3d"), list) else None
        best["pos_3d"] = [float(pos2d[0]), float(old_pos3[1]) if old_pos3 and len(old_pos3) >= 2 else 0.0, float(pos2d[1])]
    best["consolidated_detection_count"] = len(group)
    best["position_confidence"] = round(max(_position_confidence(det) for det in group), 6)
    best["raw_positions_2d"] = [
        [round(float(pos[0]), 4), round(float(pos[1]), 4)]
        for pos in (_pos2d_from_detection(det) for det in group)
        if pos is not None
    ][:20]
    best["consolidation_sources"] = [
        {
            "view_index": det.get("_view_index"),
            "position_source": det.get("position_source"),
            "position_confidence": det.get("position_confidence"),
            "depth_median": det.get("depth_median"),
        }
        for det in group[:20]
    ]
    return best


def consolidate_detections(
    detections: Iterable[Dict[str, Any]],
    *,
    distance_threshold: float = 0.6,
    clip_threshold: float = 0.90,
    clip_distance_limit: float = 1.5,
) -> List[Dict[str, Any]]:
    groups: List[List[Dict[str, Any]]] = []
    for det in detections:
        label = sanitize_detection_label(det.get("label"))
        if is_noise_detection_label(label):
            continue
        pos = _pos2d_from_detection(det)
        if pos is None:
            continue
        emb = det.get("clip_embedding") if isinstance(det.get("clip_embedding"), list) else []
        placed = False
        for group in groups:
            head = group[0]
            if sanitize_detection_label(head.get("label")) != label:
                continue
            head_pos = _pos2d_from_detection(head)
            head_emb = head.get("clip_embedding") if isinstance(head.get("clip_embedding"), list) else []
            dist = euclidean_2d(pos, head_pos) if head_pos is not None else float("inf")
            dist_ok = dist <= distance_threshold
            clip_ok = bool(emb and head_emb and dist <= clip_distance_limit and _clip_cosine(emb, head_emb) >= clip_threshold)
            if dist_ok or clip_ok:
                group.append(det)
                placed = True
                break
        if not placed:
            groups.append([det])
    return [_consolidate_detection_group(group) for group in groups]


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
        tracks: List[Dict[str, Any]] = []
        for obj in sorted(self._objects.values(), key=lambda item: item.obj_id):
            item = obj.to_dict()
            stats = obj.cooccur_stats if isinstance(obj.cooccur_stats, dict) else {}
            for key in ("position_history", "position_confidence", "smoothed_pos_2d"):
                if key in stats:
                    item[key] = stats[key]
            tracks.append(item)
        return {
            "scene_name": self.scene_name,
            "track_count": len(self._objects),
            "tracks": tracks,
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

    def _init_position_stats(self, obj: Object, det: Dict[str, Any], pos2d: List[float]) -> None:
        confidence = _position_confidence(det)
        stats = obj.cooccur_stats if isinstance(obj.cooccur_stats, dict) else {}
        stats["raw_pos_2d"] = [round(float(pos2d[0]), 4), round(float(pos2d[1]), 4)]
        stats["smoothed_pos_2d"] = [round(float(pos2d[0]), 4), round(float(pos2d[1]), 4)]
        stats["position_confidence"] = round(float(confidence), 6)
        stats["position_source"] = det.get("position_source")
        stats["position_history"] = [
            {
                "pos_2d": [round(float(pos2d[0]), 4), round(float(pos2d[1]), 4)],
                "confidence": round(float(confidence), 6),
                "source": det.get("position_source"),
                "consolidated_detection_count": int(det.get("consolidated_detection_count", 1) or 1),
            }
        ]
        stats["position_confidence_history"] = [round(float(confidence), 6)]
        obj.cooccur_stats = stats

    def _update_position_stats(self, obj: Object, det: Dict[str, Any], pos2d: List[float]) -> List[float]:
        confidence = _position_confidence(det)
        stats = obj.cooccur_stats if isinstance(obj.cooccur_stats, dict) else {}
        old = stats.get("smoothed_pos_2d") if isinstance(stats.get("smoothed_pos_2d"), list) else obj.pos_2d
        if isinstance(old, list) and len(old) >= 2:
            old_pos = [float(old[0]), float(old[1])]
        else:
            old_pos = [float(pos2d[0]), float(pos2d[1])]
        alpha = 0.25 + 0.45 * confidence
        smoothed = [
            old_pos[0] * (1.0 - alpha) + float(pos2d[0]) * alpha,
            old_pos[1] * (1.0 - alpha) + float(pos2d[1]) * alpha,
        ]
        history = list(stats.get("position_history", []))
        history.append(
            {
                "pos_2d": [round(float(pos2d[0]), 4), round(float(pos2d[1]), 4)],
                "smoothed_pos_2d": [round(float(smoothed[0]), 4), round(float(smoothed[1]), 4)],
                "confidence": round(float(confidence), 6),
                "source": det.get("position_source"),
                "consolidated_detection_count": int(det.get("consolidated_detection_count", 1) or 1),
            }
        )
        conf_history = list(stats.get("position_confidence_history", []))
        conf_history.append(round(float(confidence), 6))
        stats["raw_pos_2d"] = [round(float(pos2d[0]), 4), round(float(pos2d[1]), 4)]
        stats["smoothed_pos_2d"] = [round(float(smoothed[0]), 4), round(float(smoothed[1]), 4)]
        stats["position_confidence"] = round(float(confidence), 6)
        stats["position_source"] = det.get("position_source")
        stats["position_history"] = history[-100:]
        stats["position_confidence_history"] = conf_history[-100:]
        obj.cooccur_stats = stats
        return smoothed

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
        batch_seen_track_ids: set[str] = set()
        for det in consolidate_detections(detections):
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
                        "last_seen_layout_id": layout_id,
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
                self._init_position_stats(obj, det, pos2d)
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
                details.update(
                    {
                        "position_confidence": _position_confidence(det),
                        "position_source": det.get("position_source"),
                        "raw_pos_2d": [round(float(pos2d[0]), 4), round(float(pos2d[1]), 4)],
                        "smoothed_pos_2d": obj.cooccur_stats.get("smoothed_pos_2d"),
                        "consolidated_detection_count": int(det.get("consolidated_detection_count", 1) or 1),
                        "merge_reason": "new_track",
                    }
                )
                events.append(MemoryEvent("new", track_id, label, state_index, layout_id, step, details))
                batch_seen_track_ids.add(track_id)
                continue

            event_type = "merged"
            pos_conf = _position_confidence(det)
            adaptive_merge_distance = self.merge_distance if pos_conf >= 0.5 else 1.25
            stats_for_match = match.cooccur_stats or {}
            same_update_batch = (
                match.obj_id in batch_seen_track_ids
                or (
                int(stats_for_match.get("last_seen_state_index", -999999) or -999999) == int(state_index)
                and str(stats_for_match.get("last_seen_layout_id") or "") == str(layout_id)
                and any(int(item.get("step", -1) or -1) == int(step) for item in stats_for_match.get("source_observations", []) if isinstance(item, dict))
                )
            )
            if dist > adaptive_merge_distance and not same_update_batch and (
                normalize_label(match.label) == label and dist <= self.migration_distance
                or clip_score >= self.clip_migration_threshold
            ):
                event_type = "migrated"
            elif dist > adaptive_merge_distance:
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
                        "last_seen_layout_id": layout_id,
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
                self._init_position_stats(obj, det, pos2d)
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
                details.update(
                    {
                        "position_confidence": _position_confidence(det),
                        "position_source": det.get("position_source"),
                        "raw_pos_2d": [round(float(pos2d[0]), 4), round(float(pos2d[1]), 4)],
                        "smoothed_pos_2d": obj.cooccur_stats.get("smoothed_pos_2d"),
                        "consolidated_detection_count": int(det.get("consolidated_detection_count", 1) or 1),
                        "merge_reason": "same_label_far_new_track",
                    }
                )
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
                batch_seen_track_ids.add(track_id)
                continue

            stats = match.cooccur_stats or {}
            history = list(stats.get("source_observations", []))
            history.append({"state_index": int(state_index), "layout_id": layout_id, "step": int(step)})
            stats.update(
                {
                    "scene_name": self.scene_name,
                    "last_seen_state_index": int(state_index),
                    "last_seen_layout_id": layout_id,
                    "seen_count": int(stats.get("seen_count", 0) or 0) + 1,
                    "source_observations": history[-50:],
                }
            )
            if "first_seen_state_index" not in stats:
                stats["first_seen_state_index"] = int(state_index)
            match.cooccur_stats = stats
            match.last_update_time = now
            smoothed_pos2d = self._update_position_stats(match, det, pos2d)
            match.pos_2d = smoothed_pos2d
            match.pos_3d = [float(smoothed_pos2d[0]), float(pos3d[1]) if len(pos3d) >= 2 else 0.0, float(smoothed_pos2d[1])]
            match.region = _region_from_pos2d(smoothed_pos2d)
            match.exist_prob = min(1.0, float(match.exist_prob or 1.0) + 0.15)
            if clip_embedding:
                match.clip_embedding = clip_embedding
            if bbox is not None:
                match.bbox_3d = {"source_bbox_xyxy": bbox}
            details = self._update_room_memory(
                match,
                room_id=observed_room_id,
                pos2d=smoothed_pos2d,
                state_index=state_index,
                layout_id=layout_id,
                step=step,
                event_type=event_type,
                now=now,
            )
            details.update({"distance": None if not math.isfinite(dist) else dist, "clip_score": clip_score})
            details.update(
                {
                    "position_confidence": _position_confidence(det),
                    "position_source": det.get("position_source"),
                    "raw_pos_2d": [round(float(pos2d[0]), 4), round(float(pos2d[1]), 4)],
                    "smoothed_pos_2d": [round(float(smoothed_pos2d[0]), 4), round(float(smoothed_pos2d[1]), 4)],
                    "consolidated_detection_count": int(det.get("consolidated_detection_count", 1) or 1),
                    "merge_reason": event_type,
                }
            )
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
            batch_seen_track_ids.add(match.obj_id)
        return events

    def compact_duplicates(
        self,
        *,
        state_index: int,
        layout_id: str,
        step: int,
        distance_threshold: float = 1.0,
        clip_threshold: float = 0.88,
    ) -> List[MemoryEvent]:
        events: List[MemoryEvent] = []
        removed: set[str] = set()
        objects = sorted(self._objects.values(), key=lambda obj: obj.obj_id)
        for i, a in enumerate(objects):
            if a.obj_id in removed or not a.pos_2d:
                continue
            for b in objects[i + 1 :]:
                if b.obj_id in removed or not b.pos_2d:
                    continue
                if normalize_label(a.label) != normalize_label(b.label):
                    continue
                room_a = str((a.cooccur_stats or {}).get("current_room_id") or a.room_id or "")
                room_b = str((b.cooccur_stats or {}).get("current_room_id") or b.room_id or "")
                if room_a != room_b:
                    continue
                dist = euclidean_2d(a.pos_2d, b.pos_2d)
                clip_score = _clip_cosine(a.clip_embedding, b.clip_embedding)
                if dist > distance_threshold or clip_score < clip_threshold:
                    continue

                stats_a = a.cooccur_stats or {}
                stats_b = b.cooccur_stats or {}
                seen_a = int(stats_a.get("seen_count", 0) or 0)
                seen_b = int(stats_b.get("seen_count", 0) or 0)
                conf_a = float(stats_a.get("position_confidence", 0.5) or 0.5)
                conf_b = float(stats_b.get("position_confidence", 0.5) or 0.5)
                keep, drop = (a, b) if (seen_a, conf_a, a.obj_id) >= (seen_b, conf_b, b.obj_id) else (b, a)
                keep_stats = keep.cooccur_stats or {}
                drop_stats = drop.cooccur_stats or {}
                keep_stats["seen_count"] = int(keep_stats.get("seen_count", 0) or 0) + int(drop_stats.get("seen_count", 0) or 0)
                keep_stats["miss_count"] = int(keep_stats.get("miss_count", 0) or 0) + int(drop_stats.get("miss_count", 0) or 0)
                keep_stats["negative_feedback_count"] = int(keep_stats.get("negative_feedback_count", 0) or 0) + int(
                    drop_stats.get("negative_feedback_count", 0) or 0
                )
                keep_stats["position_history"] = (list(keep_stats.get("position_history", [])) + list(drop_stats.get("position_history", [])))[-100:]
                keep_stats["position_confidence_history"] = (
                    list(keep_stats.get("position_confidence_history", []))
                    + list(drop_stats.get("position_confidence_history", []))
                )[-100:]
                keep_stats["duplicate_compaction"] = {
                    "merged_track_id": drop.obj_id,
                    "distance": round(float(dist), 6),
                    "clip_score": round(float(clip_score), 6),
                    "layout_id": layout_id,
                    "state_index": int(state_index),
                    "step": int(step),
                }
                keep.cooccur_stats = keep_stats
                keep.exist_prob = max(float(keep.exist_prob or 0.0), float(drop.exist_prob or 0.0))
                if not keep.clip_embedding and drop.clip_embedding:
                    keep.clip_embedding = drop.clip_embedding

                for room in self.rooms.values():
                    if drop.obj_id not in room.object_refs:
                        continue
                    drop_ref = room.object_refs.pop(drop.obj_id)
                    keep_ref = room.object_refs.get(keep.obj_id)
                    if keep_ref is None:
                        drop_ref.obj_id = keep.obj_id
                        room.object_refs[keep.obj_id] = drop_ref
                    else:
                        keep_ref.seen_count += drop_ref.seen_count
                        keep_ref.negative_count += drop_ref.negative_count
                        keep_ref.positions_2d = (keep_ref.positions_2d + drop_ref.positions_2d)[-50:]
                        keep_ref.room_exist_prob = max(keep_ref.room_exist_prob, drop_ref.room_exist_prob)

                removed.add(drop.obj_id)
                self._objects.pop(drop.obj_id, None)
                events.append(
                    MemoryEvent(
                        "duplicate_compaction",
                        keep.obj_id,
                        keep.label,
                        state_index,
                        layout_id,
                        step,
                        {
                            "merged_track_id": drop.obj_id,
                            "room_id": room_a or None,
                            "distance": round(float(dist), 6),
                            "clip_score": round(float(clip_score), 6),
                            "duplicate_compaction": True,
                        },
                    )
                )
                break
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
