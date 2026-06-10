from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from RAANav.AgenticRAG.semantic_map import Floor, Object, Room

from .common import euclidean_2d
from .dataset_index import DatasetIndex
from .layout_to_floors import layout_to_floors, virtual_time


def _iter_objects(floors: Iterable[Floor]) -> Iterable[Object]:
    for floor in floors:
        for room in floor.rooms:
            for obj in room.objects:
                yield obj


def _relation_weight(a: Object, b: Object) -> Tuple[int, Dict[str, int]]:
    stats_a = a.cooccur_stats or {}
    stats_b = b.cooccur_stats or {}
    weight = 0
    detail = {
        "same_room": 0,
        "same_target_receptacle": 0,
        "same_assigned_receptacle": 0,
        "near_2d": 0,
    }
    if stats_a.get("sampled_region_id") == stats_b.get("sampled_region_id"):
        detail["same_room"] = 1
        weight += 1
    if stats_a.get("target_instance_id") == stats_b.get("target_instance_id"):
        detail["same_target_receptacle"] = 1
        weight += 2
    if stats_a.get("assigned_target_instance_id") == stats_b.get("assigned_target_instance_id"):
        detail["same_assigned_receptacle"] = 1
        weight += 3
    if a.pos_2d and b.pos_2d:
        try:
            if euclidean_2d(a.pos_2d, b.pos_2d) <= 2.0:
                detail["near_2d"] = 1
                weight += 2
        except Exception:
            pass
    return weight, detail


def _add_relations(objects: List[Object], rel_acc: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            weight, detail = _relation_weight(a, b)
            if weight <= 0:
                continue
            for src, dst in ((a, b), (b, a)):
                entry = rel_acc[src.obj_id].setdefault(
                    dst.obj_id,
                    {
                        "weight_sum": 0,
                        "same_room": 0,
                        "same_target_receptacle": 0,
                        "same_assigned_receptacle": 0,
                        "near_2d": 0,
                    },
                )
                entry["weight_sum"] += weight
                for key, val in detail.items():
                    entry[key] += val


def merge_layout_floors(layout_floors: List[List[Floor]]) -> List[Floor]:
    latest: Dict[str, Object] = {}
    counts: Dict[str, int] = defaultdict(int)
    rel_acc: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for floors in layout_floors:
        objs = list(_iter_objects(floors))
        _add_relations(objs, rel_acc)
        for obj in objs:
            latest[obj.obj_id] = obj
            counts[obj.obj_id] += 1
    for oid, obj in latest.items():
        obj.N = counts.get(oid, 1)
        rels = rel_acc.get(oid, {})
        obj.cooccur_stats = obj.cooccur_stats or {}
        obj.cooccur_stats["relations"] = rels
        robjs = {}
        for rel_oid, detail in rels.items():
            nr = int(detail.get("weight_sum", 0))
            if nr <= 0:
                continue
            robjs[rel_oid] = {
                "Nt": max(1, counts.get(oid, 1)),
                "Nr": nr,
                "Rcfd": round(min(1.0, nr / 6.0), 4),
            }
        obj.R_objs = robjs
    rooms: Dict[str, List[Object]] = defaultdict(list)
    for obj in latest.values():
        rooms[obj.room_id or "unknown"].append(obj)
    room_objs = [
        Room(
            room_id=rid,
            room_name={"1": f"region_{rid}"},
            objects=sorted(objects, key=lambda o: o.obj_id),
            floor_id="F0",
            N=len(objects),
        )
        for rid, objects in sorted(rooms.items(), key=lambda kv: kv[0])
    ]
    return [Floor(floor_id="F0", rooms=room_objs, description=f"merged_layouts={len(layout_floors)}")]


def build_memory_from_seen_layouts(
    index: DatasetIndex,
    scene_name: str,
    state_index: int,
    seen_layout_count_before: int,
) -> List[Floor]:
    entries = index.history_layouts(scene_name, state_index, seen_layout_count_before)
    if not entries:
        return [Floor(floor_id="F0", rooms=[], description="empty_memory")]
    layouts = [layout_to_floors(entry.layout_path, entry.layout_index) for entry in entries]
    return merge_layout_floors(layouts)

