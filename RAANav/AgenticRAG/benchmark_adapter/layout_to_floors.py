from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from RAANav.AgenticRAG.semantic_map import Floor, Object, Room

from .common import (
    agentic_root,
    as_path,
    bbox_from_object,
    footprint_region,
    normalize_label,
    object_stem,
    position_2d,
    read_json,
)


PLACEMENT_STABILITY = {
    "floor_only": 0.75,
    "small_tabletop": 0.35,
    "large_tabletop": 0.45,
    "soft_surface": 0.30,
    "floor_or_large_surface": 0.55,
    "tabletop_or_floor": 0.50,
}


def _load_stability_priors() -> Dict[str, float]:
    path = agentic_root() / "config" / "stability_priors.yaml"
    if yaml is None or not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    priors = data.get("stability_priors", data)
    out: Dict[str, float] = {}
    if isinstance(priors, dict):
        for key, value in priors.items():
            try:
                out[normalize_label(key)] = max(0.0, min(1.0, float(value)))
            except Exception:
                continue
    return out


def stability_for_object(obj: Dict[str, Any], priors: Optional[Dict[str, float]] = None) -> float:
    priors = priors if priors is not None else _load_stability_priors()
    label = object_stem(obj)
    if label in priors:
        return priors[label]
    base_label = "_".join([p for p in label.split("_") if not p.isdigit()])
    if base_label in priors:
        return priors[base_label]
    placement_class = str((obj.get("object_profile") or {}).get("placement_class", "")).strip()
    return PLACEMENT_STABILITY.get(placement_class, 0.5)


def virtual_time(state_index: int) -> str:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return (start + timedelta(days=int(state_index))).isoformat().replace("+00:00", "Z")


def object_to_raanav(obj: Dict[str, Any], *, state_index: int, scene_name: str, priors: Dict[str, float]) -> Object:
    obj_id_raw = obj.get("id", obj.get("model_id", obj.get("name", "object")))
    obj_id = str(obj_id_raw)
    label = object_stem(obj)
    pos3 = obj.get("position") if isinstance(obj.get("position"), list) else None
    pos2 = position_2d(obj)
    room_id = str(obj.get("sampled_region_id", "unknown"))
    profile = obj.get("object_profile") or {}
    image_stem = label
    image_path = Path("objects_images") / f"{image_stem}.webp"
    imgs = {"1": [image_path.as_posix()]} if as_path(image_path).exists() else {}
    cooccur_stats = {
        "scene_name": scene_name,
        "model_id": obj.get("model_id"),
        "name": obj.get("name"),
        "sampled_region_id": obj.get("sampled_region_id"),
        "target_instance_id": obj.get("target_instance_id"),
        "assigned_target_instance_id": obj.get("assigned_target_instance_id"),
        "placement_target_source": obj.get("placement_target_source"),
        "placement_class": profile.get("placement_class"),
        "relations": {},
    }
    return Object(
        obj_id=obj_id,
        label=label,
        region=footprint_region(obj),
        stability=stability_for_object(obj, priors),
        clip_embedding=[],
        cfd=1.0,
        room_id=room_id,
        R_objs={},
        imgs=imgs,
        N=1,
        description={
            "1": f"{obj.get('name', label)} ({obj.get('model_id', label)}), placement_class={profile.get('placement_class', 'unknown')}"
        },
        last_update_time=virtual_time(state_index),
        cooccur_stats=cooccur_stats,
        exist_prob=1.0,
        pos_3d=pos3,
        pos_2d=pos2,
        bbox_3d=bbox_from_object(obj),
    )


def layout_to_floors(
    layout_path: Union[str, Path],
    state_index: int,
    scene_info_path: Optional[Union[str, Path]] = None,
    surfaces_path: Optional[Union[str, Path]] = None,
) -> List[Floor]:
    layout = read_json(layout_path)
    scene_name = Path(str(layout.get("scene", ""))).parent.name or str(layout.get("scene", "unknown"))
    priors = _load_stability_priors()
    rooms: Dict[str, List[Object]] = defaultdict(list)
    y_values: List[float] = []
    for raw_obj in layout.get("objects", []):
        if not isinstance(raw_obj, dict):
            continue
        obj = object_to_raanav(raw_obj, state_index=state_index, scene_name=scene_name, priors=priors)
        rooms[obj.room_id or "unknown"].append(obj)
        if obj.pos_3d and len(obj.pos_3d) >= 2:
            try:
                y_values.append(float(obj.pos_3d[1]))
            except Exception:
                pass
    room_objs = [
        Room(
            room_id=rid,
            room_name={"1": f"region_{rid}"},
            objects=objects,
            floor_id="F0",
            N=len(objects),
            description={"1": f"sampled_region_id={rid}"},
        )
        for rid, objects in sorted(rooms.items(), key=lambda kv: kv[0])
    ]
    if y_values:
        z_range = {"z_min": min(y_values) - 0.5, "z_max": max(y_values) + 0.5}
    else:
        z_range = {"z_min": 0.0, "z_max": 4.0}
    return [Floor(floor_id="F0", rooms=room_objs, z_range=z_range, description=f"layout={Path(layout_path).name}")]
