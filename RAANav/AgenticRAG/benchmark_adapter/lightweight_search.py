from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

from RAANav.AgenticRAG.semantic_map import Floor, Object

from .common import euclidean_2d, normalize_label, token_overlap
from .episode_to_queries import QuerySpec


@dataclass
class Peak:
    obj_id: str
    label: str
    world_x: float
    world_z: float
    score: float
    reason: str


@dataclass
class SearchResult:
    found: bool
    sss: int
    visited_points: List[List[float]]
    final_dist: float
    mra: bool
    ghr3: bool
    ghr5: bool
    min_dist: float
    peaks: List[Dict[str, Any]]


def iter_objects(floors: Iterable[Floor]) -> Iterable[Object]:
    for floor in floors:
        for room in floor.rooms:
            for obj in room.objects:
                yield obj


def _score_object(obj: Object, query: QuerySpec) -> tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []
    if obj.obj_id == query.target_object_id:
        score += 1.0
        reasons.append("same_obj_id")
    if normalize_label(obj.label) == query.query_label:
        score += 0.9
        reasons.append("same_label")
    overlap = token_overlap(obj.label, query.query_label)
    if overlap > 0:
        score += 0.25 * overlap
        reasons.append(f"token_overlap={overlap:.2f}")
    stats = obj.cooccur_stats or {}
    if query.sampled_region_id is not None and str(stats.get("sampled_region_id")) == str(query.sampled_region_id):
        score += 0.35
        reasons.append("same_region_prior")
    if query.target_instance_id is not None and str(stats.get("target_instance_id")) == str(query.target_instance_id):
        score += 0.45
        reasons.append("same_receptacle_prior")
    try:
        score *= 0.5 + 0.5 * float(obj.exist_prob)
    except Exception:
        pass
    try:
        score *= 0.75 + 0.25 * float(obj.stability or 0.5)
    except Exception:
        pass
    return score, reasons


def rank_memory_objects(floors: List[Floor], query: QuerySpec, max_candidates: int = 10) -> List[Peak]:
    peaks: List[Peak] = []
    for obj in iter_objects(floors):
        if not obj.pos_2d:
            continue
        score, reasons = _score_object(obj, query)
        if score <= 0:
            continue
        try:
            x, z = float(obj.pos_2d[0]), float(obj.pos_2d[1])
        except Exception:
            continue
        peaks.append(
            Peak(
                obj_id=obj.obj_id,
                label=obj.label,
                world_x=x,
                world_z=z,
                score=round(score, 6),
                reason=",".join(reasons),
            )
        )
    peaks.sort(key=lambda p: (-p.score, p.obj_id))
    deduped: List[Peak] = []
    for peak in peaks:
        if len(deduped) >= max_candidates:
            break
        if any(math.hypot(peak.world_x - old.world_x, peak.world_z - old.world_z) < 0.25 for old in deduped):
            continue
        deduped.append(peak)
    return deduped


def search_memory(
    floors: List[Floor],
    query: QuerySpec,
    gt_xz: List[float],
    success_radius: float,
    max_candidates: int = 10,
) -> SearchResult:
    peaks = rank_memory_objects(floors, query, max_candidates=max_candidates)
    visited: List[List[float]] = []
    found = False
    final_dist = float("inf")
    sss = 0
    for peak in peaks[:max_candidates]:
        sss += 1
        point = [peak.world_x, peak.world_z]
        visited.append(point)
        dist = euclidean_2d(point, gt_xz)
        final_dist = dist
        if dist <= success_radius:
            found = True
            break
    if not peaks:
        sss = 0
    distances = [euclidean_2d([p.world_x, p.world_z], gt_xz) for p in peaks]
    min_dist = min(distances) if distances else float("inf")
    mra = bool(distances and distances[0] <= success_radius)
    ghr3 = any(d <= success_radius for d in distances[:3])
    ghr5 = any(d <= success_radius for d in distances[:5])
    return SearchResult(
        found=found,
        sss=sss,
        visited_points=visited,
        final_dist=final_dist,
        mra=mra,
        ghr3=ghr3,
        ghr5=ghr5,
        min_dist=min_dist,
        peaks=[asdict(p) for p in peaks],
    )

