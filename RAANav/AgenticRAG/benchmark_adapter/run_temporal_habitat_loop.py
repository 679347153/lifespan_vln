from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from benchmark.evaluate import evaluate as benchmark_evaluate
from benchmark.habitat_adapter import horizontal_distance
from benchmark.schemas import (
    Episode,
    Pose,
    Subtask,
    SubtaskTrace,
    Trajectory,
    TrajectoryStep,
    read_episode,
    to_json_dict,
)

from .common import as_path, euclidean_2d, normalize_label, write_json
from .dataset_index import DatasetIndex
from .episode_to_queries import (
    QuerySpec,
    TargetNameMap,
    WEAK_ALIAS_SIMILARITY_CAP,
    aliases_for_label,
    load_target_name_map,
    query_from_subtask,
    query_label_match_strength,
    query_label_matches,
    query_label_similarity,
)
from .formal_scoring import CandidateScore, FusionParams, candidates_to_dict, rank_candidates
from .habitat_vision_loop import HabitatVisionLoop
from .image_goal_index import ImageGoalIndex
from .offline_eval import _aggregate_group, _safe_json_value, load_episodes_for_args
from .temporal_memory import SceneMemory


DEFAULT_DETECT_LABELS = [
    "chair",
    "table",
    "sofa",
    "bed",
    "desk",
    "shelf",
    "cabinet",
    "lamp",
    "plant",
    "picture",
    "mirror",
    "clock",
    "vase",
    "bottle",
    "cup",
    "bowl",
    "book",
    "camera",
    "megaphone",
    "alarm clock",
    "tea set",
    "cake",
    "apple",
    "chess set",
]


def _episode_sort_key(episode: Episode) -> Tuple[str, int, str, str]:
    return (episode.scene_name, int(episode.state_index), episode.layout_id, episode.episode_id)


def _load_sorted_episodes(args: argparse.Namespace, index: DatasetIndex) -> Tuple[Path, List[Episode]]:
    episodes_root, paths = load_episodes_for_args(args, index)
    episodes = [read_episode(path) for path in paths]
    episodes.sort(key=_episode_sort_key)
    if args.layout_limit and args.layout_limit > 0:
        allowed: Dict[str, set[str]] = defaultdict(set)
        for ep in episodes:
            if len(allowed[ep.scene_name]) < int(args.layout_limit):
                allowed[ep.scene_name].add(ep.layout_id)
        episodes = [ep for ep in episodes if ep.layout_id in allowed[ep.scene_name]]
    if args.episodes_per_layout and args.episodes_per_layout > 0:
        grouped_count: Dict[Tuple[str, str], int] = defaultdict(int)
        kept: List[Episode] = []
        for ep in episodes:
            key = (ep.scene_name, ep.layout_id)
            if grouped_count[key] >= int(args.episodes_per_layout):
                continue
            grouped_count[key] += 1
            kept.append(ep)
        episodes = kept
    if not episodes:
        raise FileNotFoundError("No episodes left after temporal filtering.")
    return episodes_root, episodes


def _label_for_prompt(value: Any) -> str:
    return normalize_label(value).replace("_", " ")


def _collect_prompt_labels(
    episodes: Iterable[Episode],
    *,
    target_name_map: Optional[TargetNameMap] = None,
    normalize_target_names: bool = True,
) -> List[str]:
    labels: List[str] = []
    seen = set()
    for label in DEFAULT_DETECT_LABELS:
        norm = normalize_label(label)
        if norm not in seen:
            seen.add(norm)
            labels.append(_label_for_prompt(label))
    for ep in episodes:
        for subtask in ep.subtasks:
            for raw in (
                subtask.target_object,
                (subtask.metadata or {}).get("model_id"),
            ):
                for norm in aliases_for_label(
                    raw,
                    target_name_map=target_name_map,
                    normalize_target_names=normalize_target_names,
                ):
                    if norm and norm not in seen:
                        seen.add(norm)
                        labels.append(_label_for_prompt(norm))
    return labels


def _prompt_for_query(query: QuerySpec, global_labels: List[str], memory: SceneMemory, max_labels: int) -> str:
    labels: List[str] = []
    seen = set()
    for raw in [*(query.alias_labels or []), query.query_label, query.target_object]:
        norm = normalize_label(raw)
        if norm and norm not in seen:
            seen.add(norm)
            labels.append(_label_for_prompt(norm))
    for obj in memory.objects():
        norm = normalize_label(obj.label)
        if norm and norm not in seen:
            seen.add(norm)
            labels.append(_label_for_prompt(norm))
    for label in global_labels:
        norm = normalize_label(label)
        if norm and norm not in seen:
            seen.add(norm)
            labels.append(label)
        if len(labels) >= max_labels:
            break
    return " . ".join(labels[:max_labels])


def _detection_pos2d(det: Dict[str, Any]) -> Optional[List[float]]:
    pos = det.get("pos_2d")
    if isinstance(pos, list) and len(pos) >= 2:
        return [float(pos[0]), float(pos[1])]
    if isinstance(pos, dict) and "x" in pos:
        return [float(pos["x"]), float(pos.get("z", pos.get("y", 0.0)))]
    pos3 = det.get("pos_3d")
    if isinstance(pos3, list) and len(pos3) >= 3:
        return [float(pos3[0]), float(pos3[2])]
    return None


def _target_detected(
    detections: Iterable[Dict[str, Any]],
    query: QuerySpec,
    *,
    subtask: Optional[Subtask] = None,
    confirmation_mode: str = "spatial",
    spatial_margin: float = 1.0,
) -> bool:
    for det in detections:
        if not query_label_matches(det.get("label"), query):
            continue
        if confirmation_mode == "semantic" or subtask is None:
            return True
        det_pos = _detection_pos2d(det)
        if det_pos is None:
            continue
        target_pos = [float(subtask.target_position.x), float(subtask.target_position.z)]
        threshold = max(float(subtask.success_radius), 0.0) + max(0.0, float(spatial_margin))
        if euclidean_2d(det_pos, target_pos) <= threshold:
            return True
    return False


def _target_detection_diagnostics(
    detections: Iterable[Dict[str, Any]],
    query: QuerySpec,
    *,
    subtask: Optional[Subtask] = None,
    confirmation_mode: str = "spatial",
    spatial_margin: float = 1.0,
) -> Dict[str, Any]:
    threshold: Optional[float] = None
    target_pos: Optional[List[float]] = None
    if subtask is not None:
        target_pos = [float(subtask.target_position.x), float(subtask.target_position.z)]
        threshold = max(float(subtask.success_radius), 0.0) + max(0.0, float(spatial_margin))

    semantic_count = 0
    positioned_count = 0
    closest_dist: Optional[float] = None
    closest_label: Optional[str] = None
    closest_pos: Optional[List[float]] = None
    confirmed = False
    for det in detections:
        if not query_label_matches(det.get("label"), query):
            continue
        semantic_count += 1
        if confirmation_mode == "semantic" or subtask is None:
            confirmed = True
        det_pos = _detection_pos2d(det)
        if det_pos is None or target_pos is None:
            continue
        positioned_count += 1
        dist = euclidean_2d(det_pos, target_pos)
        if closest_dist is None or dist < closest_dist:
            closest_dist = dist
            closest_label = normalize_label(det.get("label"))
            closest_pos = [round(float(det_pos[0]), 6), round(float(det_pos[1]), 6)]
        if threshold is not None and dist <= threshold:
            confirmed = True

    return {
        "target_semantic_match_count": semantic_count,
        "target_positioned_match_count": positioned_count,
        "target_detection_confirmed": confirmed,
        "target_confirm_threshold": round(float(threshold), 6) if threshold is not None else None,
        "closest_target_detection_distance": round(float(closest_dist), 6) if closest_dist is not None else None,
        "closest_target_detection_label": closest_label,
        "closest_target_detection_pos": closest_pos,
        "target_detection_distance_minus_threshold": (
            round(float(closest_dist) - float(threshold), 6)
            if closest_dist is not None and threshold is not None
            else None
        ),
    }


def _merge_target_detection_diagnostics(current: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    if not current:
        return dict(update)
    merged = dict(current)
    merged["target_semantic_match_count"] = int(merged.get("target_semantic_match_count", 0) or 0) + int(
        update.get("target_semantic_match_count", 0) or 0
    )
    merged["target_positioned_match_count"] = int(merged.get("target_positioned_match_count", 0) or 0) + int(
        update.get("target_positioned_match_count", 0) or 0
    )
    merged["target_detection_confirmed"] = bool(merged.get("target_detection_confirmed")) or bool(
        update.get("target_detection_confirmed")
    )
    old_dist = merged.get("closest_target_detection_distance")
    new_dist = update.get("closest_target_detection_distance")
    if old_dist is None or (new_dist is not None and float(new_dist) < float(old_dist)):
        merged["closest_target_detection_distance"] = new_dist
        merged["closest_target_detection_label"] = update.get("closest_target_detection_label")
        merged["closest_target_detection_pos"] = update.get("closest_target_detection_pos")
        merged["target_detection_distance_minus_threshold"] = update.get("target_detection_distance_minus_threshold")
    if merged.get("target_confirm_threshold") is None:
        merged["target_confirm_threshold"] = update.get("target_confirm_threshold")
    return merged


def _annotate_candidate_debug(
    candidates: List[Dict[str, Any]],
    query: QuerySpec,
    subtask: Subtask,
) -> List[Dict[str, Any]]:
    target = [float(subtask.target_position.x), float(subtask.target_position.z)]
    out: List[Dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        item = dict(candidate)
        item["pre_nav_rank"] = rank
        match_strength = query_label_match_strength(item.get("label"), query)
        item["label_alias_match_strength"] = match_strength
        item["label_alias_match"] = match_strength == "strong"
        item["label_weak_alias_match"] = match_strength == "weak"
        item["label_query_similarity"] = round(float(query_label_similarity(item.get("label"), query)), 6)
        if "world_x" in item and "world_z" in item:
            dist = euclidean_2d([item["world_x"], item["world_z"]], target)
            item["distance_to_target"] = round(float(dist), 6)
            item["within_success_radius"] = bool(dist <= float(subtask.success_radius))
        out.append(item)
    return out


def _pose_from_candidate(candidate: CandidateScore, current_pose: Pose) -> Pose:
    return Pose(float(candidate.world_x), float(current_pose.y), float(candidate.world_z), float(current_pose.yaw))


def _path_distance(loop: HabitatVisionLoop, start: Pose, goal: Pose) -> float:
    if loop.adapter is None:
        return euclidean_2d([start.x, start.z], [goal.x, goal.z])
    try:
        snapped = loop.adapter.snap_pose(goal)
        points = loop.adapter.shortest_path_points(start, snapped)
    except Exception:
        return euclidean_2d([start.x, start.z], [goal.x, goal.z])
    if not points:
        return euclidean_2d([start.x, start.z], [goal.x, goal.z])
    dist = 0.0
    last = start
    for point in points:
        dist += math.sqrt((float(point.x) - float(last.x)) ** 2 + (float(point.z) - float(last.z)) ** 2)
        last = point
    return float(dist)


def _candidate_room_id(candidate: CandidateScore, memory: SceneMemory) -> str:
    obj = {item.obj_id: item for item in memory.objects()}.get(candidate.obj_id)
    if obj is None:
        return "unknown"
    stats = obj.cooccur_stats or {}
    return str(stats.get("current_room_id") or obj.room_id or "unknown")


def _candidate_room_belief(candidate: CandidateScore, memory: SceneMemory, room_id: str) -> float:
    obj = {item.obj_id: item for item in memory.objects()}.get(candidate.obj_id)
    if obj is None:
        return 0.0
    belief = (obj.cooccur_stats or {}).get("room_belief")
    if not isinstance(belief, dict):
        return 0.0
    try:
        return max(0.0, min(1.0, float(belief.get(room_id, 0.0))))
    except Exception:
        return 0.0


def _room_representative_pose(candidates: List[CandidateScore], current_pose: Pose) -> Pose:
    if not candidates:
        return current_pose
    weights = [max(0.001, float(c.S_final)) for c in candidates]
    total = sum(weights)
    x = sum(w * float(c.world_x) for w, c in zip(weights, candidates)) / total
    z = sum(w * float(c.world_z) for w, c in zip(weights, candidates)) / total
    return Pose(float(x), float(current_pose.y), float(z), float(current_pose.yaw))


def _approach_pose_for_candidate(
    loop: HabitatVisionLoop,
    candidate: CandidateScore,
    current_pose: Pose,
    *,
    recent_poses: List[Pose],
) -> Tuple[Pose, Dict[str, Any]]:
    center_x, center_z = float(candidate.world_x), float(candidate.world_z)
    if loop.adapter is None:
        pose = Pose(center_x, float(current_pose.y), center_z, float(current_pose.yaw))
        return pose, {"source": "object_center_no_adapter"}
    best_pose: Optional[Pose] = None
    best_score = float("inf")
    best_terms: Dict[str, Any] = {}
    radii = [0.8, 1.1, 1.5]
    angles = [i * (math.pi / 4.0) for i in range(8)]
    for radius in radii:
        for angle in angles:
            raw = Pose(
                center_x + radius * math.cos(angle),
                float(current_pose.y),
                center_z + radius * math.sin(angle),
                math.degrees(math.atan2(center_x - (center_x + radius * math.cos(angle)), center_z - (center_z + radius * math.sin(angle)))),
            )
            try:
                snapped = loop.adapter.snap_pose(raw)
            except Exception:
                continue
            path_dist = _path_distance(loop, current_pose, snapped)
            object_dist = euclidean_2d([snapped.x, snapped.z], [center_x, center_z])
            revisit = 0.0
            for prev in recent_poses[-6:]:
                d = euclidean_2d([snapped.x, snapped.z], [prev.x, prev.z])
                if d < 1.0:
                    revisit += 1.0 - d
            score = path_dist + 0.5 * abs(object_dist - 1.1) + 0.75 * revisit
            if score < best_score:
                best_pose = snapped
                best_score = score
                best_terms = {
                    "source": "object_approach_pose",
                    "object_distance": round(float(object_dist), 6),
                    "geodesic_distance": round(float(path_dist), 6),
                    "revisit_penalty": round(float(revisit), 6),
                }
    if best_pose is None:
        pose = loop.adapter.snap_pose(Pose(center_x, float(current_pose.y), center_z, float(current_pose.yaw)))
        return pose, {"source": "object_center_snap_fallback"}
    return best_pose, best_terms


def _select_room_level_goal(
    candidates: List[CandidateScore],
    memory: SceneMemory,
    loop: HabitatVisionLoop,
    current_pose: Pose,
    args: argparse.Namespace,
    *,
    previous_room_id: Optional[str],
    room_commit_remaining: int,
    recent_rooms: List[str],
    recent_poses: List[Pose],
    failed_rooms: Dict[str, int],
    query: QuerySpec,
) -> Tuple[Optional[CandidateScore], Optional[Pose], Dict[str, Any], int]:
    if not candidates:
        return None, None, {"selected_pose_source": "no_candidate"}, 0
    room_groups: Dict[str, List[CandidateScore]] = defaultdict(list)
    for cand in candidates:
        room_groups[_candidate_room_id(cand, memory)].append(cand)
    room_rows: List[Dict[str, Any]] = []
    raw_distances: Dict[str, float] = {}
    for room_id, items in room_groups.items():
        top_items = sorted(items, key=lambda c: -float(c.S_final))[: max(1, int(args.room_top_k_objects))]
        rep_pose = _room_representative_pose(top_items, current_pose)
        raw_distances[room_id] = _path_distance(loop, current_pose, rep_pose)
    max_dist = max(raw_distances.values(), default=1.0) or 1.0
    for room_id, items in room_groups.items():
        ranked = sorted(items, key=lambda c: -float(c.S_final))
        top_items = ranked[: max(1, int(args.room_top_k_objects))]
        object_scores = [float(c.S_final) for c in top_items]
        max_object_score = max(object_scores) if object_scores else 0.0
        mean_top3 = sum(object_scores[:3]) / max(1, len(object_scores[:3]))
        target_room_belief = max((_candidate_room_belief(c, memory, room_id) for c in top_items), default=0.0)
        exploration_gain = 1.0 / math.sqrt(1.0 + len(memory.rooms.get(room_id).object_refs if room_id in memory.rooms else []))
        normalized_dist = raw_distances.get(room_id, max_dist) / max_dist
        room_revisit_penalty = sum(1.0 for rid in recent_rooms[-int(args.room_revisit_window) :] if rid == room_id) / max(
            1, int(args.room_revisit_window)
        )
        room_revisit_penalty += float(failed_rooms.get(room_id, 0)) * float(args.room_negative_penalty_weight)
        room_switch_penalty = 1.0 if previous_room_id and room_id != previous_room_id and room_commit_remaining > 0 else 0.0
        score = (
            0.45 * max_object_score
            + 0.20 * mean_top3
            + 0.15 * target_room_belief
            + 0.10 * exploration_gain
            - float(args.room_distance_weight) * normalized_dist
            - 0.20 * room_revisit_penalty
            - 0.15 * room_switch_penalty
        )
        room_rows.append(
            {
                "room_id": room_id,
                "score": round(float(score), 6),
                "terms": {
                    "max_object_score": round(float(max_object_score), 6),
                    "mean_top3_object_score": round(float(mean_top3), 6),
                    "target_room_belief": round(float(target_room_belief), 6),
                    "exploration_gain": round(float(exploration_gain), 6),
                    "normalized_geodesic_distance": round(float(normalized_dist), 6),
                    "room_revisit_penalty": round(float(room_revisit_penalty), 6),
                    "room_switch_penalty": round(float(room_switch_penalty), 6),
                },
                "top_object_ids": [c.obj_id for c in top_items],
                "top_labels": [c.label for c in top_items],
            }
        )
    room_rows.sort(key=lambda item: (-float(item["score"]), str(item["room_id"])))
    selected_room = room_rows[0]
    switch_reason = "best_room"
    if previous_room_id and room_commit_remaining > 0:
        previous = next((row for row in room_rows if row["room_id"] == previous_room_id), None)
        if previous is not None and float(selected_room["score"]) <= float(previous["score"]) + float(args.room_switch_margin):
            selected_room = previous
            switch_reason = "commit_previous_room"
    selected_room_id = str(selected_room["room_id"])
    room_candidates = sorted(room_groups[selected_room_id], key=lambda c: -float(c.S_final))
    selected_candidate = next((c for c in room_candidates if query_label_match_strength(c.label, query) == "strong"), None)
    selected_pose_source = "object_approach_pose"
    approach_details: Dict[str, Any] = {}
    if selected_candidate is None:
        selected_candidate = room_candidates[0] if room_candidates else None
        representative = _room_representative_pose(room_candidates[: max(1, int(args.room_top_k_objects))], current_pose)
        selected_pose_source = "room_representative_pose"
        try:
            target_pose = loop.adapter.snap_pose(representative) if loop.adapter is not None else representative
        except Exception:
            target_pose = representative
    else:
        target_pose, approach_details = _approach_pose_for_candidate(loop, selected_candidate, current_pose, recent_poses=recent_poses)
        selected_pose_source = str(approach_details.get("source") or "object_approach_pose")
    if selected_candidate is None:
        return None, None, {"selected_pose_source": "empty_selected_room", "room_scores_top": room_rows[:5]}, 0
    next_commit = max(0, int(args.room_commit_rounds) - 1) if switch_reason == "best_room" else max(0, room_commit_remaining - 1)
    planning_info = {
        "planning_mode": "room",
        "selected_room_id": selected_room_id,
        "room_scores_top": room_rows[:5],
        "room_score_terms": selected_room.get("terms", {}),
        "selected_pose_source": selected_pose_source,
        "approach_pose": to_json_dict(target_pose),
        "approach_terms": approach_details,
        "room_switch_reason": switch_reason,
        "revisit_penalty": selected_room.get("terms", {}).get("room_revisit_penalty"),
    }
    return selected_candidate, target_pose, planning_info, next_commit


def _fallback_pose(loop: HabitatVisionLoop, episode: Episode, subtask: Subtask, round_idx: int) -> Pose:
    if loop.adapter is None or loop.adapter.sim is None:
        rng = random.Random(episode.seed + round_idx)
        return Pose(
            float(loop.current_pose.x + rng.uniform(-1.0, 1.0)),  # type: ignore[union-attr]
            float(loop.current_pose.y),  # type: ignore[union-attr]
            float(loop.current_pose.z + rng.uniform(-1.0, 1.0)),  # type: ignore[union-attr]
            float(loop.current_pose.yaw),  # type: ignore[union-attr]
        )
    point = loop.adapter.pathfinder.get_random_navigable_point()
    if point is None:
        return loop.current_pose or episode.start_pose
    return Pose(float(point[0]), float(point[1]), float(point[2]), float((loop.current_pose or episode.start_pose).yaw))


def _rank_with_remote_clip(
    memory: SceneMemory,
    query: QuerySpec,
    state_index: int,
    round_idx: int,
    image_index: ImageGoalIndex,
    loop: HabitatVisionLoop,
    *,
    max_candidates: int,
    clip_min_score: float,
    remote_clip_include_weak_aliases: bool = False,
    remote_clip_none_min_score: float = 0.25,
    remote_clip_none_score_cap: float = 0.30,
    remote_clip_weak_min_score: float = 0.25,
) -> List[CandidateScore]:
    params = FusionParams()
    candidates = rank_candidates(
        memory.floors(),
        query,
        state_index,
        exploration_round=round_idx,
        max_candidates=max_candidates,
        image_index=image_index,
        params=params,
    )
    by_id = {cand.obj_id: cand for cand in candidates}
    objects = [obj for obj in memory.objects() if obj.clip_embedding and obj.pos_2d]
    if not objects:
        return candidates
    if query.language_prompt:
        text = query.language_prompt
    else:
        text_aliases = list(query.strong_alias_labels or [query.query_label])
        if remote_clip_include_weak_aliases:
            text_aliases.extend(query.weak_alias_labels or [])
        text = " . ".join(alias.replace("_", " ") for alias in text_aliases[:6])
    sims = loop.detector.clip_text_image_similarity(
        text,
        [obj.clip_embedding for obj in objects],
        ids=[obj.obj_id for obj in objects],
        labels=[obj.label for obj in objects],
        top_k=max(max_candidates, 20),
        min_score=clip_min_score,
    )
    obj_by_id = {obj.obj_id: obj for obj in objects}
    for item in sims:
        obj_id = str(item.get("id", ""))
        raw_score = float(item.get("score", 0.0) or 0.0)
        obj = obj_by_id.get(obj_id)
        if obj is None or not obj.pos_2d:
            continue
        match_strength = query_label_match_strength(obj.label, query)
        if match_strength == "weak" and raw_score < float(remote_clip_weak_min_score):
            continue
        if match_strength == "none" and raw_score < float(remote_clip_none_min_score):
            continue
        if match_strength == "weak":
            score = min(raw_score, WEAK_ALIAS_SIMILARITY_CAP)
        elif match_strength == "none":
            score = min(raw_score, float(remote_clip_none_score_cap))
        else:
            score = raw_score
        score_note = f"remote_clip={raw_score:.4f}"
        if match_strength == "weak" and score != raw_score:
            score_note += f",weak_alias_cap={score:.4f}"
        if match_strength == "none" and score != raw_score:
            score_note += f",none_alias_cap={score:.4f}"
        if match_strength == "none":
            score_note += f",none_alias_min={float(remote_clip_none_min_score):.4f}"
        if match_strength == "weak":
            score_note += f",weak_alias_min={float(remote_clip_weak_min_score):.4f}"
        if obj_id in by_id:
            cand = by_id[obj_id]
            cand.R_sim = round(max(float(cand.R_sim), score), 6)
            cand.S_rag = round(max(float(cand.S_rag), score), 6)
            cand.S_final = round(max(float(cand.S_final), score), 6)
            cand.sim_backend = f"{cand.sim_backend}+remote_clip"
            cand.reason = f"{cand.reason},{score_note}"
            continue
        stability = max(0.0, min(1.0, float(obj.stability if obj.stability is not None else 0.5)))
        exist_prob = max(0.0, min(1.0, float(obj.exist_prob if obj.exist_prob is not None else 1.0)))
        candidates.append(
            CandidateScore(
                obj_id=obj.obj_id,
                label=obj.label,
                world_x=float(obj.pos_2d[0]),
                world_z=float(obj.pos_2d[1]),
                R_sim=round(score, 6),
                R_cfd=0.0,
                G_ts=1.0,
                S_rag=round(score * exist_prob, 6),
                S_agent=0.0,
                w_agent=0.0,
                w_rag=1.0,
                S_final=round(score * exist_prob, 6),
                stability=round(stability, 6),
                exist_prob=round(exist_prob, 6),
                negative_feedback_count=int((obj.cooccur_stats or {}).get("negative_feedback_count", 0) or 0),
                stability_bucket="high" if stability >= 0.67 else "medium" if stability >= 0.34 else "low",
                query_modality=image_index.query_modality(query),
                sim_backend="remote_clip_text_image",
                reason=score_note,
            )
        )
    candidates.sort(key=lambda cand: (-float(cand.S_final), cand.obj_id))
    return candidates[:max_candidates]


def _trace_distance(final_pose: Pose, subtask: Subtask) -> float:
    return horizontal_distance(final_pose, subtask.target_position)


def run_episode_temporal(
    episode: Episode,
    memory: SceneMemory,
    loop: HabitatVisionLoop,
    args: argparse.Namespace,
    *,
    global_labels: List[str],
) -> Tuple[Trajectory, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    start_time = time.time()
    image_index = ImageGoalIndex(images_dir=args.images_dir)
    current_pose = loop.current_pose or episode.start_pose
    total_steps = 0
    found_count = 0
    step_records: List[TrajectoryStep] = [
        TrajectoryStep(t=0, position=current_pose, action="RESET", completed_subtask_ids=[])
    ]
    traces: List[SubtaskTrace] = []
    diagnostics: List[Dict[str, Any]] = []
    candidate_records: List[Dict[str, Any]] = []
    observation_records: List[Dict[str, Any]] = []
    memory_events: List[Dict[str, Any]] = []
    feedback_records: List[Dict[str, Any]] = []

    for subtask in episode.subtasks:
        query = query_from_subtask(
            episode,
            subtask,
            target_name_map=args.target_name_map_data,
            normalize_target_names=bool(args.normalize_target_names),
        )
        path_length = 0.0
        subtask_steps = 0
        perception_found = False
        fallback_count = 0
        last_candidates: List[Dict[str, Any]] = []
        final_pose = current_pose
        target_detection_summary: Dict[str, Any] = {}
        previous_room_id: Optional[str] = None
        room_commit_remaining = 0
        recent_rooms: List[str] = []
        recent_poses: List[Pose] = []
        failed_rooms: Dict[str, int] = defaultdict(int)

        for round_idx in range(max(1, int(args.max_rounds))):
            if subtask_steps >= int(args.max_steps_per_subtask):
                break
            text_prompt = _prompt_for_query(query, global_labels, memory, int(args.max_detect_labels))
            obs = loop.observe(
                n_views=args.n_views,
                text_prompt=text_prompt,
                step=total_steps,
                state_index=episode.state_index,
                layout_id=episode.layout_id,
            )
            observation_records.extend(obs.records)
            for event in memory.update_from_detections(
                obs.detections,
                state_index=episode.state_index,
                layout_id=episode.layout_id,
                step=total_steps,
            ):
                event_record = event.to_dict()
                event_record.update(
                    {
                        "episode_id": episode.episode_id,
                        "subtask_id": subtask.subtask_id,
                        "scene_name": episode.scene_name,
                        "round": round_idx,
                        "observation_phase": "pre_move",
                    }
                )
                memory_events.append(event_record)
            obs_target_diag = _target_detection_diagnostics(
                obs.detections,
                query,
                subtask=subtask,
                confirmation_mode=args.target_confirmation_mode,
                spatial_margin=args.target_confirmation_margin,
            )
            target_detection_summary = _merge_target_detection_diagnostics(target_detection_summary, obs_target_diag)
            if obs_target_diag.get("target_detection_confirmed"):
                perception_found = True
                final_pose = loop.current_pose or current_pose
                break

            candidates = _rank_with_remote_clip(
                memory,
                query,
                episode.state_index,
                round_idx,
                image_index,
                loop,
                max_candidates=args.max_candidates,
                clip_min_score=args.clip_min_score,
                remote_clip_include_weak_aliases=bool(args.remote_clip_include_weak_aliases),
                remote_clip_none_min_score=float(args.remote_clip_none_min_score),
                remote_clip_none_score_cap=float(args.remote_clip_none_score_cap),
                remote_clip_weak_min_score=float(args.remote_clip_weak_min_score),
            )
            last_candidates = _annotate_candidate_debug(candidates_to_dict(candidates), query, subtask)
            selected_candidate: Optional[CandidateScore] = None
            planning_info: Dict[str, Any] = {"planning_mode": args.planning_mode}
            if candidates:
                if args.planning_mode == "room":
                    selected_candidate, planned_pose, planning_info, room_commit_remaining = _select_room_level_goal(
                        candidates,
                        memory,
                        loop,
                        loop.current_pose or current_pose,
                        args,
                        previous_room_id=previous_room_id,
                        room_commit_remaining=room_commit_remaining,
                        recent_rooms=recent_rooms,
                        recent_poses=recent_poses,
                        failed_rooms=failed_rooms,
                        query=query,
                    )
                    if selected_candidate is not None and planned_pose is not None:
                        target_pose = planned_pose
                    else:
                        selected_candidate = candidates[0]
                        target_pose = _pose_from_candidate(selected_candidate, loop.current_pose or current_pose)
                        planning_info = {"planning_mode": "room", "selected_pose_source": "object_top1_fallback"}
                else:
                    selected_candidate = candidates[0]
                    target_pose = _pose_from_candidate(selected_candidate, loop.current_pose or current_pose)
                    planning_info = {"planning_mode": "object", "selected_pose_source": "object_center"}
                fallback_reason = None
            else:
                target_pose = _fallback_pose(loop, episode, subtask, round_idx)
                fallback_reason = "no_candidate"
                fallback_count += 1
                planning_info = {"planning_mode": args.planning_mode, "selected_pose_source": "random_navmesh_fallback"}

            remaining_steps = max(1, int(args.max_steps_per_subtask) - subtask_steps)
            step_result = loop.step_to(target_pose, max_micro_steps=min(int(args.micro_steps_per_round), remaining_steps))
            subtask_steps += step_result.steps
            total_steps += step_result.steps
            path_length += step_result.path_length
            current_pose = step_result.final_pose
            final_pose = step_result.final_pose
            selected_candidate_record = None
            if selected_candidate is not None:
                selected_candidate_record = next(
                    (item for item in last_candidates if item.get("obj_id") == selected_candidate.obj_id),
                    None,
                )
            if selected_candidate_record is None and last_candidates:
                selected_candidate_record = last_candidates[0]
            candidate_records.append(
                _safe_json_value(
                    {
                        "episode_id": episode.episode_id,
                        "subtask_id": subtask.subtask_id,
                        "scene_name": episode.scene_name,
                        "layout_id": episode.layout_id,
                        "state_index": episode.state_index,
                        "round": round_idx,
                        "fallback_reason": fallback_reason,
                        "selected_candidate": selected_candidate_record,
                        "top_candidates": last_candidates,
                        **planning_info,
                        "target_position": [subtask.target_position.x, subtask.target_position.z],
                        "success_radius": subtask.success_radius,
                        "planned_pose": to_json_dict(target_pose),
                        "final_pose": to_json_dict(final_pose),
                        "path": step_result.path,
                        "path_length": step_result.path_length,
                        "micro_steps": step_result.steps,
                    }
                )
            )
            if planning_info.get("selected_room_id"):
                previous_room_id = str(planning_info.get("selected_room_id"))
                recent_rooms.append(previous_room_id)
                recent_rooms = recent_rooms[-20:]
            recent_poses.append(target_pose)
            recent_poses = recent_poses[-20:]

            text_prompt = _prompt_for_query(query, global_labels, memory, int(args.max_detect_labels))
            post_obs = loop.observe(
                n_views=args.n_views,
                text_prompt=text_prompt,
                step=total_steps,
                state_index=episode.state_index,
                layout_id=episode.layout_id,
            )
            observation_records.extend(post_obs.records)
            for event in memory.update_from_detections(
                post_obs.detections,
                state_index=episode.state_index,
                layout_id=episode.layout_id,
                step=total_steps,
            ):
                event_record = event.to_dict()
                event_record.update(
                    {
                        "episode_id": episode.episode_id,
                        "subtask_id": subtask.subtask_id,
                        "scene_name": episode.scene_name,
                        "round": round_idx,
                        "observation_phase": "post_move",
                    }
                )
                memory_events.append(event_record)
            post_target_diag = _target_detection_diagnostics(
                post_obs.detections,
                query,
                subtask=subtask,
                confirmation_mode=args.target_confirmation_mode,
                spatial_margin=args.target_confirmation_margin,
            )
            target_detection_summary = _merge_target_detection_diagnostics(target_detection_summary, post_target_diag)
            if post_target_diag.get("target_detection_confirmed"):
                perception_found = True
                break
            if selected_candidate is not None:
                event = memory.record_negative_feedback(
                    selected_candidate.obj_id,
                    state_index=episode.state_index,
                    layout_id=episode.layout_id,
                    step=total_steps,
                    reason="candidate_reached_target_not_detected",
                )
                if event is not None:
                    record = event.to_dict()
                    feedback_records.append(record)
                    memory_events.append(record)
                selected_room = planning_info.get("selected_room_id")
                if selected_room:
                    failed_rooms[str(selected_room)] += 1

        final_dist = _trace_distance(final_pose, subtask)
        benchmark_success = final_dist <= float(subtask.success_radius)
        if benchmark_success:
            found_count += 1
        trace = SubtaskTrace(
            subtask_id=subtask.subtask_id,
            final_pose=final_pose,
            path_length=round(float(path_length), 6),
            steps=max(1, int(subtask_steps)),
            elapsed_seconds=round(time.time() - start_time, 6),
            reported_success=perception_found,
            metadata={
                "perception_found": perception_found,
                "benchmark_distance_to_goal": final_dist,
                "fallback_count": fallback_count,
                "target_confirmation_mode": args.target_confirmation_mode,
                "target_confirmation_margin": args.target_confirmation_margin,
                "runner": "temporal_habitat_loop",
            },
        )
        traces.append(trace)
        step_records.append(
            TrajectoryStep(
                t=total_steps,
                position=final_pose,
                action="TEMPORAL_HABITAT_SEARCH",
                completed_subtask_ids=[subtask.subtask_id],
            )
        )
        dists = [
            euclidean_2d([cand["world_x"], cand["world_z"]], [subtask.target_position.x, subtask.target_position.z])
            for cand in last_candidates
            if "world_x" in cand and "world_z" in cand
        ]
        diagnostics.append(
            _safe_json_value(
                {
                    "episode_id": episode.episode_id,
                    "subtask_id": subtask.subtask_id,
                    "scene_name": episode.scene_name,
                    "layout_id": episode.layout_id,
                    "state_index": episode.state_index,
                    "seen_layout_count_before": episode.seen_layout_count_before,
                    "task_type": subtask.task_type,
                    "target_object": subtask.target_object,
                    "query_label": query.query_label,
                    "alias_labels": query.alias_labels,
                    "strong_alias_labels": query.strong_alias_labels,
                    "weak_alias_labels": query.weak_alias_labels,
                    "query_modality": image_index.query_modality(query),
                    "perception_found": perception_found,
                    "found": benchmark_success,
                    "target_confirmation_mode": args.target_confirmation_mode,
                    "target_confirmation_margin": args.target_confirmation_margin,
                    "success_radius": subtask.success_radius,
                    "sss": trace.steps,
                    "mra": bool(dists and dists[0] <= float(subtask.success_radius)),
                    "ghr3": any(d <= float(subtask.success_radius) for d in dists[:3]),
                    "ghr5": any(d <= float(subtask.success_radius) for d in dists[:5]),
                    "ghr10": any(d <= float(subtask.success_radius) for d in dists[:10]),
                    "min_dist": min(dists) if dists else None,
                    "top1_dist": dists[0] if dists else None,
                    "top3_min_dist": min(dists[:3]) if dists[:3] else None,
                    "top5_min_dist": min(dists[:5]) if dists[:5] else None,
                    "top10_min_dist": min(dists[:10]) if dists[:10] else None,
                    "final_dist": final_dist,
                    "final_minus_best_candidate_dist": (
                        final_dist - min(dists) if dists else None
                    ),
                    "fallback_count": fallback_count,
                    "memory_track_count": len(memory),
                    "peaks": last_candidates,
                    "target_position": [subtask.target_position.x, subtask.target_position.z],
                    **target_detection_summary,
                }
            )
        )

    trajectory = Trajectory(
        episode_id=episode.episode_id,
        steps=step_records,
        finished=True,
        finish_reason="all_subtasks_processed",
        elapsed_seconds=round(time.time() - start_time, 6),
        agent_id="raanav_temporal_habitat_loop",
        scene_name=episode.scene_name,
        layout_id=episode.layout_id,
        seen_layout_count_before=episode.seen_layout_count_before,
        subtask_traces=traces,
        metadata={
            "runner": "RAANav.AgenticRAG.benchmark_adapter.run_temporal_habitat_loop",
            "memory_metrics": {
                "dynamic_memory_correct": found_count,
                "dynamic_memory_total": len(episode.subtasks),
                "fixed_memory_correct": 0,
                "fixed_memory_total": 0,
            },
        },
    )
    return trajectory, diagnostics, candidate_records, observation_records, memory_events, feedback_records


def _write_memory_snapshot(output_dir: Path, memory: SceneMemory, episode: Episode, suffix: str) -> Path:
    path = (
        output_dir
        / "memory_snapshots"
        / episode.scene_name
        / f"state_{int(episode.state_index):03d}_{episode.layout_id}_{suffix}.json"
    )
    write_json(path, memory.to_dict())
    return path


def _temporal_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped_state: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    grouped_seen: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_state[int(record.get("state_index", 0))].append(record)
        grouped_seen[int(record.get("seen_layout_count_before", 0))].append(record)
    return {
        "overall": _aggregate_group(records),
        "by_state_index": {str(k): _aggregate_group(v) for k, v in sorted(grouped_state.items())},
        "by_seen_layout_count_before": {str(k): _aggregate_group(v) for k, v in sorted(grouped_seen.items())},
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.perception_backend != "remote":
        raise ValueError("run_temporal_habitat_loop requires --perception-backend remote for the real-vision path.")
    index = DatasetIndex(args.split_manifest)
    episodes_root, episodes = _load_sorted_episodes(args, index)
    output_dir = as_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.target_name_map_data = load_target_name_map(args.target_name_map)
    global_labels = _collect_prompt_labels(
        episodes,
        target_name_map=args.target_name_map_data,
        normalize_target_names=bool(args.normalize_target_names),
    )

    run_path = output_dir / "run.jsonl"
    diag_path = output_dir / "memory_diagnostics.jsonl"
    candidate_path = output_dir / "candidate_traces.jsonl"
    observation_path = output_dir / "observation_traces.jsonl"
    memory_updates_path = output_dir / "memory_updates.jsonl"
    feedback_path = output_dir / "negative_feedback.jsonl"
    transitions_path = output_dir / "layout_transitions.jsonl"

    memories: Dict[str, SceneMemory] = {}
    all_diagnostics: List[Dict[str, Any]] = []
    warnings: List[str] = []
    exported_geometry_scenes: set[str] = set()
    current_layout_key: Optional[Tuple[str, str]] = None
    current_layout_episode: Optional[Episode] = None
    loop: Optional[HabitatVisionLoop] = None

    try:
        loop = HabitatVisionLoop(
            data_dir=args.data_dir,
            objects_dir=args.objects_dir,
            load_layout_objects=bool(args.load_layout_objects),
            require_habitat=not bool(args.euclidean_fallback),
            sensor_width=args.sensor_width,
            sensor_height=args.sensor_height,
            max_depth=args.max_depth,
            step_size=args.step_size,
            remote_vision_base_url=args.remote_vision_base_url,
            remote_vision_use_ssh_tunnel=bool(args.remote_vision_use_ssh_tunnel),
            remote_vision_ssh_host=args.remote_vision_ssh_host,
            remote_vision_ssh_port=args.remote_vision_ssh_port,
            remote_vision_ssh_user=args.remote_vision_ssh_user,
            remote_vision_ssh_password=args.remote_vision_ssh_password,
            remote_vision_remote_port=args.remote_vision_remote_port,
            remote_vision_local_port=args.remote_vision_local_port,
        )
        with run_path.open("w", encoding="utf-8") as run_f, diag_path.open("w", encoding="utf-8") as diag_f, candidate_path.open("w", encoding="utf-8") as cand_f, observation_path.open("w", encoding="utf-8") as obs_f, memory_updates_path.open("w", encoding="utf-8") as mem_f, feedback_path.open("w", encoding="utf-8") as fb_f, transitions_path.open("w", encoding="utf-8") as trans_f:
            for episode in episodes:
                warnings.extend(index.validate_episode_layout(episode))
                memory = memories.setdefault(episode.scene_name, SceneMemory(episode.scene_name))
                layout_key = (episode.scene_name, episode.layout_id)
                if current_layout_key != layout_key:
                    if current_layout_key is not None and current_layout_episode is not None:
                        snapshot_path = _write_memory_snapshot(
                            output_dir,
                            memories[current_layout_key[0]],
                            current_layout_episode,
                            "after_layout",
                        )
                        trans_f.write(
                            json.dumps(
                                {
                                    "event": "layout_exit",
                                    "scene_name": current_layout_key[0],
                                    "layout_id": current_layout_key[1],
                                    "memory_snapshot": str(snapshot_path),
                                    "track_count": len(memories[current_layout_key[0]]),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    loop.reset_layout(
                        episode.scene_name,
                        as_path(episode.scene_state.layout_path),
                        episode.start_pose,
                        sample_start_seed=episode.seed if bool(args.sample_start_pose) else None,
                    )
                    if bool(args.export_scene_geometry) and episode.scene_name not in exported_geometry_scenes:
                        try:
                            if loop.adapter is not None:
                                geometry = loop.adapter.export_scene_geometry(
                                    grid_resolution=float(args.geometry_grid_resolution),
                                    max_random_points=int(args.geometry_max_random_points),
                                )
                                write_json(output_dir / "scene_geometry" / f"{episode.scene_name}.json", geometry)
                                exported_geometry_scenes.add(episode.scene_name)
                        except Exception as exc:
                            warnings.append(f"scene_geometry_export_failed:{episode.scene_name}:{exc}")
                    current_layout_key = layout_key
                    current_layout_episode = episode
                    trans_f.write(
                        json.dumps(
                            {
                                "event": "layout_enter",
                                "scene_name": episode.scene_name,
                                "layout_id": episode.layout_id,
                                "state_index": episode.state_index,
                                "layout_path": episode.scene_state.layout_path,
                                "track_count": len(memory),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                else:
                    loop.set_pose(
                        episode.start_pose,
                        sample_start_seed=episode.seed if bool(args.sample_start_pose) else None,
                    )

                trajectory, diagnostics, candidates, observations, memory_events, feedback = run_episode_temporal(
                    episode,
                    memory,
                    loop,
                    args,
                    global_labels=global_labels,
                )
                compaction_events = memory.compact_duplicates(
                    state_index=episode.state_index,
                    layout_id=episode.layout_id,
                    step=max((int(step.t) for step in trajectory.steps), default=0),
                )
                for event in compaction_events:
                    event_record = event.to_dict()
                    event_record.update(
                        {
                            "episode_id": episode.episode_id,
                            "scene_name": episode.scene_name,
                            "round": None,
                            "observation_phase": "after_episode_compaction",
                        }
                    )
                    memory_events.append(event_record)
                run_f.write(json.dumps(to_json_dict(trajectory), ensure_ascii=False) + "\n")
                for record in diagnostics:
                    all_diagnostics.append(record)
                    diag_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                for record in candidates:
                    cand_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                for record in observations:
                    obs_f.write(json.dumps(_safe_json_value(record), ensure_ascii=False) + "\n")
                for record in memory_events:
                    mem_f.write(json.dumps(_safe_json_value(record), ensure_ascii=False) + "\n")
                for record in feedback:
                    fb_f.write(json.dumps(_safe_json_value(record), ensure_ascii=False) + "\n")
                _write_memory_snapshot(output_dir, memory, episode, "after_episode")

            if current_layout_key is not None and current_layout_episode is not None:
                snapshot_path = _write_memory_snapshot(
                    output_dir,
                    memories[current_layout_key[0]],
                    current_layout_episode,
                    "after_layout",
                )
                trans_f.write(
                    json.dumps(
                        {
                            "event": "layout_exit",
                            "scene_name": current_layout_key[0],
                            "layout_id": current_layout_key[1],
                            "memory_snapshot": str(snapshot_path),
                            "track_count": len(memories[current_layout_key[0]]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        for scene_name, memory in memories.items():
            write_json(output_dir / "scene_memory" / scene_name / "scene_memory_final.json", memory.to_dict())
        write_json(output_dir / "temporal_summary.json", _temporal_summary(all_diagnostics))
        write_json(output_dir / "adapter_warnings.json", {"warnings": warnings, "warning_count": len(warnings)})
        write_json(
            output_dir / "target_name_normalization.json",
            {
                "normalize_target_names": bool(args.normalize_target_names),
                "target_name_map": args.target_name_map_data,
                "remote_clip_retrieval": {
                    "clip_min_score": float(args.clip_min_score),
                    "include_weak_aliases": bool(args.remote_clip_include_weak_aliases),
                    "none_min_score": float(args.remote_clip_none_min_score),
                    "none_score_cap": float(args.remote_clip_none_score_cap),
                    "weak_min_score": float(args.remote_clip_weak_min_score),
                    "weak_score_cap": WEAK_ALIAS_SIMILARITY_CAP,
                },
                "planning": {
                    "planning_mode": args.planning_mode,
                    "room_top_k_objects": int(args.room_top_k_objects),
                    "room_commit_rounds": int(args.room_commit_rounds),
                    "room_switch_margin": float(args.room_switch_margin),
                    "room_revisit_window": int(args.room_revisit_window),
                    "room_negative_penalty_weight": float(args.room_negative_penalty_weight),
                    "room_distance_weight": float(args.room_distance_weight),
                },
            },
        )
        if loop is not None:
            write_json(output_dir / "perception_summary.json", loop.perception_summary())

        bench_args = argparse.Namespace(
            episodes=str(episodes_root),
            trajectories=str(run_path),
            output_dir=str(output_dir),
            data_dir=args.data_dir,
            objects_dir=args.objects_dir,
            euclidean_fallback=bool(args.euclidean_fallback),
        )
        benchmark_result = benchmark_evaluate(bench_args)
    finally:
        if loop is not None:
            loop.close()

    result = {
        "episodes": len(episodes),
        "output_dir": str(output_dir),
        "run_jsonl": str(run_path),
        "memory_diagnostics": str(diag_path),
        "candidate_traces": str(candidate_path),
        "observation_traces": str(observation_path),
        "memory_updates": str(memory_updates_path),
        "negative_feedback": str(feedback_path),
        "layout_transitions": str(transitions_path),
        "benchmark_summary": benchmark_result.get("summary", {}),
        "warnings": len(warnings),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal Habitat RGB-D closed-loop benchmark.")
    parser.add_argument("--split-manifest", default="benchmark/splits/benchmark_split_longterm_v1.json")
    parser.add_argument("--episodes", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--layout-limit", type=int, default=0)
    parser.add_argument("--episodes-per-layout", type=int, default=0)
    parser.add_argument("--output-dir", default="benchmark/eval/longterm_habitat_loop")
    parser.add_argument("--images-dir", default="objects_images")
    parser.add_argument("--data-dir", default="hm3d")
    parser.add_argument("--objects-dir", default="objects")
    parser.add_argument("--load-layout-objects", action="store_true")
    parser.add_argument("--euclidean-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-start-pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--perception-backend", choices=("remote",), default="remote")
    parser.add_argument("--remote-vision-base-url", default=None)
    parser.add_argument("--remote-vision-use-ssh-tunnel", action="store_true")
    parser.add_argument("--remote-vision-ssh-host", default="7.216.187.6")
    parser.add_argument("--remote-vision-ssh-port", type=int, default=30180)
    parser.add_argument("--remote-vision-ssh-user", default="root")
    parser.add_argument("--remote-vision-ssh-password", default=None)
    parser.add_argument("--remote-vision-remote-port", type=int, default=8010)
    parser.add_argument("--remote-vision-local-port", type=int, default=None)
    parser.add_argument("--max-steps-per-subtask", type=int, default=80)
    parser.add_argument("--micro-steps-per-round", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--max-detect-labels", type=int, default=80)
    parser.add_argument(
        "--planning-mode",
        choices=("object", "room"),
        default="room",
        help="Use legacy object Top-1 planning or room-first planning.",
    )
    parser.add_argument("--room-top-k-objects", type=int, default=5)
    parser.add_argument("--room-commit-rounds", type=int, default=2)
    parser.add_argument("--room-switch-margin", type=float, default=0.12)
    parser.add_argument("--room-revisit-window", type=int, default=4)
    parser.add_argument("--room-negative-penalty-weight", type=float, default=0.20)
    parser.add_argument("--room-distance-weight", type=float, default=0.25)
    parser.add_argument("--n-views", type=int, default=4)
    parser.add_argument("--sensor-width", type=int, default=640)
    parser.add_argument("--sensor-height", type=int, default=480)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--clip-min-score", type=float, default=0.2)
    parser.add_argument(
        "--remote-clip-include-weak-aliases",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include weak target aliases in the remote CLIP text query. "
            "Disabled by default to avoid broad words such as cup/bottle/stool dominating retrieval."
        ),
    )
    parser.add_argument(
        "--remote-clip-none-min-score",
        type=float,
        default=0.25,
        help="Minimum remote CLIP score required for candidates whose label has no strong/weak alias match.",
    )
    parser.add_argument(
        "--remote-clip-none-score-cap",
        type=float,
        default=0.30,
        help="Maximum S_final contribution allowed for remote CLIP candidates with no alias match.",
    )
    parser.add_argument(
        "--remote-clip-weak-min-score",
        type=float,
        default=0.25,
        help="Minimum remote CLIP score required for candidates whose label only matches a weak alias.",
    )
    parser.add_argument(
        "--export-scene-geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export scene_geometry/<scene>.json for 2D/3D visual debugging.",
    )
    parser.add_argument(
        "--geometry-grid-resolution",
        type=float,
        default=0.25,
        help="Meters per cell when sampling navmesh footprint for visualization.",
    )
    parser.add_argument(
        "--geometry-max-random-points",
        type=int,
        default=6000,
        help="Maximum random navigable points used to estimate visualization bounds.",
    )
    parser.add_argument(
        "--target-name-map",
        default=None,
        help="Optional JSON object mapping raw target names to canonical labels or alias lists.",
    )
    parser.add_argument(
        "--normalize-target-names",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable built-in target aliases such as chess_set->chess and camera_01->camera.",
    )
    parser.add_argument(
        "--target-confirmation-mode",
        choices=("spatial", "semantic"),
        default="spatial",
        help="Use spatial target confirmation for benchmark runs, or semantic for legacy label-only confirmation.",
    )
    parser.add_argument(
        "--target-confirmation-margin",
        type=float,
        default=1.0,
        help="Extra meters added to subtask.success_radius when spatially confirming a target detection.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
