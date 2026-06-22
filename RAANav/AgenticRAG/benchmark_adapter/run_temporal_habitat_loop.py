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
from benchmark.habitat_adapter import euclidean_distance
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

from .common import as_path, euclidean_2d, normalize_label, token_overlap, write_json
from .dataset_index import DatasetIndex
from .episode_to_queries import QuerySpec, query_from_subtask
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


def _collect_prompt_labels(episodes: Iterable[Episode]) -> List[str]:
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
                norm = normalize_label(raw)
                if norm and norm not in seen:
                    seen.add(norm)
                    labels.append(_label_for_prompt(norm))
    return labels


def _prompt_for_query(query: QuerySpec, global_labels: List[str], memory: SceneMemory, max_labels: int) -> str:
    labels: List[str] = []
    seen = set()
    for raw in (query.query_label, query.target_object):
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


def _target_detected(detections: Iterable[Dict[str, Any]], query: QuerySpec) -> bool:
    query_label = normalize_label(query.query_label)
    target = normalize_label(query.target_object)
    for det in detections:
        label = normalize_label(det.get("label"))
        if not label:
            continue
        if label in {query_label, target}:
            return True
        if token_overlap(label, query_label) >= 0.6:
            return True
    return False


def _pose_from_candidate(candidate: CandidateScore, current_pose: Pose) -> Pose:
    return Pose(float(candidate.world_x), float(current_pose.y), float(candidate.world_z), float(current_pose.yaw))


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
    text = query.language_prompt or query.query_label.replace("_", " ")
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
        score = float(item.get("score", 0.0) or 0.0)
        obj = obj_by_id.get(obj_id)
        if obj is None or not obj.pos_2d:
            continue
        if obj_id in by_id:
            cand = by_id[obj_id]
            cand.R_sim = round(max(float(cand.R_sim), score), 6)
            cand.S_rag = round(max(float(cand.S_rag), score), 6)
            cand.S_final = round(max(float(cand.S_final), score), 6)
            cand.sim_backend = f"{cand.sim_backend}+remote_clip"
            cand.reason = f"{cand.reason},remote_clip={score:.4f}"
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
                reason=f"remote_clip={score:.4f}",
            )
        )
    candidates.sort(key=lambda cand: (-float(cand.S_final), cand.obj_id))
    return candidates[:max_candidates]


def _trace_distance(final_pose: Pose, subtask: Subtask) -> float:
    return euclidean_distance(final_pose, subtask.target_position)


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
        query = query_from_subtask(episode, subtask)
        path_length = 0.0
        subtask_steps = 0
        perception_found = False
        fallback_count = 0
        last_candidates: List[Dict[str, Any]] = []
        final_pose = current_pose

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
                memory_events.append(event.to_dict())
            if _target_detected(obs.detections, query):
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
            )
            last_candidates = candidates_to_dict(candidates)
            if candidates:
                target_pose = _pose_from_candidate(candidates[0], loop.current_pose or current_pose)
                fallback_reason = None
            else:
                target_pose = _fallback_pose(loop, episode, subtask, round_idx)
                fallback_reason = "no_candidate"
                fallback_count += 1

            remaining_steps = max(1, int(args.max_steps_per_subtask) - subtask_steps)
            step_result = loop.step_to(target_pose, max_micro_steps=min(int(args.micro_steps_per_round), remaining_steps))
            subtask_steps += step_result.steps
            total_steps += step_result.steps
            path_length += step_result.path_length
            current_pose = step_result.final_pose
            final_pose = step_result.final_pose
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
                        "selected_candidate": last_candidates[0] if last_candidates else None,
                        "top_candidates": last_candidates,
                        "planned_pose": to_json_dict(target_pose),
                        "final_pose": to_json_dict(final_pose),
                        "path_length": step_result.path_length,
                        "micro_steps": step_result.steps,
                    }
                )
            )

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
                memory_events.append(event.to_dict())
            if _target_detected(post_obs.detections, query):
                perception_found = True
                break
            if candidates:
                event = memory.record_negative_feedback(
                    candidates[0].obj_id,
                    state_index=episode.state_index,
                    layout_id=episode.layout_id,
                    step=total_steps,
                    reason="candidate_reached_target_not_detected",
                )
                if event is not None:
                    record = event.to_dict()
                    feedback_records.append(record)
                    memory_events.append(record)

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
                    "query_modality": image_index.query_modality(query),
                    "perception_found": perception_found,
                    "found": benchmark_success,
                    "sss": trace.steps,
                    "mra": bool(dists and dists[0] <= float(subtask.success_radius)),
                    "ghr3": any(d <= float(subtask.success_radius) for d in dists[:3]),
                    "ghr5": any(d <= float(subtask.success_radius) for d in dists[:5]),
                    "min_dist": min(dists) if dists else None,
                    "final_dist": final_dist,
                    "fallback_count": fallback_count,
                    "memory_track_count": len(memory),
                    "peaks": last_candidates,
                    "target_position": [subtask.target_position.x, subtask.target_position.z],
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
    global_labels = _collect_prompt_labels(episodes)

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
    parser.add_argument("--n-views", type=int, default=4)
    parser.add_argument("--sensor-width", type=int, default=640)
    parser.add_argument("--sensor-height", type=int, default=480)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--step-size", type=float, default=0.5)
    parser.add_argument("--clip-min-score", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
