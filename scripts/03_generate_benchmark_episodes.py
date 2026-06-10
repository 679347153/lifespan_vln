#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from benchmark.schemas import (
    Episode,
    Pose,
    SceneState,
    Subtask,
    read_split_manifest,
    validate_episode,
    write_json,
)

TASK_TYPES = ["open_vocab", "image_goal", "language_goal"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate benchmark episodes")
    p.add_argument("--split-manifest", default="benchmark/splits/benchmark_split_v1.json")
    p.add_argument("--states-per-scene", type=int, default=15)
    p.add_argument("--train-episodes-per-scene", type=int, default=5000)
    p.add_argument("--val-episodes-per-scene", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-root", default="benchmark/episodes/v1")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def list_scene_objects(scene_name: str) -> List[str]:
    prob_dir = Path("results/probabilities") / scene_name
    if not prob_dir.is_dir():
        # fallback names when no probability files exist yet
        return [
            "mug_01",
            "vase_01",
            "book_01",
            "bottle_01",
            "clock_01",
            "chair_01",
        ]

    objects = []
    for p in sorted(prob_dir.glob("*_probs.json")):
        stem = p.name
        objects.append(stem.replace("_probs.json", ""))
    return objects or ["mug_01", "vase_01", "book_01"]


def make_scene_state(scene_name: str, state_index: int, objects: List[str], rng: random.Random) -> SceneState:
    if not objects:
        objects = ["mug_01", "vase_01"]

    shuffled = objects[:]
    rng.shuffle(shuffled)
    split_at = max(1, len(shuffled) // 2)
    fixed_objects = sorted(shuffled[:split_at])
    movable_objects = sorted(shuffled[split_at:])

    transition_hint = {
        "mechanism": "hybrid",
        "rule_component": {
            "period": "weekly",
            "description": "Object shift triggered every 1-2 virtual weeks",
        },
        "prob_component": {
            "stay_prob": 0.55,
            "move_prob": 0.45,
        },
    }

    return SceneState(
        state_id=f"state_{state_index:02d}",
        time_index=state_index,
        fixed_objects=fixed_objects,
        movable_objects=movable_objects,
        transition_hint=transition_hint,
    )


def make_balanced_subtasks(scene_objects: List[str], rng: random.Random) -> List[Subtask]:
    n = rng.randint(5, 10)
    base = n // 3
    rem = n % 3
    counts = {
        "open_vocab": base,
        "image_goal": base,
        "language_goal": base,
    }
    for t in TASK_TYPES[:rem]:
        counts[t] += 1

    subtasks: List[Subtask] = []
    idx = 0
    for task_type in TASK_TYPES:
        for _ in range(counts[task_type]):
            obj = rng.choice(scene_objects)
            idx += 1
            image_path = f"objects_images/{obj}.jpg" if task_type == "image_goal" else None
            prompt = (
                f"Find object '{obj}' in the scene"
                if task_type != "language_goal"
                else f"Please navigate and locate the household object: {obj}."
            )
            subtasks.append(
                Subtask(
                    subtask_id=f"st_{idx:02d}",
                    task_type=task_type,  # type: ignore[arg-type]
                    target_object=obj,
                    prompt=prompt,
                    image_path=image_path,
                )
            )

    rng.shuffle(subtasks)
    return subtasks


def make_episode(split: str, scene_name: str, state: SceneState, episode_index: int, global_seed: int, scene_objects: List[str], rng: random.Random) -> Episode:
    ep_seed = (global_seed * 1_000_003 + episode_index * 97 + state.time_index * 13) % (2**31 - 1)
    start_pose = Pose(
        x=rng.uniform(-2.0, 2.0),
        y=rng.uniform(0.0, 1.0),
        z=rng.uniform(-2.0, 2.0),
        yaw=rng.uniform(-3.1416, 3.1416),
    )

    subtasks = make_balanced_subtasks(scene_objects, rng)
    episode = Episode(
        episode_id=f"{scene_name}_{state.state_id}_ep_{episode_index:05d}",
        split=split,  # type: ignore[arg-type]
        scene_name=scene_name,
        scene_state=state,
        seed=ep_seed,
        start_pose=start_pose,
        max_steps=rng.randint(250, 600),
        subtasks=subtasks,
        metadata={
            "benchmark_version": "v1",
            "subtask_count": len(subtasks),
            "task_mix": "balanced_3way",
        },
    )
    validate_episode(episode)
    return episode


def generate_for_split(split: str, scenes: List[str], episodes_per_scene: int, states_per_scene: int, output_root: Path, seed: int, dry_run: bool) -> int:
    total = 0
    rng = random.Random(seed)

    for scene_name in scenes:
        scene_objects = list_scene_objects(scene_name)
        for state_idx in range(1, states_per_scene + 1):
            state_rng = random.Random(rng.randint(0, 2**31 - 1))
            scene_state = make_scene_state(scene_name, state_idx, scene_objects, state_rng)

            # Distribute per-scene episodes across states.
            base = episodes_per_scene // states_per_scene
            rem = episodes_per_scene % states_per_scene
            state_episodes = base + (1 if state_idx <= rem else 0)

            for i in range(1, state_episodes + 1):
                ep_rng = random.Random(state_rng.randint(0, 2**31 - 1))
                ep = make_episode(split, scene_name, scene_state, i, seed, scene_objects, ep_rng)
                total += 1

                if dry_run:
                    continue

                out = (
                    output_root
                    / split
                    / scene_name
                    / scene_state.state_id
                    / f"{ep.episode_id}.json"
                )
                write_json(out, ep)

    return total


def main() -> None:
    args = parse_args()
    manifest = read_split_manifest(Path(args.split_manifest))

    out_root = Path(args.output_root)
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    train_total = generate_for_split(
        split="train",
        scenes=manifest.train_scenes,
        episodes_per_scene=args.train_episodes_per_scene,
        states_per_scene=args.states_per_scene,
        output_root=out_root,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    val_total = generate_for_split(
        split="val",
        scenes=manifest.val_scenes,
        episodes_per_scene=args.val_episodes_per_scene,
        states_per_scene=args.states_per_scene,
        output_root=out_root,
        seed=args.seed + 7,
        dry_run=args.dry_run,
    )

    print("[Done] Benchmark episode generation complete")
    print(f"  train episodes: {train_total}")
    print(f"  val episodes:   {val_total}")
    print(f"  dry_run:        {args.dry_run}")


if __name__ == "__main__":
    main()
