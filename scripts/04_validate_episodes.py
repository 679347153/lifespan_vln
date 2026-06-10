#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate generated benchmark episodes")
    p.add_argument("--episodes-root", default="benchmark/episodes/v1")
    p.add_argument("--expected-train-scenes", type=int, default=60)
    p.add_argument("--expected-val-scenes", type=int, default=20)
    p.add_argument("--expected-states-per-scene", type=int, default=15)
    p.add_argument("--expected-train-episodes-per-scene", type=int, default=5000)
    p.add_argument("--expected-val-episodes-per-scene", type=int, default=10)
    return p.parse_args()


def count_scene_states(scene_dir: Path) -> tuple[int, int]:
    state_dirs = sorted([p for p in scene_dir.iterdir() if p.is_dir()])
    state_count = len(state_dirs)
    episode_count = 0
    for s in state_dirs:
        episode_count += len(list(s.glob("*.json")))
    return state_count, episode_count


def validate_split(root: Path, split: str, expected_scenes: int, expected_states: int, expected_eps_per_scene: int) -> list[str]:
    errors: list[str] = []
    split_dir = root / split
    if not split_dir.is_dir():
        return [f"Missing split directory: {split_dir}"]

    scene_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    if len(scene_dirs) != expected_scenes:
        errors.append(f"{split}: expected {expected_scenes} scenes, got {len(scene_dirs)}")

    for scene_dir in scene_dirs:
        state_count, episode_count = count_scene_states(scene_dir)
        if state_count != expected_states:
            errors.append(
                f"{split}/{scene_dir.name}: expected {expected_states} states, got {state_count}"
            )
        if episode_count != expected_eps_per_scene:
            errors.append(
                f"{split}/{scene_dir.name}: expected {expected_eps_per_scene} episodes, got {episode_count}"
            )

    return errors


def check_subtask_rules(root: Path) -> list[str]:
    errors: list[str] = []
    episode_files = list(root.glob("**/*.json"))
    for ep_file in episode_files[:2000]:
        # sample-check first 2000 files for speed
        with ep_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        subtasks = data.get("subtasks", [])
        if not (5 <= len(subtasks) <= 10):
            errors.append(f"{ep_file}: subtask count out of range")
            continue

        counts = Counter(st["task_type"] for st in subtasks)
        spread = max(counts.values()) - min(counts.values())
        if spread > 1:
            errors.append(f"{ep_file}: unbalanced subtask mix {dict(counts)}")

    return errors


def main() -> None:
    args = parse_args()
    root = Path(args.episodes_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Episodes root not found: {root}")

    errors = []
    errors += validate_split(
        root,
        "train",
        args.expected_train_scenes,
        args.expected_states_per_scene,
        args.expected_train_episodes_per_scene,
    )
    errors += validate_split(
        root,
        "val",
        args.expected_val_scenes,
        args.expected_states_per_scene,
        args.expected_val_episodes_per_scene,
    )
    errors += check_subtask_rules(root)

    if errors:
        print("[FAIL] Episode validation found issues:")
        for item in errors[:50]:
            print(f"  - {item}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
        raise SystemExit(1)

    print("[PASS] Episode dataset validation succeeded")


if __name__ == "__main__":
    main()
