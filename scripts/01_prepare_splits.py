#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from hm3d_paths import list_available_scenes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare benchmark split manifest")
    p.add_argument("--train-count", type=int, default=60)
    p.add_argument("--val-count", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="benchmark/splits/benchmark_split_v1.json")
    p.add_argument(
        "--allow-reuse-scenes",
        action="store_true",
        help="Allow reusing scenes if available scenes are fewer than requested",
    )
    return p.parse_args()


def choose_splits(scenes: list[str], train_count: int, val_count: int, seed: int, allow_reuse: bool) -> tuple[list[str], list[str]]:
    if train_count <= 0 or val_count <= 0:
        raise ValueError("train-count and val-count must be positive")

    rng = random.Random(seed)
    pool = sorted(scenes)

    required = train_count + val_count
    if len(pool) < required and not allow_reuse:
        raise ValueError(
            f"Need {required} scenes, but only {len(pool)} available. "
            "Use --allow-reuse-scenes or add more scenes."
        )

    if len(pool) >= required:
        rng.shuffle(pool)
        train = sorted(pool[:train_count])
        val = sorted(pool[train_count:train_count + val_count])
        return train, val

    expanded = []
    while len(expanded) < required:
        shuffled = pool[:]
        rng.shuffle(shuffled)
        expanded.extend(shuffled)

    chosen = expanded[:required]
    train = sorted(chosen[:train_count])
    val = sorted(chosen[train_count:train_count + val_count])
    return train, val


def main() -> None:
    args = parse_args()
    scenes = list_available_scenes(require_semantic=True)
    train, val = choose_splits(scenes, args.train_count, args.val_count, args.seed, args.allow_reuse_scenes)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "version": "v1",
        "seed": args.seed,
        "available_scene_count": len(scenes),
        "train_scenes": train,
        "val_scenes": val,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[Done] Wrote split manifest: {out_path}")
    print(f"  train: {len(train)} scenes")
    print(f"  val:   {len(val)} scenes")


if __name__ == "__main__":
    main()
