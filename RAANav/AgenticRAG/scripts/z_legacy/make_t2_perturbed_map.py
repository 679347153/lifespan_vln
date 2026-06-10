#!/usr/bin/env python3
import argparse
import copy
import json
import random
from pathlib import Path


def iter_objects(map_data):
    for f_idx, floor in enumerate(map_data):
        for r_idx, room in enumerate(floor.get("rooms", [])):
            for o_idx, obj in enumerate(room.get("objects", [])):
                yield f_idx, r_idx, o_idx, room, obj


def shift_region(region, dx, dy):
    if not isinstance(region, list):
        return region
    out = []
    for p in region:
        if isinstance(p, list) and len(p) >= 2:
            out.append([float(p[0]) + dx, float(p[1]) + dy])
        else:
            out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser(description="Create a perturbed T2 map_now from T1 map.")
    parser.add_argument("--input", required=True, help="Input map json path")
    parser.add_argument("--output", required=True, help="Output perturbed map json path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop_ratio", type=float, default=0.25, help="Ratio of objects to remove")
    parser.add_argument("--move_ratio", type=float, default=0.35, help="Ratio of remaining objects to move")
    parser.add_argument("--max_shift", type=float, default=1.2, help="Max xy shift in meters")
    parser.add_argument("--time_now", type=float, default=120.0, help="Timestamp assigned to surviving objects")
    args = parser.parse_args()

    random.seed(args.seed)
    in_path = Path(args.input)
    out_path = Path(args.output)
    data = json.loads(in_path.read_text(encoding="utf-8"))
    data2 = copy.deepcopy(data)

    all_refs = list(iter_objects(data2))
    total = len(all_refs)
    if total == 0:
        out_path.write_text(json.dumps(data2, ensure_ascii=False, indent=2), encoding="utf-8")
        print("No objects found; wrote unchanged map.")
        return

    n_drop = max(1, int(total * args.drop_ratio))
    drop_set = set(random.sample(range(total), min(n_drop, total)))

    kept_indices = [idx for idx in range(total) if idx not in drop_set]
    n_move = max(1, int(len(kept_indices) * args.move_ratio)) if kept_indices else 0
    move_set = set(random.sample(kept_indices, min(n_move, len(kept_indices)))) if kept_indices else set()

    removed = 0
    moved = 0

    # remove in reverse order per room to keep indices stable
    room_to_obj_indices = {}
    for idx, (f_idx, r_idx, o_idx, _room, _obj) in enumerate(all_refs):
        if idx in drop_set:
            room_to_obj_indices.setdefault((f_idx, r_idx), []).append(o_idx)

    for (f_idx, r_idx), obj_indices in room_to_obj_indices.items():
        room = data2[f_idx]["rooms"][r_idx]
        for o_idx in sorted(obj_indices, reverse=True):
            if 0 <= o_idx < len(room.get("objects", [])):
                room["objects"].pop(o_idx)
                removed += 1

    # rebuild refs after deletion then move subset by global order mapping through original idx
    refs_after = list(iter_objects(data2))
    # map surviving original global idx -> new ref order
    surviving_original = [idx for idx in range(total) if idx not in drop_set]
    for new_pos, orig_idx in enumerate(surviving_original):
        if orig_idx not in move_set:
            continue
        if new_pos >= len(refs_after):
            continue
        f_idx, r_idx, o_idx, room, obj = refs_after[new_pos]
        dx = random.uniform(-args.max_shift, args.max_shift)
        dy = random.uniform(-args.max_shift, args.max_shift)

        region = obj.get("region")
        if region is not None:
            obj["region"] = shift_region(region, dx, dy)

        pcd = obj.get("pcd")
        if isinstance(pcd, list) and len(pcd) >= 2:
            obj["pcd"] = [float(pcd[0]) + dx, float(pcd[1]) + dy, float(pcd[2]) if len(pcd) > 2 else 0.0]

        obj["last_update_time"] = args.time_now
        moved += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data2, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"T2 generated: total={total}, removed={removed}, moved={moved}, kept={total-removed}")


if __name__ == "__main__":
    main()
