#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def collect(map_data):
    objs = []
    for floor in map_data:
        for room in floor.get("rooms", []):
            for obj in room.get("objects", []):
                name = obj.get("name", "unknown")
                pcd = obj.get("pcd", [None, None, None])
                exist_prob = float(obj.get("exist_prob", 1.0))
                objs.append({"name": name, "pcd": pcd, "exist_prob": exist_prob})
    return objs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1", required=True)
    parser.add_argument("--t2", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    t1 = json.loads(Path(args.t1).read_text(encoding="utf-8"))
    t2 = json.loads(Path(args.t2).read_text(encoding="utf-8"))
    merged = json.loads(Path(args.merged).read_text(encoding="utf-8"))

    t1_objs = collect(t1)
    t2_objs = collect(t2)
    m_objs = collect(merged)

    out = {
        "t1_object_count": len(t1_objs),
        "t2_object_count": len(t2_objs),
        "merged_object_count": len(m_objs),
        "merged_exist_prob_stats": {
            "min": min((o["exist_prob"] for o in m_objs), default=0.0),
            "max": max((o["exist_prob"] for o in m_objs), default=0.0),
            "avg": (sum(o["exist_prob"] for o in m_objs) / len(m_objs)) if m_objs else 0.0,
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
