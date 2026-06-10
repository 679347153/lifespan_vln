#!/usr/bin/env python3
"""
Minimal test runner for GMM feature pipeline.

Usage:
  python scripts/min_gmm_test.py --target umbrella_stand

Prints key information:
- CLIP-selected target instances
- Self scores per target instance
- Top related items by GMM_score (limited)
- Top agent Robj_scores (labels)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GMM_map_Create.GMM_map_calcualte import build_GMM_feature_set


def _top_k(items: List[Dict[str, Any]], key: str, k: int = 10) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda x: x.get(key, 0), reverse=True)[:k]


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal GMM pipeline tester")
    parser.add_argument("--target", type=str, default="umbrella_stand", help="Target label to search (e.g., umbrella_stand)")
    parser.add_argument("--config", type=str, default="config/map.yaml", help="Path to config yaml")
    parser.add_argument("--top", type=int, default=10, help="How many top related items to print")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[WARN] Config not found: {config_path} — continuing with defaults inside code.")

    try:
        meta = build_GMM_feature_set(args.target, config_path=config_path)
    except Exception as e:
        print(f"[ERROR] Failed to build GMM feature set: {e}")
        return 2

    # 1) CLIP-selected target instances
    targets_from_clip = meta.get("targets_from_clip", [])
    print("\n=== Targets from CLIP (threshold selected) ===")
    print(f"count = {len(targets_from_clip)}")
    for t in targets_from_clip:
        print(f" - {t.get('obj_id')} | clip_score={t.get('score')}")

    # 2) Self scores per target instance
    targets_self = meta.get("targets_self", [])
    print("\n=== Target self scores (sorted) ===")
    for s in _top_k(targets_self, key="GMM_self_score", k=max(3, min(len(targets_self), args.top))):
        print(f" - {s.get('obj_id')}: self_score={s.get('GMM_self_score')}")

    # 3) Related items by GMM_score
    related = meta.get("related", [])
    print("\n=== Related items by GMM_score (top) ===")
    top_related = _top_k(related, key="GMM_score", k=max(5, args.top))
    for r in top_related:
        print(
            " - target={tid} | obj={oid} ({lb}) | GMM={gmm} | Rscore={rs} | Nr/N={nrn} | Rcfd={rcfd} | stability={st}".format(
                tid=r.get("target_id"),
                oid=r.get("obj_id"),
                lb=r.get("label"),
                gmm=r.get("GMM_score"),
                rs=r.get("Rscore"),
                nrn=r.get("Nr_over_N"),
                rcfd=r.get("Rcfd"),
                st=r.get("stability"),
            )
        )
    print(f"total related items: {len(related)}")

    # 4) Agent Robj_scores summary (labels)
    label_scores: Dict[str, float] = meta.get("Robj_scores", {})
    print("\n=== Agent label Robj_scores (top) ===")
    if not label_scores:
        print("[WARN] No Robj_scores loaded. Check agent output_path in config or run RScore agent first.")
    else:
        top_labels = sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)[:max(5, args.top)]
        for lb, sc in top_labels:
            print(f" - {lb}: {sc}")

    # 5) Aggregated by obj_id (ensure ID-level consolidation)
    print("\n=== Related aggregated by obj_id (top by GMM_score) ===")
    by_obj: Dict[str, Dict[str, Any]] = meta.get("related_by_obj", {})
    if by_obj:
        flat = list(by_obj.values())
        for r in _top_k(flat, key="GMM_score", k=max(5, args.top)):
            print(
                " - obj={oid} ({lb}) | GMM={gmm} | best_from_target={tid} | Nr/N={nrn} | Rcfd={rcfd} | Rscore={rs}".format(
                    oid=r.get("obj_id"),
                    lb=r.get("label"),
                    gmm=r.get("GMM_score"),
                    tid=r.get("best_from_target"),
                    nrn=r.get("Nr_over_N"),
                    rcfd=r.get("Rcfd"),
                    rs=r.get("Rscore"),
                )
            )
        # Highlight one expected object if present
        key_oid = "shoe_cabinet_1_R3"
        if key_oid in by_obj:
            r = by_obj[key_oid]
            print("\n[CHECK] Found expected object:")
            print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print("[WARN] related_by_obj is empty (check build_GMM_feature_set aggregation path)")

    # 6) Compact mapping (obj_id -> GMM_score)
    print("\n=== Compact mapping: obj_id -> GMM_score ===")
    compact: Dict[str, float] = meta.get("GMM_scores_by_obj", {})
    top_compact = sorted(compact.items(), key=lambda kv: kv[1], reverse=True)[:max(10, args.top)]
    for oid, sc in top_compact:
        print(f" - {oid}: {sc}")

    # 5) Compact JSON dump path (optional)
    result_path = Path("scripts/_last_min_gmm_result.json")
    try:
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] Full result saved to {result_path}")
    except Exception as e:
        print(f"[WARN] Failed to save result JSON: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
