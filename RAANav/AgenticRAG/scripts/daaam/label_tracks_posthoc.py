#!/usr/bin/env python3
"""
Post-hoc CLIP labeling of DAAAM frontend tracks.

Runs CLIP zero-shot classification on the best observation of each track
(after the fast frontend pipeline completes), following DAAAM's architecture
of separating fast segmentation/tracking from slow semantic labeling.

Usage:
  LD_PRELOAD=/tmp/libvitjot_stub.so conda run -n daaam_p0 python \
      AgenticRAG/scripts/daaam/label_tracks_posthoc.py \
      --data-path /tmp/daaam_apartment \
      --frontend-output /tmp/daaam_frontend_output \
      --output /tmp/daaam_labeled_summary.json
"""

import sys, argparse, json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import cv2

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from clip_labeler import CLIPZeroShotLabeler


def load_rgb(data_path: Path, frame_id: int) -> np.ndarray:
    """Load RGB frame from ImageSequenceDataset."""
    rgb_path = data_path / "rgb" / f"{frame_id:06d}.png"
    img = cv2.imread(str(rgb_path))
    if img is None:
        raise FileNotFoundError(str(rgb_path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def label_tracks(
    summary: dict,
    data_path: Path,
    labeler: CLIPZeroShotLabeler,
) -> dict:
    """For each track: pick best frame (largest mask area), run CLIP, assign label."""
    tracks = summary.get("tracks", {})
    all_frames_path = Path(summary.get("_all_frames_path", ""))

    # Build per-frame index: frame_id → {track_id → {mask_idx, bbox, mask_area}}
    frame_index: Dict[int, Dict[int, dict]] = {}
    if all_frames_path.exists():
        with open(all_frames_path) as f:
            all_frames = json.load(f)
        for fe in all_frames:
            fid = fe["frame_id"]
            frame_index[fid] = {}
            for tr in fe.get("tracks", []):
                frame_index[fid][tr["track_id"]] = tr

    labeled = {}
    n_total = len(tracks)

    for i, (tid_str, track) in enumerate(tracks.items()):
        tid = int(tid_str)
        best_label = {"label": "obj", "confidence": 0.0, "clip_embedding": None}

        # Find best frame for this track (largest mask area)
        best_frame = None
        best_area = 0
        for fid, trk_dict in frame_index.items():
            if tid in trk_dict:
                area = trk_dict[tid].get("mask_area_pixels", 0)
                if area > best_area:
                    best_area = area
                    best_frame = fid

        if best_frame is not None and best_area > 100:
            try:
                rgb = load_rgb(data_path, best_frame)
                # Create a mask for this track from its bbox (approximate)
                trk_info = frame_index[best_frame][tid]
                bbox = trk_info.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = bbox
                    # Create approximate mask from bbox (better than nothing)
                    mask = np.zeros(rgb.shape[:2], dtype=bool)
                    # Tighten to valid range
                    H, W = rgb.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    if x2 > x1 and y2 > y1:
                        mask[y1:y2, x1:x2] = True
                        labels = labeler.label_masks(rgb, [mask])
                        if labels:
                            best_label = labels[0]
            except Exception as e:
                pass  # skip frames that fail to load

        labeled[tid_str] = best_label
        if (i + 1) % 50 == 0:
            n_labeled = sum(1 for v in labeled.values() if v['confidence'] > 0.01)
            print(f"  Labeled {i+1}/{n_total} tracks ({n_labeled} with labels)...")

    return labeled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="ImageSequenceDataset root")
    parser.add_argument("--frontend-output", required=True, help="Frontend pipeline output dir")
    parser.add_argument("--output", required=True, help="Output summary.json path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tracks", type=int, default=None)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    frontend_dir = Path(args.frontend_output)

    # Load original summary
    summary_path = frontend_dir / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    # Add pointer to all_frames
    summary["_all_frames_path"] = str(frontend_dir / "all_frames.json")

    print(f"Loading CLIP labeler...")
    labeler = CLIPZeroShotLabeler(device=args.device)

    tracks = summary.get("tracks", {})
    if args.max_tracks:
        tracks = dict(list(tracks.items())[:args.max_tracks])
        summary["tracks"] = tracks

    print(f"Labeling {len(tracks)} tracks from {data_path}...")
    labels = label_tracks(summary, data_path, labeler)

    # Update summary with labels
    n_labeled = 0
    for tid_str, lbl in labels.items():
        if tid_str in summary["tracks"]:
            summary["tracks"][tid_str]["label"] = lbl["label"]
            summary["tracks"][tid_str]["label_confidence"] = lbl["confidence"]
            summary["tracks"][tid_str]["clip_embedding"] = lbl.get("clip_embedding")
            if lbl["confidence"] > 0.01:
                n_labeled += 1

    summary["clip_model"] = "ViT-B-16"
    summary["labeled_tracks"] = n_labeled
    summary["total_tracks"] = len(tracks)

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. {n_labeled}/{len(tracks)} tracks labeled.")
    print(f"Output: {args.output}")

    # Print label distribution
    from collections import Counter
    label_counts = Counter(
        summary["tracks"][tid]["label"]
        for tid in summary["tracks"]
        if summary["tracks"][tid].get("label_confidence", 0) > 0.01
    )
    print(f"\nLabel distribution (top 15):")
    for label, count in label_counts.most_common(15):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
