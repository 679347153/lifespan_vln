#!/usr/bin/env python3
"""
Visualize a single frame with FastSAM masks + CLIP labels overlaid.

Usage:
  LD_PRELOAD=/tmp/libvitjot_stub.so conda run -n daaam_p0 python \
      AgenticRAG/scripts/daaam/visualize_labeled_frame.py \
      --rgb /tmp/daaam_apartment/rgb/000000.png \
      --output /tmp/labeled_frame.png
"""

import sys, argparse
from pathlib import Path
import numpy as np
import cv2

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

DAAAM_ROOT = Path("/home/adminer/agentRAG/参考开源代码/DAAAM")
sys.path.insert(0, str(DAAAM_ROOT / "src"))

from clip_labeler import CLIPZeroShotLabeler
from daaam.utils.segmentation import UniversalSegmenter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--output", default="/tmp/labeled_frame.png")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-mask-area", type=int, default=300)
    args = parser.parse_args()

    # Load image
    img = cv2.imread(args.rgb)
    if img is None:
        raise FileNotFoundError(args.rgb)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    # FastSAM
    print("Running FastSAM...")
    seg = UniversalSegmenter(
        model_checkpoint_path=str(DAAAM_ROOT / "checkpoints" / "fastsam" / "FastSAM-s.pt"),
        model_config_path=str(DAAAM_ROOT / "config" / "fastsam" / "fastsam_config.yaml"),
        device=args.device,
        min_mask_region_area=args.min_mask_area,
    )
    dets, masks = seg(rgb)
    print(f"  {len(masks)} masks")

    # CLIP labeling
    print("Running CLIP labeling...")
    labeler = CLIPZeroShotLabeler(device=args.device)
    labels = labeler.label_masks(rgb, masks)
    print(f"  {len(labels)} labels assigned")

    # Draw visualization
    vis = rgb.copy()
    rng = np.random.RandomState(42)

    for i, (det, mask, lbl) in enumerate(zip(dets, masks, labels)):
        x1, y1, x2, y2 = map(int, det[:4])
        conf = lbl['confidence']
        label = lbl['label'] if conf > 0.01 else '?'
        color = rng.randint(0, 255, 3).tolist()

        # Draw filled mask with alpha
        m = mask.astype(bool)
        if m.any():
            alpha = 0.3
            vis[m] = (vis[m] * (1 - alpha) + np.array(color, dtype=np.uint8) * alpha).astype(np.uint8)

        # Draw bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        # Label
        cv2.putText(vis, f"{label} ({conf:.2f})",
                    (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, color, 1)

    # Legend
    from collections import Counter
    label_counts = Counter(
        lbl['label'] for lbl in labels if lbl['confidence'] > 0.01
    )
    y_offset = H - 10
    for label, count in label_counts.most_common(10):
        y_offset -= 14
        cv2.putText(vis, f"{label}: {count}",
                    (5, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255, 255, 255), 1)

    cv2.putText(vis, f"FastSAM + CLIP | {len(masks)} masks | {len(label_counts)} unique labels",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imwrite(args.output, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved: {args.output}")
    print(f"Label distribution: {dict(label_counts.most_common(10))}")


if __name__ == "__main__":
    main()
