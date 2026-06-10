#!/usr/bin/env python3
"""
CLIP Zero-Shot Classifier for FastSAM masks.

Replaces GroundingDINO: FastSAM gives class-agnostic masks, CLIP assigns
semantic labels by comparing each masked crop against a label bank of
65 indoor categories.

Design follows DAAAM's CLIPHandler pattern (daaam.utils.embedding).

Usage (standalone test):
  LD_PRELOAD=/tmp/libvitjot_stub.so conda run -n daaam_p0 python \
      AgenticRAG/scripts/daaam/clip_labeler.py --image /tmp/test.png
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
from PIL import Image
import torch

# DAAAM import path
DAAAM_ROOT = Path("/home/adminer/agentRAG/参考开源代码/DAAAM")
sys.path.insert(0, str(DAAAM_ROOT / "src"))

# 65 indoor categories (from RAANav's stability_priors.yaml detect_labels)
INDOOR_LABELS = [
    "chair", "table", "desk", "couch", "sofa", "bed", "bookshelf", "bookcase",
    "cabinet", "kitchen cabinet", "cupboard", "drawer", "shelf", "counter",
    "refrigerator", "fridge", "microwave", "oven", "stove", "toaster",
    "coffee machine", "dishwasher", "sink", "faucet", "bathtub", "toilet",
    "shower", "mirror", "towel", "clock", "vase", "plant", "potted plant",
    "painting", "picture", "tv", "television", "monitor", "screen", "laptop",
    "computer", "keyboard", "mouse", "speaker", "lamp", "ceiling lamp",
    "chandelier", "light", "curtain", "blinds", "pillow", "cushion",
    "blanket", "rug", "carpet", "mat", "box", "bin", "basket",
    "statue", "sculpture", "decoration", "toy", "book",
]

# Labels that CLIP often confuses — explicit disambiguation
LABEL_ALIASES = {
    "couch": "sofa",
    "bookcase": "bookshelf",
    "fridge": "refrigerator",
    "television": "tv",
    "picture": "painting",
    "cushion": "pillow",
    "carpet": "rug",
    "mat": "rug",
    "sculpture": "statue",
    "cupboard": "cabinet",
    "chandelier": "ceiling lamp",
}


class CLIPZeroShotLabeler:
    """Assign semantic labels to image masks using CLIP zero-shot classification.

    Follows DAAAM's CLIPHandler pattern (open_clip, batch encoding, L2-norm).
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        device: str = "cpu",
        label_bank: Optional[List[str]] = None,
    ):
        # The model is cached on the robot workstation. Avoid per-run network
        # metadata checks because they can stall live simulation iterations.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import open_clip

        self.device = device
        self.model_name = model_name
        self.label_bank = label_bank or INDOOR_LABELS

        # Deduplicate aliases → canonical mapping
        self.aliases: Dict[str, str] = {}
        canonical_set = set()
        filtered_labels = []
        for lbl in self.label_bank:
            canon = LABEL_ALIASES.get(lbl, lbl)
            if canon not in canonical_set:
                canonical_set.add(canon)
                filtered_labels.append(canon)
            if canon != lbl:
                self.aliases[lbl] = canon

        self.canonical_labels = filtered_labels
        print(f"[CLIPLabeler] {len(self.canonical_labels)} canonical labels "
              f"(from {len(self.label_bank)} raw, {len(self.aliases)} aliases)")

        # Load model
        print(f"[CLIPLabeler] Loading {model_name} ({pretrained}) on {device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        # Pre-compute text embeddings for all labels
        self._build_label_index()

    def _build_label_index(self):
        """Pre-compute and cache text embeddings for the label bank."""
        prompts = [f"an indoor photo of a {lbl}" for lbl in self.canonical_labels]
        tokens = self.tokenizer(prompts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features  # [N_labels, dim]
        self.text_features_np = text_features.cpu().numpy()
        print(f"[CLIPLabeler] Text embeddings built: {self.text_features.shape}")

    @torch.no_grad()
    def label_masks(
        self,
        rgb: np.ndarray,         # HxWx3 uint8
        masks: List[np.ndarray], # List of HxW bool
        min_crop_size: int = 20,
    ) -> List[dict]:
        """Assign labels to masks via CLIP zero-shot.

        Returns list of {label, confidence, clip_embedding} in same order as masks.
        """
        import torch

        if not masks:
            return []

        crops = []
        valid_indices = []
        for i, mask in enumerate(masks):
            crop = self._crop_from_mask(rgb, mask)
            if crop is not None and crop.size[0] >= min_crop_size and crop.size[1] >= min_crop_size:
                crops.append(crop)
                valid_indices.append(i)

        if not crops:
            return [{"label": "unknown", "confidence": 0.0, "clip_embedding": None}
                    for _ in masks]

        # Batch encode crops
        tensors = torch.stack([self.preprocess(c) for c in crops]).to(self.device)
        image_features = self.model.encode_image(tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity against label bank, scaled the same way CLIP was trained.
        # A fixed temperature such as 0.5 makes the 53-way distribution almost
        # uniform, which hides useful top-k differences during object search.
        similarities = image_features @ self.text_features.T  # [N_crops, N_labels]
        if hasattr(self.model, "logit_scale"):
            probs = (similarities * self.model.logit_scale.exp()).softmax(dim=-1)
        else:
            probs = (similarities * 100.0).softmax(dim=-1)
        best_indices = probs.argmax(dim=-1)                   # [N_crops]
        best_scores = probs.max(dim=-1).values                # [N_crops]

        best_indices_np = best_indices.cpu().numpy()
        best_scores_np = best_scores.cpu().numpy()
        image_features_np = image_features.cpu().numpy()

        # Map back to original mask order
        results = []
        crop_idx = 0
        for i in range(len(masks)):
            if i in valid_indices and crop_idx < len(crops):
                label = self.canonical_labels[int(best_indices_np[crop_idx])]
                score = float(best_scores_np[crop_idx])
                embedding = image_features_np[crop_idx].tolist()
                crop_idx += 1
            else:
                label = "unknown"
                score = 0.0
                embedding = None
            results.append({"label": label, "confidence": score, "clip_embedding": embedding})

        return results

    def _crop_from_mask(self, rgb: np.ndarray, mask: np.ndarray) -> Optional[Image.Image]:
        """Extract the masked region as a square PIL crop (following DAAAM's crop pattern)."""
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None

        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        # Add 10% padding
        h, w = y2 - y1, x2 - x1
        pad_y = max(1, int(h * 0.1))
        pad_x = max(1, int(w * 0.1))

        H, W = rgb.shape[:2]
        y1 = max(0, y1 - pad_y)
        y2 = min(H, y2 + pad_y)
        x1 = max(0, x1 - pad_x)
        x2 = min(W, x2 + pad_x)

        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Apply mask to crop (zero out background)
        crop_mask = mask[y1:y2, x1:x2]
        crop_masked = crop.copy()
        crop_masked[~crop_mask] = 0

        return Image.fromarray(crop_masked)


# ── Standalone test ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Quick test: run FastSAM + CLIP labeling on a single image
    from daaam.utils.segmentation import UniversalSegmenter

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # FastSAM
    seg = UniversalSegmenter(
        model_checkpoint_path=str(
            DAAAM_ROOT / "checkpoints" / "fastsam" / "FastSAM-s.pt"
        ),
        model_config_path=str(DAAAM_ROOT / "config" / "fastsam" / "fastsam_config.yaml"),
        device=args.device,
        min_mask_region_area=300,
    )
    dets, masks = seg(rgb)
    print(f"FastSAM: {len(masks)} masks")

    # CLIP labeling
    labeler = CLIPZeroShotLabeler(device=args.device)
    labels = labeler.label_masks(rgb, masks)

    # Show results
    for i, (det, lbl) in enumerate(zip(dets, labels)):
        x1, y1, x2, y2 = map(int, det[:4])
        print(f"  Mask {i}: [{x1},{y1},{x2},{y2}] → {lbl['label']} ({lbl['confidence']:.3f})")

    # Draw visualization
    vis = rgb.copy()
    rng = np.random.RandomState(42)
    for i, (det, lbl) in enumerate(zip(dets, labels)):
        x1, y1, x2, y2 = map(int, det[:4])
        color = rng.randint(0, 255, 3).tolist()
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{lbl['label']} {lbl['confidence']:.2f}",
                    (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, color, 1)

    out_path = "/tmp/clip_labeler_test.png"
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved: {out_path}")
