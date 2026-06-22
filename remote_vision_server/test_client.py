from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - handled at runtime.
    requests = None  # type: ignore[assignment]


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def parse_labels(raw: str) -> List[str]:
    return [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the remote vision service through an SSH tunnel.")
    parser.add_argument("--base-url", default="http://127.0.0.1:50220")
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", default="chair,table,sofa,bed,cabinet,lamp")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.35)
    parser.add_argument("--return-mask-png", action="store_true")
    parser.add_argument("--return-clip-embedding", action="store_true")
    parser.add_argument("--clip-mode", choices=["mask_only", "full"], default="mask_only")
    parser.add_argument("--clip-model", default=None, help="Server-side CLIP model path or model id.")
    parser.add_argument("--clip-online", action="store_true", help="Allow server-side online CLIP loading.")
    parser.add_argument("--clip-query-text", default=None, help="Optional text query to rank returned CLIP image embeddings.")
    parser.add_argument("--clip-min-score", type=float, default=0.0)
    args = parser.parse_args()
    if requests is None:
        raise RuntimeError("Missing dependency: requests. Install it with: pip install requests")

    payload: Dict[str, Any] = {
        "image_base64": image_to_data_url(Path(args.image)),
        "labels": parse_labels(args.labels),
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "return_mask_png": bool(args.return_mask_png),
        "return_clip_embedding": bool(args.return_clip_embedding),
        "clip_mode": args.clip_mode,
    }
    if args.clip_model:
        payload["clip_model"] = args.clip_model
    if args.clip_online:
        payload["clip_local_files_only"] = False
    response = requests.post(f"{args.base_url.rstrip('/')}/v1/detect_segment", json=payload, timeout=120)
    if response.status_code >= 400:
        print(response.text)
        response.raise_for_status()
    data = response.json()
    if args.clip_query_text:
        detections = data.get("detections", [])
        embeddings = [det.get("clip_embedding") for det in detections if det.get("clip_embedding")]
        ids = [str(i) for i, det in enumerate(detections) if det.get("clip_embedding")]
        labels = [str(det.get("label", "")) for det in detections if det.get("clip_embedding")]
        if embeddings:
            sim_payload: Dict[str, Any] = {
                "text": args.clip_query_text,
                "image_embeddings": embeddings,
                "ids": ids,
                "labels": labels,
                "top_k": len(embeddings),
                "min_score": args.clip_min_score,
            }
            if args.clip_model:
                sim_payload["clip_model"] = args.clip_model
            if args.clip_online:
                sim_payload["clip_local_files_only"] = False
            sim_resp = requests.post(
                f"{args.base_url.rstrip('/')}/v1/clip/text_image_similarity",
                json=sim_payload,
                timeout=120,
            )
            if sim_resp.status_code >= 400:
                print(sim_resp.text)
                sim_resp.raise_for_status()
            data["clip_text_image_similarity"] = sim_resp.json()
        else:
            data["clip_text_image_similarity"] = {"results": [], "reason": "no_clip_embeddings"}
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
