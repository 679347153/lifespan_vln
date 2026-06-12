from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List

import requests


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
    args = parser.parse_args()

    payload: Dict[str, Any] = {
        "image_base64": image_to_data_url(Path(args.image)),
        "labels": parse_labels(args.labels),
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "return_mask_png": bool(args.return_mask_png),
    }
    response = requests.post(f"{args.base_url.rstrip('/')}/v1/detect_segment", json=payload, timeout=120)
    if response.status_code >= 400:
        print(response.text)
        response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
