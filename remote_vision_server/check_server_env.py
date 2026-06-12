from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def repo_root() -> Path:
    return Path(os.environ.get("REMOTE_VISION_REPO_ROOT", Path.cwd())).resolve()


def exists(path: Optional[str]) -> bool:
    return bool(path) and Path(path).expanduser().exists()


def setup_may_skip_extensions(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "def get_extensions" in text and "return None" in text


def find_groundingdino_extension(root: Path) -> list[str]:
    package_root = root / "groundingdino"
    if not package_root.exists():
        return []
    return [str(path) for path in package_root.rglob("_C*.so")]


def resolve_bert() -> Optional[str]:
    explicit = (
        os.environ.get("BERT_BASE_UNCASED_PATH")
        or os.environ.get("REMOTE_VISION_BERT_PATH")
        or os.environ.get("GROUNDINGDINO_TEXT_ENCODER_PATH")
    )
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    root = repo_root()
    candidates.extend([root / "bert-base-uncased", root.parent / "bert-base-uncased"])
    for env_name in ["HF_HOME", "TRANSFORMERS_CACHE"]:
        base = os.environ.get(env_name)
        if base:
            candidates.append(Path(base) / "bert-base-uncased")
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_dir() and (candidate / "config.json").exists() and (candidate / "vocab.txt").exists():
            return str(candidate)
    return None


def try_import(module: str) -> Dict[str, Any]:
    try:
        if module == "groundingdino._C":
            import torch  # noqa: F401
        imported = importlib.import_module(module)
        return {"ok": True, "module": module, "file": getattr(imported, "__file__", None)}
    except Exception as exc:
        return {"ok": False, "module": module, "error": repr(exc)}


def resolve_torch_lib_dir() -> Optional[str]:
    try:
        import torch

        path = Path(torch.__file__).resolve().parent / "lib"
        return str(path) if path.exists() else None
    except Exception:
        return None


def main() -> int:
    root = repo_root()
    gdino_root = root / "third_party/GroundingDINO"
    gdino_setup = gdino_root / "setup.py"
    gdino_config = os.environ.get(
        "GDINO_CONFIG",
        str(gdino_root / "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    )
    gdino_checkpoint = os.environ.get(
        "GDINO_CHECKPOINT",
        str(root / "RAANav/AgenticRAG/checkpoints/groundingdino_swint_ogc.pth"),
    )
    mobilesam_checkpoint = os.environ.get(
        "MOBILESAM_CHECKPOINT",
        str(root / "RAANav/AgenticRAG/checkpoints/mobile_sam.pt"),
    )

    report: Dict[str, Any] = {
        "python": sys.executable,
        "repo_root": str(root),
        "env": {
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "FORCE_CUDA": os.environ.get("FORCE_CUDA"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "TORCH_LIB_DIR": os.environ.get("TORCH_LIB_DIR"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
        "paths": {
            "bert_base_uncased": resolve_bert(),
            "torch_lib_dir": resolve_torch_lib_dir(),
            "gdino_setup": str(gdino_setup),
            "gdino_setup_exists": gdino_setup.exists(),
            "gdino_setup_may_skip_extensions": setup_may_skip_extensions(gdino_setup),
            "groundingdino_extension_files": find_groundingdino_extension(gdino_root),
            "gdino_config": gdino_config,
            "gdino_checkpoint": gdino_checkpoint,
            "mobilesam_checkpoint": mobilesam_checkpoint,
            "gdino_config_exists": exists(gdino_config),
            "gdino_checkpoint_exists": exists(gdino_checkpoint),
            "mobilesam_checkpoint_exists": exists(mobilesam_checkpoint),
        },
        "imports": {
            "groundingdino": try_import("groundingdino"),
            "groundingdino._C": try_import("groundingdino._C"),
            "mobile_sam": try_import("mobile_sam"),
        },
    }
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        report["torch"] = {"error": repr(exc)}

    print(json.dumps(report, ensure_ascii=False, indent=2))

    failed = []
    if not report["paths"]["bert_base_uncased"]:
        failed.append("bert-base-uncased not found")
    for key in ["gdino_config_exists", "gdino_checkpoint_exists", "mobilesam_checkpoint_exists"]:
        if not report["paths"][key]:
            failed.append(key)
    if report["paths"]["gdino_setup_may_skip_extensions"]:
        failed.append("GroundingDINO setup.py appears to skip get_extensions")
    if not report["paths"]["groundingdino_extension_files"]:
        failed.append("no groundingdino/_C*.so extension file found")
    if not report["paths"]["torch_lib_dir"]:
        failed.append("torch lib dir not found")
    if not report["imports"]["groundingdino._C"]["ok"]:
        failed.append("groundingdino._C import failed")
    if failed:
        print("\nFAILED: " + ", ".join(failed), file=sys.stderr)
        return 1
    print("\nOK: remote vision server environment looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
