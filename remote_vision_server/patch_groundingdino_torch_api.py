from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict


REPLACEMENTS: Dict[str, str] = {
    "AT_DISPATCH_FLOATING_TYPES(value.type(),": "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),",
    "AT_DISPATCH_FLOATING_TYPES(grad_output.type(),": "AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(),",
    "value.data<scalar_t>()": "value.data_ptr<scalar_t>()",
    "sampling_loc.data<scalar_t>()": "sampling_loc.data_ptr<scalar_t>()",
    "attn_weight.data<scalar_t>()": "attn_weight.data_ptr<scalar_t>()",
    "columns.data<scalar_t>()": "columns.data_ptr<scalar_t>()",
    "grad_output_g.data<scalar_t>()": "grad_output_g.data_ptr<scalar_t>()",
    "grad_value.data<scalar_t>()": "grad_value.data_ptr<scalar_t>()",
    "grad_sampling_loc.data<scalar_t>()": "grad_sampling_loc.data_ptr<scalar_t>()",
    "grad_attn_weight.data<scalar_t>()": "grad_attn_weight.data_ptr<scalar_t>()",
    "spatial_shapes.data<int64_t>()": "spatial_shapes.data_ptr<int64_t>()",
    "level_start_index.data<int64_t>()": "level_start_index.data_ptr<int64_t>()",
}

HEADER_REPLACEMENTS: Dict[str, str] = {
    "value.type().is_cuda()": "value.is_cuda()",
}


def patch_file(path: Path, replacements: Dict[str, str]) -> int:
    text = path.read_text(encoding="utf-8")
    original = text
    count = 0
    for old, new in replacements.items():
        occurrences = text.count(old)
        if occurrences:
            text = text.replace(old, new)
            count += occurrences
    if text != original:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(text, encoding="utf-8")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch GroundingDINO ms_deform_attn sources for newer PyTorch C++ APIs."
    )
    parser.add_argument(
        "--groundingdino-root",
        default="third_party/GroundingDINO",
        help="Path to the GroundingDINO checkout.",
    )
    args = parser.parse_args()

    root = Path(args.groundingdino_root).expanduser().resolve()
    csrc = root / "groundingdino/models/GroundingDINO/csrc/MsDeformAttn"
    cuda_file = csrc / "ms_deform_attn_cuda.cu"
    header_file = csrc / "ms_deform_attn.h"

    missing = [str(path) for path in [cuda_file, header_file] if not path.exists()]
    if missing:
        raise SystemExit("Missing GroundingDINO source files:\n" + "\n".join(missing))

    cuda_count = patch_file(cuda_file, REPLACEMENTS)
    header_count = patch_file(header_file, HEADER_REPLACEMENTS)

    print(f"Patched {cuda_file}: {cuda_count} replacements")
    print(f"Patched {header_file}: {header_count} replacements")
    print("Backups were written next to modified files as *.bak if they did not already exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
