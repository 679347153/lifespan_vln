from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def polygon_from_xz_bounds(x_min: float, x_max: float, z_min: float, z_max: float) -> List[Dict[str, float]]:
    return [
        {'x': float(x_min), 'y': float(z_min)},
        {'x': float(x_max), 'y': float(z_min)},
        {'x': float(x_max), 'y': float(z_max)},
        {'x': float(x_min), 'y': float(z_max)},
    ]


def read_ply_points(ply_path: Path) -> np.ndarray:
    # 优先 open3d，失败则尝试简单 ascii 解析
    try:
        import open3d as o3d  # type: ignore

        pcd = o3d.io.read_point_cloud(str(ply_path))
        arr = np.asarray(pcd.points, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3]
    except Exception:
        pass

    points: List[List[float]] = []
    with ply_path.open('r', encoding='utf-8', errors='ignore') as f:
        header = True
        for line in f:
            s = line.strip()
            if header:
                if s.lower() == 'end_header':
                    header = False
                continue
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except Exception:
                continue
            points.append([x, y, z])
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def load_mask_feats(mask_feats_path: Path) -> Optional[np.ndarray]:
    if torch is None or not mask_feats_path.exists():
        return None
    try:
        obj = torch.load(str(mask_feats_path), map_location='cpu')
    except Exception:
        return None

    if hasattr(obj, 'detach') and hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
        arr = obj.detach().cpu().numpy()
        return np.asarray(arr)

    if isinstance(obj, np.ndarray):
        return obj

    if isinstance(obj, (list, tuple)):
        try:
            arr = np.asarray(obj)
            if arr.ndim >= 2:
                return arr
        except Exception:
            return None

    if isinstance(obj, dict):
        for k in ('mask_feats', 'features', 'feats'):
            if k in obj:
                v = obj[k]
                try:
                    if hasattr(v, 'detach') and hasattr(v, 'cpu') and hasattr(v, 'numpy'):
                        return np.asarray(v.detach().cpu().numpy())
                    return np.asarray(v)
                except Exception:
                    continue
    return None


def parse_obj_index(p: Path) -> int:
    m = re.search(r'pcd_(\d+)\.ply$', p.name)
    return int(m.group(1)) if m else 10**9


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert HOV-SG minimal outputs to AgenticRAG map json')
    parser.add_argument('--hovsg_scene_dir', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--floor_id', type=str, default='F1')
    parser.add_argument('--room_id', type=str, default='R1')
    parser.add_argument('--room_name', type=str, default='unknown_room')
    parser.add_argument('--default_stability', type=float, default=0.5)
    parser.add_argument('--default_cfd', type=float, default=0.5)
    parser.add_argument('--object_crops_json', type=str, default='')
    parser.add_argument('--room_keyframes_json', type=str, default='')
    args = parser.parse_args()

    scene_dir = Path(args.hovsg_scene_dir)
    output_json = Path(args.output_json)

    objects_dir = scene_dir / 'objects'
    full_pcd_path = scene_dir / 'full_pcd.ply'
    mask_feats_path = scene_dir / 'mask_feats.pt'

    if not objects_dir.exists():
        raise FileNotFoundError(f'objects dir not found: {objects_dir}')

    object_files = sorted(objects_dir.glob('pcd_*.ply'), key=parse_obj_index)
    if not object_files:
        raise RuntimeError(f'no object ply files found in {objects_dir}')

    mask_feats = load_mask_feats(mask_feats_path)
    ts = now_iso_utc()

    object_crops_map: Dict[str, List[str]] = {}
    if args.object_crops_json:
        p = Path(args.object_crops_json)
        if p.exists():
            try:
                with p.open('r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    object_crops_map = {str(k): list(v) for k, v in loaded.items() if isinstance(v, list)}
            except Exception:
                object_crops_map = {}

    room_keyframes_map: Dict[str, List[str]] = {}
    if args.room_keyframes_json:
        p = Path(args.room_keyframes_json)
        if p.exists():
            try:
                with p.open('r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    room_keyframes_map = {str(k): list(v) for k, v in loaded.items() if isinstance(v, list)}
            except Exception:
                room_keyframes_map = {}

    # 房间范围来自 full_pcd，如不可用则由对象集合估计
    full_pts = read_ply_points(full_pcd_path) if full_pcd_path.exists() else np.zeros((0, 3), dtype=np.float32)
    all_pts_cache: List[np.ndarray] = []

    objects: List[Dict[str, Any]] = []
    seq = 0
    for pf in object_files:
        pts = read_ply_points(pf)
        if pts.shape[0] == 0:
            continue
        all_pts_cache.append(pts)
        x_min, y_min, z_min = pts.min(axis=0).tolist()
        x_max, y_max, z_max = pts.max(axis=0).tolist()
        region = polygon_from_xz_bounds(x_min, x_max, z_min, z_max)

        idx = parse_obj_index(pf)
        feat: List[float] = []
        if mask_feats is not None and mask_feats.ndim >= 2 and 0 <= idx < mask_feats.shape[0]:
            try:
                feat = mask_feats[idx].astype(np.float32).tolist()
            except Exception:
                feat = []

        seq += 1
        label = 'unknown_object'
        obj_id = f'{label}_{seq}_{args.room_id}'
        crop_paths = object_crops_map.get(pf.stem, [])
        imgs_field = {'1': crop_paths} if crop_paths else {'1': [str(pf)]}
        objects.append(
            {
                'obj_id': obj_id,
                'label': label,
                'room_id': args.room_id,
                'region': region,
                'stability': float(args.default_stability),
                'clip_embedding': feat,
                'cfd': float(args.default_cfd),
                'R_objs': {},
                'imgs': imgs_field,
                'N': 1,
                'description': {'1': f'converted from {pf.name}'},
                'last_update_time': ts,
                'cooccur_stats': {},
                'exist_prob': 1.0,
            }
        )

    if full_pts.shape[0] == 0 and all_pts_cache:
        full_pts = np.concatenate(all_pts_cache, axis=0)

    if full_pts.shape[0] == 0:
        room_region = polygon_from_xz_bounds(0.0, 5.0, 0.0, 5.0)
        z_min, z_max = 0.0, 3.0
    else:
        x_min, y_min, z_min = full_pts.min(axis=0).tolist()
        x_max, y_max, z_max = full_pts.max(axis=0).tolist()
        room_region = polygon_from_xz_bounds(x_min, x_max, z_min, z_max)

    room = {
        'room_id': args.room_id,
        'room_name': {'1': args.room_name},
        'objects': objects,
        'Region': room_region,
        'floor_id': args.floor_id,
        'obj_ids': [o['obj_id'] for o in objects],
        'N': 1,
        'imgs': {'1': room_keyframes_map.get(args.room_id, [])},
        'description': {'1': 'converted from hovsg minimal outputs'},
    }

    floor = {
        'floor_id': args.floor_id,
        'z_range': {'z_min': float(z_min), 'z_max': float(z_max)},
        'rooms': [room],
        'room_ids': [args.room_id],
        'description': 'auto-generated from hovsg minimal outputs',
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open('w', encoding='utf-8') as f:
        json.dump([floor], f, ensure_ascii=False, indent=2)

    print(f'output: {output_json}')
    print(f'objects: {len(objects)}')
    with_feat = sum(1 for o in objects if o.get('clip_embedding'))
    print(f'objects_with_clip_embedding: {with_feat}')


if __name__ == '__main__':
    main()
