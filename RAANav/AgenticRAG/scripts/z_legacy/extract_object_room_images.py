from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import open3d as o3d


def parse_obj_index(name: str) -> int:
    try:
        return int(name.replace('pcd_', '').replace('.ply', ''))
    except Exception:
        return 10**9


def load_pose(pose_path: Path) -> np.ndarray:
    vals = np.fromstring(pose_path.read_text().strip(), sep='\t', dtype=np.float64)
    if vals.size != 16:
        raise ValueError(f'bad pose format: {pose_path}')
    return vals.reshape(4, 4)


def project_points(points_w: np.ndarray, twc: np.ndarray, rwc: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # habitat camera forward is -Z in camera frame
    pc = (rwc.T @ (points_w - twc).T).T
    depth = -pc[:, 2]
    fx = w / 2.0
    fy = h / 2.0
    cx = w / 2.0
    cy = h / 2.0
    eps = 1e-6
    u = fx * (pc[:, 0] / np.maximum(depth, eps)) + cx
    v = fy * (-pc[:, 1] / np.maximum(depth, eps)) + cy
    uv = np.stack([u, v], axis=1)
    valid = (depth > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return uv, valid, depth


def crop_with_bbox(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2, x1:x2]


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract object crops and room keyframes from rgb+semantic+pose')
    parser.add_argument('--hovsg_scene_dir', required=True)
    parser.add_argument('--capture_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--room_id', default='R1')
    parser.add_argument('--object_topk', type=int, default=3)
    parser.add_argument('--room_topk', type=int, default=6)
    args = parser.parse_args()

    scene_dir = Path(args.hovsg_scene_dir)
    cap_dir = Path(args.capture_dir)
    out_dir = Path(args.output_dir)
    obj_out_dir = out_dir / 'objects'
    room_out_dir = out_dir / 'rooms' / args.room_id
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    room_out_dir.mkdir(parents=True, exist_ok=True)

    rgb_dir = cap_dir / 'rgb'
    sem_dir = cap_dir / 'semantic'
    pose_dir = cap_dir / 'pose'

    rgb_files = sorted(rgb_dir.glob('*.png'))
    if not rgb_files:
        raise RuntimeError(f'no rgb files in {rgb_dir}')

    frame_ids = [p.stem for p in rgb_files]
    poses: Dict[str, np.ndarray] = {}
    semantics: Dict[str, np.ndarray] = {}
    sharpness_vals: Dict[str, float] = {}
    sem_cover_vals: Dict[str, float] = {}

    # frame-level stats for room keyframes
    for rf in rgb_files:
        fid = rf.stem
        img = cv2.imread(str(rf), cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_vals[fid] = sharp

        sem_path = sem_dir / f'{fid}.npy'
        if sem_path.exists():
            sem = np.load(str(sem_path))
            semantics[fid] = sem
            sem_cover = float(np.count_nonzero(sem)) / float(sem.size) if sem.size else 0.0
        else:
            sem_cover = 0.0
        sem_cover_vals[fid] = sem_cover

        pose_path = pose_dir / f'{fid}.txt'
        if pose_path.exists():
            poses[fid] = load_pose(pose_path)

    # normalize frame-level metrics
    all_sharp = np.array(list(sharpness_vals.values()), dtype=np.float64)
    smin = float(all_sharp.min()) if all_sharp.size else 0.0
    smax = float(all_sharp.max()) if all_sharp.size else 1.0

    def sharp_norm(fid: str) -> float:
        v = sharpness_vals.get(fid, 0.0)
        return (v - smin) / (smax - smin + 1e-8)

    # room keyframes
    room_scores = []
    for fid in frame_ids:
        score = 0.6 * sharp_norm(fid) + 0.4 * sem_cover_vals.get(fid, 0.0)
        room_scores.append((score, fid))
    room_scores.sort(reverse=True)

    room_imgs = []
    for rank, (_score, fid) in enumerate(room_scores[: max(1, args.room_topk)], start=1):
        src = rgb_dir / f'{fid}.png'
        dst = room_out_dir / f'room_{rank:02d}_{fid}.jpg'
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            continue
        cv2.imwrite(str(dst), img)
        room_imgs.append(str(dst))

    # object crops
    obj_files = sorted((scene_dir / 'objects').glob('pcd_*.ply'), key=lambda p: parse_obj_index(p.name))
    object_crops: Dict[str, List[str]] = {}

    for pf in obj_files:
        stem = pf.stem
        pcd = o3d.io.read_point_cloud(str(pf))
        pts = np.asarray(pcd.points)
        if pts.shape[0] == 0:
            object_crops[stem] = []
            continue

        centroid = pts.mean(axis=0)
        candidates: List[Tuple[float, str, Tuple[int, int, int, int]]] = []

        for fid in frame_ids:
            pose = poses.get(fid)
            if pose is None:
                continue
            img_path = rgb_dir / f'{fid}.png'
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]

            rwc = pose[:3, :3]
            twc = pose[:3, 3]

            uv, valid, _depth = project_points(pts, twc, rwc, w, h)
            vis_ratio = float(np.count_nonzero(valid)) / float(valid.size)
            if vis_ratio < 0.01:
                continue

            uv_valid = uv[valid]
            x1, y1 = np.floor(uv_valid.min(axis=0)).astype(int)
            x2, y2 = np.ceil(uv_valid.max(axis=0)).astype(int)

            # margin
            mx = int(0.08 * (x2 - x1 + 1)) + 4
            my = int(0.08 * (y2 - y1 + 1)) + 4
            x1 -= mx
            x2 += mx
            y1 -= my
            y2 += my

            # frontness + distance
            cam_to_obj = centroid - twc
            dist = float(np.linalg.norm(cam_to_obj)) + 1e-8
            dir_vec = cam_to_obj / dist
            forward = -rwc[:, 2]
            front = max(0.0, float(np.dot(dir_vec, forward)))

            # semantic support in bbox
            sem = semantics.get(fid)
            sem_support = 0.0
            if sem is not None:
                yy1 = max(0, min(h - 1, y1))
                yy2 = max(1, min(h, y2))
                xx1 = max(0, min(w - 1, x1))
                xx2 = max(1, min(w, x2))
                if yy2 > yy1 and xx2 > xx1:
                    crop_sem = sem[yy1:yy2, xx1:xx2]
                    sem_support = float(np.count_nonzero(crop_sem)) / float(crop_sem.size) if crop_sem.size else 0.0

            score = 0.50 * vis_ratio + 0.25 * front + 0.15 * sem_support + 0.10 * sharp_norm(fid)
            candidates.append((score, fid, (x1, y1, x2, y2)))

        candidates.sort(reverse=True, key=lambda x: x[0])

        crops = []
        topk = candidates[: max(1, args.object_topk)]
        obj_dir = obj_out_dir / stem
        obj_dir.mkdir(parents=True, exist_ok=True)
        for rank, (_s, fid, (x1, y1, x2, y2)) in enumerate(topk, start=1):
            img = cv2.imread(str(rgb_dir / f'{fid}.png'), cv2.IMREAD_COLOR)
            if img is None:
                continue
            crop = crop_with_bbox(img, x1, y1, x2, y2)
            dst = obj_dir / f'{stem}_top{rank}_{fid}.jpg'
            cv2.imwrite(str(dst), crop)
            crops.append(str(dst))

        # fallback: 没有候选时至少给一张全图中心裁剪
        if not crops and frame_ids:
            fid = frame_ids[0]
            img = cv2.imread(str(rgb_dir / f'{fid}.png'), cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                ch, cw = int(h * 0.4), int(w * 0.4)
                cy, cx = h // 2, w // 2
                crop = crop_with_bbox(img, cx - cw // 2, cy - ch // 2, cx + cw // 2, cy + ch // 2)
                dst = obj_dir / f'{stem}_fallback_{fid}.jpg'
                cv2.imwrite(str(dst), crop)
                crops.append(str(dst))

        object_crops[stem] = crops

    object_json = out_dir / 'object_crops.json'
    room_json = out_dir / 'room_keyframes.json'
    with object_json.open('w', encoding='utf-8') as f:
        json.dump(object_crops, f, ensure_ascii=False, indent=2)
    with room_json.open('w', encoding='utf-8') as f:
        json.dump({args.room_id: room_imgs}, f, ensure_ascii=False, indent=2)

    print('saved', object_json)
    print('saved', room_json)
    print('objects_with_crops', sum(1 for v in object_crops.values() if v))
    print('room_frames', len(room_imgs))


if __name__ == '__main__':
    main()
