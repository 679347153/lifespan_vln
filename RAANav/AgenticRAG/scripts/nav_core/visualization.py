"""轨迹可视化 — 从 sim_nav_loop.py 提取."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import cv2
import numpy as np


def visualize_trajectory_on_occ_grid(
    occ_grid,
    positions: List[list],
    step_log: List[Dict],
    target: str,
    output_dir: str,
):
    """绘制 agent 轨迹在深度构建的占据栅格上."""
    base = occ_grid.to_image(navigable=True, show_layers=True)

    pts = []
    for p in positions:
        gc = occ_grid.world_to_grid(np.array([p[0], p[2]]))
        pts.append((int(gc[0]), int(gc[1])))

    for i in range(1, len(pts)):
        cv2.line(base, pts[i - 1], pts[i], (0, 200, 255), 1)

    if pts:
        cv2.circle(base, pts[0], 5, (255, 0, 0), -1)
        cv2.circle(base, pts[-1], 5, (0, 0, 255), -1)

    for sl in step_log:
        if sl.get("target_visible"):
            p = sl["position"]
            gc = occ_grid.world_to_grid(np.array([p[0], p[2]]))
            cv2.drawMarker(base, (int(gc[0]), int(gc[1])), (0, 255, 0), cv2.MARKER_STAR, 12, 2)

    cv2.putText(base, f'Target: "{target}" | Steps: {len(step_log)}',
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    path = os.path.join(output_dir, "trajectory_depth.png")
    cv2.imwrite(path, base)
    print(f"  轨迹图(深度栅格): {path}")


def visualize_trajectory(
    grid: np.ndarray,
    grid_meta: Dict,
    positions: List[list],
    step_log: List[Dict],
    target: str,
    output_dir: str,
):
    """绘制 agent 轨迹在占据栅格上."""
    H, W = grid.shape
    resolution = grid_meta["resolution"]
    origin_x = grid_meta["origin_x"]
    origin_z = grid_meta["origin_z"]

    base = np.zeros((H, W, 3), dtype=np.uint8)
    base[grid == 1] = [200, 200, 200]
    base[grid == 2] = [40, 40, 40]

    pts = []
    for p in positions:
        c = int(round((p[0] - origin_x) / resolution))
        r = int(round((p[2] - origin_z) / resolution))
        pts.append((c, r))

    for i in range(1, len(pts)):
        cv2.line(base, pts[i - 1], pts[i], (0, 200, 255), 1)

    if pts:
        cv2.circle(base, pts[0], 5, (255, 0, 0), -1)
        cv2.circle(base, pts[-1], 5, (0, 0, 255), -1)

    for sl in step_log:
        if sl.get("target_visible"):
            p = sl["position"]
            c = int(round((p[0] - origin_x) / resolution))
            r = int(round((p[2] - origin_z) / resolution))
            cv2.drawMarker(base, (c, r), (0, 255, 0), cv2.MARKER_STAR, 12, 2)

    cv2.putText(base, f'Target: "{target}" | Steps: {len(step_log)}',
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    path = os.path.join(output_dir, "trajectory.png")
    cv2.imwrite(path, base)
    print(f"  轨迹图: {path}")
