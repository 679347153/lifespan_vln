from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
import plotly.graph_objects as go


def read_pcd(p: Path) -> Tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(str(p))
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if cols.size == 0:
        cols = np.ones_like(pts) * 0.75
    return pts, cols


def downsample(pts: np.ndarray, cols: np.ndarray, max_n: int) -> Tuple[np.ndarray, np.ndarray]:
    if pts.shape[0] <= max_n:
        return pts, cols
    idx = np.random.choice(pts.shape[0], size=max_n, replace=False)
    return pts[idx], cols[idx]


def rgb_str(cols: np.ndarray) -> np.ndarray:
    vals = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
    return np.array([f"rgb({r},{g},{b})" for r, g, b in vals], dtype=object)


def main() -> None:
    np.random.seed(42)

    root = Path('/home/adminer/agentRAG')
    out_dir = root / '组会汇报_2026-03-21'
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_dir = root / 'HOV-SG/output_min/hm3dsem/00824-Dd4bFSTQ8gi'
    full_pcd = scene_dir / 'full_pcd.ply'
    objects_dir = scene_dir / 'objects'

    pts, cols = read_pcd(full_pcd)
    pts_ds, cols_ds = downsample(pts, cols, max_n=90000)

    # 1) 原始房间点云（仅 full_pcd）
    fig0 = go.Figure()
    fig0.add_trace(
        go.Scatter3d(
            x=pts_ds[:, 0], y=pts_ds[:, 1], z=pts_ds[:, 2],
            mode='markers',
            marker=dict(size=1.2, color=rgb_str(cols_ds), opacity=0.85),
            name='原始房间点云(full_pcd)'
        )
    )
    fig0.update_layout(
        title='原始房间点云（full_pcd）',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        margin=dict(l=0, r=0, t=40, b=0),
        template='plotly_dark'
    )
    out0 = out_dir / '06_3D交互可视化_原始房间点云.html'
    fig0.write_html(str(out0), include_plotlyjs='cdn', full_html=True)

    # 2) 原始点云 + 对象中心点/索引（帮助解释“我们提取到了什么”）
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter3d(
            x=pts_ds[:, 0], y=pts_ds[:, 1], z=pts_ds[:, 2],
            mode='markers',
            marker=dict(size=1.0, color='rgba(180,180,180,0.45)'),
            name='原始点云背景'
        )
    )

    centers = []
    labels = []
    for i, pf in enumerate(sorted(objects_dir.glob('pcd_*.ply'))):
        op, _ = read_pcd(pf)
        if op.shape[0] == 0:
            continue
        c = op.mean(axis=0)
        centers.append(c)
        labels.append(pf.stem)

    if centers:
        c = np.asarray(centers)
        fig1.add_trace(
            go.Scatter3d(
                x=c[:, 0], y=c[:, 1], z=c[:, 2],
                mode='markers+text',
                text=labels,
                textposition='top center',
                marker=dict(size=5, color='red', opacity=0.95),
                name='提取对象中心(索引)'
            )
        )

    fig1.update_layout(
        title='原始点云 + 提取对象中心（用于解释当前进展）',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        margin=dict(l=0, r=0, t=40, b=0),
        template='plotly_dark'
    )
    out1 = out_dir / '07_3D交互可视化_原始点云与对象中心.html'
    fig1.write_html(str(out1), include_plotlyjs='cdn', full_html=True)

    print('saved', out0)
    print('saved', out1)


if __name__ == '__main__':
    main()
