from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
import plotly.graph_objects as go


def read_points_colors(ply_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if cols.size == 0:
        cols = np.ones_like(pts) * 0.7
    return pts, cols


def downsample(pts: np.ndarray, cols: np.ndarray, max_n: int) -> Tuple[np.ndarray, np.ndarray]:
    n = pts.shape[0]
    if n <= max_n:
        return pts, cols
    idx = np.random.choice(n, size=max_n, replace=False)
    return pts[idx], cols[idx]


def rgb_str(cols: np.ndarray) -> np.ndarray:
    cols = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
    return np.array([f"rgb({r},{g},{b})" for r, g, b in cols], dtype=object)


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    root = Path('/home/adminer/agentRAG')
    scene_dir = root / 'HOV-SG/output_min/hm3dsem/00824-Dd4bFSTQ8gi'
    out_dir = root / '组会汇报_2026-03-21'
    out_dir.mkdir(parents=True, exist_ok=True)

    full_ply = scene_dir / 'full_pcd.ply'
    masked_ply = scene_dir / 'masked_pcd.ply'
    objects_dir = scene_dir / 'objects'

    traces = []

    # Layer 1: full scene
    if full_ply.exists():
        pts, cols = read_points_colors(full_ply)
        pts, cols = downsample(pts, cols, max_n=50000)
        traces.append(
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=1.2, color=rgb_str(cols), opacity=0.45),
                name='L0_FullScene',
                visible=True,
            )
        )

    # Layer 2: masked scene
    if masked_ply.exists():
        pts, cols = read_points_colors(masked_ply)
        pts, cols = downsample(pts, cols, max_n=35000)
        traces.append(
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=1.6, color=rgb_str(cols), opacity=0.55),
                name='L1_MaskedScene',
                visible=False,
            )
        )

    # Layer 3: each object
    obj_files = sorted(objects_dir.glob('pcd_*.ply'))
    palette = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
        '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62', '#8da0cb',
    ]
    for i, pf in enumerate(obj_files):
        pts, _ = read_points_colors(pf)
        if pts.shape[0] == 0:
            continue
        pts, _ = downsample(pts, np.zeros_like(pts), max_n=8000)
        color = palette[i % len(palette)]
        traces.append(
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=2.4, color=color, opacity=0.9),
                name=f'L2_{pf.stem}',
                visible=True if i < 6 else 'legendonly',
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title='HOV-SG 3D Interactive View (Scene 00824)',
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
        ),
        legend=dict(itemsizing='constant'),
        margin=dict(l=0, r=0, t=40, b=0),
        template='plotly_dark',
    )

    out_html = out_dir / '04_3D交互可视化_全景与对象分层.html'
    fig.write_html(str(out_html), include_plotlyjs='cdn', full_html=True)

    # Objects-only view for cleaner demo
    obj_only = go.Figure()
    for tr in traces:
        if str(tr.name).startswith('L2_'):
            obj_only.add_trace(tr)
    obj_only.update_layout(
        title='HOV-SG 3D Interactive View (Objects Only)',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        margin=dict(l=0, r=0, t=40, b=0),
        template='plotly_dark',
    )
    out_html2 = out_dir / '05_3D交互可视化_对象层.html'
    obj_only.write_html(str(out_html2), include_plotlyjs='cdn', full_html=True)

    print('saved', out_html)
    print('saved', out_html2)


if __name__ == '__main__':
    main()
