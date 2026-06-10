#!/usr/bin/env python3
"""实时探索演示脚本 — 展示 GT-free 探索模式构建点云 & 房间分割.

功能:
  1. 在 Habitat 场景中自主探索 (无 navmesh, 纯深度占据栅格)
  2. 逐步累积 3D 彩色点云
  3. 实时可视化:
     - Open3D 窗口: 3D 彩色点云 + agent 轨迹
     - Matplotlib 窗口: 2D 占据栅格 + 探索进度 + 房间分割
  4. 探索结束后: 房间分割 + Voronoi 拓扑图

用法:
  python scripts/exploration_demo.py \
    --scene-dir experiment_data/hm3d/val/00814-p53SfW6mjZe \
    --dataset-config experiment_data/hm3d/hm3d_val_scene_dataset_config.json \
    --max-steps 40 --n-views 4

参考: HOV-SG visualize_graph.py / visualize_query_graph.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("TkAgg")          # 交互式后端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# ---- 项目根目录 ----
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from semantic_map_Create.occupancy_grid import (
    CameraIntrinsics,
    OccupancyGrid,
    camera_to_world,
    depth_to_local_pointcloud,
    filter_points_by_height,
)
from semantic_map_Create.room_segmentation import segment_rooms_from_occ_grid
from semantic_map_Create.voronoi_graph import build_voronoi_graph


# =====================================================================
# Habitat 仿真器工具
# =====================================================================

def _make_sim(scene_dir: str, dataset_config: str, resolution=(480, 640), hfov=90.0):
    """创建 Habitat 仿真器 (RGB + depth)."""
    import habitat_sim

    glb = os.path.join(scene_dir, os.path.basename(scene_dir).split("-", 1)[1] + ".basis.glb")
    if not os.path.isfile(glb):
        candidates = [f for f in os.listdir(scene_dir) if f.endswith(".basis.glb")]
        if not candidates:
            raise FileNotFoundError(f"找不到 .basis.glb: {scene_dir}")
        glb = os.path.join(scene_dir, candidates[0])

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = os.path.abspath(dataset_config)
    sim_cfg.scene_id = glb
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0
    sim_cfg.load_semantic_mesh = True

    color_spec = habitat_sim.CameraSensorSpec()
    color_spec.uuid = "color"
    color_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_spec.resolution = list(resolution)
    color_spec.hfov = hfov

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = list(resolution)
    depth_spec.hfov = hfov

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [color_spec, depth_spec]
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def _get_agent_pose(sim) -> Tuple[np.ndarray, float]:
    """返回 (position [x,y,z], heading_deg)."""
    state = sim.get_agent(0).get_state()
    pos = state.position
    q = state.rotation
    # 从四元数提取 yaw: agent 面朝 [-sin(θ), 0, -cos(θ)]
    import quaternion as quat_mod
    fwd = quat_mod.rotate_vectors(q, np.array([0, 0, -1.0]))
    heading = math.degrees(math.atan2(-fwd[0], -fwd[2]))
    return np.array(pos, dtype=np.float32), heading


def _set_agent_pose(sim, pos, heading_deg):
    """传送 agent 到指定位置和朝向."""
    import quaternion as quat_mod
    state = sim.get_agent(0).get_state()
    state.position = np.array(pos, dtype=np.float32)
    theta = math.radians(heading_deg)
    state.rotation = quat_mod.from_rotation_vector([0, theta, 0])
    sim.get_agent(0).set_state(state, reset_sensors=True)


def _snap_to_free(occ_grid: OccupancyGrid, pos_xz: np.ndarray, max_r=50) -> np.ndarray:
    """BFS 找最近 free cell (世界坐标 xz)."""
    gc = occ_grid.world_to_grid(pos_xz).astype(int)
    col, row = gc[0], gc[1]
    rows, cols = occ_grid.grid.shape
    if 0 <= row < rows and 0 <= col < cols and occ_grid.grid[row, col] == OccupancyGrid.FREE:
        return pos_xz
    visited = set()
    queue = deque([(col, row)])
    visited.add((col, row))
    while queue:
        c, r = queue.popleft()
        if abs(c - col) > max_r or abs(r - row) > max_r:
            break
        if 0 <= r < rows and 0 <= c < cols and occ_grid.grid[r, c] == OccupancyGrid.FREE:
            return occ_grid.grid_to_world(np.array([c, r], dtype=np.float64))
        for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nc, nr = c + dc, r + dr
            if (nc, nr) not in visited:
                visited.add((nc, nr))
                queue.append((nc, nr))
    return pos_xz  # fallback


# =====================================================================
# 覆盖路径规划 (简化版)
# =====================================================================

def _plan_coverage_waypoints(
    occ_grid: OccupancyGrid,
    start_pos: np.ndarray,
    spacing: float = 2.0,
    max_waypoints: int = 60,
) -> List[np.ndarray]:
    """在已知/未知空间均匀采样路径点, 贪心最近邻排序."""
    free_cells = np.argwhere(occ_grid.grid == OccupancyGrid.FREE)
    if len(free_cells) == 0:
        return []

    # 扩大覆盖: 在 free 区域边界之外也做网格采样
    grid = occ_grid.grid
    rows, cols = grid.shape
    spacing_px = max(1, int(spacing / occ_grid.resolution))

    candidates = []
    for r in range(0, rows, spacing_px):
        for c in range(0, cols, spacing_px):
            if grid[r, c] == OccupancyGrid.FREE:
                wpt = occ_grid.grid_to_world(np.array([c, r], dtype=np.float64))
                candidates.append(wpt)
    if not candidates:
        return []

    candidates = np.array(candidates)
    if len(candidates) > max_waypoints:
        idx = np.random.choice(len(candidates), max_waypoints, replace=False)
        candidates = candidates[idx]

    # 贪心最近邻排序
    ordered = []
    cur = start_pos[[0, 2]] if len(start_pos) == 3 else start_pos
    remaining = list(range(len(candidates)))
    while remaining:
        dists = np.linalg.norm(candidates[remaining] - cur, axis=1)
        nearest = remaining[np.argmin(dists)]
        ordered.append(candidates[nearest])
        cur = candidates[nearest]
        remaining.remove(nearest)
    return ordered


# =====================================================================
# frontier 探索 (增量)
# =====================================================================

def _pick_frontier_target(occ_grid: OccupancyGrid, agent_xz: np.ndarray) -> Optional[np.ndarray]:
    """选择最近的 frontier 区域中心."""
    frontiers = occ_grid.get_frontiers(min_area_m2=0.3)
    if len(frontiers) == 0:
        return None
    dists = np.linalg.norm(frontiers - agent_xz, axis=1)
    # 偏向中等距离: 太近没信息量, 太远不可靠
    scores = -dists  # 简化: 选最近的
    best = np.argmax(scores)
    return frontiers[best]


# =====================================================================
# A* 路径规划
# =====================================================================

def _astar_path(occ_grid: OccupancyGrid, start_xz, goal_xz) -> Optional[List[np.ndarray]]:
    """在占据栅格上做 A* 路径规划, 返回世界坐标路径点."""
    nav_map = occ_grid.get_navigable_map(layer="merged")
    sc = occ_grid.world_to_grid(np.array(start_xz)).astype(int)
    gc = occ_grid.world_to_grid(np.array(goal_xz)).astype(int)
    rows, cols = nav_map.shape

    def clamp(c, r):
        return max(0, min(c, cols - 1)), max(0, min(r, rows - 1))

    sc = clamp(sc[0], sc[1])
    gc = clamp(gc[0], gc[1])

    if not nav_map[gc[1], gc[0]]:
        return None

    import heapq
    DIRS = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
            (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414)]

    open_set = [(0.0, sc)]
    g_score = {sc: 0.0}
    came_from = {}
    closed = set()

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == gc:
            path = []
            while current in came_from:
                wc = occ_grid.grid_to_world(np.array(current, dtype=np.float64))
                path.append(wc)
                current = came_from[current]
            path.append(occ_grid.grid_to_world(np.array(sc, dtype=np.float64)))
            path.reverse()
            # 降采样: 每隔 step_size 取一个
            step_size = max(1, int(0.5 / occ_grid.resolution))
            return path[::step_size]
        if current in closed:
            continue
        closed.add(current)
        for dc, dr, cost in DIRS:
            nc, nr = current[0] + dc, current[1] + dr
            if 0 <= nc < cols and 0 <= nr < rows and nav_map[nr, nc]:
                ng = g_score[current] + cost
                if ng < g_score.get((nc, nr), float('inf')):
                    g_score[(nc, nr)] = ng
                    f = ng + heuristic((nc, nr), gc)
                    heapq.heappush(open_set, (f, (nc, nr)))
                    came_from[(nc, nr)] = current
    return None


# =====================================================================
# 可视化
# =====================================================================

class ExplorationVisualizer:
    """双窗口实时可视化: Open3D (3D点云) + Matplotlib (2D栅格)."""

    def __init__(self, use_open3d: bool = True):
        self.use_open3d = use_open3d
        self._pcd_points = []    # 累积世界坐标点, (N,3)
        self._pcd_colors = []    # 对应 RGB, (N,3) float [0,1]
        self._trajectory = []    # agent 轨迹 [(x,y,z), ...]

        # Matplotlib
        self.fig = plt.figure("探索演示 — 2D 栅格 & 房间分割", figsize=(14, 6))
        gs = GridSpec(1, 3, figure=self.fig, width_ratios=[1, 1, 1])
        self.ax_occ = self.fig.add_subplot(gs[0])
        self.ax_occ.set_title("占据栅格")
        self.ax_room = self.fig.add_subplot(gs[1])
        self.ax_room.set_title("房间分割")
        self.ax_stats = self.fig.add_subplot(gs[2])
        self.ax_stats.set_title("探索统计")
        self.fig.tight_layout(pad=2)

        # Open3D
        self.o3d_vis = None
        self.o3d_pcd = None
        self.o3d_traj = None
        if self.use_open3d:
            self._init_open3d()

    def _init_open3d(self):
        import open3d as o3d
        self.o3d_vis = o3d.visualization.Visualizer()
        self.o3d_vis.create_window("探索演示 — 3D 点云", width=960, height=720)

        self.o3d_pcd = o3d.geometry.PointCloud()
        self.o3d_vis.add_geometry(self.o3d_pcd)

        self.o3d_traj = o3d.geometry.LineSet()
        self.o3d_vis.add_geometry(self.o3d_traj)

        # 坐标轴
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        self.o3d_vis.add_geometry(axes)

        opt = self.o3d_vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.1, 0.1, 0.1])

    def add_pointcloud(self, points_world: np.ndarray, colors_rgb: np.ndarray):
        """增量添加点云 (N,3) + (N,3) float [0,1]."""
        if len(points_world) == 0:
            return
        # 下采样: 保留每 4 个点 (避免过多点拖慢渲染)
        step = 4
        self._pcd_points.append(points_world[::step])
        self._pcd_colors.append(colors_rgb[::step])

    def add_trajectory_point(self, pos: np.ndarray):
        self._trajectory.append(pos.copy())

    def update_3d(self):
        """刷新 Open3D 3D 视图."""
        if not self.use_open3d or self.o3d_vis is None:
            return
        import open3d as o3d

        if len(self._pcd_points) == 0:
            return

        all_pts = np.concatenate(self._pcd_points, axis=0)
        all_clr = np.concatenate(self._pcd_colors, axis=0)

        self.o3d_pcd.points = o3d.utility.Vector3dVector(all_pts)
        self.o3d_pcd.colors = o3d.utility.Vector3dVector(all_clr)
        self.o3d_vis.update_geometry(self.o3d_pcd)

        # 轨迹线
        if len(self._trajectory) >= 2:
            traj = np.array(self._trajectory)
            lines = [[i, i + 1] for i in range(len(traj) - 1)]
            self.o3d_traj.points = o3d.utility.Vector3dVector(traj)
            self.o3d_traj.lines = o3d.utility.Vector2iVector(lines)
            self.o3d_traj.colors = o3d.utility.Vector3dVector(
                [[1.0, 0.2, 0.2]] * len(lines)  # 红色轨迹
            )
            self.o3d_vis.update_geometry(self.o3d_traj)

        self.o3d_vis.poll_events()
        self.o3d_vis.update_renderer()

    def update_2d(self, occ_grid: OccupancyGrid, step: int, total_steps: int,
                  n_points: int, room_result=None, voronoi_graph=None):
        """刷新 Matplotlib 2D 视图."""
        # — 占据栅格 —
        self.ax_occ.cla()
        self.ax_occ.set_title(f"占据栅格 (step {step}/{total_steps})")
        grid = occ_grid.grid.copy().astype(np.float32)
        # 着色: unknown=灰, free=白, occupied=黑
        vis = np.full((*grid.shape, 3), 0.5, dtype=np.float32)
        vis[grid == OccupancyGrid.FREE] = [1.0, 1.0, 1.0]
        vis[grid == OccupancyGrid.OCCUPIED] = [0.0, 0.0, 0.0]
        # 已探索区域蓝色边界
        explored = occ_grid.explored.astype(np.uint8)
        contours, _ = cv2.findContours(explored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis_bgr = (vis[:, :, ::-1] * 255).astype(np.uint8)
        cv2.drawContours(vis_bgr, contours, -1, (255, 100, 0), 1)
        vis = vis_bgr[:, :, ::-1].astype(np.float32) / 255.0

        # 绘制轨迹
        if len(self._trajectory) >= 2:
            for i in range(len(self._trajectory) - 1):
                p1_xz = np.array([self._trajectory[i][0], self._trajectory[i][2]])
                p2_xz = np.array([self._trajectory[i + 1][0], self._trajectory[i + 1][2]])
                g1 = occ_grid.world_to_grid(p1_xz).astype(int)
                g2 = occ_grid.world_to_grid(p2_xz).astype(int)
                vis_u8 = (vis * 255).astype(np.uint8)
                cv2.line(vis_u8, (g1[0], g1[1]), (g2[0], g2[1]), (255, 50, 50), 2)
                vis = vis_u8.astype(np.float32) / 255.0
        # 当前位置
        if self._trajectory:
            cur = self._trajectory[-1]
            gc = occ_grid.world_to_grid(np.array([cur[0], cur[2]])).astype(int)
            vis_u8 = (vis * 255).astype(np.uint8)
            cv2.circle(vis_u8, (gc[0], gc[1]), 5, (0, 255, 0), -1)
            vis = vis_u8.astype(np.float32) / 255.0

        self.ax_occ.imshow(vis)
        self.ax_occ.axis("off")

        # — 房间分割 —
        self.ax_room.cla()
        if room_result is not None and room_result["n_rooms"] > 0:
            self.ax_room.set_title(f"房间分割 ({room_result['n_rooms']} 个房间)")
            labels = room_result["room_labels"]
            n = room_result["n_rooms"]
            cmap = plt.cm.get_cmap("Set3", max(n + 1, 3))
            room_vis = np.zeros((*labels.shape, 3), dtype=np.float32)
            for rid in range(1, n + 1):
                color = cmap(rid)[:3]
                room_vis[labels == rid] = color
            room_vis[labels == 0] = [0.2, 0.2, 0.2]

            # 绘制 Voronoi 图
            if voronoi_graph is not None:
                import networkx as _nx
                room_u8 = (room_vis * 255).astype(np.uint8)
                for u, v, data in voronoi_graph.edges(data=True):
                    p1 = voronoi_graph.nodes[u].get("pos_grid")
                    p2 = voronoi_graph.nodes[v].get("pos_grid")
                    if p1 is not None and p2 is not None:
                        cv2.line(room_u8,
                                 (int(p1[0]), int(p1[1])),
                                 (int(p2[0]), int(p2[1])),
                                 (255, 255, 0), 1)
                for nid, ndata in voronoi_graph.nodes(data=True):
                    pg = ndata.get("pos_grid")
                    if pg is not None:
                        cv2.circle(room_u8, (int(pg[0]), int(pg[1])), 3, (0, 0, 255), -1)
                room_vis = room_u8.astype(np.float32) / 255.0

            # 房间中心标注
            for i, center in enumerate(room_result["room_centers"]):
                gc = occ_grid.world_to_grid(np.array(center)).astype(int)
                self.ax_room.text(gc[0], gc[1], f"R{i+1}", color="white",
                                  fontsize=8, ha="center", va="center",
                                  fontweight="bold",
                                  bbox=dict(boxstyle="round,pad=0.2",
                                            facecolor="black", alpha=0.6))
            self.ax_room.imshow(room_vis)
        else:
            self.ax_room.set_title("房间分割 (待计算)")
            self.ax_room.text(0.5, 0.5, "探索中...", ha="center", va="center",
                              transform=self.ax_room.transAxes, fontsize=14)
        self.ax_room.axis("off")

        # — 统计 —
        self.ax_stats.cla()
        self.ax_stats.set_title("探索统计")
        self.ax_stats.axis("off")
        grid = occ_grid.grid
        n_free = int(np.sum(grid == OccupancyGrid.FREE))
        n_occ = int(np.sum(grid == OccupancyGrid.OCCUPIED))
        n_unknown = int(np.sum(grid == OccupancyGrid.UNKNOWN))
        n_explored = int(np.sum(occ_grid.explored))
        total_cells = grid.size
        coverage = n_explored / max(total_cells, 1) * 100

        stats_text = (
            f"步数: {step} / {total_steps}\n"
            f"累积点云: {n_points:,} 点\n"
            f"轨迹长度: {self._get_traj_length():.1f} m\n"
            f"\n--- 栅格统计 ---\n"
            f"Free:    {n_free:>8,}\n"
            f"Occupied:{n_occ:>8,}\n"
            f"Unknown: {n_unknown:>8,}\n"
            f"已探索:  {coverage:.1f}%\n"
        )
        if room_result is not None:
            stats_text += (
                f"\n--- 房间分割 ---\n"
                f"房间数: {room_result['n_rooms']}\n"
            )
            for i, area in enumerate(room_result["room_areas_m2"]):
                stats_text += f"  R{i+1}: {area:.1f} m²\n"

        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                           fontsize=9, verticalalignment="top", fontfamily="monospace")
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def _get_traj_length(self) -> float:
        if len(self._trajectory) < 2:
            return 0.0
        traj = np.array(self._trajectory)
        return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))

    def show_final(self, occ_grid, room_result, voronoi_graph, output_dir: str):
        """最终可视化 & 保存."""
        self.update_2d(occ_grid, step=-1, total_steps=-1,
                       n_points=sum(len(p) for p in self._pcd_points),
                       room_result=room_result, voronoi_graph=voronoi_graph)
        self.ax_occ.set_title("占据栅格 (最终)")
        self.ax_room.set_title(f"房间分割 + Voronoi ({room_result['n_rooms']} 房间)")

        os.makedirs(output_dir, exist_ok=True)
        self.fig.savefig(os.path.join(output_dir, "exploration_2d_final.png"), dpi=150)
        print(f"  [保存] 2D 栅格: {output_dir}/exploration_2d_final.png")

        # 保存 3D 点云 PLY
        if len(self._pcd_points) > 0:
            import open3d as o3d
            all_pts = np.concatenate(self._pcd_points, axis=0)
            all_clr = np.concatenate(self._pcd_colors, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_pts)
            pcd.colors = o3d.utility.Vector3dVector(all_clr)
            ply_path = os.path.join(output_dir, "exploration_pointcloud.ply")
            o3d.io.write_point_cloud(ply_path, pcd)
            print(f"  [保存] 3D 点云: {ply_path} ({len(all_pts):,} 点)")

        # 保存房间分割的点云 (按房间着色)
        if room_result is not None and len(self._pcd_points) > 0:
            self._save_room_colored_pointcloud(occ_grid, room_result, output_dir)

    def _save_room_colored_pointcloud(self, occ_grid, room_result, output_dir):
        """将点云按房间着色并保存."""
        import open3d as o3d
        all_pts = np.concatenate(self._pcd_points, axis=0)
        labels = room_result["room_labels"]
        n = room_result["n_rooms"]
        cmap = plt.cm.get_cmap("Set3", max(n + 1, 3))

        # 每个 3D 点投影到栅格, 查房间 ID
        xz = all_pts[:, [0, 2]]
        gc = occ_grid.world_to_grid(xz).astype(int)
        rows, cols = labels.shape
        valid = (gc[:, 0] >= 0) & (gc[:, 0] < cols) & (gc[:, 1] >= 0) & (gc[:, 1] < rows)

        room_colors = np.full((len(all_pts), 3), 0.3, dtype=np.float32)
        gc_valid = gc[valid]
        room_ids = labels[gc_valid[:, 1], gc_valid[:, 0]]
        for rid in range(1, n + 1):
            mask_local = room_ids == rid
            idx_global = np.where(valid)[0][mask_local]
            room_colors[idx_global] = cmap(rid)[:3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        pcd.colors = o3d.utility.Vector3dVector(room_colors)
        ply_path = os.path.join(output_dir, "exploration_rooms_colored.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"  [保存] 房间着色点云: {ply_path}")

    def close(self):
        if self.o3d_vis is not None:
            self.o3d_vis.destroy_window()
        plt.close(self.fig)


# =====================================================================
# 主探索循环
# =====================================================================

def run_exploration_demo(
    scene_dir: str,
    dataset_config: str,
    max_steps: int = 40,
    n_views: int = 4,
    step_size: float = 0.5,
    max_depth: float = 5.0,
    output_dir: str = "exploration_demo_output",
    update_interval: int = 2,            # 每 N 步刷新可视化
    room_segment_interval: int = 10,     # 每 N 步重新分割房间
    use_open3d: bool = True,
):
    """运行完整探索演示."""

    print("=" * 60)
    print("  实时探索演示 (GT-free)")
    print(f"  场景: {os.path.basename(scene_dir)}")
    print(f"  最大步数: {max_steps}, 环视: {n_views}×{360 // n_views}°")
    print(f"  步长: {step_size}m, 最大深度: {max_depth}m")
    print("=" * 60)

    # 1. 初始化仿真器
    print("\n[初始化] 创建 Habitat 仿真器 ...")
    sim = _make_sim(scene_dir, dataset_config)
    intrinsics = CameraIntrinsics(hfov_deg=90.0, height=480, width=640)

    agent_pos, agent_heading = _get_agent_pose(sim)
    print(f"  起始位置: {agent_pos}")

    # 2. 初始化占据栅格 (无 navmesh)
    print("[初始化] 创建占据栅格 (from_agent_position) ...")
    occ_grid = OccupancyGrid.from_agent_position(
        agent_pos, resolution=0.05, agent_radius=0.18, initial_extent=15.0
    )

    # 3. 初始化可视化器
    viz = ExplorationVisualizer(use_open3d=use_open3d)
    room_result = None
    voronoi_graph_obj = None
    total_points = 0

    # 4. 起点处首次观测
    print("[Phase 0] 起点处首次环视观测 ...\n")
    _observe_panoramic(sim, occ_grid, intrinsics, viz, n_views, max_depth)
    agent_pos, _ = _get_agent_pose(sim)
    viz.add_trajectory_point(agent_pos)
    total_points = sum(len(p) for p in viz._pcd_points)

    viz.update_2d(occ_grid, 0, max_steps, total_points)
    viz.update_3d()

    # 5. 探索主循环
    print("[探索] 开始自主探索 ...\n")
    for step in range(1, max_steps + 1):
        agent_pos, agent_heading = _get_agent_pose(sim)
        agent_xz = np.array([agent_pos[0], agent_pos[2]])

        # 选择下一个目标
        target_xz = _pick_frontier_target(occ_grid, agent_xz)
        if target_xz is None:
            # 无 frontier → 随机选一个已知 free 点
            free_cells = np.argwhere(occ_grid.grid == OccupancyGrid.FREE)
            if len(free_cells) > 0:
                idx = np.random.randint(len(free_cells))
                target_xz = occ_grid.grid_to_world(
                    np.array([free_cells[idx][1], free_cells[idx][0]], dtype=np.float64)
                )
            else:
                print(f"  step {step}: 无可达目标, 停止探索")
                break

        # A* 路径规划
        path = _astar_path(occ_grid, agent_xz, target_xz)
        if path is None or len(path) < 2:
            # 直线前进 fallback
            direction = target_xz - agent_xz
            dist = np.linalg.norm(direction)
            if dist > 0:
                target_xz = agent_xz + direction / dist * min(dist, step_size * 2)
            target_xz = _snap_to_free(occ_grid, target_xz)
            path = [agent_xz, target_xz]

        # 沿路径移动 (每步只走一小段)
        wp_idx = min(3, len(path) - 1)  # 每个宏步走 ~3 个路径点
        next_wp = path[wp_idx]
        next_wp = _snap_to_free(occ_grid, next_wp)

        # 计算朝向
        dx = next_wp[0] - agent_xz[0]
        dz = next_wp[1] - agent_xz[1]
        heading = math.degrees(math.atan2(-dx, -dz))

        # 传送到新位置
        new_pos = np.array([next_wp[0], agent_pos[1], next_wp[1]], dtype=np.float32)
        _set_agent_pose(sim, new_pos, heading)

        # 环视观测
        _observe_panoramic(sim, occ_grid, intrinsics, viz, n_views, max_depth)
        agent_pos, _ = _get_agent_pose(sim)
        viz.add_trajectory_point(agent_pos)
        total_points = sum(len(p) for p in viz._pcd_points)

        # 房间分割 (周期性)
        if step % room_segment_interval == 0 or step == max_steps:
            try:
                room_result = segment_rooms_from_occ_grid(occ_grid, min_room_area_m2=2.0)
                voronoi_graph_obj = build_voronoi_graph(occ_grid, room_result["room_labels"])
                n_rooms = room_result["n_rooms"]
                n_nodes = voronoi_graph_obj.number_of_nodes()
                n_edges = voronoi_graph_obj.number_of_edges()
                print(f"  step {step}: 房间={n_rooms}, Voronoi 节点={n_nodes}, 边={n_edges}")
            except Exception as e:
                print(f"  step {step}: 房间分割异常: {e}")

        # 更新可视化
        if step % update_interval == 0 or step == max_steps:
            n_free = int(np.sum(occ_grid.grid == OccupancyGrid.FREE))
            print(f"  step {step}/{max_steps}: 点={total_points:,}, free={n_free:,}, "
                  f"轨迹={viz._get_traj_length():.1f}m")
            viz.update_2d(occ_grid, step, max_steps, total_points, room_result, voronoi_graph_obj)
            viz.update_3d()

    # 6. 最终房间分割 & Voronoi
    print("\n[Phase Final] 最终房间分割 + Voronoi 拓扑图 ...")
    try:
        room_result = segment_rooms_from_occ_grid(occ_grid, min_room_area_m2=2.0)
        voronoi_graph_obj = build_voronoi_graph(occ_grid, room_result["room_labels"])
        print(f"  房间数: {room_result['n_rooms']}")
        for i, (center, area) in enumerate(
            zip(room_result["room_centers"], room_result["room_areas_m2"])
        ):
            print(f"    R{i+1}: 中心=({center[0]:.1f}, {center[1]:.1f}), 面积={area:.1f} m²")
        print(f"  Voronoi: {voronoi_graph_obj.number_of_nodes()} 节点, "
              f"{voronoi_graph_obj.number_of_edges()} 边")
    except Exception as e:
        print(f"  房间分割失败: {e}")
        if room_result is None:
            room_result = {"n_rooms": 0, "room_labels": np.zeros_like(occ_grid.grid, dtype=np.int32),
                           "room_centers": [], "room_areas_m2": [], "room_pixel_counts": []}

    # 7. 保存结果
    print(f"\n[保存] 输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    occ_grid.save(output_dir, prefix="exploration")
    viz.show_final(occ_grid, room_result, voronoi_graph_obj, output_dir)

    # 8. 保存占据栅格可视化
    print("\n" + "=" * 60)
    print("  探索演示完成!")
    print(f"  总步数: {max_steps}")
    print(f"  累积点云: {total_points:,} 点")
    print(f"  轨迹长度: {viz._get_traj_length():.1f} m")
    print(f"  房间数: {room_result['n_rooms']}")
    print(f"  输出: {output_dir}/")
    print("=" * 60)

    # 等待用户关闭窗口
    print("\n[提示] 关闭 Matplotlib 窗口以结束程序")
    plt.show()
    viz.close()
    sim.close()


def _observe_panoramic(sim, occ_grid, intrinsics, viz, n_views, max_depth):
    """360° 环视观测, 更新栅格 + 累积点云."""
    agent_pos, base_heading = _get_agent_pose(sim)
    angle_step = 360.0 / n_views

    for v in range(n_views):
        heading = base_heading + v * angle_step
        _set_agent_pose(sim, agent_pos, heading)
        obs = sim.get_sensor_observations()

        rgb = obs["color"][:, :, :3]              # (H,W,3) uint8
        depth = obs["depth"]                       # (H,W) float32 米

        # 更新占据栅格
        occ_grid.update_from_habitat_depth(
            depth, intrinsics, agent_pos, heading, max_depth=max_depth,
        )

        # 深度 → 世界坐标点云 (含颜色)
        pc_cam = depth_to_local_pointcloud(depth, intrinsics, max_depth)
        if len(pc_cam) == 0:
            continue
        pc_world = camera_to_world(pc_cam, agent_pos, heading)

        # 获取对应颜色: 反投影到像素坐标
        H, W = depth.shape
        valid = (depth > 0) & (depth < max_depth)
        v_idx, u_idx = np.where(valid)
        colors = rgb[v_idx, u_idx].astype(np.float32) / 255.0

        viz.add_pointcloud(pc_world, colors)

    # 恢复原始朝向
    _set_agent_pose(sim, agent_pos, base_heading)


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="实时探索演示 — GT-free 点云构建 & 房间分割")
    parser.add_argument("--scene-dir", required=True, help="HM3D 场景目录")
    parser.add_argument("--dataset-config", required=True, help="scene_dataset_config.json")
    parser.add_argument("--max-steps", type=int, default=40, help="最大探索步数")
    parser.add_argument("--n-views", type=int, default=4, help="环视视角数")
    parser.add_argument("--step-size", type=float, default=0.5, help="每步移动距离 (m)")
    parser.add_argument("--max-depth", type=float, default=5.0, help="深度最大值 (m)")
    parser.add_argument("--output-dir", default="exploration_demo_output", help="输出目录")
    parser.add_argument("--update-interval", type=int, default=2, help="可视化刷新间隔 (步)")
    parser.add_argument("--room-interval", type=int, default=10, help="房间分割间隔 (步)")
    parser.add_argument("--no-open3d", action="store_true", help="禁用 Open3D 3D 窗口")
    args = parser.parse_args()

    run_exploration_demo(
        scene_dir=args.scene_dir,
        dataset_config=args.dataset_config,
        max_steps=args.max_steps,
        n_views=args.n_views,
        step_size=args.step_size,
        max_depth=args.max_depth,
        output_dir=args.output_dir,
        update_interval=args.update_interval,
        room_segment_interval=args.room_interval,
        use_open3d=not args.no_open3d,
    )


if __name__ == "__main__":
    main()
