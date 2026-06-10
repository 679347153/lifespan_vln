"""2D 占据栅格地图构建模块 — 三层架构.

功能:
  1. 从深度图 + 相机位姿增量构建 2D 占据栅格
  2. 从 Habitat navmesh 直接生成导航可通行地图
  3. 三层分离架构:
     - Layer 1 (persistent): 墙壁/结构 + stability≥阈值物体, 用于 A* 全局规划
     - Layer 2 (short_term): 最近 K 帧深度累积, 处理临时障碍
     - Layer 3 (current_frame): 当前深度帧, 实时避障
  4. Frontier 检测 (已探索 vs 未探索边界)
  5. 实时碰撞检测
  6. 输出可直接用于 GMM 概率场叠加的 M_free(x,y) 掩码

坐标约定 (HM3D):
  World: x-right, y-up, z-back
  Grid:  col ↔ world-x, row ↔ world-z (鸟瞰图)

参考:
  - BeliefMapNav/vlfm/mapping/obstacle_map.py (实时增量更新 + fog-of-war + frontier)
  - DovSG/dovsg/navigation/occupancy_map.py (高度过滤 + 阈值)
  - osmAG-LLM/HM3DSEM_navigation/render_hm3d.py (hfov=90° 深度反投影)
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)


# ------------------------------------------------------------------
# 相机内参工具
# ------------------------------------------------------------------

class CameraIntrinsics:
    """从 hfov + 分辨率计算相机内参.

    参考 BeliefMapNav / HOV-SG / osmAG-LLM 各项目一致的计算方式:
      fx = W / (2 * tan(hfov/2))
      fy = H / (2 * tan(vfov/2)),  vfov = 2 * atan(tan(hfov/2) * H/W)
    """

    def __init__(self, hfov_deg: float = 90.0, height: int = 480, width: int = 640):
        self.hfov_deg = hfov_deg
        self.hfov = math.radians(hfov_deg)
        self.height = height
        self.width = width
        self.fx = width / (2.0 * math.tan(self.hfov / 2.0))
        vfov = 2.0 * math.atan(math.tan(self.hfov / 2.0) * height / width)
        self.fy = height / (2.0 * math.tan(vfov / 2.0))
        self.cx = width / 2.0
        self.cy = height / 2.0

    def to_dict(self) -> Dict[str, float]:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}


# ------------------------------------------------------------------
# 深度图 → 世界坐标点云
# ------------------------------------------------------------------

def depth_to_local_pointcloud(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_depth: float = 10.0,
) -> np.ndarray:
    """深度图 → 相机坐标系 3D 点云.

    Args:
        depth: (H, W) float32, 单位: 米 (habitat depth sensor 直接返回)
        intrinsics: 相机内参
        max_depth: 最大有效深度

    Returns:
        (N, 3) camera frame [x_right, y_down, z_forward]
    """
    H, W = depth.shape[:2]
    valid = (depth > 0) & (depth < max_depth)
    v_idx, u_idx = np.where(valid)
    z = depth[v_idx, u_idx].astype(np.float32)
    x = (u_idx - intrinsics.cx) * z / intrinsics.fx
    y = (v_idx - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x, y, z], axis=-1)


def camera_to_world(
    points_cam: np.ndarray,
    agent_pos: np.ndarray,
    agent_heading_deg: float,
) -> np.ndarray:
    """相机坐标系 → Habitat 世界坐标系.

    depth_to_local_pointcloud 产出的局部坐标: x=right, y=down, z=forward
    Habitat 世界坐标: x=right, y=up, z 轴 (agent 默认面向 -Z)

    heading θ 对应的 agent 朝向: [-sin(θ), 0, -cos(θ)]
    agent 右方向:            [cos(θ), 0, -sin(θ)]

    变换:
      world_x =  cam_x * cos(θ) - cam_z * sin(θ) + agent_x
      world_y =  agent_y - cam_y                    (Y翻转: cam下→world上)
      world_z = -cam_x * sin(θ) - cam_z * cos(θ) + agent_z
    """
    if len(points_cam) == 0:
        return np.empty((0, 3), dtype=np.float32)

    theta = math.radians(agent_heading_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy, cz = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    wx = cx * cos_t - cz * sin_t + agent_pos[0]
    wy = agent_pos[1] - cy
    wz = -cx * sin_t - cz * cos_t + agent_pos[2]
    return np.stack([wx, wy, wz], axis=-1).astype(np.float32)


def filter_points_by_height(
    points_world: np.ndarray,
    min_height: float = 0.15,
    max_height: float = 1.2,
    agent_y: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """高度过滤, 返回 (free_points, obstacle_points).

    free: 低于 min_height (地面)
    obstacle: min_height ≤ y ≤ max_height
    """
    y = points_world[:, 1]
    floor_y = agent_y
    rel_y = y - floor_y
    free_mask = rel_y < min_height
    obs_mask = (rel_y >= min_height) & (rel_y <= max_height)
    return points_world[free_mask], points_world[obs_mask]


class OccupancyGrid:
    """2D 占据栅格地图 — 三层架构.

    层级:
      persistent (Layer 1): 结构性障碍 + 高 stability 物体, A* 全局规划用
      short_term (Layer 2): 最近 K 帧深度累积, 临时障碍
      current_frame (Layer 3): 当前深度帧, 实时避障

    用法:
      A) 从 Habitat navmesh 一次性生成 (作为 persistent 层基线)
      B) 从深度帧增量更新 (在线导航: update_from_habitat_depth)
    """

    FREE = 1
    OCCUPIED = 2
    UNKNOWN = 0

    def __init__(
        self,
        resolution: float = 0.05,
        world_min: Optional[np.ndarray] = None,
        world_max: Optional[np.ndarray] = None,
        grid_size: Optional[Tuple[int, int]] = None,
        agent_radius: float = 0.18,
        short_term_frames: int = 20,
        stability_threshold: float = 0.80,
    ):
        """
        Args:
            resolution: 栅格分辨率 (m/pixel)
            world_min: 世界坐标最小值 [x_min, z_min] (2D)
            world_max: 世界坐标最大值 [x_max, z_max] (2D)
            grid_size: (rows, cols), 若提供则直接使用
            agent_radius: 机器人半径 (m), 用于膨胀障碍物
            short_term_frames: 短期层保留的最近帧数
            stability_threshold: 持久层入库的 stability 阈值
        """
        self.resolution = resolution
        self.agent_radius = agent_radius
        self.short_term_frames = short_term_frames
        self.stability_threshold = stability_threshold

        if grid_size is not None:
            self._rows, self._cols = grid_size
            if world_min is not None:
                self._origin = np.array(world_min, dtype=np.float64)
            else:
                self._origin = np.array([0.0, 0.0])
        elif world_min is not None and world_max is not None:
            self._origin = np.array(world_min, dtype=np.float64)
            extent = np.array(world_max) - self._origin
            self._cols = int(np.ceil(extent[0] / resolution)) + 1
            self._rows = int(np.ceil(extent[1] / resolution)) + 1
        else:
            # 延迟初始化
            self._origin = np.array([0.0, 0.0])
            self._rows = 0
            self._cols = 0

        # 主栅格 (兼容旧接口): 0=unknown, 1=free, 2=occupied
        self.grid = np.zeros((self._rows, self._cols), dtype=np.uint8)
        # 探索标记: True=已观测
        self.explored = np.zeros((self._rows, self._cols), dtype=bool)

        # --- 三层栅格 ---
        # Layer 1: 持久层 (结构性障碍)
        self.persistent = np.zeros((self._rows, self._cols), dtype=np.uint8)
        # Layer 2: 短期层 (最近 K 帧)
        self._short_term_history: List[np.ndarray] = []
        self.short_term = np.zeros((self._rows, self._cols), dtype=np.uint8)
        # Layer 3: 当前帧
        self.current_frame = np.zeros((self._rows, self._cols), dtype=np.uint8)

        # --- 墙壁密度图 (用于房间分割, 仅累积高海拔障碍点) ---
        # 类似 HOV-SG: 只取 1.2m+ 的点投影为 2D 密度直方图,
        # 排除低矮家具 (桌椅沙发 < 1.2m), 只保留墙壁/高柜等结构
        self.wall_density = np.zeros((self._rows, self._cols), dtype=np.float32)

        # --- 全密度图 (HOV-SG xyz_full 等价物, 用于 outside_boundary) ---
        # 累积所有深度投影点 (不限高度), 用于确定建筑占地轮廓
        self.full_density = np.zeros((self._rows, self._cols), dtype=np.float32)

        # navmesh 确认的 free 掩码 (from_navmesh_fast 设置, 保护地面不被深度覆盖)
        self._navmesh_free: Optional[np.ndarray] = None

        # 膨胀核
        kernel_size = max(3, int(round(agent_radius * 2 / resolution)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self._dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._rows, self._cols)

    @property
    def free_mask(self) -> np.ndarray:
        """返回 M_free(x,y) 掩码, 1=可通行, 0=不可通行. 用于 GMM 概率场约束."""
        return (self.grid == 1).astype(np.uint8)

    @property
    def obstacle_mask(self) -> np.ndarray:
        """返回障碍物掩码, 1=障碍, 0=非障碍."""
        return (self.grid == 2).astype(np.uint8)

    # ------------------------------------------------------------------
    # 坐标转换
    # ------------------------------------------------------------------
    def world_to_grid(self, points_xz: np.ndarray) -> np.ndarray:
        """世界坐标 (x, z) -> 栅格坐标 (col, row).

        Args:
            points_xz: shape (..., 2), 世界坐标 [x, z]

        Returns:
            grid_coords: shape (..., 2), [col, row] (整数)
        """
        pts = np.asarray(points_xz, dtype=np.float64)
        relative = pts - self._origin
        col = np.round(relative[..., 0] / self.resolution).astype(int)
        row = np.round(relative[..., 1] / self.resolution).astype(int)
        return np.stack([col, row], axis=-1)

    def grid_to_world(self, grid_coords: np.ndarray) -> np.ndarray:
        """栅格坐标 (col, row) -> 世界坐标 (x, z)."""
        gc = np.asarray(grid_coords, dtype=np.float64)
        x = gc[..., 0] * self.resolution + self._origin[0]
        z = gc[..., 1] * self.resolution + self._origin[1]
        return np.stack([x, z], axis=-1)

    def _ensure_capacity(self, col: np.ndarray, row: np.ndarray):
        """动态扩展栅格以容纳新坐标 (增量模式时使用)."""
        max_col = int(np.max(col)) + 1 if len(col) > 0 else 0
        max_row = int(np.max(row)) + 1 if len(row) > 0 else 0
        min_col = int(np.min(col)) if len(col) > 0 else 0
        min_row = int(np.min(row)) if len(row) > 0 else 0

        if min_col < 0 or min_row < 0 or max_col > self._cols or max_row > self._rows:
            # 需要扩展 - 添加 padding
            pad = 50
            new_col_offset = max(0, -min_col) + pad
            new_row_offset = max(0, -min_row) + pad
            new_cols = max(self._cols + new_col_offset, max_col + pad) + new_col_offset
            new_rows = max(self._rows + new_row_offset, max_row + pad) + new_row_offset

            new_grid = np.zeros((new_rows, new_cols), dtype=np.uint8)
            new_explored = np.zeros((new_rows, new_cols), dtype=bool)
            new_persistent = np.zeros((new_rows, new_cols), dtype=np.uint8)
            new_short_term = np.zeros((new_rows, new_cols), dtype=np.uint8)
            new_current_frame = np.zeros((new_rows, new_cols), dtype=np.uint8)
            new_wall_density = np.zeros((new_rows, new_cols), dtype=np.float32)
            new_full_density = np.zeros((new_rows, new_cols), dtype=np.float32)
            new_navmesh_free = None
            if self._navmesh_free is not None:
                new_navmesh_free = np.zeros((new_rows, new_cols), dtype=bool)

            if self._rows > 0 and self._cols > 0:
                r_sl = slice(new_row_offset, new_row_offset + self._rows)
                c_sl = slice(new_col_offset, new_col_offset + self._cols)
                new_grid[r_sl, c_sl] = self.grid
                new_explored[r_sl, c_sl] = self.explored
                new_persistent[r_sl, c_sl] = self.persistent
                new_short_term[r_sl, c_sl] = self.short_term
                new_current_frame[r_sl, c_sl] = self.current_frame
                new_wall_density[r_sl, c_sl] = self.wall_density
                new_full_density[r_sl, c_sl] = self.full_density
                if self._navmesh_free is not None and new_navmesh_free is not None:
                    new_navmesh_free[r_sl, c_sl] = self._navmesh_free

            self.grid = new_grid
            self.explored = new_explored
            self.persistent = new_persistent
            self.short_term = new_short_term
            self.current_frame = new_current_frame
            self.wall_density = new_wall_density
            self.full_density = new_full_density
            self._navmesh_free = new_navmesh_free
            # 同步扩展 short_term 历史帧
            new_history = []
            for frame in self._short_term_history:
                new_frame = np.zeros((new_rows, new_cols), dtype=np.uint8)
                if self._rows > 0 and self._cols > 0:
                    new_frame[r_sl, c_sl] = frame
                new_history.append(new_frame)
            self._short_term_history = new_history
            self._origin -= np.array([new_col_offset * self.resolution,
                                       new_row_offset * self.resolution])
            self._rows = new_rows
            self._cols = new_cols

    # ------------------------------------------------------------------
    # 模式 A: 从 Habitat navmesh 生成
    # ------------------------------------------------------------------
    @classmethod
    def from_navmesh(
        cls,
        sim,
        resolution: float = 0.05,
        agent_radius: float = 0.18,
        padding: float = 1.0,
    ) -> "OccupancyGrid":
        """从 Habitat Simulator 的 navmesh 生成占据栅格.

        采样 navmesh 上的可导航点来构建自由空间，再用场景 AABB 确定边界。

        Args:
            sim: habitat_sim.Simulator 实例
            resolution: 栅格分辨率 (m/px)
            agent_radius: 机器人半径
            padding: 边界外扩 (m)
        """
        pathfinder = sim.pathfinder

        # 采样可导航点确定场景 xz 范围 (scene_aabb 有时返回 0)
        sample_pts = []
        for _ in range(10000):
            pt = pathfinder.get_random_navigable_point()
            if not np.isnan(pt[0]):
                sample_pts.append(pt)
        if not sample_pts:
            raise RuntimeError("无法从 navmesh 采样可导航点")
        sample_pts = np.array(sample_pts)
        sample_xz = sample_pts[:, [0, 2]]

        aabb_min = sample_xz.min(axis=0)
        aabb_max = sample_xz.max(axis=0)
        world_min = aabb_min - padding
        world_max = aabb_max + padding

        grid = cls(
            resolution=resolution,
            world_min=world_min,
            world_max=world_max,
            agent_radius=agent_radius,
        )

        # 密集采样可导航点
        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols):
                world_xz = grid.grid_to_world(np.array([c, r]))
                # 在 y 轴上取多个高度测试
                for y_probe in [0.0, 0.5, 1.0, 1.5, 2.0]:
                    pt_3d = np.array([world_xz[0], y_probe, world_xz[1]])
                    if pathfinder.is_navigable(pt_3d, max_y_delta=0.5):
                        grid.grid[r, c] = 1  # free
                        grid.explored[r, c] = True
                        break

        # 标记已知范围内非 free 为 occupied
        grid.grid[(grid.grid == 0) & grid.explored] = 2

        # 基于可通行结果标记所有 unknown 在场景边界内的为 occupied
        for r in range(rows):
            for c in range(cols):
                if grid.grid[r, c] == 0:
                    grid.grid[r, c] = 2
                    grid.explored[r, c] = True

        print(f"[OccupancyGrid.from_navmesh] shape={grid.shape}, "
              f"free={np.sum(grid.grid == 1)}, occupied={np.sum(grid.grid == 2)}")
        return grid

    @classmethod
    def from_navmesh_fast(
        cls,
        sim,
        resolution: float = 0.05,
        agent_radius: float = 0.18,
        padding: float = 1.0,
        num_samples: int = 50000,
    ) -> "OccupancyGrid":
        """从 navmesh 快速采样构建栅格 (不逐像素查询, 改用随机采样+内核填充).

        适用于大场景，速度更快。

        Args:
            sim: habitat_sim.Simulator
            resolution: m/px
            agent_radius: 机器人半径
            padding: 边界 padding
            num_samples: 随机采样可导航点数量
        """
        pathfinder = sim.pathfinder

        # 先采样一批可导航点来确定场景范围 (scene_aabb 有时返回 0)
        nav_points = []
        for _ in range(num_samples):
            pt = pathfinder.get_random_navigable_point()
            if not np.isnan(pt[0]):
                nav_points.append(pt)

        if not nav_points:
            print("[WARNING] 未采样到任何可导航点!")
            return cls(resolution=resolution, agent_radius=agent_radius)

        nav_points = np.array(nav_points)

        # 从采样点确定场景 xz 范围
        nav_xz = nav_points[:, [0, 2]]
        aabb_min = nav_xz.min(axis=0)
        aabb_max = nav_xz.max(axis=0)
        world_min = aabb_min - padding
        world_max = aabb_max + padding

        grid = cls(
            resolution=resolution,
            world_min=world_min,
            world_max=world_max,
            agent_radius=agent_radius,
        )

        # 转换到栅格坐标
        grid_coords = grid.world_to_grid(nav_xz)

        # 标记 free 像元
        valid_mask = (
            (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < grid._cols) &
            (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < grid._rows)
        )
        valid_coords = grid_coords[valid_mask]
        grid.grid[valid_coords[:, 1], valid_coords[:, 0]] = 1
        grid.explored[valid_coords[:, 1], valid_coords[:, 0]] = True

        # 对自由空间做 morphological closing，填补采样稀疏的间隙
        free_binary = (grid.grid == 1).astype(np.uint8)
        kernel_size = max(3, int(np.ceil(agent_radius / resolution)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        free_closed = cv2.morphologyEx(free_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        grid.grid[free_closed == 1] = 1

        # 获取最大连通自由区域 (去除采样噪声孤岛)
        free_binary = (grid.grid == 1).astype(np.uint8)
        n_components, labels, stats, _ = cv2.connectedComponentsWithStats(
            free_binary, connectivity=8
        )
        if n_components > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            main_label = np.argmax(areas) + 1
            # 小连通域 → 退回 UNKNOWN (不是 OCCUPIED，我们不知道那里有什么)
            grid.grid[(labels != main_label) & (grid.grid == 1)] = 0

        # 在 free 边界 dilate 一圈标记 OCCUPIED (墙壁/结构边界)
        free_binary = (grid.grid == 1).astype(np.uint8)
        wall_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size + 2, kernel_size + 2),
        )
        dilated_free = cv2.dilate(free_binary, wall_kernel, iterations=1)
        boundary = (dilated_free == 1) & (free_binary == 0)
        grid.grid[boundary] = 2  # 边界 → OCCUPIED

        # explored 只标记 free 区域及其边界，其余保持未探索
        grid.explored[free_binary == 1] = True
        grid.explored[boundary] = True
        # 不设 grid.explored[:] = True — 保留大量 UNKNOWN 供深度更新逐步填充

        # 保存 navmesh 确认的 free 掩码 — 深度更新时保护这些 cell
        # (上方墙壁/天花板的深度点投影到 XZ 平面会落在地面 free cell 上,
        #  不应覆盖 navmesh ground truth)
        grid._navmesh_free = free_binary.astype(bool).copy()

        print(f"[OccupancyGrid.from_navmesh_fast] shape={grid.shape}, "
              f"free={np.sum(grid.grid == 1)}, occupied={np.sum(grid.grid == 2)}, "
              f"unknown={np.sum(grid.grid == 0)}, sampled={len(nav_points)} points")
        return grid

    # ------------------------------------------------------------------
    # 模式 A2: 从 Agent 位置初始化 (无 navmesh, 纯深度增量)
    # ------------------------------------------------------------------
    @classmethod
    def from_agent_position(
        cls,
        agent_pos: np.ndarray,
        resolution: float = 0.05,
        agent_radius: float = 0.18,
        initial_extent: float = 15.0,
        **kwargs,
    ) -> "OccupancyGrid":
        """从 Agent 初始位置创建空白栅格, 后续由深度帧增量填充.

        不依赖 navmesh, 通过 _ensure_capacity 动态扩展.

        Args:
            agent_pos: Agent 世界坐标 (x, y, z) 或 (x, z)
            initial_extent: 初始栅格覆盖半径 (m), 默认 15m
        """
        if len(agent_pos) >= 3:
            ax, az = float(agent_pos[0]), float(agent_pos[2])
        else:
            ax, az = float(agent_pos[0]), float(agent_pos[1])

        world_min = np.array([ax - initial_extent, az - initial_extent])
        world_max = np.array([ax + initial_extent, az + initial_extent])

        grid = cls(
            resolution=resolution,
            world_min=world_min,
            world_max=world_max,
            agent_radius=agent_radius,
            **kwargs,
        )
        # 全部标记为 UNKNOWN = 0, 由深度帧逐步揭示
        # 在 agent 脚下标记一个小圆为 free (避免 A* 起点被封锁)
        gc = grid.world_to_grid(np.array([ax, az]))
        r_px = max(3, int(agent_radius * 2 / resolution))
        cv2.circle(grid.grid, (int(gc[0]), int(gc[1])), r_px, 1, -1)
        # explored 是 bool 数组, cv2.circle 需要 uint8; 用 grid.grid>0 替代
        explored_u8 = grid.explored.astype(np.uint8)
        cv2.circle(explored_u8, (int(gc[0]), int(gc[1])), r_px, 1, -1)
        grid.explored = explored_u8.astype(bool)

        print(f"[OccupancyGrid.from_agent_position] shape={grid.shape}, "
              f"center=({ax:.2f}, {az:.2f}), extent={initial_extent}m")
        return grid

    # ------------------------------------------------------------------
    # 模式 B: 从深度帧增量更新
    # ------------------------------------------------------------------
    def update_from_depth(
        self,
        depth_image: np.ndarray,
        camera_pose: np.ndarray,
        camera_intrinsics: Dict[str, float],
        depth_scale: float = 1000.0,
        min_height: float = 0.2,
        max_height: float = 1.5,
        max_depth: float = 5.0,
    ) -> None:
        """从单帧深度图更新栅格地图.

        Args:
            depth_image: HxW 深度图 (uint16 或 float)
            camera_pose: 4x4 相机到世界的变换矩阵
            camera_intrinsics: {"fx", "fy", "cx", "cy"}
            depth_scale: 深度值到米的缩放因子 (uint16 时常为 1000)
            min_height: 障碍物最小高度 (m)
            max_height: 障碍物最大高度 (m)
            max_depth: 最大有效深度 (m)
        """
        # 1. 深度图 -> 3D 点云 (相机坐标系)
        points_cam = self._depth_to_pointcloud(
            depth_image, camera_intrinsics, depth_scale, max_depth
        )
        if points_cam.shape[0] == 0:
            return

        # 2. 变换到世界坐标
        points_world = self._transform_points(points_cam, camera_pose)

        # 3. 高度过滤 (y 轴)
        # 地面层: y < min_height -> 标记为 free
        # 障碍物层: min_height <= y <= max_height -> 标记为 occupied
        floor_height = camera_pose[1, 3] - 1.5  # 估计地面高度

        y_vals = points_world[:, 1]
        relative_y = y_vals - floor_height

        free_mask = relative_y < min_height
        obstacle_mask = (relative_y >= min_height) & (relative_y <= max_height)

        # 4. 投影到 xz 平面并转换为栅格坐标
        free_xz = points_world[free_mask][:, [0, 2]]
        obstacle_xz = points_world[obstacle_mask][:, [0, 2]]

        if len(free_xz) > 0:
            free_gc = self.world_to_grid(free_xz)
            self._ensure_capacity(free_gc[:, 0], free_gc[:, 1])
            free_gc = self.world_to_grid(free_xz)  # 重新计算 (origin 可能已变)
            valid = (
                (free_gc[:, 0] >= 0) & (free_gc[:, 0] < self._cols) &
                (free_gc[:, 1] >= 0) & (free_gc[:, 1] < self._rows)
            )
            fc = free_gc[valid]
            # 只标记未知区域为 free（不覆盖已知 occupied）
            for i in range(fc.shape[0]):
                r, c = fc[i, 1], fc[i, 0]
                if self.grid[r, c] != 2:
                    self.grid[r, c] = 1
                self.explored[r, c] = True

        if len(obstacle_xz) > 0:
            obs_gc = self.world_to_grid(obstacle_xz)
            self._ensure_capacity(obs_gc[:, 0], obs_gc[:, 1])
            obs_gc = self.world_to_grid(obstacle_xz)
            valid = (
                (obs_gc[:, 0] >= 0) & (obs_gc[:, 0] < self._cols) &
                (obs_gc[:, 1] >= 0) & (obs_gc[:, 1] < self._rows)
            )
            oc = obs_gc[valid]
            self.grid[oc[:, 1], oc[:, 0]] = 2
            self.explored[oc[:, 1], oc[:, 0]] = True

    @staticmethod
    def _depth_to_pointcloud(
        depth: np.ndarray,
        intrinsics: Dict[str, float],
        depth_scale: float,
        max_depth: float,
    ) -> np.ndarray:
        """将深度图转换为 3D 点云 (相机坐标系).

        Pinhole 模型: X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy

        Returns:
            points: (N, 3) 相机坐标系点云 [X, Y, Z]
        """
        depth_f = depth.astype(np.float32) / depth_scale
        H, W = depth_f.shape[:2]

        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics.get("cx", W / 2.0)
        cy = intrinsics.get("cy", H / 2.0)

        # 创建像素坐标网格
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))

        # 有效深度掩码
        valid = (depth_f > 0) & (depth_f < max_depth)
        u_valid = u_grid[valid]
        v_valid = v_grid[valid]
        z_valid = depth_f[valid]

        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy

        return np.stack([x, y, z_valid], axis=-1)

    @staticmethod
    def _transform_points(points: np.ndarray, tf_matrix: np.ndarray) -> np.ndarray:
        """用 4x4 齐次变换矩阵变换点云."""
        N = points.shape[0]
        ones = np.ones((N, 1), dtype=np.float64)
        pts_homo = np.hstack([points.astype(np.float64), ones])
        transformed = (tf_matrix @ pts_homo.T).T
        return transformed[:, :3]

    # ------------------------------------------------------------------
    # 模式 C: 从 Habitat 深度传感器增量更新 (三层架构)
    # ------------------------------------------------------------------
    def update_from_habitat_depth(
        self,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        agent_pos: np.ndarray,
        agent_heading_deg: float,
        min_height: float = 0.15,
        max_height: float = 1.2,
        max_depth: float = 5.0,
    ) -> None:
        """从 Habitat 深度传感器的一帧更新三层栅格.

        流程 (参考 BeliefMapNav obstacle_map.update_map):
          1. depth → local pointcloud (camera frame)
          2. camera → world transform
          3. 高度过滤: free (地面) + obstacle
          4. 投影到栅格并更新三层

        Args:
            depth: (H, W) float32 深度图 (米, habitat 深度传感器直接返回)
            intrinsics: CameraIntrinsics 实例
            agent_pos: agent 世界坐标 [x, y, z]
            agent_heading_deg: agent 朝向角 (度)
            min_height: 障碍物最低高度
            max_height: 障碍物最高高度
            max_depth: 深度有效上限 (米)
        """
        # 1. 深度 → 相机坐标系点云
        pc_cam = depth_to_local_pointcloud(depth, intrinsics, max_depth)
        if len(pc_cam) == 0:
            return

        # 2. → 世界坐标系
        pc_world = camera_to_world(pc_cam, agent_pos, agent_heading_deg)

        # 3. 高度过滤
        free_pts, obs_pts = filter_points_by_height(
            pc_world, min_height, max_height, agent_y=agent_pos[1],
        )

        # 4. 投影到栅格 — free
        if len(free_pts) > 0:
            free_xz = free_pts[:, [0, 2]]
            free_gc = self.world_to_grid(free_xz)
            self._ensure_capacity(free_gc[:, 0], free_gc[:, 1])
            free_gc = self.world_to_grid(free_xz)
            valid = (
                (free_gc[:, 0] >= 0) & (free_gc[:, 0] < self._cols) &
                (free_gc[:, 1] >= 0) & (free_gc[:, 1] < self._rows)
            )
            fc = free_gc[valid]
            self.grid[fc[:, 1], fc[:, 0]] = np.where(
                self.grid[fc[:, 1], fc[:, 0]] != self.OCCUPIED,
                self.FREE,
                self.grid[fc[:, 1], fc[:, 0]],
            )
            self.explored[fc[:, 1], fc[:, 0]] = True

        # 5. 投影到栅格 — obstacle (当前帧 + 短期层 + 主栅格)
        frame_mask = np.zeros((self._rows, self._cols), dtype=np.uint8)
        if len(obs_pts) > 0:
            obs_xz = obs_pts[:, [0, 2]]
            obs_gc = self.world_to_grid(obs_xz)
            self._ensure_capacity(obs_gc[:, 0], obs_gc[:, 1])
            obs_gc = self.world_to_grid(obs_xz)
            valid = (
                (obs_gc[:, 0] >= 0) & (obs_gc[:, 0] < self._cols) &
                (obs_gc[:, 1] >= 0) & (obs_gc[:, 1] < self._rows)
            )
            oc = obs_gc[valid]
            # 主栅格: UNKNOWN → OCCUPIED
            # navmesh 模式下保护 navmesh 确认的 FREE (上方墙壁/天花板投影不应覆盖地面)
            if self._navmesh_free is not None:
                protect = self._navmesh_free[oc[:, 1], oc[:, 0]]
                oc_unprotected = oc[~protect]
                if len(oc_unprotected) > 0:
                    self.grid[oc_unprotected[:, 1], oc_unprotected[:, 0]] = self.OCCUPIED
            else:
                self.grid[oc[:, 1], oc[:, 0]] = self.OCCUPIED
            self.explored[oc[:, 1], oc[:, 0]] = True
            frame_mask[oc[:, 1], oc[:, 0]] = 1

        # --- Layer 3: 当前帧 ---
        self.current_frame = frame_mask

        # --- Layer 2: 短期层累积 ---
        self._short_term_history.append(frame_mask)
        if len(self._short_term_history) > self.short_term_frames:
            self._short_term_history.pop(0)
        self.short_term = np.zeros((self._rows, self._cols), dtype=np.uint8)
        for fr in self._short_term_history:
            if fr.shape == self.short_term.shape:
                np.maximum(self.short_term, fr, out=self.short_term)

        # --- 更新已探索区域 (fog-of-war 射线追踪) ---
        self._ray_mark_fog_of_war(agent_pos, pc_world)

        # --- 更新墙壁密度图 (仅高海拔点, 用于房间分割) ---
        # y_rel 是相机坐标系高度差: 相机上方 0.15m+ ≈ 离地 ~1.65m+, 排除低矮家具
        wall_min_h = min_height  # 0.15m — 捕获墙壁结构 (之前误用 max_height=1.2m 导致过滤过严)
        wall_max_h = 3.0  # 天花板上限
        y_rel = pc_world[:, 1] - agent_pos[1]
        wall_mask = (y_rel >= wall_min_h) & (y_rel < wall_max_h)
        wall_pts = pc_world[wall_mask]
        if len(wall_pts) > 0:
            wall_xz = wall_pts[:, [0, 2]]
            wall_gc = self.world_to_grid(wall_xz)
            valid_w = (
                (wall_gc[:, 0] >= 0) & (wall_gc[:, 0] < self._cols) &
                (wall_gc[:, 1] >= 0) & (wall_gc[:, 1] < self._rows)
            )
            wc = wall_gc[valid_w]
            np.add.at(self.wall_density, (wc[:, 1], wc[:, 0]), 1.0)

        # --- 更新全密度图 (所有深度点, 用于 outside_boundary) ---
        # 等价于 HOV-SG 的 xyz_full → histogram2d (不限高度)
        all_xz = pc_world[:, [0, 2]]
        all_gc = self.world_to_grid(all_xz)
        valid_a = (
            (all_gc[:, 0] >= 0) & (all_gc[:, 0] < self._cols) &
            (all_gc[:, 1] >= 0) & (all_gc[:, 1] < self._rows)
        )
        ac = all_gc[valid_a]
        if len(ac) > 0:
            np.add.at(self.full_density, (ac[:, 1], ac[:, 0]), 1.0)

    def _update_explored_circle(self, agent_pos: np.ndarray, radius_m: float):
        """将 agent 周围圆形区域标记为已探索 (保留兼容, navmesh 模式仍可用)."""
        xz = np.array([agent_pos[0], agent_pos[2]])
        gc = self.world_to_grid(xz)
        col, row = int(gc[0]), int(gc[1])
        r_px = int(round(radius_m / self.resolution))
        if 0 <= col < self._cols and 0 <= row < self._rows:
            cv2.circle(
                self.explored.view(np.uint8),
                (col, row), r_px, 1, -1,
            )

    def _ray_mark_fog_of_war(
        self,
        agent_pos: np.ndarray,
        pc_world: np.ndarray,
        n_angular_bins: int = 120,
    ):
        """基于深度扇形扫描的 fog-of-war 探索标记 (参考 HOV-SG occupancy 构建).

        方法: 将深度点按水平角度分箱, 每个方向取最近障碍点距离作为
        可见范围, 用 cv2.fillPoly 填充扇形区域, 高效标记所有可见格子.

        关键改进 (vs 旧 Bresenham 射线版):
          - 扇形填充而非稀疏射线 → 无间隙
          - 取最近障碍 (而非最远) → 不穿墙
          - 非障碍方向用 max_range 深度 fallback
        """
        if len(pc_world) < 10:
            self._update_explored_circle(agent_pos, self.agent_radius * 3)
            return

        agent_xz = np.array([agent_pos[0], agent_pos[2]])
        obs_pts = pc_world[:, [0, 2]]

        dx = obs_pts[:, 0] - agent_xz[0]
        dz = obs_pts[:, 1] - agent_xz[1]
        angles = np.arctan2(dz, dx)
        dists = np.sqrt(dx ** 2 + dz ** 2)

        bin_edges = np.linspace(-np.pi, np.pi, n_angular_bins + 1)
        bin_idx = np.digitize(angles, bin_edges) - 1
        np.clip(bin_idx, 0, n_angular_bins - 1, out=bin_idx)

        # 每个角度方向取最远可见深度点距离 (所有点, 而非仅障碍)
        max_range = np.max(dists) if len(dists) > 0 else 3.0
        bin_range = np.full(n_angular_bins, 1.0)  # 默认 1m (无点的方向)
        for b in range(n_angular_bins):
            mask = bin_idx == b
            if np.any(mask):
                bin_range[b] = np.max(dists[mask])

        # 构建扇形多边形 (在栅格坐标系)
        agent_gc = self.world_to_grid(agent_xz)
        ac, ar = int(agent_gc[0]), int(agent_gc[1])
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        polygon_pts = [(ac, ar)]  # 从 agent 位置开始
        for b in range(n_angular_bins):
            r_m = bin_range[b]
            ang = bin_centers[b]
            ex = agent_xz[0] + r_m * np.cos(ang)
            ez = agent_xz[1] + r_m * np.sin(ang)
            egc = self.world_to_grid(np.array([ex, ez]))
            polygon_pts.append((int(egc[0]), int(egc[1])))
        # 闭合
        polygon_pts.append(polygon_pts[1])

        pts_arr = np.array(polygon_pts, dtype=np.int32).reshape(-1, 1, 2)

        # 用 fillPoly 标记 explored
        explored_u8 = self.explored.view(np.uint8)
        cv2.fillPoly(explored_u8, [pts_arr], 1)

        # 同时标记 UNKNOWN → FREE (在多边形区域内)
        fow_mask = np.zeros((self._rows, self._cols), dtype=np.uint8)
        cv2.fillPoly(fow_mask, [pts_arr], 1)
        update_cells = (fow_mask > 0) & (self.grid == self.UNKNOWN)
        self.grid[update_cells] = self.FREE

        # agent 脚下小范围保证 FREE
        r_px = max(2, int(self.agent_radius * 2 / self.resolution))
        if 0 <= ac < self._cols and 0 <= ar < self._rows:
            foot_mask = np.zeros((self._rows, self._cols), dtype=np.uint8)
            cv2.circle(foot_mask, (ac, ar), r_px, 1, -1)
            self.grid[(foot_mask > 0) & (self.grid == self.UNKNOWN)] = self.FREE

    def _bresenham_explore(self, c0: int, r0: int, c1: int, r1: int):
        """Bresenham 线追踪: 标记沿线栅格为 explored, UNKNOWN→FREE."""
        dc = abs(c1 - c0)
        dr = abs(r1 - r0)
        sc = 1 if c0 < c1 else -1
        sr = 1 if r0 < r1 else -1
        err = dc - dr
        c, r = c0, r0

        while True:
            if 0 <= r < self._rows and 0 <= c < self._cols:
                self.explored[r, c] = True
                if self.grid[r, c] == self.UNKNOWN:
                    self.grid[r, c] = self.FREE
            else:
                break
            if c == c1 and r == r1:
                break
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr

    def update_persistent_from_objects(self, objects: list):
        """根据语义地图物体更新持久层.

        规则:
          - stability >= threshold → 持久障碍
          - 此方法应在地图合并后调用
        """
        for obj in objects:
            stability = getattr(obj, 'stability', 0.5)
            if stability >= self.stability_threshold:
                pos_2d = getattr(obj, 'pos_2d', None)
                if pos_2d and 'x' in pos_2d and 'y' in pos_2d:
                    xz = np.array([pos_2d['x'], pos_2d['y']])
                    gc = self.world_to_grid(xz)
                    col, row = int(gc[0]), int(gc[1])
                    if 0 <= col < self._cols and 0 <= row < self._rows:
                        # 用物体近似半径标记
                        r_px = max(1, int(0.3 / self.resolution))
                        cv2.circle(self.persistent, (col, row), r_px, self.OCCUPIED, -1)

    def init_persistent_from_navmesh(self):
        """将当前主栅格的 occupied 区域复制到持久层 (navmesh 初始化后调用)."""
        self.persistent = np.where(
            self.grid == self.OCCUPIED, self.OCCUPIED, self.UNKNOWN
        ).astype(np.uint8)

    # ------------------------------------------------------------------
    # 三层合并 + 导航查询
    # ------------------------------------------------------------------
    def get_merged_obstacle_mask(self) -> np.ndarray:
        """合并三层得到完整障碍物掩码.

        Returns:
            (rows, cols) uint8, 1=obstacle, 0=free/unknown
        """
        merged = np.zeros((self._rows, self._cols), dtype=np.uint8)
        merged = np.maximum(merged, (self.persistent == self.OCCUPIED).astype(np.uint8))
        merged = np.maximum(merged, self.short_term)
        merged = np.maximum(merged, self.current_frame)
        return merged

    def get_navigable_map(self, layer: str = "merged") -> np.ndarray:
        """获取膨胀后的可导航地图 (兼容旧接口 + 新三层).

        Args:
            layer: "persistent" | "short_term" | "current" | "merged" | "legacy"

        Returns:
            (rows, cols) uint8, 1=可通行, 0=不可通行
        """
        if layer == "legacy":
            return self.dilate_obstacles()

        if layer == "persistent":
            occ = (self.persistent == self.OCCUPIED).astype(np.uint8)
        elif layer == "short_term":
            occ = self.short_term.copy()
        elif layer == "current":
            occ = self.current_frame.copy()
        else:  # merged
            occ = self.get_merged_obstacle_mask()

        dilated = cv2.dilate(occ, self._dilate_kernel, iterations=1)
        navigable = ((dilated == 0) & self.explored).astype(np.uint8)
        return navigable

    # ------------------------------------------------------------------
    # Frontier 检测
    # ------------------------------------------------------------------
    def get_frontiers(self, min_area_m2: float = 0.5) -> np.ndarray:
        """检测 frontier: 已探索可通行区域与未探索区域的边界.

        参考 BeliefMapNav obstacle_map._get_frontiers

        Args:
            min_area_m2: 最小 frontier 区域面积 (m²)

        Returns:
            (N, 2) 世界坐标 [[wx, wz], ...]
        """
        nav = self.get_navigable_map("merged")
        explored_u8 = self.explored.astype(np.uint8)
        nav_and_explored = (nav > 0) & self.explored

        # frontier = 已探索且可通行 的邻域中有未探索的
        explored_dilated = cv2.dilate(explored_u8, np.ones((3, 3), np.uint8), iterations=2)
        frontier_mask = (explored_dilated > 0) & (~self.explored)
        # 还需要邻接可通行区域 (避免在障碍物外围产生虚假 frontier)
        nav_dilated = cv2.dilate(nav_and_explored.astype(np.uint8), np.ones((3, 3), np.uint8))
        frontier_mask = frontier_mask & (nav_dilated > 0)

        if not np.any(frontier_mask):
            return np.array([])

        frontier_u8 = frontier_mask.astype(np.uint8) * 255
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier_u8, connectivity=8,
        )

        min_area_px = int(min_area_m2 / (self.resolution ** 2))
        results = []
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area_px:
                continue
            c_col, c_row = centroids[i]
            world_xz = self.grid_to_world(np.array([int(c_col), int(c_row)]))
            results.append([float(world_xz[0]), float(world_xz[1])])

        return np.array(results) if results else np.array([])

    # ------------------------------------------------------------------
    # 实时避障
    # ------------------------------------------------------------------
    def check_collision_ahead(
        self,
        agent_pos: np.ndarray,
        heading_deg: float,
        check_distance: float = 1.0,
        n_samples: int = 5,
    ) -> bool:
        """检查前方是否有障碍物 (实时避障, 基于 current_frame).

        Returns:
            True = 前方碰撞, 应停止/绕行
        """
        theta = math.radians(heading_deg)
        sin_t, cos_t = math.sin(theta), math.cos(theta)

        # 合并当前帧 + 短期层判断
        occ = np.maximum(self.current_frame, self.short_term)
        dilated = cv2.dilate(occ, self._dilate_kernel, iterations=1)

        for i in range(1, n_samples + 1):
            d = check_distance * i / n_samples
            wx = agent_pos[0] + d * sin_t
            wz = agent_pos[2] + d * cos_t
            xz = np.array([wx, wz])
            gc = self.world_to_grid(xz)
            col, row = int(gc[0]), int(gc[1])
            if 0 <= col < self._cols and 0 <= row < self._rows:
                if dilated[row, col] > 0:
                    return True
            else:
                return True
        return False

    def is_navigable_at(self, wx: float, wz: float) -> bool:
        """查询某世界坐标点是否可通行 (merged 层)."""
        xz = np.array([wx, wz])
        gc = self.world_to_grid(xz)
        col, row = int(gc[0]), int(gc[1])
        if 0 <= col < self._cols and 0 <= row < self._rows:
            nav = self.get_navigable_map("merged")
            return bool(nav[row, col] > 0)
        return False

    # ------------------------------------------------------------------
    # 后处理
    # ------------------------------------------------------------------
    def dilate_obstacles(self, extra_radius: Optional[float] = None) -> np.ndarray:
        """膨胀障碍物 (用于导航安全边距), 返回膨胀后的 free_mask.

        Args:
            extra_radius: 额外膨胀半径 (m), 默认使用 agent_radius
        """
        radius = extra_radius if extra_radius is not None else self.agent_radius
        kernel_size = max(3, int(np.ceil(radius / self.resolution) * 2 + 1))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        obs = self.obstacle_mask
        dilated_obs = cv2.dilate(obs, kernel, iterations=1)
        # 膨胀后的可通行区域
        navigable = ((self.grid == 1) & (dilated_obs == 0)).astype(np.uint8)
        return navigable

    def get_navigable_mask(self) -> np.ndarray:
        """获取考虑机器人半径的可导航掩码 M_free."""
        return self.dilate_obstacles()

    # ------------------------------------------------------------------
    # 可视化 & 保存
    # ------------------------------------------------------------------
    def to_image(self, navigable: bool = True, show_layers: bool = False) -> np.ndarray:
        """生成可视化 BGR 图像.

        颜色定义:
          - 白色: 可通行 (free)
          - 黑色: 障碍物 (occupied)
          - 灰色: 未探索 (unknown)
          - 绿色: 可导航 (膨胀后仍可通行)
          - 深蓝: 持久层障碍
          - 橙色: 短期层障碍
          - 红色: 当前帧障碍
        """
        img = np.full((self._rows, self._cols, 3), 128, dtype=np.uint8)  # 灰色=未知
        img[self.grid == 1] = [255, 255, 255]  # 白色=free
        img[self.grid == 2] = [0, 0, 0]  # 黑色=occupied

        if navigable:
            nav_mask = self.get_navigable_mask()
            img[nav_mask == 1] = [200, 255, 200]  # 浅绿=navigable

        if show_layers:
            # 覆盖显示三层
            img[self.persistent == self.OCCUPIED] = [100, 50, 20]  # 深蓝
            img[self.short_term > 0] = [0, 140, 255]  # 橙色
            img[self.current_frame > 0] = [0, 0, 255]  # 红色

        return img

    def save(self, output_dir: str, prefix: str = "occupancy") -> Dict[str, str]:
        """保存栅格地图及元数据.

        保存文件:
          - {prefix}_grid.npy: 栅格数据
          - {prefix}_meta.npz: 元数据 (origin, resolution, shape)
          - {prefix}_vis.png: 可视化图像
          - {prefix}_free_mask.png: M_free 掩码 (二值)
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        grid_path = os.path.join(output_dir, f"{prefix}_grid.npy")
        np.save(grid_path, self.grid)
        paths["grid"] = grid_path

        meta_path = os.path.join(output_dir, f"{prefix}_meta.npz")
        np.savez(
            meta_path,
            origin=self._origin,
            resolution=np.array([self.resolution]),
            shape=np.array(self.shape),
            agent_radius=np.array([self.agent_radius]),
        )
        paths["meta"] = meta_path

        vis_path = os.path.join(output_dir, f"{prefix}_vis.png")
        cv2.imwrite(vis_path, self.to_image())
        paths["vis"] = vis_path

        free_path = os.path.join(output_dir, f"{prefix}_free_mask.png")
        cv2.imwrite(free_path, self.free_mask * 255)
        paths["free_mask"] = free_path

        nav_path = os.path.join(output_dir, f"{prefix}_navigable_mask.png")
        cv2.imwrite(nav_path, self.get_navigable_mask() * 255)
        paths["navigable_mask"] = nav_path

        print(f"[OccupancyGrid.save] 已保存到 {output_dir}/")
        return paths

    @classmethod
    def load(cls, output_dir: str, prefix: str = "occupancy") -> "OccupancyGrid":
        """从保存文件加载栅格地图."""
        meta = np.load(os.path.join(output_dir, f"{prefix}_meta.npz"))
        grid_data = np.load(os.path.join(output_dir, f"{prefix}_grid.npy"))

        origin = meta["origin"]
        resolution = float(meta["resolution"][0])
        agent_radius = float(meta["agent_radius"][0])
        rows, cols = int(meta["shape"][0]), int(meta["shape"][1])

        grid = cls(
            resolution=resolution,
            world_min=origin,
            grid_size=(rows, cols),
            agent_radius=agent_radius,
        )
        grid.grid = grid_data
        grid.explored = (grid_data > 0)
        return grid

    # ------------------------------------------------------------------
    # 物体定位辅助
    # ------------------------------------------------------------------
    def plot_objects_on_grid(
        self,
        objects_pos_2d: List[Dict[str, float]],
        labels: Optional[List[str]] = None,
    ) -> np.ndarray:
        """在栅格地图上标注物体位置, 返回 BGR 图像."""
        img = self.to_image(navigable=True)
        for i, pos in enumerate(objects_pos_2d):
            xz = np.array([pos["x"], pos["y"]])
            gc = self.world_to_grid(xz)
            col, row = int(gc[0]), int(gc[1])
            if 0 <= col < self._cols and 0 <= row < self._rows:
                cv2.circle(img, (col, row), 3, (0, 0, 255), -1)
                if labels and i < len(labels):
                    cv2.putText(
                        img, labels[i], (col + 5, row - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1,
                    )
        return img
