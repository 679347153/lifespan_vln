"""基于占据栅格的 A* 路径规划器.

功能:
  1. 在 OccupancyGrid 上执行 A* 搜索 (8 连通)
  2. 障碍物惩罚: 靠近障碍物的路径代价更高 (scipy EDT)
  3. 路径平滑: Bresenham 视线简化冗余路径点
  4. 被占据起/终点 BFS 寻找最近可通行点
  5. 返回世界坐标路径点, 可直接送入导航循环

参考:
  - DovSG/dovsg/navigation/astar_planner.py (A* + obstacle punishment + Bresenham)
  - osmAG-LLM/HM3DSEM_navigation/Astar.py (EDT cost map + NetworkX)

坐标约定:
  Grid: (row, col), row ↔ world-z, col ↔ world-x
  World: (x, z), x=col*res+origin_x, z=row*res+origin_z
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


# ------------------------------------------------------------------
# 8-连通邻居
# ------------------------------------------------------------------
_DIRS_8 = [(-1, -1), (-1, 0), (-1, 1),
           (0, -1),           (0, 1),
           (1, -1),  (1, 0),  (1, 1)]

_SQRT2 = math.sqrt(2.0)


def _move_cost(dr: int, dc: int) -> float:
    """8-连通移动代价: 对角 √2, 四邻 1."""
    return _SQRT2 if (dr != 0 and dc != 0) else 1.0


# ------------------------------------------------------------------
# A* 核心
# ------------------------------------------------------------------
class GridAStarPlanner:
    """基于占据栅格的 A* 路径规划器.

    用法:
        planner = GridAStarPlanner(occ_grid)
        waypoints = planner.plan(start_xz, goal_xz, step_size=0.5)
    """

    def __init__(
        self,
        occ_grid,  # OccupancyGrid instance
        obstacle_weight: float = 5.0,
        obstacle_decay_cells: int = 6,
        use_navmesh_grid: bool = True,
        unknown_cost: float = 0.5,
    ):
        """
        Args:
            occ_grid: OccupancyGrid 实例
            obstacle_weight: 靠近障碍物的额外代价权重
            obstacle_decay_cells: 障碍物惩罚衰减距离 (栅格单元数)
            use_navmesh_grid: True=用 navmesh grid (FREE/OCCUPIED) 做A*规划;
                              False=用合并三层 (merged) 做规划.
                              navmesh 模式更稳定, 因为深度障碍不会阻断已知走廊.
            unknown_cost: UNKNOWN 格子的额外通行代价 (仅 non-navmesh 模式).
                          >0 允许 A* 穿越未探索区域 (轻微惩罚), 0=不惩罚.
        """
        self.occ_grid = occ_grid
        self.obstacle_weight = obstacle_weight
        self.obstacle_decay_cells = obstacle_decay_cells
        self.use_navmesh_grid = use_navmesh_grid
        self.unknown_cost = unknown_cost
        self._cost_map: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 代价地图 (EDT)
    # ------------------------------------------------------------------
    def build_cost_map(self) -> np.ndarray:
        """基于障碍物距离变换构建代价地图.

        代价 = obstacle_weight / max(distance_to_obstacle, 1)
        距离超过 obstacle_decay_cells 的区域代价为 0.
        """
        occ_mask = self._get_occ_mask()  # 1=occ, 0=free
        # EDT: 计算每个 free 格到最近障碍物的距离
        free_mask = (occ_mask == 0)
        if not np.any(free_mask):
            self._cost_map = np.full_like(occ_mask, dtype=np.float32, fill_value=self.obstacle_weight)
            return self._cost_map

        dist = distance_transform_edt(free_mask).astype(np.float32)
        # 距离截断
        dist = np.clip(dist, 0, self.obstacle_decay_cells)
        # 代价: 越近障碍物越大 (避免除零)
        safe_dist = np.maximum(dist, 1e-6)
        cost = np.where(
            dist > 0,
            self.obstacle_weight / safe_dist,
            self.obstacle_weight,  # 障碍物本身
        )
        # 超过衰减距离的区域代价归零
        cost[dist >= self.obstacle_decay_cells] = 0.0
        self._cost_map = cost
        return self._cost_map

    def _get_cost_map(self) -> np.ndarray:
        if self._cost_map is None:
            self.build_cost_map()
        return self._cost_map

    # ------------------------------------------------------------------
    # 起/终点修正
    # ------------------------------------------------------------------
    def _get_occ_mask(self) -> np.ndarray:
        """缓存的障碍物掩码 (与 cost_map 同步失效).

        navmesh模式: 来自 grid==OCCUPIED (navmesh 确定的墙壁/结构)
        merged模式:  来自三层 OR (含深度检测的临时障碍)
        """
        if not hasattr(self, '_occ_mask_cache') or self._occ_mask_cache is None:
            if self.use_navmesh_grid:
                self._occ_mask_cache = (self.occ_grid.grid == self.occ_grid.OCCUPIED).astype(np.uint8)
            else:
                self._occ_mask_cache = self.occ_grid.get_merged_obstacle_mask()
        return self._occ_mask_cache

    def invalidate_cost_map(self):
        """栅格更新后应调用此方法使代价地图失效."""
        self._cost_map = None
        self._occ_mask_cache = None

    def _is_occupied(self, row: int, col: int) -> bool:
        rows, cols = self.occ_grid.shape
        if row < 0 or col < 0 or row >= rows or col >= cols:
            return True
        occ = self._get_occ_mask()
        return bool(occ[row, col])

    def _is_passable(self, row: int, col: int) -> bool:
        """格子是否可通行 (A* 用).

        navmesh 模式: 只允许 FREE (排除 UNKNOWN 和 OCCUPIED).
        non-navmesh 模式: 允许 FREE 和 UNKNOWN (用 unknown_cost 惩罚), 只排除 OCCUPIED.
        """
        rows, cols = self.occ_grid.shape
        if row < 0 or col < 0 or row >= rows or col >= cols:
            return False
        cell = self.occ_grid.grid[row, col]
        if cell == self.occ_grid.OCCUPIED:
            return False
        if self.use_navmesh_grid:
            return cell == self.occ_grid.FREE
        # non-navmesh: FREE + UNKNOWN 都可通行
        return True

    def _nearest_free_cell(self, row: int, col: int, max_radius: int = 50) -> Optional[Tuple[int, int]]:
        """BFS 寻找最近可通行格.

        Args:
            max_radius: 最大搜索半径 (栅格单元)

        Returns:
            (row, col) or None
        """
        if self._is_passable(row, col):
            return (row, col)

        rows, cols = self.occ_grid.shape
        visited = set()
        queue = [(row, col)]
        visited.add((row, col))

        while queue:
            cr, cc = queue.pop(0)
            for dr, dc in _DIRS_8:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in visited:
                    continue
                if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                    continue
                # 距离限制
                if abs(nr - row) > max_radius or abs(nc - col) > max_radius:
                    continue
                visited.add((nr, nc))
                if self._is_passable(nr, nc):
                    return (nr, nc)
                queue.append((nr, nc))

        return None

    # ------------------------------------------------------------------
    # A* 搜索
    # ------------------------------------------------------------------
    def _astar_grid(
        self, start_rc: Tuple[int, int], goal_rc: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """在栅格上执行 A* (8-连通).

        Args:
            start_rc: (row, col)
            goal_rc: (row, col)

        Returns:
            路径 [(row, col), ...] 从 start 到 goal, 或 None 如果不可达
        """
        rows, cols = self.occ_grid.shape
        cost_map = self._get_cost_map()

        sr, sc = start_rc
        gr, gc = goal_rc

        # 启发式: 八邻域 Octile distance
        def heuristic(r, c):
            dr = abs(r - gr)
            dc = abs(c - gc)
            return max(dr, dc) + (_SQRT2 - 1) * min(dr, dc)

        # min-heap: (f, g, row, col)
        open_heap = [(heuristic(sr, sc), 0.0, sr, sc)]
        came_from = {}
        g_score = {(sr, sc): 0.0}

        while open_heap:
            f, g, r, c = heapq.heappop(open_heap)

            if (r, c) == (gr, gc):
                # 重建路径
                path = [(gr, gc)]
                cur = (gr, gc)
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                path.reverse()
                return path

            # 已经有更好的路径到此节点
            if g > g_score.get((r, c), float('inf')):
                continue

            for dr, dc in _DIRS_8:
                nr, nc = r + dr, c + dc
                if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                    continue
                if not self._is_passable(nr, nc):
                    continue
                move_g = _move_cost(dr, dc)
                # 额外代价: EDT 惩罚
                extra = cost_map[nr, nc] if cost_map is not None else 0.0
                # UNKNOWN 格子额外代价 (鼓励走已知路, 但不禁止穿越)
                if (not self.use_navmesh_grid
                        and self.occ_grid.grid[nr, nc] == self.occ_grid.UNKNOWN):
                    extra += self.unknown_cost
                new_g = g + move_g + extra

                if new_g < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = new_g
                    f_new = new_g + heuristic(nr, nc)
                    heapq.heappush(open_heap, (f_new, new_g, nr, nc))
                    came_from[(nr, nc)] = (r, c)

        return None  # 不可达

    # ------------------------------------------------------------------
    # 路径平滑 (Bresenham 视线检查)
    # ------------------------------------------------------------------
    def _line_of_sight(self, r0: int, c0: int, r1: int, c1: int) -> bool:
        """Bresenham 视线检查: 两点之间是否可通行."""
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc

        r, c = r0, c0
        while True:
            if not self._is_passable(r, c):
                return False
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        return True

    def _smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """去除视线可达的中间路径点.

        贪心: 从当前点向后找最远的视线可达点, 跳到那里.
        """
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            # 从后向前找最远可达
            best_j = i + 1
            for j in range(len(path) - 1, i, -1):
                if self._line_of_sight(path[i][0], path[i][1], path[j][0], path[j][1]):
                    best_j = j
                    break
            smoothed.append(path[best_j])
            i = best_j

        return smoothed

    # ------------------------------------------------------------------
    # 公开接口: plan
    # ------------------------------------------------------------------
    def plan(
        self,
        start_xz: Tuple[float, float],
        goal_xz: Tuple[float, float],
        step_size: float = 0.5,
        smooth: bool = True,
    ) -> Optional[List[Tuple[float, float]]]:
        """从世界坐标 (x, z) 规划路径.

        Args:
            start_xz: 起点 (world_x, world_z)
            goal_xz: 终点 (world_x, world_z)
            step_size: 返回路径点间距 (米), 0 表示不重采样
            smooth: 是否执行 Bresenham 路径平滑

        Returns:
            世界坐标路径点 [(x, z), ...], 包含起点和终点. 不可达返回 None.
        """
        # 世界坐标 → 栅格坐标 (col, row)
        start_cr = self.occ_grid.world_to_grid(np.array(start_xz))  # [col, row]
        goal_cr = self.occ_grid.world_to_grid(np.array(goal_xz))

        start_rc = (int(start_cr[1]), int(start_cr[0]))  # (row, col)
        goal_rc = (int(goal_cr[1]), int(goal_cr[0]))

        # 修正被占据的起/终点
        start_rc = self._nearest_free_cell(*start_rc, max_radius=80)
        if start_rc is None:
            print(f"  [A* plan] 起点附近无可通行格")
            return None
        goal_rc = self._nearest_free_cell(*goal_rc, max_radius=80)
        if goal_rc is None:
            print(f"  [A* plan] 终点附近无可通行格")
            return None

        # A* 搜索
        grid_path = self._astar_grid(start_rc, goal_rc)
        if grid_path is None:
            dist = math.sqrt((start_rc[0]-goal_rc[0])**2 + (start_rc[1]-goal_rc[1])**2)
            print(f"  [A* plan] 无路径: start_rc={start_rc}, goal_rc={goal_rc}, "
                  f"dist={dist:.0f}cells, grid={self.occ_grid.shape}")
            return None

        # 路径平滑
        if smooth and len(grid_path) > 2:
            grid_path = self._smooth_path(grid_path)

        # 栅格 → 世界坐标
        world_path = []
        for r, c in grid_path:
            xz = self.occ_grid.grid_to_world(np.array([c, r]))  # [col, row] → (x, z)
            world_path.append((float(xz[0]), float(xz[1])))

        # 强制起终点精确
        world_path[0] = (start_xz[0], start_xz[1])
        if len(world_path) > 1:
            world_path[-1] = (goal_xz[0], goal_xz[1])

        # 按 step_size 重采样
        if step_size > 0 and len(world_path) > 1:
            world_path = self._resample_path(world_path, step_size)

        return world_path

    def plan_3d(
        self,
        start_xyz: List[float],
        goal_xyz: List[float],
        step_size: float = 0.5,
        smooth: bool = True,
    ) -> Optional[List[List[float]]]:
        """3D 接口: 输入/输出 [x, y, z], y 保持起点高度.

        方便直接替换 get_path_waypoints 的返回格式.
        """
        start_xz = (start_xyz[0], start_xyz[2])
        goal_xz = (goal_xyz[0], goal_xyz[2])
        path_2d = self.plan(start_xz, goal_xz, step_size=step_size, smooth=smooth)
        if path_2d is None:
            return None
        y = start_xyz[1]  # 保持高度
        return [[x, y, z] for x, z in path_2d]

    # ------------------------------------------------------------------
    # 重采样
    # ------------------------------------------------------------------
    @staticmethod
    def _resample_path(
        path: List[Tuple[float, float]], step_size: float,
    ) -> List[Tuple[float, float]]:
        """按等距 step_size 重采样路径, 保留起终点."""
        if len(path) < 2:
            return path

        resampled = [path[0]]
        accum = 0.0

        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dz = path[i][1] - path[i - 1][1]
            seg_len = math.sqrt(dx * dx + dz * dz)
            if seg_len < 1e-6:
                continue

            remaining = seg_len
            prev = path[i - 1]

            while accum + remaining >= step_size:
                frac = (step_size - accum) / seg_len
                # 插值
                nx = prev[0] + frac * dx
                nz = prev[1] + frac * dz
                resampled.append((nx, nz))
                # 更新基准
                remaining -= (step_size - accum)
                accum = 0.0
                prev = (nx, nz)
                # 重新计算增量
                dx = path[i][0] - prev[0]
                dz = path[i][1] - prev[1]
                seg_len = math.sqrt(dx * dx + dz * dz)
                if seg_len < 1e-6:
                    break
            else:
                accum += remaining

        # 始终包含终点
        end = path[-1]
        if len(resampled) < 2 or (
            math.sqrt((resampled[-1][0] - end[0]) ** 2 + (resampled[-1][1] - end[1]) ** 2) > 0.05
        ):
            resampled.append(end)

        return resampled

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------
    def visualize_path(
        self,
        path_xz: List[Tuple[float, float]],
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """在栅格图上绘制路径.

        Returns:
            BGR image
        """
        # 基础图
        img = self.occ_grid.to_image(navigable=True)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 绘制路径
        pts_grid = []
        for x, z in path_xz:
            cr = self.occ_grid.world_to_grid(np.array([x, z]))
            pts_grid.append((int(cr[0]), int(cr[1])))

        for i in range(len(pts_grid) - 1):
            cv2.line(img, pts_grid[i], pts_grid[i + 1], (0, 255, 0), 2)

        # 起终点标记
        if pts_grid:
            cv2.circle(img, pts_grid[0], 5, (255, 0, 0), -1)   # 蓝色起点
            cv2.circle(img, pts_grid[-1], 5, (0, 0, 255), -1)  # 红色终点

        if save_path:
            cv2.imwrite(save_path, img)

        return img
