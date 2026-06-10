"""HabitatAgent — Habitat simulator agent 封装.

从 sim_nav_loop.py 提取, 封装 position/heading/observe/navigate 操作.
支持两种导航后端:
  1) navmesh (pathfinder) — 仅用于 GT 对照实验
  2) occ_grid (A*) — 无 navmesh 模式, 由深度帧增量构建
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

import habitat_sim

if TYPE_CHECKING:
    from semantic_map_Create.occupancy_grid import OccupancyGrid

# 结构性物体过滤集合
STRUCTURAL_CATEGORIES = {
    "wall", "floor", "ceiling", "unknown", "misc", "void", "objects",
    "column", "beam", "railing", "stair", "stairs",
}


class HabitatAgent:
    """封装 Habitat simulator 的 agent 操作."""

    def __init__(self, sim: habitat_sim.Simulator, use_navmesh: bool = True):
        self.sim = sim
        self.agent = sim.get_agent(0)
        self.semantic_scene = sim.semantic_scene
        self.use_navmesh = use_navmesh
        self._occ_grid: Optional["OccupancyGrid"] = None
        self._astar: Optional[Any] = None

        # 建立 semantic id → object info 映射
        self._sem_id_to_obj: Dict[int, Any] = {}
        for obj in self.semantic_scene.objects:
            if obj is not None and obj.category is not None:
                self._sem_id_to_obj[obj.semantic_id] = obj

        self._nav_points = []

    def set_occ_grid(self, occ_grid: "OccupancyGrid"):
        """注入占据栅格, 启用 A* 导航后端."""
        self._occ_grid = occ_grid
        self._astar = None  # 下次路径规划时重建

    def get_position(self) -> np.ndarray:
        """当前 agent 世界坐标 [x, y, z]."""
        state = self.agent.get_state()
        p = state.position
        return np.array([float(p[0]), float(p[1]), float(p[2])])

    def get_rotation(self) -> np.ndarray:
        state = self.agent.get_state()
        return np.array([state.rotation.x, state.rotation.y,
                         state.rotation.z, state.rotation.w])

    def get_heading_deg(self) -> float:
        """返回当前 agent 水平朝向角 (度).

        Habitat 默认面向 -Z, heading θ → 朝向 [-sin(θ), 0, -cos(θ)].
        """
        q = self.get_rotation()
        siny_cosp = 2.0 * (q[3] * q[1] + q[0] * q[2])
        cosy_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return math.degrees(yaw)

    def set_position(self, pos: list):
        state = self.agent.get_state()
        state.position = np.array(pos, dtype=np.float32)
        self.agent.set_state(state)

    def set_heading(self, heading_deg: float):
        """设置 agent 水平朝向 (度, Y轴旋转)."""
        import quaternion as _qt
        rad = math.radians(heading_deg)
        q = _qt.from_euler_angles([0, rad, 0])
        state = self.agent.get_state()
        state.rotation = q
        self.agent.set_state(state)

    def face_toward(self, target_pos: list):
        """让 agent 面朝目标位置.

        Habitat 默认朝向 -Z, heading θ 对应朝向 [-sin(θ), 0, -cos(θ)].
        给定目标方向 (dx, dz), 求 θ 使 [-sin(θ), -cos(θ)] ∝ [dx, dz]:
          θ = atan2(-dx, -dz)
        """
        pos = self.get_position()
        dx = target_pos[0] - pos[0]
        dz = target_pos[2] - pos[2]
        heading = math.degrees(math.atan2(-dx, -dz))
        self.set_heading(heading)

    def navigate_to(self, target_pos: list) -> bool:
        """传送到目标位置.

        navmesh 模式: 先用 pathfinder 验证可达性.
        occ_grid 模式: 用 occ_grid 验证可行性, 直接传送.
        """
        if self.use_navmesh:
            path = habitat_sim.ShortestPath()
            path.requested_start = self.get_position().tolist()
            path.requested_end = target_pos
            found = self.sim.pathfinder.find_path(path)
            if found and path.geodesic_distance < float("inf"):
                state = self.agent.get_state()
                state.position = np.array(target_pos, dtype=np.float32)
                self.agent.set_state(state)
                dx = target_pos[0] - path.requested_start[0]
                dz = target_pos[2] - path.requested_start[2]
                if abs(dx) > 0.01 or abs(dz) > 0.01:
                    self.set_heading(math.degrees(math.atan2(-dx, -dz)))
                return True
            return False
        else:
            # occ_grid 模式: 检查目标点可行性后传送
            prev_pos = self.get_position()
            tp = np.array(target_pos, dtype=np.float32)
            if self._occ_grid is not None:
                tp = self._snap_to_free(tp)
            state = self.agent.get_state()
            state.position = tp
            self.agent.set_state(state)
            dx = float(tp[0]) - prev_pos[0]
            dz = float(tp[2]) - prev_pos[2]
            if abs(dx) > 0.01 or abs(dz) > 0.01:
                self.set_heading(math.degrees(math.atan2(-dx, -dz)))
            return True

    def get_path_waypoints(self, target_pos: list, step_size: float = 0.5) -> Optional[List[list]]:
        """计算路径并按 step_size 重采样.

        navmesh 模式: pathfinder shortest_path.
        occ_grid 模式: A* on occupancy grid.
        """
        if self.use_navmesh:
            return self._get_path_waypoints_navmesh(target_pos, step_size)
        else:
            return self._get_path_waypoints_astar(target_pos, step_size)

    def _get_path_waypoints_navmesh(self, target_pos: list, step_size: float) -> Optional[List[list]]:
        """navmesh 后端路径规划."""
        path = habitat_sim.ShortestPath()
        path.requested_start = self.get_position().tolist()
        path.requested_end = target_pos
        found = self.sim.pathfinder.find_path(path)
        if not found or path.geodesic_distance >= float("inf") or len(path.points) < 2:
            return None
        raw_pts = [[float(p[0]), float(p[1]), float(p[2])] for p in path.points]
        return self._resample_path(raw_pts, step_size)

    def _get_path_waypoints_astar(self, target_pos: list, step_size: float) -> Optional[List[list]]:
        """A* on occ_grid 后端路径规划."""
        if self._occ_grid is None:
            return None
        from semantic_map_Create.astar_planner import AStarPlanner
        planner = AStarPlanner(self._occ_grid, use_navmesh_grid=False)
        pos = self.get_position()
        start_xz = np.array([pos[0], pos[2]])
        end_xz = np.array([target_pos[0], target_pos[2]])
        path_xz = planner.plan(start_xz, end_xz)
        if path_xz is None or len(path_xz) < 2:
            return None
        y_val = float(pos[1])
        raw_pts = [[float(p[0]), y_val, float(p[1])] for p in path_xz]
        return self._resample_path(raw_pts, step_size)

    @staticmethod
    def _resample_path(raw_pts: List[list], step_size: float) -> List[list]:
        """将路径按 step_size 等距重采样."""
        resampled = []
        accum = 0.0
        for i in range(1, len(raw_pts)):
            dx = raw_pts[i][0] - raw_pts[i-1][0]
            dy = raw_pts[i][1] - raw_pts[i-1][1]
            dz = raw_pts[i][2] - raw_pts[i-1][2]
            seg_len = math.sqrt(dx*dx + dy*dy + dz*dz)
            if seg_len < 1e-6:
                continue
            accum += seg_len
            if accum >= step_size:
                resampled.append(raw_pts[i])
                accum = 0.0
        end = raw_pts[-1]
        if not resampled or HabitatAgent._dist3(resampled[-1], end) > 0.1:
            resampled.append(end)
        return resampled if resampled else [end]

    def move_to_waypoint(self, waypoint: list):
        """移动到路径点 — 连续步进模式 (非传送).

        使用小步前进 + 转向动作序列, 而非直接设置位置.
        navmesh 模式: habitat 内置碰撞检测 (agent.act).
        occ_grid 模式: 小步 teleport + OCC Grid 碰撞检测.
        """
        target = np.array(waypoint, dtype=np.float32)
        forward_amount = 0.25   # 每步前进距离 (m)
        turn_amount = 10.0      # 每步转向角度 (度)
        reach_threshold = forward_amount * 1.5
        max_substeps = 40       # 防止无限循环

        for _ in range(max_substeps):
            pos = self.get_position()
            dx = float(target[0]) - pos[0]
            dz = float(target[2]) - pos[2]
            dist = math.sqrt(dx * dx + dz * dz)
            if dist < reach_threshold:
                # 到达, 面朝目标方向
                if dist > 0.05:
                    self.face_toward(waypoint)
                return

            # 计算朝向差
            desired_heading = math.degrees(math.atan2(-dx, -dz))
            current_heading = self.get_heading_deg()
            diff = (desired_heading - current_heading + 180) % 360 - 180

            if abs(diff) > turn_amount * 0.6:
                # 需要转向
                if self.use_navmesh:
                    if diff > 0:
                        self.agent.act("turn_left")
                    else:
                        self.agent.act("turn_right")
                else:
                    # occ_grid 模式: 直接设置朝向 (无碰撞问题)
                    step = min(abs(diff), turn_amount)
                    self.set_heading(current_heading + step * (1 if diff > 0 else -1))
                continue

            # 前进一步
            prev_pos = pos.copy()
            if self.use_navmesh:
                self.agent.act("move_forward")
                new_pos = self.get_position()
                moved = math.sqrt((new_pos[0]-prev_pos[0])**2 +
                                  (new_pos[2]-prev_pos[2])**2)
                if moved < 0.01:
                    # 碰撞: 尝试小角度偏转绕行
                    escaped = False
                    for nudge in [15, -15, 30, -30, 45, -45]:
                        self.set_heading(current_heading + nudge)
                        self.agent.act("move_forward")
                        nudge_pos = self.get_position()
                        nudge_moved = math.sqrt(
                            (nudge_pos[0]-prev_pos[0])**2 +
                            (nudge_pos[2]-prev_pos[2])**2)
                        if nudge_moved > 0.05:
                            escaped = True
                            break
                    if not escaped:
                        return  # 所有方向都被阻挡, 放弃
            else:
                # occ_grid 模式: 小步 teleport + 碰撞检测
                heading_rad = math.radians(current_heading)
                step_x = -math.sin(heading_rad) * forward_amount
                step_z = -math.cos(heading_rad) * forward_amount
                next_pos = np.array([pos[0] + step_x, pos[1], pos[2] + step_z],
                                    dtype=np.float32)
                if self._occ_grid is not None:
                    xz = np.array([float(next_pos[0]), float(next_pos[2])])
                    if not self._occ_grid.is_navigable_at(xz[0], xz[1]):
                        # 尝试 snap 到附近 free cell
                        next_pos = self._snap_to_free(next_pos)
                        snapped_dist = math.sqrt(
                            (next_pos[0]-pos[0])**2 + (next_pos[2]-pos[2])**2)
                        if snapped_dist < 0.01:
                            return  # 真的走不了
                state = self.agent.get_state()
                state.position = next_pos
                self.agent.set_state(state)

    def move_to_waypoint_teleport(self, waypoint: list):
        """传送到路径点 (旧版, 调试用).

        navmesh 模式: snap 到 navmesh 确保不穿墙.
        occ_grid 模式: snap 到最近 free cell.
        """
        prev_pos = self.get_position()
        if self.use_navmesh:
            snapped = self.sim.pathfinder.snap_point(np.array(waypoint, dtype=np.float32))
            if self.sim.pathfinder.is_navigable(snapped):
                target = np.array([float(snapped[0]), float(snapped[1]), float(snapped[2])], dtype=np.float32)
            else:
                target = np.array(waypoint, dtype=np.float32)
        else:
            target = self._snap_to_free(np.array(waypoint, dtype=np.float32))
        state = self.agent.get_state()
        state.position = target
        self.agent.set_state(state)
        dx = float(target[0]) - prev_pos[0]
        dz = float(target[2]) - prev_pos[2]
        if abs(dx) > 0.01 or abs(dz) > 0.01:
            self.set_heading(math.degrees(math.atan2(-dx, -dz)))

    def _snap_to_free(self, pos_3d: np.ndarray) -> np.ndarray:
        """将 3D 位置 snap 到 occ_grid 最近 free cell. 无 occ_grid 时原样返回."""
        if self._occ_grid is None:
            return pos_3d
        xz = np.array([float(pos_3d[0]), float(pos_3d[2])])
        if self._occ_grid.is_navigable_at(xz[0], xz[1]):
            return pos_3d
        # 搜索最近 free cell (BFS in grid)
        gc = self._occ_grid.world_to_grid(xz)
        c0, r0 = int(gc[0]), int(gc[1])
        rows, cols = self._occ_grid.shape
        from collections import deque
        visited = set()
        queue = deque([(r0, c0)])
        visited.add((r0, c0))
        while queue:
            r, c = queue.popleft()
            if 0 <= r < rows and 0 <= c < cols and self._occ_grid.grid[r, c] == 1:
                world_xz = self._occ_grid.grid_to_world(np.array([c, r]))
                y_val = float(pos_3d[1]) if len(pos_3d) >= 3 else 0.0
                return np.array([world_xz[0], y_val, world_xz[1]], dtype=np.float32)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if (nr, nc) not in visited and 0 <= nr < rows and 0 <= nc < cols:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return pos_3d  # fallback

    @staticmethod
    def _dist3(a: list, b: list) -> float:
        return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))

    def observe(self) -> Dict[str, np.ndarray]:
        return self.sim.get_sensor_observations()

    def panoramic_observe(self, n_views: int = 4) -> List[Dict[str, np.ndarray]]:
        """360° 环视采集, 结束后恢复原始朝向."""
        original_rot = self.get_rotation()
        heading_start = self.get_heading_deg()
        step_deg = 360.0 / n_views
        observations = []
        for i in range(n_views):
            self.set_heading(heading_start + i * step_deg)
            observations.append(self.sim.get_sensor_observations())
        # 恢复
        state = self.agent.get_state()
        import quaternion as _qt
        state.rotation = _qt.quaternion(
            original_rot[3], original_rot[0], original_rot[1], original_rot[2]
        )
        self.agent.set_state(state)
        return observations

    def get_visible_objects(self, panoramic: bool = False, n_views: int = 4) -> List[Dict[str, Any]]:
        """从语义图像中提取可见物体."""
        all_obs = self.panoramic_observe(n_views) if panoramic else [self.observe()]
        pixel_counter: Dict[int, int] = {}
        for obs in all_obs:
            if "semantic" not in obs:
                continue
            unique_ids, counts = np.unique(obs["semantic"], return_counts=True)
            for sem_id, px_count in zip(unique_ids, counts):
                sem_id = int(sem_id)
                pixel_counter[sem_id] = pixel_counter.get(sem_id, 0) + int(px_count)

        visible = []
        for sem_id, px_count in pixel_counter.items():
            if sem_id == 0:
                continue
            obj_info = self._sem_id_to_obj.get(sem_id)
            if obj_info is None:
                continue
            cat_name = obj_info.category.name("").strip().lower()
            if cat_name in STRUCTURAL_CATEGORIES or not cat_name:
                continue
            if px_count < 100:
                continue
            visible.append({
                "semantic_id": sem_id, "label": cat_name,
                "pixel_count": px_count, "object_ref": obj_info,
            })
        return visible

    def get_random_navigable_point(self, floor_y_threshold: float = 1.5) -> list:
        """获取随机可行点 (同楼层).

        navmesh 模式: pathfinder 采样, 限制与当前 agent Y 的差值.
        occ_grid 模式: 从 free cells 随机采样.
        """
        if self.use_navmesh:
            agent_y = float(self.get_position()[1])
            for _ in range(30):
                p = self.sim.pathfinder.get_random_navigable_point()
                if abs(float(p[1]) - agent_y) < floor_y_threshold:
                    return [float(p[0]), float(p[1]), float(p[2])]
            # 采样 30 次仍未命中同层 → 返回最后一次结果
            return [float(p[0]), float(p[1]), float(p[2])]
        elif self._occ_grid is not None:
            free_ys, free_xs = np.where(self._occ_grid.grid == 1)
            if len(free_ys) == 0:
                p = self.get_position()
                return [float(p[0]), float(p[1]), float(p[2])]
            idx = np.random.randint(0, len(free_ys))
            world_xz = self._occ_grid.grid_to_world(np.array([free_xs[idx], free_ys[idx]]))
            y_val = float(self.get_position()[1])
            return [float(world_xz[0]), y_val, float(world_xz[1])]
        else:
            p = self.sim.pathfinder.get_random_navigable_point()
            return [float(p[0]), float(p[1]), float(p[2])]

    def get_gt_target_positions(self, target_label: str) -> List[List[float]]:
        """获取 GT 语义标注中目标物体的所有 3D 中心坐标."""
        positions = []
        target_lower = target_label.strip().lower()
        for obj in self.semantic_scene.objects:
            if obj is None or obj.category is None:
                continue
            cat = obj.category.name("").strip().lower()
            if target_lower in cat or cat in target_lower:
                center = obj.aabb.center()
                positions.append([float(center[0]), float(center[1]), float(center[2])])
        return positions
