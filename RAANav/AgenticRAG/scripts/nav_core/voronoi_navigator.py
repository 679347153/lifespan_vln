"""Voronoi 辅助导航 — 基于 Voronoi 拓扑图的路径规划与探索目标选择.

提供:
  1. plan_voronoi_path()  — Voronoi 图上的粗粒度路径规划 (用于跨房间导航)
  2. select_voronoi_exploration_target() — 选择未访问的 Voronoi 节点作为探索目标
  3. get_unvisited_voronoi_nodes() — 获取所有未访问的 Voronoi 节点
  4. StuckDetector — 卡住检测 + 恢复建议

Voronoi 路径规划流水线:
  Agent → 最近 Voronoi 节点 → 图上最短路径 → 目标最近节点 → 目标点
  优点: 路径经过房间中心/走廊, 比纯 A* 更好的视野 + 更少穿墙风险
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None


# ---------------------------------------------------------------------------
# Voronoi 路径规划
# ---------------------------------------------------------------------------

def plan_voronoi_path(
    G,  # nx.Graph — Voronoi 导航图
    start_world: Tuple[float, float],  # (x, z)
    goal_world: Tuple[float, float],   # (x, z)
    max_snap_dist: float = 5.0,
) -> Optional[List[Tuple[float, float]]]:
    """在 Voronoi 图上规划粗粒度路径.

    Pipeline:
      1. 找到距 start/goal 最近的图节点
      2. 验证 snap 距离不超过 max_snap_dist (太远则说明该区域图不覆盖)
      3. nx.shortest_path (Dijkstra, weight=dist) 获取节点序列
      4. 返回世界坐标路径点列表

    Args:
        G: Voronoi 导航图 (nodes have pos_world attr)
        start_world: 起点 (x, z)
        goal_world: 终点 (x, z)
        max_snap_dist: 最大 snap 距离, 超过则返回 None

    Returns:
        list of (x, z) 路径中间点 (包含 start_snap 和 goal_snap),
        或 None (图为空 / 不连通 / snap 距离过大)
    """
    if G is None or nx is None or G.number_of_nodes() < 2:
        return None

    from semantic_map_Create.voronoi_graph import find_path_on_graph

    path = find_path_on_graph(G, start_world, goal_world)
    if path is None or len(path) < 2:
        return None

    # 验证 snap 距离
    d_start = math.sqrt(
        (path[0][0] - start_world[0]) ** 2 + (path[0][1] - start_world[1]) ** 2
    )
    d_goal = math.sqrt(
        (path[-1][0] - goal_world[0]) ** 2 + (path[-1][1] - goal_world[1]) ** 2
    )
    if d_start > max_snap_dist or d_goal > max_snap_dist:
        return None  # 图节点离 start/goal 太远, 不可靠

    return path


def voronoi_path_to_3d_waypoints(
    voronoi_path: List[Tuple[float, float]],
    y_val: float,
    step_size: float = 1.0,
) -> List[list]:
    """将 Voronoi (x,z) 路径转换为 3D waypoints 并按 step_size 重采样.

    Args:
        voronoi_path: [(x, z), ...] — plan_voronoi_path 的输出
        y_val: agent 高度 (y 坐标)
        step_size: 重采样间距

    Returns:
        [[x, y, z], ...] — 3D路径点列表
    """
    if not voronoi_path or len(voronoi_path) < 2:
        return []

    raw_pts = [[p[0], y_val, p[1]] for p in voronoi_path]

    # 重采样
    resampled = []
    accum = 0.0
    for i in range(1, len(raw_pts)):
        dx = raw_pts[i][0] - raw_pts[i - 1][0]
        dz = raw_pts[i][2] - raw_pts[i - 1][2]
        seg_len = math.sqrt(dx * dx + dz * dz)
        if seg_len < 1e-6:
            continue
        accum += seg_len
        if accum >= step_size:
            resampled.append(raw_pts[i])
            accum = 0.0
    # 确保终点在内
    end = raw_pts[-1]
    if not resampled or _dist3(resampled[-1], end) > 0.1:
        resampled.append(end)
    return resampled if resampled else [end]


# ---------------------------------------------------------------------------
# 探索目标选择
# ---------------------------------------------------------------------------

def get_unvisited_voronoi_nodes(
    G,
    visited_positions: List[list],
    visit_radius: float = 2.0,
) -> List[Dict]:
    """获取所有未被访问过的 Voronoi 节点.

    Args:
        G: Voronoi 导航图
        visited_positions: 已访问位置列表 [[x, y, z], ...]
        visit_radius: 视为 "已访问" 的半径 (米)

    Returns:
        list of {"node_id": int, "x": float, "z": float, "room_id": str}
    """
    if G is None or nx is None or G.number_of_nodes() == 0:
        return []

    # 转为 numpy 加速距离计算
    vp_xz = np.array([[vp[0], vp[2]] for vp in visited_positions]) if visited_positions else np.empty((0, 2))

    unvisited = []
    for n, data in G.nodes(data=True):
        pw = data.get("pos_world")
        if pw is None:
            continue
        nx_pos, nz_pos = pw

        if len(vp_xz) > 0:
            dists = np.sqrt((vp_xz[:, 0] - nx_pos) ** 2 + (vp_xz[:, 1] - nz_pos) ** 2)
            if dists.min() < visit_radius:
                continue  # 已访问

        unvisited.append({
            "node_id": n,
            "x": nx_pos,
            "z": nz_pos,
            "room_id": data.get("room_id", "R0"),
        })

    return unvisited


def select_voronoi_exploration_target(
    G,
    agent_pos: list,
    visited_positions: List[list],
    visit_radius: float = 2.0,
    prefer_new_rooms: bool = True,
) -> Optional[List[float]]:
    """选择最佳未访问 Voronoi 节点作为探索目标.

    策略:
      - prefer_new_rooms=True:  优先选择访问次数最少的房间中的节点 (鼓励跨房间探索)
      - prefer_new_rooms=False: 选择距已访问位置最远的节点 (类似 frontier 策略)

    Args:
        G: Voronoi 导航图
        agent_pos: 当前 agent 位置 [x, y, z]
        visited_positions: 已访问位置列表
        visit_radius: 视为 "已访问" 的半径
        prefer_new_rooms: 是否优先探索新房间

    Returns:
        [x, z] 目标坐标, 或 None (全部已访问)
    """
    unvisited = get_unvisited_voronoi_nodes(G, visited_positions, visit_radius)
    if not unvisited:
        return None

    ax, az = float(agent_pos[0]), float(agent_pos[2])

    if prefer_new_rooms:
        # 统计每个房间的已访问次数 (用最近 Voronoi 节点的 room_id 近似)
        room_visit_count: Dict[str, int] = {}
        if visited_positions:
            # 为效率, 只对 Voronoi 节点做一次最近邻, 不对每个 visited_pos
            for node_info in get_unvisited_voronoi_nodes(G, [], 0):
                rid = node_info["room_id"]
                n_x, n_z = node_info["x"], node_info["z"]
                for vp in visited_positions:
                    if math.sqrt((n_x - vp[0]) ** 2 + (n_z - vp[2]) ** 2) < visit_radius:
                        room_visit_count[rid] = room_visit_count.get(rid, 0) + 1
                        break

        # 评分: 房间访问次数越少越优先, 同房间内选最近的
        scored = []
        for info in unvisited:
            rv = room_visit_count.get(info["room_id"], 0)
            dist = math.sqrt((info["x"] - ax) ** 2 + (info["z"] - az) ** 2)
            score = rv * 100.0 + dist  # 越小越好
            scored.append((score, info))

        scored.sort(key=lambda x: x[0])
        best = scored[0][1]
        return [best["x"], best["z"]]
    else:
        best = None
        best_min_dist = -1.0
        for info in unvisited:
            if visited_positions:
                min_d = min(
                    math.sqrt((info["x"] - vp[0]) ** 2 + (info["z"] - vp[2]) ** 2)
                    for vp in visited_positions
                )
            else:
                min_d = 0.0
            if min_d > best_min_dist:
                best_min_dist = min_d
                best = info
        if best is None:
            return None
        return [best["x"], best["z"]]


# ---------------------------------------------------------------------------
# 卡住检测 + 恢复
# ---------------------------------------------------------------------------

class StuckDetector:
    """检测 agent 是否卡住 (位移过小).

    判定逻辑: 最近 window 步内的总位移 < min_displacement → 卡住.
    """

    def __init__(
        self,
        window: int = 5,
        min_displacement: float = 0.5,
    ):
        self.window = window
        self.min_displacement = min_displacement
        self._positions: List[list] = []
        self._stuck_count = 0

    def update(self, pos: list) -> bool:
        """更新位置, 返回 True 如果判定卡住."""
        self._positions.append([float(pos[0]), float(pos[1]), float(pos[2])])

        if len(self._positions) < self.window:
            return False

        recent = self._positions[-self.window:]
        dx = recent[-1][0] - recent[0][0]
        dz = recent[-1][2] - recent[0][2]
        displacement = math.sqrt(dx * dx + dz * dz)

        if displacement < self.min_displacement:
            self._stuck_count += 1
            return True
        else:
            self._stuck_count = 0
            return False

    @property
    def stuck_count(self) -> int:
        return self._stuck_count

    def reset(self):
        self._stuck_count = 0
        self._positions.clear()


def recover_from_stuck(
    agent,
    occ_grid,
    visited_positions: List[list],
    stuck_count: int,
    use_navmesh: bool = True,
) -> Optional[list]:
    """卡住恢复: 根据卡住次数选择不同恢复策略.

    恢复链:
      stuck_count == 1: 随机扰动 (附近 free cell)
      stuck_count == 2: 回退到上次有效位置
      stuck_count >= 3: 随机导航点 (更激进的脱困)

    Args:
        agent: HabitatAgent
        occ_grid: 占据栅格
        visited_positions: 已访问位置列表
        stuck_count: 连续卡住次数
        use_navmesh: 是否使用 navmesh

    Returns:
        恢复目标位置 [x, y, z], 或 None (无法恢复)
    """
    pos = agent.get_position()
    y_val = float(pos[1])

    if stuck_count <= 1:
        # 策略 1: 随机扰动 — 在当前位置附近找一个 free cell
        for _ in range(10):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(1.0, 3.0)
            nx_pos = float(pos[0]) + dist * np.cos(angle)
            nz_pos = float(pos[2]) + dist * np.sin(angle)
            if occ_grid is not None and occ_grid.is_navigable_at(nx_pos, nz_pos):
                print(f"    [卡住恢复] 策略1: 随机扰动 d={dist:.1f}m")
                return [nx_pos, y_val, nz_pos]
        # fallback
        print(f"    [卡住恢复] 策略1失败, 升级到策略2")
        stuck_count = 2

    if stuck_count == 2:
        # 策略 2: 回退到之前有效位置
        if len(visited_positions) > 10:
            backtrack_pos = visited_positions[-10]
            print(f"    [卡住恢复] 策略2: 回退到 10 步前位置")
            return backtrack_pos
        elif len(visited_positions) > 3:
            backtrack_pos = visited_positions[0]
            print(f"    [卡住恢复] 策略2: 回退到起始位置")
            return backtrack_pos

    # 策略 3: 随机导航点
    if use_navmesh:
        rp = agent.get_random_navigable_point()
        if rp is not None and not np.isnan(rp).any():
            print(f"    [卡住恢复] 策略3: 随机导航点")
            return list(rp)
    else:
        # 从 occ_grid 随机选一个远处 free cell
        if occ_grid is not None:
            free_ys, free_xs = np.where(occ_grid.grid == 1)
            if len(free_ys) > 0:
                idx = np.random.randint(len(free_ys))
                w = occ_grid.grid_to_world(np.array([free_xs[idx], free_ys[idx]]))
                print(f"    [卡住恢复] 策略3: 随机 free cell")
                return [float(w[0]), y_val, float(w[1])]

    return None


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _dist3(a: list, b: list) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
