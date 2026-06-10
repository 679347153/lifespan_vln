"""Voronoi 拓扑导航图 — 基于占据栅格的稀疏导航图构建.

Pipeline (参考 HOV-SG navigation_graph.py):
  1. 自由空间 → 边界提取 (binary_erosion)
  2. 边界点 → Voronoi 图
  3. 过滤: 只保留自由空间内的 Voronoi 边
  4. 稀疏化: 移除 degree-2 节点
  5. 输出: networkx.Graph (节点带世界坐标, 边带距离)

用于任务模式导航: 识别房间间连接通道, 高效跨房间路径规划.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None


def build_voronoi_graph(
    occ_grid,
    room_labels: Optional[np.ndarray] = None,
    min_node_dist_m: float = 0.3,
    boundary_erosion_iter: int = 1,
) -> "nx.Graph":
    """从占据栅格构建 Voronoi 拓扑导航图.

    Args:
        occ_grid: OccupancyGrid 实例
        room_labels: 房间标签图 (可选, 用于标注节点所属房间)
        min_node_dist_m: 最小节点间距 (米), 用于后处理去重
        boundary_erosion_iter: 边界提取腐蚀迭代次数

    Returns:
        networkx.Graph: 节点属性 {pos_world: (x,z), pos_grid: (col,row), room_id: str}
                        边属性 {dist: float (世界坐标距离)}
    """
    if nx is None:
        raise ImportError("networkx is required: pip install networkx")
    from scipy.spatial import Voronoi
    from scipy.ndimage import binary_erosion

    grid = occ_grid.grid
    resolution = occ_grid.resolution
    rows, cols = grid.shape

    # Step 1: 构建自由空间掩码
    free_map = (grid == 1).astype(np.uint8)

    # 获取最大连通区域 (排除孤立小区域)
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(free_map, connectivity=8)
    if n_comp > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        main_label = np.argmax(areas) + 1
        free_map = (labels == main_label).astype(np.uint8)

    # Step 2: 边界提取
    eroded = binary_erosion(free_map, iterations=boundary_erosion_iter).astype(np.uint8)
    boundary = free_map - eroded
    boundary_pixels = np.argwhere(boundary == 1)  # (row, col) pairs

    if len(boundary_pixels) < 10:
        print("[VoronoiGraph] 边界点太少, 返回空图")
        return nx.Graph()

    # 降采样边界点 (避免 Voronoi 计算过慢)
    max_boundary_points = 5000
    if len(boundary_pixels) > max_boundary_points:
        indices = np.random.choice(len(boundary_pixels), max_boundary_points, replace=False)
        boundary_pixels = boundary_pixels[indices]

    # Step 3: 计算 Voronoi 图
    try:
        vor = Voronoi(boundary_pixels)
    except Exception as e:
        print(f"[VoronoiGraph] Voronoi 计算失败: {e}")
        return nx.Graph()

    # Step 4: 构建图 — 只保留自由空间内的边
    G = nx.Graph()
    node_id_map = {}  # (row, col) → node_id

    def _get_node_id(r, c):
        key = (int(r), int(c))
        if key not in node_id_map:
            nid = len(node_id_map)
            node_id_map[key] = nid
            world_xz = occ_grid.grid_to_world(np.array([c, r]))
            attrs = {
                "pos_world": (float(world_xz[0]), float(world_xz[1])),
                "pos_grid": (int(c), int(r)),
            }
            if room_labels is not None and 0 <= int(r) < rows and 0 <= int(c) < cols:
                rl = room_labels[int(r), int(c)]
                attrs["room_id"] = f"R{rl}" if rl > 0 else "R0"
            G.add_node(nid, **attrs)
        return node_id_map[key]

    for simplex in vor.ridge_vertices:
        if -1 in simplex:
            continue  # 跳过无穷远边

        p1 = vor.vertices[simplex[0]]
        p2 = vor.vertices[simplex[1]]
        r1, c1 = int(round(p1[0])), int(round(p1[1]))
        r2, c2 = int(round(p2[0])), int(round(p2[1]))

        # 检查两端点是否在自由空间内
        if not (0 <= r1 < rows and 0 <= c1 < cols and
                0 <= r2 < rows and 0 <= c2 < cols):
            continue
        if free_map[r1, c1] == 0 or free_map[r2, c2] == 0:
            continue

        # 检查边的中点也在自由空间内 (避免穿墙)
        rm, cm = (r1 + r2) // 2, (c1 + c2) // 2
        if 0 <= rm < rows and 0 <= cm < cols and free_map[rm, cm] == 0:
            continue

        n1 = _get_node_id(r1, c1)
        n2 = _get_node_id(r2, c2)
        if n1 != n2:
            dist_px = np.sqrt((r1 - r2)**2 + (c1 - c2)**2)
            dist_m = dist_px * resolution
            G.add_edge(n1, n2, dist=dist_m)

    print(f"[VoronoiGraph] 原始图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    # Step 5: 稀疏化 — 移除 degree-2 节点 (直线走廊上的冗余点)
    G = _sparsify_graph(G, min_node_dist_m)

    print(f"[VoronoiGraph] 稀疏后: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G


def _sparsify_graph(G: "nx.Graph", min_dist: float) -> "nx.Graph":
    """移除 degree-2 节点, 合并直线段为单边."""
    if nx is None:
        return G

    # 迭代移除 degree-2 节点
    changed = True
    while changed:
        changed = False
        deg2_nodes = [n for n in G.nodes if G.degree(n) == 2]
        for node in deg2_nodes:
            neighbors = list(G.neighbors(node))
            if len(neighbors) != 2:
                continue
            n1, n2 = neighbors
            d1 = G.edges[node, n1].get("dist", 0)
            d2 = G.edges[node, n2].get("dist", 0)
            total = d1 + d2
            # 合并: 移除中间节点, 连接两端
            G.add_edge(n1, n2, dist=total)
            G.remove_node(node)
            changed = True
            break  # restart iteration after modification

    # 合并过近的节点
    merged = True
    while merged:
        merged = False
        for n1, n2 in list(G.edges):
            if not G.has_node(n1) or not G.has_node(n2):
                continue
            dist = G.edges[n1, n2].get("dist", 0)
            if dist < min_dist:
                # 保留 degree 较高的节点
                if G.degree(n1) >= G.degree(n2):
                    keep, remove = n1, n2
                else:
                    keep, remove = n2, n1
                for neighbor in list(G.neighbors(remove)):
                    if neighbor != keep and G.has_node(neighbor):
                        d = G.edges[remove, neighbor].get("dist", 0)
                        if G.has_edge(keep, neighbor):
                            existing = G.edges[keep, neighbor].get("dist", float("inf"))
                            G.edges[keep, neighbor]["dist"] = min(existing, d)
                        else:
                            G.add_edge(keep, neighbor, dist=d)
                G.remove_node(remove)
                merged = True
                break

    return G


def find_path_on_graph(
    G: "nx.Graph",
    start_world: Tuple[float, float],
    goal_world: Tuple[float, float],
    occ_grid=None,
) -> Optional[List[Tuple[float, float]]]:
    """在 Voronoi 图上找最短路径.

    Args:
        G: Voronoi 导航图
        start_world: 起点世界坐标 (x, z)
        goal_world: 终点世界坐标 (x, z)

    Returns:
        list of (x, z) 路径点, 或 None
    """
    if nx is None or G.number_of_nodes() == 0:
        return None

    # 找到最近的图节点
    def _nearest_node(wx, wz):
        best_node = None
        best_dist = float("inf")
        for n, data in G.nodes(data=True):
            pw = data.get("pos_world")
            if pw is None:
                continue
            d = (pw[0] - wx)**2 + (pw[1] - wz)**2
            if d < best_dist:
                best_dist = d
                best_node = n
        return best_node

    src = _nearest_node(*start_world)
    dst = _nearest_node(*goal_world)
    if src is None or dst is None:
        return None

    try:
        path_nodes = nx.shortest_path(G, src, dst, weight="dist")
    except nx.NetworkXNoPath:
        return None

    path_world = []
    for n in path_nodes:
        pw = G.nodes[n].get("pos_world")
        if pw:
            path_world.append(pw)
    return path_world


class VoronoiNavigator:
    """Voronoi 导航器 — 管理 Voronoi 图并提供路径规划接口.
    
    使用方式:
      1. 深度探索: Voronoi 图节点作为探索目标（优先访问门口/交叉点）
      2. 任务模式: Voronoi 图上做最短路径 → 走房间中间 → 更快到达目标区域
      
    路径规划 fallback:
      Voronoi → navmesh shortest_path → A* on grid
    """

    def __init__(self, occ_grid, voronoi_graph: Optional["nx.Graph"] = None):
        self.occ_grid = occ_grid
        self.graph = voronoi_graph
        self._visited_nodes: set = set()

    def set_graph(self, G: "nx.Graph"):
        self.graph = G

    def has_graph(self) -> bool:
        return self.graph is not None and self.graph.number_of_nodes() > 0

    def plan_path_3d(
        self,
        start_pos: list,
        goal_pos: list,
        step_size: float = 1.0,
        y_val: Optional[float] = None,
    ) -> Optional[List[list]]:
        """在 Voronoi 图上规划路径, 返回 3D waypoints.

        路径结构: start → (snap到最近节点) → Voronoi路径 → (snap到目标) → goal
        
        Args:
            start_pos: [x, y, z] 起点
            goal_pos: [x, y, z] 目标
            step_size: 重采样间距
            y_val: y坐标 (楼层高度), None则从start_pos取
            
        Returns:
            3D waypoints [[x,y,z], ...] 或 None (无路径)
        """
        if not self.has_graph():
            return None

        sx, sz = float(start_pos[0]), float(start_pos[2])
        gx, gz = float(goal_pos[0]), float(goal_pos[2])
        yv = y_val if y_val is not None else float(start_pos[1])

        path_xz = find_path_on_graph(self.graph, (sx, sz), (gx, gz), self.occ_grid)
        if path_xz is None or len(path_xz) == 0:
            return None

        # 构建完整路径: start → voronoi节点序列 → goal
        full_path_xz = [(sx, sz)]
        for p in path_xz:
            # 避免重复起点
            if abs(p[0] - full_path_xz[-1][0]) > 0.1 or abs(p[1] - full_path_xz[-1][1]) > 0.1:
                full_path_xz.append(p)
        # 添加终点
        if abs(gx - full_path_xz[-1][0]) > 0.1 or abs(gz - full_path_xz[-1][1]) > 0.1:
            full_path_xz.append((gx, gz))

        # 重采样为等距 waypoints
        waypoints_3d = []
        accum = 0.0
        for i in range(1, len(full_path_xz)):
            dx = full_path_xz[i][0] - full_path_xz[i-1][0]
            dz = full_path_xz[i][1] - full_path_xz[i-1][1]
            seg_len = (dx**2 + dz**2) ** 0.5
            if seg_len < 1e-6:
                continue
            accum += seg_len
            if accum >= step_size:
                waypoints_3d.append([full_path_xz[i][0], yv, full_path_xz[i][1]])
                accum = 0.0
        # 确保终点
        end_3d = [full_path_xz[-1][0], yv, full_path_xz[-1][1]]
        if not waypoints_3d or self._dist2(waypoints_3d[-1], end_3d) > 0.1:
            waypoints_3d.append(end_3d)
        
        return waypoints_3d if waypoints_3d else None

    def get_unvisited_exploration_targets(
        self,
        agent_pos: list,
        visited_positions: List[list],
        prefer_junctions: bool = True,
        max_targets: int = 5,
    ) -> List[Dict]:
        """获取未访问的探索目标 (用于深度探索).
        
        优先级:
          1. 交叉点（degree >= 3, 通常是门口/走廊交汇）
          2. 跨房间连接点
          3. 距已访问位置最远的节点
          
        Args:
            agent_pos: [x, y, z] 当前位置
            visited_positions: 已访问位置列表
            prefer_junctions: 优先选择交叉点
            max_targets: 返回最多几个目标
            
        Returns:
            [{"node_id": int, "pos_world": (x,z), "score": float, "type": str}, ...]
        """
        if not self.has_graph():
            return []

        ax, az = float(agent_pos[0]), float(agent_pos[2])
        candidates = []

        for n, data in self.graph.nodes(data=True):
            pw = data.get("pos_world")
            if pw is None:
                continue
            
            # 距离已访问位置的最小距离（越远越好）
            min_visit_dist = float("inf")
            for vp in visited_positions:
                d = ((pw[0] - vp[0])**2 + (pw[1] - vp[2])**2) ** 0.5
                if d < min_visit_dist:
                    min_visit_dist = d
            
            # 太近的跳过 (< 1m 距离最近已访问点)
            if min_visit_dist < 1.0:
                self._visited_nodes.add(n)
                continue

            deg = self.graph.degree(n)
            room_id = data.get("room_id", "")
            
            # 评分
            score = min_visit_dist  # 基础分: 距离
            node_type = "normal"
            
            if prefer_junctions and deg >= 3:
                score *= 2.0  # 交叉点加倍
                node_type = "junction"
            
            # 跨房间连接点额外加分
            neighbor_rooms = set()
            for nb in self.graph.neighbors(n):
                nb_room = self.graph.nodes[nb].get("room_id", "")
                if nb_room:
                    neighbor_rooms.add(nb_room)
            if len(neighbor_rooms) > 1:
                score *= 1.5
                node_type = "cross_room"

            candidates.append({
                "node_id": n,
                "pos_world": pw,
                "score": score,
                "type": node_type,
                "degree": deg,
                "room_id": room_id,
            })

        # 按分数排序, 取 top-k
        candidates.sort(key=lambda x: -x["score"])
        return candidates[:max_targets]

    @staticmethod
    def _dist2(a, b):
        return ((a[0]-b[0])**2 + (a[2]-b[2])**2) ** 0.5

    # ------------------------------------------------------------------
    # 序列化 / 反序列化
    # ------------------------------------------------------------------

    def save(self, path: str):
        """将 Voronoi 导航图保存到 JSON 文件.

        格式: node_link_data (networkx 标准序列化).
        同时将 tuple 类型的 pos_world / pos_grid 转为 list 以确保 JSON 兼容.
        """
        import json
        if self.graph is None or self.graph.number_of_nodes() == 0:
            print(f"[VoronoiNavigator] 无导航图, 跳过保存")
            return
        data = nx.node_link_data(self.graph)
        # tuple → list for JSON
        for node in data.get("nodes", []):
            for key in ("pos_world", "pos_grid"):
                if isinstance(node.get(key), tuple):
                    node[key] = list(node[key])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[VoronoiNavigator] 已保存: {path} "
              f"({self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边)")

    @classmethod
    def load(cls, path: str, occ_grid=None) -> "VoronoiNavigator":
        """从 JSON 文件加载 Voronoi 导航图.

        Args:
            path: JSON 文件路径
            occ_grid: 可选 OccupancyGrid (用于后续路径规划的碰撞检测)
        """
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        G = nx.node_link_graph(data)
        # list → tuple for internal consistency
        for n in G.nodes:
            for key in ("pos_world", "pos_grid"):
                v = G.nodes[n].get(key)
                if isinstance(v, list):
                    G.nodes[n][key] = tuple(v)
        nav = cls(occ_grid, voronoi_graph=G)
        print(f"[VoronoiNavigator] 已加载: {path} "
              f"({G.number_of_nodes()} 节点, {G.number_of_edges()} 边)")
        return nav


def visualize_voronoi_graph(
    G: "nx.Graph",
    occ_grid,
    room_labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """在占据栅格上可视化 Voronoi 图.

    Returns:
        BGR image
    """
    # 基础: 栅格图像
    vis = occ_grid.to_image(navigable=False, show_layers=False)

    if room_labels is not None:
        # 半透明房间着色
        n_rooms = int(room_labels.max())
        np.random.seed(42)
        for rid in range(1, n_rooms + 1):
            mask = room_labels == rid
            color = np.random.randint(60, 200, size=3).tolist()
            vis[mask] = [
                int(vis[mask, 0].mean() * 0.5 + color[0] * 0.5),
                int(vis[mask, 1].mean() * 0.5 + color[1] * 0.5),
                int(vis[mask, 2].mean() * 0.5 + color[2] * 0.5),
            ]

    # 画边
    for n1, n2 in G.edges:
        g1 = G.nodes[n1].get("pos_grid")
        g2 = G.nodes[n2].get("pos_grid")
        if g1 and g2:
            cv2.line(vis, g1, g2, (0, 200, 255), 1, cv2.LINE_AA)

    # 画节点
    for n, data in G.nodes(data=True):
        g = data.get("pos_grid")
        if g is None:
            continue
        deg = G.degree(n)
        if deg >= 3:
            # 交叉点: 大圆
            cv2.circle(vis, g, 4, (0, 0, 255), -1)
        else:
            cv2.circle(vis, g, 2, (0, 255, 0), -1)

    if save_path:
        cv2.imwrite(save_path, vis)

    return vis
