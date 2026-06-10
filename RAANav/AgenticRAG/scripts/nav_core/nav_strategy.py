"""导航策略 — 负向高斯核场 + Frontier 探索.

从 sim_nav_loop.py 提取.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class NegativeGaussianField:
    """管理负向高斯核 — 假阴性位置注入负向高斯抑制已搜过的失败区域."""

    def __init__(
        self,
        sigma_neg_ratio: float = 2.0,
        decay_halflife_steps: int = 10,
        r_trigger: float = 3.0,
        miss_threshold: int = 2,
    ):
        self.sigma_neg_ratio = sigma_neg_ratio
        self.decay_rate = math.log(2) / max(1, decay_halflife_steps)
        self.r_trigger = r_trigger
        self.miss_threshold = miss_threshold
        self.kernels: List[Dict] = []
        self.miss_counts: Dict[str, int] = {}

    def check_and_inject(
        self,
        agent_pos: np.ndarray,
        visible_labels: Set[str],
        watch_list: List[Dict],
        current_step: int,
        sigma_base: float = 1.5,
    ) -> List[str]:
        """检查假阴性并注入负向高斯核."""
        ax, az = float(agent_pos[0]), float(agent_pos[2])
        injected = []

        for obj in watch_list:
            oid = obj["obj_id"]
            ox, oz = obj["wx"], obj["wz"]
            dist = math.sqrt((ax - ox) ** 2 + (az - oz) ** 2)

            if dist > self.r_trigger:
                continue

            if obj["label"] in visible_labels:
                self.miss_counts[oid] = 0
                continue

            self.miss_counts[oid] = self.miss_counts.get(oid, 0) + 1

            if self.miss_counts[oid] >= self.miss_threshold:
                stability = obj.get("stability", 0.5)
                gamma_neg = 0.1 + 0.25 * (1.0 - stability)
                alpha = 0.5 * (1.0 - gamma_neg)
                sigma = sigma_base * self.sigma_neg_ratio

                self.kernels.append({
                    "wx": ox, "wz": oz,
                    "alpha": alpha, "sigma": sigma,
                    "inject_step": current_step,
                    "obj_id": oid,
                })
                injected.append(oid)
                self.miss_counts[oid] = 0
                print(f"    [负核注入] {oid} α={alpha:.3f} σ={sigma:.1f}m (stability={stability:.2f})")

        return injected

    def apply(
        self,
        prob_field: np.ndarray,
        grid_meta: Dict,
        current_step: int,
    ) -> np.ndarray:
        """将所有活跃负向核叠加到概率场."""
        if not self.kernels or prob_field is None:
            return prob_field

        result = prob_field.copy()
        resolution = grid_meta["resolution"]
        origin_x = grid_meta["origin_x"]
        origin_z = grid_meta["origin_z"]
        H, W = result.shape

        for k in self.kernels:
            age = current_step - k["inject_step"]
            effective_alpha = k["alpha"] * math.exp(-self.decay_rate * age)
            if effective_alpha < 0.01:
                continue

            sigma_px = k["sigma"] / resolution
            col = int(round((k["wx"] - origin_x) / resolution))
            row = int(round((k["wz"] - origin_z) / resolution))

            win = int(math.ceil(3 * sigma_px))
            r0, r1 = max(0, row - win), min(H, row + win + 1)
            c0, c1 = max(0, col - win), min(W, col + win + 1)
            if r0 >= r1 or c0 >= c1:
                continue

            yy, xx = np.mgrid[r0:r1, c0:c1]
            neg = effective_alpha * np.exp(
                -((yy - row) ** 2 + (xx - col) ** 2) / (2 * sigma_px ** 2)
            )
            result[r0:r1, c0:c1] -= neg

        np.maximum(result, 0.0, out=result)
        total = result.sum()
        if total > 0:
            result /= total
        return result

    def cleanup(self, current_step: int, max_age: int = 50):
        self.kernels = [k for k in self.kernels
                        if current_step - k["inject_step"] < max_age]

    @property
    def n_active(self) -> int:
        return len(self.kernels)


def frontier_exploration_step(
    agent,  # HabitatAgent
    prob_field_base: Optional[np.ndarray],
    neg_field: NegativeGaussianField,
    grid_meta: Optional[Dict],
    grid: Optional[np.ndarray],
    visited_positions: List[list],
    current_step: int,
    flatness_ratio: float = 3.0,
    occ_grid=None,
) -> Tuple[list, str]:
    """决定下一个导航目标 (负向高斯感知版).

    策略 (三层决策):
    1. 对基础概率场施加负向高斯修正
    2. 叠加访问密度惩罚
    3. 如果修正后场变 "平坦" → Frontier 模式
    4. 否则 → GMM 引导模式
    5. 无概率场时 → Frontier 回退 (从 occ_grid 检测 frontier)

    Returns:
        (target_position, mode_str)  mode_str = "gmm" | "frontier" | "random"
    """
    if prob_field_base is None or grid_meta is None or grid is None:
        # 无概率场: 优先使用 frontier 探索 (替代纯随机)
        if occ_grid is not None:
            target = _frontier_fallback(agent, occ_grid, visited_positions)
            if target is not None:
                return target
        return agent.get_random_navigable_point(), "random"

    resolution = grid_meta["resolution"]
    origin_x = grid_meta["origin_x"]
    origin_z = grid_meta["origin_z"]
    H, W = grid.shape
    # 可通行掩码: FREE 或 UNKNOWN (non-navmesh 模式 A* 可穿越 UNKNOWN)
    free_mask = (grid == 1)
    passable_mask = (grid != 2)  # 排除 OCCUPIED, 允许 FREE + UNKNOWN

    prob_corrected = neg_field.apply(prob_field_base, grid_meta, current_step)

    # 访问密度场
    visit_density = np.zeros((H, W), dtype=np.float64)
    visit_sigma_px = 2.0 / resolution
    for vp in visited_positions:
        vc = int(round((vp[0] - origin_x) / resolution))
        vr = int(round((vp[2] - origin_z) / resolution))
        win = int(math.ceil(3 * visit_sigma_px))
        r0, r1 = max(0, vr - win), min(H, vr + win + 1)
        c0, c1 = max(0, vc - win), min(W, vc + win + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        yy, xx = np.mgrid[r0:r1, c0:c1]
        g = np.exp(-((yy - vr) ** 2 + (xx - vc) ** 2) / (2 * visit_sigma_px ** 2))
        visit_density[r0:r1, c0:c1] += g

    free_vals = prob_corrected[free_mask] if free_mask.any() else np.array([0.0])
    prob_max = free_vals.max()
    prob_mean = free_vals.mean()
    is_flat = (prob_max < 1e-10) or (prob_mean > 0 and prob_max / prob_mean < flatness_ratio)

    if is_flat:
        score = -visit_density.copy()
        score[~passable_mask] = -np.inf
        mode = "frontier"
    else:
        beta = prob_max / 3.0
        score = prob_corrected - beta * visit_density
        score[~passable_mask] = -np.inf
        mode = "gmm"

    max_idx = np.unravel_index(np.argmax(score), score.shape)
    r, c = max_idx
    wx = origin_x + c * resolution
    wz = origin_z + r * resolution

    if agent.use_navmesh:
        import magnum
        agent_y = float(agent.get_position()[1])
        nav_point = agent.sim.pathfinder.snap_point(
            magnum.Vector3(wx, agent_y, wz)
        )
        # 楼层验证: snap 后检查 Y 是否仍在同楼层 (防止 snap 到其他楼层)
        if (agent.sim.pathfinder.is_navigable(nav_point)
                and abs(float(nav_point[1]) - agent_y) < 1.5):
            return [float(nav_point[0]), float(nav_point[1]), float(nav_point[2])], mode
    else:
        # occ_grid 模式: 直接检查栅格可行性
        if agent._occ_grid is not None and agent._occ_grid.is_navigable_at(wx, wz):
            y_val = float(agent.get_position()[1])
            return [wx, y_val, wz], mode
        elif agent._occ_grid is not None:
            # snap 到最近 free cell
            snapped = agent._snap_to_free(np.array([wx, float(agent.get_position()[1]), wz]))
            return [float(snapped[0]), float(snapped[1]), float(snapped[2])], mode

    return agent.get_random_navigable_point(), "random"


def _frontier_fallback(
    agent,
    occ_grid,
    visited_positions: List[list],
    min_area_m2: float = 0.3,
) -> Optional[Tuple[list, str]]:
    """Frontier 探索回退: 当无 GMM 概率场时, 选择最远 frontier.

    评分: 选择距离所有已访问位置最远的 frontier (鼓励广度探索).
    """
    frontiers = occ_grid.get_frontiers(min_area_m2=min_area_m2)
    if len(frontiers) == 0:
        return None

    best_frontier = None
    best_score = -np.inf

    for i in range(len(frontiers)):
        fxy = frontiers[i]
        if visited_positions:
            dists = [
                math.sqrt((fxy[0] - vp[0]) ** 2 + (fxy[1] - vp[2]) ** 2)
                for vp in visited_positions
            ]
            score = min(dists)  # 距最近已访问点的距离
        else:
            score = 0.0
        if score > best_score:
            best_score = score
            best_frontier = fxy

    if best_frontier is None:
        return None

    y_val = float(agent.get_position()[1])
    print(f"    [Frontier] {len(frontiers)} 个候选, "
          f"选择距已访问最远者 ({best_score:.1f}m)")
    return [float(best_frontier[0]), y_val, float(best_frontier[1])], "frontier"


def generate_investigation_viewpoints(
    prob_field: np.ndarray,
    grid: np.ndarray,
    grid_meta: Dict,
    agent_y: float,
    n_viewpoints: int = 3,
    standoff_m: float = 1.5,
) -> List[list]:
    """在 GMM 概率峰值周围生成多个调查视点.

    策略:
      1. 找到 prob_field 的全局峰值 (物体最可能所在位置, 可能在 OCCUPIED cell)
      2. 在峰值周围 standoff_m 距离的圆上均匀采样 n_viewpoints 个候选方向
      3. 对每个方向, 找到最近的 FREE cell 作为观测位置

    Returns:
        视点列表 [[x, y, z], ...], 可能少于 n_viewpoints (如果附近无 free cell)
    """
    resolution = grid_meta["resolution"]
    origin_x = grid_meta["origin_x"]
    origin_z = grid_meta["origin_z"]
    H, W = grid.shape

    # 找全局峰值
    peak_idx = np.unravel_index(np.argmax(prob_field), prob_field.shape)
    peak_r, peak_c = peak_idx
    peak_wx = origin_x + peak_c * resolution
    peak_wz = origin_z + peak_r * resolution

    viewpoints = []
    angles = np.linspace(0, 2 * np.pi, n_viewpoints, endpoint=False)
    # 加入随机偏移避免对称失效
    angles += np.random.uniform(0, 2 * np.pi / n_viewpoints)

    free_mask = (grid == 1)

    for angle in angles:
        # 候选世界坐标 (距峰值 standoff_m)
        cand_wx = peak_wx + standoff_m * np.cos(angle)
        cand_wz = peak_wz + standoff_m * np.sin(angle)

        # 转栅格坐标
        cand_c = int(round((cand_wx - origin_x) / resolution))
        cand_r = int(round((cand_wz - origin_z) / resolution))

        # BFS 找最近 FREE cell (搜索半径限制为 2m)
        max_search_px = int(2.0 / resolution)
        found = _bfs_nearest_free(grid, cand_r, cand_c, free_mask, max_search_px)
        if found is not None:
            vp_r, vp_c = found
            vp_wx = origin_x + vp_c * resolution
            vp_wz = origin_z + vp_r * resolution
            # 去重: 与已有视点距离 > 0.5m
            too_close = False
            for existing in viewpoints:
                d = math.sqrt((vp_wx - existing[0]) ** 2 + (vp_wz - existing[2]) ** 2)
                if d < 0.5:
                    too_close = True
                    break
            if not too_close:
                viewpoints.append([vp_wx, agent_y, vp_wz])

    return viewpoints


def _bfs_nearest_free(
    grid: np.ndarray,
    start_r: int, start_c: int,
    free_mask: np.ndarray,
    max_dist_px: int,
) -> Optional[Tuple[int, int]]:
    """BFS 搜索最近 FREE cell."""
    H, W = grid.shape
    from collections import deque
    # clamp start to grid bounds
    start_r = max(0, min(H - 1, start_r))
    start_c = max(0, min(W - 1, start_c))
    if free_mask[start_r, start_c]:
        return (start_r, start_c)
    visited = set()
    queue = deque([(start_r, start_c, 0)])
    visited.add((start_r, start_c))
    while queue:
        r, c, dist = queue.popleft()
        if dist > max_dist_px:
            break
        if 0 <= r < H and 0 <= c < W and free_mask[r, c]:
            return (r, c)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and 0 <= nr < H and 0 <= nc < W:
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))
    return None
