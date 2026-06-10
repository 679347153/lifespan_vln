"""楼层检测 — 基于高度直方图的多楼层分离.

Pipeline (参考 HOV-SG graph.py):
  1. 累积 3D 点云的 Y 坐标直方图
  2. 高斯平滑
  3. 峰值检测 (find_peaks)
  4. DBSCAN 聚类峰值
  5. 配对峰值为楼层边界 [y_min, y_max]

适配 AgenticRAG: 从 OccupancyGrid 的深度帧累积中获取高度分布.
对于单楼层场景 (大多数 HM3D), 返回单个楼层.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FloorLevel:
    """检测到的楼层."""
    floor_id: int
    y_min: float        # 楼层底部高度 (Habitat Y轴, 向上为正)
    y_max: float        # 楼层顶部高度
    y_zero: float       # 地面高度 (最低可行走面)
    height: float       # 楼层净高 (y_max - y_zero)
    point_count: int = 0


class FloorDetector:
    """从累积的 3D 高度信息检测楼层."""

    def __init__(
        self,
        bin_resolution: float = 0.01,   # 直方图分辨率 (m)
        min_floor_sep: float = 0.2,     # 最小层间距 (m)
        peak_percentile: float = 90.0,  # 峰值检测阈值 (百分位)
        dbscan_eps: float = 1.0,        # DBSCAN 聚类 eps (bin 数)
        sigma: float = 2.0,             # 高斯平滑 sigma
    ):
        self.bin_resolution = bin_resolution
        self.min_floor_sep = min_floor_sep
        self.peak_percentile = peak_percentile
        self.dbscan_eps = dbscan_eps
        self.sigma = sigma

        # 累积高度值
        self._y_values: List[float] = []

    def add_heights(self, y_values: np.ndarray):
        """添加一批 Y 坐标值."""
        self._y_values.extend(y_values.tolist())

    def detect_floors(self) -> List[FloorLevel]:
        """执行楼层检测.

        Returns:
            list of FloorLevel, 按高度排序
        """
        if len(self._y_values) < 100:
            # 点太少, 返回单楼层
            if self._y_values:
                y_min = min(self._y_values)
                y_max = max(self._y_values)
            else:
                y_min, y_max = 0.0, 3.0
            return [FloorLevel(
                floor_id=0, y_min=y_min, y_max=y_max,
                y_zero=y_min, height=y_max - y_min,
                point_count=len(self._y_values),
            )]

        y_arr = np.array(self._y_values)
        y_min_all, y_max_all = y_arr.min(), y_arr.max()
        y_range = y_max_all - y_min_all

        if y_range < 0.5:
            # 高度范围太小, 单楼层
            return [FloorLevel(
                floor_id=0, y_min=y_min_all, y_max=y_max_all,
                y_zero=y_min_all, height=y_range,
                point_count=len(y_arr),
            )]

        n_bins = max(10, int(y_range / self.bin_resolution))
        hist_counts, bin_edges = np.histogram(y_arr, bins=n_bins)

        # 高斯平滑
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(hist_counts.astype(float), sigma=self.sigma)

        # 峰值检测
        from scipy.signal import find_peaks
        distance = max(1, int(self.min_floor_sep / self.bin_resolution))
        min_height = np.percentile(smoothed, self.peak_percentile)
        peaks, _ = find_peaks(smoothed, distance=distance, height=min_height)

        if len(peaks) < 2:
            # 不足以判断多楼层
            return [FloorLevel(
                floor_id=0, y_min=y_min_all, y_max=y_max_all,
                y_zero=y_min_all, height=y_range,
                point_count=len(y_arr),
            )]

        # DBSCAN 聚类峰值
        from sklearn.cluster import DBSCAN
        peak_locations = bin_edges[peaks]
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=1).fit(
            peak_locations.reshape(-1, 1)
        )
        labels = clustering.labels_
        n_clusters = len(set(labels))

        # 从每个聚类取代表性峰值
        clustered_peaks = []
        for cl_id in range(n_clusters):
            cluster_peak_indices = peaks[labels == cl_id]
            # 取最高的1-2个峰
            sorted_by_height = cluster_peak_indices[np.argsort(smoothed[cluster_peak_indices])]
            if cl_id == 0 or cl_id == n_clusters - 1:
                top = sorted_by_height[-1:]  # 首尾聚类取1个
            else:
                top = sorted_by_height[-2:]  # 中间聚类取2个
            clustered_peaks.extend([bin_edges[p] for p in top])

        clustered_peaks = sorted(clustered_peaks)

        # 配对为楼层边界
        floors = []
        for i in range(0, len(clustered_peaks) - 1, 2):
            floors.append((clustered_peaks[i], clustered_peaks[i + 1]))

        if not floors:
            return [FloorLevel(
                floor_id=0, y_min=y_min_all, y_max=y_max_all,
                y_zero=y_min_all, height=y_range,
                point_count=len(y_arr),
            )]

        # 扩展首尾楼层边界
        floors[0] = ((floors[0][0] + y_min_all) / 2, floors[0][1])
        floors[-1] = (floors[-1][0], (floors[-1][1] + y_max_all) / 2)

        # 构建 FloorLevel 列表
        result = []
        for idx, (fl_min, fl_max) in enumerate(floors):
            mask = (y_arr >= fl_min) & (y_arr < fl_max)
            count = int(np.sum(mask))
            fl_y_zero = float(y_arr[mask].min()) if count > 0 else fl_min
            result.append(FloorLevel(
                floor_id=idx,
                y_min=fl_min,
                y_max=fl_max,
                y_zero=fl_y_zero,
                height=fl_max - fl_y_zero,
                point_count=count,
            ))

        print(f"[FloorDetector] 检测到 {len(result)} 个楼层: "
              + ", ".join(f"F{f.floor_id}[{f.y_min:.2f}, {f.y_max:.2f}]" for f in result))
        return result

    def get_floor_for_height(self, y: float, floors: List[FloorLevel]) -> int:
        """根据高度值返回所属楼层 ID."""
        for f in floors:
            if f.y_min <= y <= f.y_max:
                return f.floor_id
        # 找最近的楼层
        dists = [min(abs(y - f.y_min), abs(y - f.y_max)) for f in floors]
        return floors[int(np.argmin(dists))].floor_id
