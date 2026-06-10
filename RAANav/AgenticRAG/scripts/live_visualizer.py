"""实时可视化模块 — 在导航过程中实时显示探索状态.

仪表盘布局 (2×2 网格):
  ┌─────────────────┬─────────────────┐
  │  RGB 帧 (正前方)  │  占据栅格 + 轨迹  │
  │  + 检测框标注     │  + Agent 位置    │
  ├─────────────────┼─────────────────┤
  │  概率热力图       │  信息面板         │
  │  + 负核 + 目标点  │  (stats/log)     │
  └─────────────────┴─────────────────┘

使用方法:
    viz = LiveVisualizer(panel_size=320)
    viz.update(rgb=..., occ_grid=..., agent_pos=..., ...)
    viz.close()

按键:
    q / ESC: 退出
    空格: 暂停/继续
    s: 保存当前帧截图
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class LiveVisualizer:
    """OpenCV 实时可视化仪表盘."""

    WINDOW_NAME = "AgenticRAG Live Explorer"

    def __init__(
        self,
        panel_size: int = 320,
        wait_ms: int = 1,
        save_dir: Optional[str] = None,
        record_video: bool = False,
        fps: int = 5,
    ):
        """
        Args:
            panel_size: 每个面板的宽高 (像素), 总窗口为 2*panel_size × 2*panel_size
            wait_ms: cv2.waitKey 延迟 (ms). 1=实时, 0=逐帧暂停
            save_dir: 如果提供, 每帧保存到此目录
            record_video: 是否录制为 mp4 视频
            fps: 视频帧率
        """
        self.ps = panel_size
        self.wait_ms = wait_ms
        self.save_dir = save_dir
        self.record_video = record_video
        self._paused = False
        self._frame_idx = 0
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._trajectory_pts: List[Tuple[int, int]] = []  # 栅格坐标轨迹

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        if record_video and save_dir:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w, h = 2 * panel_size, 2 * panel_size
            self._video_writer = cv2.VideoWriter(
                os.path.join(save_dir, "exploration.mp4"), fourcc, fps, (w, h),
            )

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 2 * panel_size, 2 * panel_size)

    # ------------------------------------------------------------------
    # 主更新接口
    # ------------------------------------------------------------------

    def update(
        self,
        step: int,
        rgb: Optional[np.ndarray] = None,
        detections: Optional[List[Dict[str, Any]]] = None,
        occ_grid=None,  # OccupancyGrid 实例
        agent_pos: Optional[np.ndarray] = None,
        nav_target: Optional[list] = None,
        prob_field: Optional[np.ndarray] = None,
        grid_meta: Optional[Dict] = None,
        neg_kernels: int = 0,
        nav_mode: str = "",
        target_name: str = "",
        target_found: bool = False,
        n_objects: int = 0,
        n_unique_labels: int = 0,
        visited_positions: Optional[List[list]] = None,
        gt_target_positions: Optional[List[List[float]]] = None,
        objects_on_map: Optional[List[Dict[str, Any]]] = None,
        n_rooms: int = 0,
        exploration_progress: float = 0.0,
        n_floors: int = 1,
        current_floor_id: str = "",
    ) -> bool:
        """更新仪表盘. 返回 False 表示用户请求退出.
        
        新增参数:
            objects_on_map: 地图上的物体 [{"label": str, "pos_3d": [x,y,z], "color": (B,G,R)}]
            n_rooms: 已检测到的房间数
            exploration_progress: 探索进度 0.0-1.0
            n_floors: 楼层总数
            current_floor_id: 当前所在楼层 ID
        """
        ps = self.ps
        canvas = np.zeros((2 * ps, 2 * ps, 3), dtype=np.uint8)

        # Panel 1 (左上): RGB + 检测框
        panel_rgb = self._make_rgb_panel(rgb, detections, target_name, step)
        canvas[0:ps, 0:ps] = panel_rgb

        # Panel 2 (右上): 占据栅格 + 轨迹
        panel_grid = self._make_grid_panel(occ_grid, agent_pos, nav_target, visited_positions,
                                           gt_target_positions, objects_on_map)
        canvas[0:ps, ps:2*ps] = panel_grid

        # Panel 3 (左下): 概率热力图 / 物体地图
        panel_prob = self._make_prob_panel(prob_field, grid_meta, agent_pos, nav_target, occ_grid,
                                           gt_target_positions, objects_on_map)
        canvas[ps:2*ps, 0:ps] = panel_prob

        # Panel 4 (右下): 信息面板
        panel_info = self._make_info_panel(
            step, nav_mode, neg_kernels, target_name, target_found,
            n_objects, n_unique_labels, agent_pos, n_rooms, exploration_progress,
            n_floors=n_floors, current_floor_id=current_floor_id,
        )
        canvas[ps:2*ps, ps:2*ps] = panel_info

        # 显示
        cv2.imshow(self.WINDOW_NAME, canvas)

        # 录制
        if self._video_writer is not None:
            self._video_writer.write(canvas)

        # 保存帧
        if self.save_dir:
            cv2.imwrite(
                os.path.join(self.save_dir, f"viz_{self._frame_idx:04d}.jpg"),
                canvas,
            )

        self._frame_idx += 1

        # 按键处理
        return self._handle_keys()

    # ------------------------------------------------------------------
    # 面板生成
    # ------------------------------------------------------------------

    def _make_rgb_panel(
        self,
        rgb: Optional[np.ndarray],
        detections: Optional[List[Dict]],
        target_name: str,
        step: int,
    ) -> np.ndarray:
        ps = self.ps
        if rgb is None:
            panel = np.zeros((ps, ps, 3), dtype=np.uint8)
            cv2.putText(panel, "No RGB", (ps//3, ps//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            return panel

        # RGB → BGR for OpenCV
        if rgb.shape[2] == 4:
            bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 画检测框
        if detections:
            for det in detections:
                bbox = det.get("bbox_xyxy")
                label = det.get("label", "")
                conf = det.get("confidence", 0)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                is_target = (label.lower() == target_name.lower())
                color = (0, 255, 0) if is_target else (0, 165, 255)
                thickness = 2 if is_target else 1
                cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness)
                text = f"{label} {conf:.2f}"
                cv2.putText(bgr, text, (x1, max(y1 - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # 缩放到 panel_size
        panel = cv2.resize(bgr, (ps, ps))
        # 标题
        cv2.putText(panel, f"Step {step} | RGB + Detections", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return panel

    def _make_grid_panel(
        self,
        occ_grid,
        agent_pos: Optional[np.ndarray],
        nav_target: Optional[list],
        visited_positions: Optional[List[list]],
        gt_target_positions: Optional[List[List[float]]] = None,
        objects_on_map: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        ps = self.ps
        if occ_grid is None:
            panel = np.zeros((ps, ps, 3), dtype=np.uint8)
            cv2.putText(panel, "No Grid", (ps//3, ps//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            return panel

        # 从 OccupancyGrid 生成彩色图
        base = occ_grid.to_image(navigable=True, show_layers=True)

        # GT 目标物体位置 (金色 ★)
        if gt_target_positions:
            for gtp in gt_target_positions:
                gc = occ_grid.world_to_grid(np.array([gtp[0], gtp[2]]))
                cv2.drawMarker(base, (int(gc[0]), int(gc[1])), (0, 215, 255),
                               cv2.MARKER_STAR, 10, 2)

        # 绘制轨迹
        if visited_positions:
            pts = []
            for p in visited_positions:
                gc = occ_grid.world_to_grid(np.array([p[0], p[2]]))
                pts.append((int(gc[0]), int(gc[1])))
            for i in range(1, len(pts)):
                cv2.line(base, pts[i - 1], pts[i], (0, 200, 255), 1)
            # 起点 (蓝)
            if pts:
                cv2.circle(base, pts[0], 4, (255, 0, 0), -1)

        # 绘制物体位置 (彩色小菱形)
        if objects_on_map and occ_grid is not None:
            for obj in objects_on_map:
                p3d = obj.get("pos_3d")
                if p3d is None:
                    continue
                gc = occ_grid.world_to_grid(np.array([p3d[0], p3d[2]]))
                cx, cy = int(gc[0]), int(gc[1])
                color = obj.get("color", (255, 200, 0))
                cv2.drawMarker(base, (cx, cy), color, cv2.MARKER_DIAMOND, 4, 1)

        # Agent 当前位置 (绿色大圆)
        if agent_pos is not None:
            gc = occ_grid.world_to_grid(np.array([agent_pos[0], agent_pos[2]]))
            cv2.circle(base, (int(gc[0]), int(gc[1])), 5, (0, 255, 0), -1)

        # 导航目标点 (红色 ×)
        if nav_target is not None:
            gc = occ_grid.world_to_grid(np.array([nav_target[0], nav_target[2]]))
            cv2.drawMarker(base, (int(gc[0]), int(gc[1])), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 8, 2)

        # 缩放
        h, w = base.shape[:2]
        scale = ps / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(base, (new_w, new_h))

        # 居中放置
        panel = np.zeros((ps, ps, 3), dtype=np.uint8)
        y_off = (ps - new_h) // 2
        x_off = (ps - new_w) // 2
        panel[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        cv2.putText(panel, "Occupancy Grid + Trajectory", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return panel

    def _make_prob_panel(
        self,
        prob_field: Optional[np.ndarray],
        grid_meta: Optional[Dict],
        agent_pos: Optional[np.ndarray],
        nav_target: Optional[list],
        occ_grid=None,
        gt_target_positions: Optional[List[List[float]]] = None,
        objects_on_map: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        ps = self.ps
        if prob_field is None:
            # 无概率场时显示物体地图摘要
            return self._make_object_map_panel(occ_grid, agent_pos, objects_on_map)

        # 归一化到 [0, 255]
        pf = prob_field.copy()
        pmax = pf.max()
        if pmax > 0:
            pf = (pf / pmax * 255).astype(np.uint8)
        else:
            pf = np.zeros_like(pf, dtype=np.uint8)

        # 应用热力图
        heatmap = cv2.applyColorMap(pf, cv2.COLORMAP_JET)

        # 叠加 Agent 和目标位置
        if grid_meta is not None:
            res = grid_meta["resolution"]
            ox = grid_meta["origin_x"]
            oz = grid_meta["origin_z"]

            if agent_pos is not None:
                c = int(round((agent_pos[0] - ox) / res))
                r = int(round((agent_pos[2] - oz) / res))
                if 0 <= r < heatmap.shape[0] and 0 <= c < heatmap.shape[1]:
                    cv2.circle(heatmap, (c, r), 4, (0, 255, 0), -1)

            if nav_target is not None:
                c = int(round((nav_target[0] - ox) / res))
                r = int(round((nav_target[2] - oz) / res))
                if 0 <= r < heatmap.shape[0] and 0 <= c < heatmap.shape[1]:
                    cv2.drawMarker(heatmap, (c, r), (255, 255, 255),
                                   cv2.MARKER_TILTED_CROSS, 8, 2)
        elif occ_grid is not None:
            if agent_pos is not None:
                gc = occ_grid.world_to_grid(np.array([agent_pos[0], agent_pos[2]]))
                cv2.circle(heatmap, (int(gc[0]), int(gc[1])), 4, (0, 255, 0), -1)
            if nav_target is not None:
                gc = occ_grid.world_to_grid(np.array([nav_target[0], nav_target[2]]))
                cv2.drawMarker(heatmap, (int(gc[0]), int(gc[1])), (255, 255, 255),
                               cv2.MARKER_TILTED_CROSS, 8, 2)

        # GT 目标位置 (金色 ★)
        if gt_target_positions:
            for gtp in gt_target_positions:
                if grid_meta is not None:
                    res = grid_meta["resolution"]
                    ox = grid_meta["origin_x"]
                    oz = grid_meta["origin_z"]
                    c = int(round((gtp[0] - ox) / res))
                    r = int(round((gtp[2] - oz) / res))
                elif occ_grid is not None:
                    gc = occ_grid.world_to_grid(np.array([gtp[0], gtp[2]]))
                    c, r = int(gc[0]), int(gc[1])
                else:
                    continue
                if 0 <= r < heatmap.shape[0] and 0 <= c < heatmap.shape[1]:
                    cv2.drawMarker(heatmap, (c, r), (0, 215, 255),
                                   cv2.MARKER_STAR, 10, 2)

        # 缩放
        h, w = heatmap.shape[:2]
        scale = ps / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(heatmap, (new_w, new_h))

        panel = np.zeros((ps, ps, 3), dtype=np.uint8)
        y_off = (ps - new_h) // 2
        x_off = (ps - new_w) // 2
        panel[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        cv2.putText(panel, f"Prob Field (max={pmax:.4f})", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return panel

    def _make_info_panel(
        self,
        step: int,
        nav_mode: str,
        neg_kernels: int,
        target_name: str,
        target_found: bool,
        n_objects: int,
        n_unique_labels: int,
        agent_pos: Optional[np.ndarray],
        n_rooms: int = 0,
        exploration_progress: float = 0.0,
        n_floors: int = 1,
        current_floor_id: str = "",
    ) -> np.ndarray:
        ps = self.ps
        panel = np.zeros((ps, ps, 3), dtype=np.uint8)

        floor_info = f"Floor: {current_floor_id}" if current_floor_id else f"Floor: F0"
        if n_floors > 1:
            floor_info += f" ({n_floors} total)"

        lines = [
            f"Target: \"{target_name}\"",
            f"Step: {step}",
            f"Mode: {nav_mode or 'init'}",
            floor_info,
            f"Map Objects: {n_objects}",
            f"Rooms: {n_rooms}",
        ]
        if exploration_progress > 0:
            pct = min(exploration_progress * 100, 100)
            lines.append(f"Coverage: {pct:.1f}%")
        if neg_kernels > 0:
            lines.append(f"Neg Kernels: {neg_kernels}")
        if n_unique_labels > 0:
            lines.append(f"Unique Labels: {n_unique_labels}")
        if agent_pos is not None:
            lines.append(f"Pos: [{agent_pos[0]:.1f}, {agent_pos[1]:.1f}, {agent_pos[2]:.1f}]")

        if target_found:
            lines.append("")
            lines.append("*** TARGET FOUND! ***")

        y = 30
        for line in lines:
            color = (0, 255, 0) if "TARGET FOUND" in line else (220, 220, 220)
            scale = 0.5 if "TARGET FOUND" in line else 0.4
            cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
            y += 22

        # 操作提示
        y = ps - 40
        hints = ["[Q/ESC] Quit  [Space] Pause  [S] Screenshot"]
        for hint in hints:
            cv2.putText(panel, hint, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
            y += 16

        cv2.putText(panel, "Info Panel", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return panel

    # ------------------------------------------------------------------
    # 物体地图面板 (替代空概率场)
    # ------------------------------------------------------------------

    def _make_object_map_panel(
        self,
        occ_grid,
        agent_pos: Optional[np.ndarray],
        objects_on_map: Optional[List[Dict[str, Any]]],
    ) -> np.ndarray:
        """当无概率场时, 显示物体分布地图."""
        ps = self.ps
        if occ_grid is None or not objects_on_map:
            panel = np.zeros((ps, ps, 3), dtype=np.uint8)
            label = "Object Map (0 objects)" if not objects_on_map else "Object Map"
            cv2.putText(panel, label, (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(panel, "Detecting...", (ps // 3, ps // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            return panel

        # 暗色栅格底图
        base = np.zeros((occ_grid.shape[0], occ_grid.shape[1], 3), dtype=np.uint8)
        free = occ_grid.grid == 1
        base[free] = (40, 40, 40)  # dark gray for free space

        # 绘制物体 (彩色圆 + 标签)
        label_counts: Dict[str, int] = {}
        for obj in objects_on_map:
            p3d = obj.get("pos_3d")
            if p3d is None:
                continue
            gc = occ_grid.world_to_grid(np.array([p3d[0], p3d[2]]))
            cx, cy = int(gc[0]), int(gc[1])
            color = obj.get("color", (0, 200, 255))
            label = obj.get("label", "?")
            cv2.circle(base, (cx, cy), 3, color, -1)
            label_counts[label] = label_counts.get(label, 0) + 1

        # Agent
        if agent_pos is not None:
            gc = occ_grid.world_to_grid(np.array([agent_pos[0], agent_pos[2]]))
            cv2.circle(base, (int(gc[0]), int(gc[1])), 5, (0, 255, 0), -1)

        # 缩放
        h, w = base.shape[:2]
        scale = ps / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(base, (new_w, new_h))

        panel = np.zeros((ps, ps, 3), dtype=np.uint8)
        y_off = (ps - new_h) // 2
        x_off = (ps - new_w) // 2
        panel[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # 右侧显示 Top 物体标签
        top_labels = sorted(label_counts.items(), key=lambda x: -x[1])[:8]
        tx = ps - 140
        ty = 30
        for lbl, cnt in top_labels:
            cv2.putText(panel, f"{lbl}: {cnt}", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)
            ty += 14

        cv2.putText(panel, f"Object Map ({len(objects_on_map)} objs, {len(label_counts)} types)",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return panel

    # ------------------------------------------------------------------
    # 按键处理
    # ------------------------------------------------------------------

    def _handle_keys(self) -> bool:
        """处理按键. 返回 False 表示退出."""
        while True:
            key = cv2.waitKey(self.wait_ms) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                return False
            if key == ord(' '):
                self._paused = not self._paused
                if self._paused:
                    print("[LiveViz] 已暂停 - 按空格继续")
                else:
                    print("[LiveViz] 继续")
            if key == ord('s') and self.save_dir:
                # 已自动保存, 打印提示
                print(f"[LiveViz] 截图已保存: viz_{self._frame_idx-1:04d}.jpg")
            if not self._paused:
                break
            # 暂停时等待下一个按键
            self.wait_ms = 0  # 阻塞等待
        # 恢复正常等待
        self.wait_ms = max(self.wait_ms, 1)
        return True

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------

    def close(self):
        """释放资源."""
        if self._video_writer is not None:
            self._video_writer.release()
        cv2.destroyWindow(self.WINDOW_NAME)
