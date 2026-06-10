"""P2: Epoch 模拟器 —— 根据 change_script 生成各 epoch 的 floors_now 并执行 merge。

核心规则 (对齐论文设计):
  - SO 物体: 位置不变, 同 obj_id 出现在 floors_now → merge 时 matched → exist_prob 恢复
  - HRO/LRO 移动超阈值: 旧 obj_id 不出现在 floors_now → 老节点 exist_prob 衰减
                        新位置以 NEW obj_id 出现 → 作为新节点加入
  - DO 消失: 不出现在 floors_now → 老节点持续衰减
  - NO 新增: 以新 obj_id 出现在 floors_now

用法:
    from eval.epoch_simulator import EpochSimulator
    sim = EpochSimulator(base_map_path, change_script, config)
    results = sim.run_all_epochs()
"""
from __future__ import annotations
import copy
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 确保项目根目录在 path 中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from semantic_map import Floor, Room, Object
from semantic_map_Update.map_Update import run_merge
from eval.change_script import ChangeScript, ObjectChange


class EpochSimulator:
    """多 epoch 模拟器: 根据 change_script 驱动 merge 循环."""

    def __init__(self, base_map_path: str, change_script: ChangeScript,
                 config: dict, match_dist_threshold: float = 1.0):
        """
        Args:
            base_map_path: T0 语义地图 JSON 路径
            change_script: 变更脚本
            config: map.yaml 配置 dict
            match_dist_threshold: 判定"移动"的距离阈值 (与 merge 匹配阈值对齐)
        """
        self.base_map_path = base_map_path
        self.cs = change_script
        self.cfg = config
        self.match_dist = match_dist_threshold

        # 加载 T0 base floors (作为 GT 模板)
        self.floors_gt: List[Floor] = Floor.from_json(base_map_path)

        # 构建 obj_id → (floor_idx, room_idx, obj_idx) 索引
        self._obj_index: Dict[str, Tuple[int, int, int]] = {}
        for fi, fl in enumerate(self.floors_gt):
            for ri, rm in enumerate(fl.rooms):
                for oi, obj in enumerate(rm.objects):
                    self._obj_index[obj.obj_id] = (fi, ri, oi)

        # obj_id 序列号计数器 (用于 NO 和移动物体的新 ID)
        self._id_counter: int = self._max_existing_seq() + 1

    def _max_existing_seq(self) -> int:
        """从现有 obj_id 中提取最大序列号."""
        max_seq = 0
        for obj_id in self._obj_index:
            parts = obj_id.split('_')
            for p in parts:
                if p.isdigit():
                    max_seq = max(max_seq, int(p))
        return max_seq

    def _gen_new_obj_id(self, label: str, room_id: str) -> str:
        """生成新的 obj_id: {label}_{seq}_{room_id}."""
        self._id_counter += 1
        return f"{label}_{self._id_counter}_{room_id}"

    def _make_new_object(self, label: str, pos_3d: List[float],
                         room_id: str) -> Object:
        """创建一个新 Object 实例 (用于 NO 或移动后的新节点)."""
        obj_id = self._gen_new_obj_id(label, room_id)
        # 构造一个简单的 0.5m 正方形 region
        cx, cz = pos_3d[0], pos_3d[2]
        region = [
            {'x': cx - 0.25, 'y': cz - 0.25},
            {'x': cx + 0.25, 'y': cz - 0.25},
            {'x': cx + 0.25, 'y': cz + 0.25},
            {'x': cx - 0.25, 'y': cz + 0.25},
        ]
        obj = Object(obj_id=obj_id, label=label, region=region)
        obj.pos_3d = list(pos_3d)
        obj.pos_2d = {'x': pos_3d[0], 'y': pos_3d[2]}
        obj.exist_prob = 1.0
        obj.N = 1
        obj.cfd = 0.5
        obj.stability = None  # 由 merge 从 stability_priors 填充
        obj.room_id = room_id
        obj.last_update_time = datetime.now(timezone.utc).strftime(
            '%Y-%m-%dT%H:%M:%SZ')
        return obj

    def generate_floors_now(self, epoch: int) -> List[Floor]:
        """生成第 epoch 轮机器人"看到的"物体列表 (floors_now).

        核心逻辑:
        1. 从 floors_gt 深拷贝 (所有 SO/未变化物体保持原样)
        2. 对 change_script 中有变化的物体: 按规则修改/移除/新增
        3. 全部 exist_prob 置为 1.0 (模拟"刚观测到")
        """
        floors_now = copy.deepcopy(self.floors_gt)

        # 收集本 epoch 需要移除的 obj_id (已移动超阈值或已消失)
        remove_ids: set = set()
        # 收集本 epoch 需要添加的新物体
        add_objects: List[Tuple[str, Object]] = []  # [(room_id, Object)]

        for change in self.cs.changes:
            if change.change_type == 'SO':
                # 静态: 保持不变
                continue

            elif change.change_type in ('HRO', 'LRO'):
                if change.obj_id and change.has_moved(epoch, self.match_dist):
                    # 移动了: 旧 ID 不出现 → 从 floors_now 中移除
                    remove_ids.add(change.obj_id)
                    # 新位置生成新物体
                    new_pos = change.get_pos_3d(epoch)
                    new_room = change.get_room(epoch) or 'R0'
                    if new_pos:
                        new_obj = self._make_new_object(
                            change.label, new_pos, new_room)
                        add_objects.append((new_room, new_obj))
                elif change.obj_id and not change.is_present(epoch):
                    # 不在场 (不应发生于 HRO/LRO, 但防御性处理)
                    remove_ids.add(change.obj_id)

            elif change.change_type == 'DO':
                if change.obj_id and not change.is_present(epoch):
                    # 物体已消失: 不出现在 floors_now
                    remove_ids.add(change.obj_id)

            elif change.change_type == 'NO':
                if change.is_present(epoch) and change.first_epoch == epoch:
                    # 本 epoch 首次出现
                    new_pos = change.get_pos_3d(epoch)
                    new_room = change.get_room(epoch) or 'R0'
                    if new_pos:
                        new_obj = self._make_new_object(
                            change.label, new_pos, new_room)
                        add_objects.append((new_room, new_obj))

        # 执行移除
        for fl in floors_now:
            for rm in fl.rooms:
                rm.objects = [o for o in rm.objects
                              if o.obj_id not in remove_ids]

        # 执行添加
        room_map: Dict[str, Room] = {}
        for fl in floors_now:
            for rm in fl.rooms:
                room_map[rm.room_id] = rm
        for room_id, obj in add_objects:
            target_room = room_map.get(room_id)
            if target_room is None:
                # 如果目标房间不存在, 放到第一个房间
                target_room = floors_now[0].rooms[0]
            target_room.objects.append(obj)

        # 全部 exist_prob 置为 1.0 + 推进时间戳
        day_offset = self.cs.epoch_days[epoch] if epoch < len(self.cs.epoch_days) else epoch
        for fl in floors_now:
            for rm in fl.rooms:
                for obj in rm.objects:
                    obj.exist_prob = 1.0
                    obj.last_update_time = (
                        datetime.now(timezone.utc) +
                        timedelta(days=day_offset)
                    ).strftime('%Y-%m-%dT%H:%M:%SZ')

        return floors_now

    def run_all_epochs(self, verbose: bool = True) -> Dict:
        """执行完整的多 epoch 模拟, 返回结果.

        Returns:
            {
                'floors_history': List[Floor],  # 最终合并后的 floors
                'epoch_snapshots': List[Dict],  # 每 epoch 的快照
                'gt_layouts': List[Dict],       # 每 epoch 的 GT 布局
            }
        """
        floors_history = copy.deepcopy(self.floors_gt)
        epoch_snapshots = []
        gt_layouts = []

        if verbose:
            print(f"\n{'='*70}")
            print(f"Epoch Simulator: {self.cs.scene_id}")
            print(f"{'='*70}")

        for epoch in range(self.cs.n_epochs):
            if verbose:
                day = self.cs.epoch_days[epoch] if epoch < len(self.cs.epoch_days) else '?'
                print(f"\n--- Epoch {epoch} (Day {day}) ---")

            # 1. 生成 floors_now
            floors_now = self.generate_floors_now(epoch)
            n_now = sum(len(rm.objects) for fl in floors_now for rm in fl.rooms)

            # 2. 执行 merge
            floors_history, warnings = run_merge(
                floors_now, floors_history, self.cfg,
                shape_check=True,
                allow_new_floors=True,
                allow_new_rooms=True,
            )
            n_hist = sum(len(rm.objects)
                         for fl in floors_history for rm in fl.rooms)

            # 3. 记录快照
            moved = self.cs.get_moved_obj_ids(epoch, self.match_dist)
            disappeared = self.cs.get_disappeared_obj_ids(epoch)
            new_labels = self.cs.get_new_obj_labels(epoch)

            snapshot = {
                'epoch': epoch,
                'day': self.cs.epoch_days[epoch] if epoch < len(self.cs.epoch_days) else epoch,
                'n_now': n_now,
                'n_history': n_hist,
                'n_moved': len(moved),
                'moved_ids': moved,
                'n_disappeared': len(disappeared),
                'disappeared_ids': disappeared,
                'n_new': len(new_labels),
                'warnings': warnings,
            }
            epoch_snapshots.append(snapshot)

            # 4. 构建 GT 布局
            gt_layout = self._build_gt_layout(epoch)
            gt_layouts.append(gt_layout)

            if verbose:
                print(f"  floors_now: {n_now} objs → merge → "
                      f"floors_history: {n_hist} objs")
                print(f"  moved: {len(moved)}, disappeared: {len(disappeared)}, "
                      f"new: {len(new_labels)}, warnings: {len(warnings)}")
                if moved:
                    print(f"    moved: {moved}")
                if warnings:
                    for w in warnings[:3]:
                        print(f"    ⚠ {w}")

        if verbose:
            print(f"\n{'='*70}")
            print(f"Simulation complete: {self.cs.n_epochs} epochs")
            print(f"Final history: {sum(len(rm.objects) for fl in floors_history for rm in fl.rooms)} objs")
            print(f"{'='*70}\n")

        return {
            'floors_history': floors_history,
            'epoch_snapshots': epoch_snapshots,
            'gt_layouts': gt_layouts,
        }

    def _build_gt_layout(self, epoch: int) -> Dict:
        """构建指定 epoch 的 GT 物体布局 (用于指标计算).

        GT 布局 = base_map 中未受变更脚本影响的物体 + 变更脚本中本 epoch 存在的物体
        """
        gt_objects = []

        # 收集变更脚本中涉及的 obj_id
        changed_ids = set()
        for c in self.cs.changes:
            if c.obj_id:
                changed_ids.add(c.obj_id)

        # 未受影响的物体: 从 base map 取
        for fl in self.floors_gt:
            for rm in fl.rooms:
                for obj in rm.objects:
                    if obj.obj_id not in changed_ids:
                        gt_objects.append({
                            'obj_id': obj.obj_id,
                            'label': obj.label,
                            'pos_3d': obj.pos_3d,
                            'pos_2d': obj.pos_2d,
                            'room_id': rm.room_id,
                            'change_type': 'SO',
                        })

        # 变更脚本中的物体
        for c in self.cs.changes:
            if not c.is_present(epoch):
                continue
            pos = c.get_pos_3d(epoch)
            room = c.get_room(epoch)
            if pos is None:
                continue
            gt_objects.append({
                'obj_id': c.obj_id or f"_new_{c.label}_{epoch}",
                'label': c.label,
                'pos_3d': pos,
                'pos_2d': {'x': pos[0], 'y': pos[2]},
                'room_id': room or 'R0',
                'change_type': c.change_type,
            })

        return {
            'epoch': epoch,
            'day': self.cs.epoch_days[epoch] if epoch < len(self.cs.epoch_days) else epoch,
            'n_objects': len(gt_objects),
            'objects': gt_objects,
        }
