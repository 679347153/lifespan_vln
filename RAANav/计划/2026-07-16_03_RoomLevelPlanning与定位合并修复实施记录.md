# 03 Room-Level Planning 与定位/合并修复实施记录

日期：2026-07-16

## 1. 实施摘要

本轮将 temporal Habitat loop 的默认导航策略从 object-centric Top-1 改为 room-first planning，并修复观测物体位置抖动和重复 track 问题。

核心目标：

```text
减少路径来回往复
降低同一物体被记录成多个 memory tracks 的概率
提升候选位置稳定性
保留 object 模式作为消融和回退
```

## 2. 已修改内容

### 2.1 RGB-D 定位修复

修改文件：

```text
RAANav/AgenticRAG/benchmark_adapter/habitat_vision_loop.py
```

变化：

```text
mask/bbox 单点定位 -> 有效深度点批量反投影 + world 坐标中位数
mask 深度质量过滤：valid_ratio 与 depth_iqr
无效或离散深度降级到 bbox_center_depth
```

每个 detection 新增诊断字段：

```text
position_confidence
position_source
depth_valid_ratio
depth_iqr
world_position_sample_count
```

### 2.2 Memory 合并与 track 平滑

修改文件：

```text
RAANav/AgenticRAG/benchmark_adapter/temporal_memory.py
```

变化：

```text
update_from_detections() 前先做同批 detection consolidation
同 label 且距离 < 0.6m 或 CLIP 相似 > 0.90 的 detection 合并
track 更新改为 position_confidence 加权平滑
低置信 detection 使用更小位置更新权重
低置信 detection 合并半径放宽到 1.25m
同一 step 内禁止把远距离同 label detection 记为 migrated
新增 compact_duplicates()，episode 后合并 same label / same room / 近距离 / 高 CLIP 相似重复 track
```

memory event 新增诊断字段：

```text
position_confidence
position_source
raw_pos_2d
smoothed_pos_2d
consolidated_detection_count
merge_reason
duplicate_compaction
```

`scene_memory_final.json` 中每个 track 顶层导出：

```text
position_history
position_confidence
smoothed_pos_2d
```

### 2.3 Room-Level Planning

修改文件：

```text
RAANav/AgenticRAG/benchmark_adapter/run_temporal_habitat_loop.py
```

新增参数：

```bash
--planning-mode object|room
--room-top-k-objects 5
--room-commit-rounds 2
--room-switch-margin 0.12
--room-revisit-window 4
--room-negative-penalty-weight 0.20
--room-distance-weight 0.25
```

默认：

```text
planning_mode = room
```

room score：

```text
S_room =
  0.45 * max_object_score
+ 0.20 * mean_top3_object_score
+ 0.15 * target_room_belief
+ 0.10 * exploration_gain
- room_distance_weight * normalized_geodesic_distance
- 0.20 * room_revisit_penalty
- 0.15 * room_switch_penalty
```

导航点选择：

```text
优先选择 selected_room 内 strong object 的 approach pose
否则使用 room representative pose
若失败则回退 object Top-1
最后才 random navmesh fallback
```

防往复机制：

```text
room commitment
room switch margin
recent room revisit penalty
failed room penalty
recent pose penalty for approach pose
```

### 2.4 可视化字段

修改文件：

```text
RAANav/AgenticRAG/benchmark_adapter/visualize_temporal_run.py
```

候选轮次表新增展示：

```text
selected_room_id
planning_mode
selected_pose_source
room_switch_reason
```

## 3. 已执行检查

静态检查：

```bash
python -m py_compile \
  RAANav/AgenticRAG/benchmark_adapter/habitat_vision_loop.py \
  RAANav/AgenticRAG/benchmark_adapter/temporal_memory.py \
  RAANav/AgenticRAG/benchmark_adapter/run_temporal_habitat_loop.py \
  RAANav/AgenticRAG/benchmark_adapter/visualize_temporal_run.py \
  RAANav/AgenticRAG/benchmark_adapter/diagnose_temporal_run.py
```

结果：

```text
通过
```

合成 memory smoke：

```text
两个近距离同 label chair detection -> 1 个 track
consolidated_detection_count = 2
smoothed_pos_2d 正常写入
compact_duplicates 后 track_count 保持稳定
```

参数 smoke：

```text
默认 planning_mode = room
--planning-mode object 可正常回退
```

补充修复：

```text
同批远距离同类物体误合并问题已修复。
原因：第二个同 label detection 会把同一批刚创建的 track 当作可迁移对象，在高 CLIP 相似时触发 migrated。
修复：update_from_detections() 内新增 batch_seen_track_ids，同一批已创建/更新的 track 若与新 detection 距离超过自适应 merge 半径，则禁止 migrated，改为新建 track。
```

补充合成验证：

```text
近距离同 label chair detection -> 1 个 track
远距离同 label chair detection -> 2 个 track
同 room 近距离高 CLIP 相似重复 track -> compact_duplicates 后 1 个 track，并产生 duplicate_compaction 事件
```

## 4. 推荐真实闭环验证命令

```bash
python -m RAANav.AgenticRAG.benchmark_adapter.run_temporal_habitat_loop \
  --split-manifest benchmark/splits/benchmark_split_longterm_v1.json \
  --split val \
  --scene 00808-y9hTuugGdiq \
  --layout-limit 1 \
  --episodes-per-layout 1 \
  --output-dir benchmark/eval/room_planning_smoke \
  --planning-mode room \
  --max-steps-per-subtask 80 \
  --max-rounds 5 \
  --micro-steps-per-round 8 \
  --n-views 4 \
  --clip-min-score 0.15 \
  --remote-clip-none-min-score 0.25 \
  --remote-clip-none-score-cap 0.30 \
  --remote-clip-weak-min-score 0.25 \
  --target-name-map RAANav/AgenticRAG/config/target_name_map_longterm_v1.json \
  --normalize-target-names \
  --target-confirmation-mode spatial \
  --target-confirmation-margin 1.0 \
  --perception-backend remote \
  --remote-vision-base-url http://127.0.0.1:50220 \
  --load-layout-objects \
  --no-euclidean-fallback
```

对照消融：

```bash
python -m RAANav.AgenticRAG.benchmark_adapter.run_temporal_habitat_loop \
  ... \
  --planning-mode object \
  --output-dir benchmark/eval/object_planning_ablation
```

## 5. 复测重点

```text
track_count 是否下降或不再异常膨胀
duplicate_compaction 事件数量
migrated / memory_events 比例是否下降
selected_room 切换次数是否下降
avg_top1_dist 是否下降
avg_final_dist 是否下降
路径往复是否减少
Found/Subtasks 是否不低于当前基线
```

## 6. 注意事项

第一版 room 仍使用 pseudo_grid，不是真实语义房间。若路径稳定性提升但成功率仍不足，下一阶段应接入 semantic region 或真实房间轮廓，并进一步改进 approach pose 的可见性估计。
