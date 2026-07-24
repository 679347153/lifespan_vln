# 输出文件说明与 Debug 指南

本文档说明 `run_temporal_habitat_loop` 的主要输出文件和常见失败定位方式。

## 1. 主要输出目录

运行目录示例：

```text
benchmark/eval/<run>/
```

常见输出：

```text
run.jsonl
memory_diagnostics.jsonl
candidate_traces.jsonl
observation_traces.jsonl
memory_updates.jsonl
negative_feedback.jsonl
layout_transitions.jsonl
summary.json
temporal_summary.json
perception_summary.json
scene_geometry/<scene>.json
scene_memory/<scene>/scene_memory_final.json
memory_snapshots/<scene>/*.json
visual_debug/index.html
videos/<episode_id>/<subtask_id>/search.mp4
```

## 2. 文件用途

### `run.jsonl`

每行对应一个 episode，包含 trajectory、subtask traces、最终 pose 和路径长度。

### `memory_diagnostics.jsonl`

每行对应一个 subtask 的诊断结果，包括：

```text
found
perception_found
mra
ghr3 / ghr5
min_dist
final_dist
fallback_count
peaks
target_detection_summary
```

### `candidate_traces.jsonl`

每行对应一次 round 的候选与选点结果：

```text
top_candidates
selected_candidate
selected_room_id
planning_mode
planned_pose
final_pose
path
fallback_reason
video_path
```

### `observation_traces.jsonl`

记录每次环视检测结果：

```text
label
bbox_xyxy
pos_2d / pos_3d
position_confidence
position_source
depth_valid_ratio
clip_embedding_dim
agent_pose
view_index
```

### `memory_updates.jsonl`

记录记忆事件：

```text
new
merged
migrated
negative_feedback
duplicate_compaction
```

可用于回放阶段性 memory。

### `scene_memory_final.json`

整个 scene 最终记忆地图。注意：这是最终状态，不等价于某个 subtask 当时拥有的记忆。

## 3. 常见失败链路

### 3.1 感知为空

现象：

```text
observation_traces.jsonl 为空或 detections 很少
failure_bucket = perception_empty
```

排查：

```bash
python log_filter.py --run "curl http://127.0.0.1:50220/health"
python log_filter.py --run "python remote_vision_server/test_client.py --base-url http://127.0.0.1:50220 --image objects_images/Camera_01.webp --prompt 'camera . chair . table .'"
```

### 3.2 候选为空

现象：

```text
candidate_recall_rate 低
rounds_without_candidates 多
fallback_reason = no_candidate
```

排查：

- 目标名称映射是否正确。
- `--clip-min-score` 是否过高。
- observation 是否写入 memory。
- remote CLIP embedding 是否返回。

### 3.3 候选接近但最终失败

现象：

```text
GHR@5 高
Found 低
avg_final_minus_candidate_min_dist 大
```

排查：

- 导航选点是否绕路或反复。
- `--planning-mode room` 是否启用。
- `selected_room_id` 是否频繁切换。
- approach pose 是否距离目标过远。
- spatial confirmation margin 是否过严。

### 3.4 物体重复 track

现象：

```text
同一 label 近距离出现多个 memory tracks
duplicate_compaction 事件多
```

排查：

- `position_confidence` 是否低。
- `depth_iqr` 是否过大。
- 同一物体多视角反投影是否偏移。
- 可适当调大合并阈值，但避免同房间多个同类物体被误合并。

## 4. 推荐 Debug 顺序

1. 看 `perception_summary.json` 和 observation label top-k。
2. 看 `diagnosis.json` 的 failure bucket。
3. 看 `candidate_traces.jsonl` 是否有候选。
4. 看 HTML 中 target、candidate、trajectory 的空间关系。
5. 切换 HTML memory 模式：
   - 阶段累计
   - 仅当前轮增量
   - 最终快照
6. 对问题 subtask 开启 `--record-search-video`，查看实时过程。
