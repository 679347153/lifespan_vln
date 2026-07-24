# 数据集与 Benchmark

本文档说明动态布局数据集、episode schema 和长期评测目标。

## 1. 数据集目标

数据集模拟同一家庭环境在不同时间点的物体布局变化。每个 `layout` 是同一个 scene 的一个时间状态，用于检验 agent 是否能利用历史经验提升后续搜索能力。

关键字段：

- `scene_name`：HM3D 场景，例如 `00808-y9hTuugGdiq`。
- `layout_id`：动态布局 ID，例如 `layout_000_seed_42`。
- `state_index`：时间状态序号。
- `seen_layout_count_before`：当前 episode 前 agent 已见过多少个 layout。
- `subtasks`：该 episode 中需要连续完成的目标搜索任务。

## 2. 动态 layout 生成链路

```text
HM3D scene
  -> export_scene_info.py
  -> query_rooms_for_objects.py
  -> sample_and_place_objects.py
  -> query_room_receptacle_objects.py
  -> assign_objects_to_receptacle_instances.py
  -> place_objects_on_instances.py
  -> batch_generate_layouts.py
```

最终 layout 通常位于：

```text
results/layouts/<scene>/batch_<time>/layout_*_seed_*.json
```

## 3. Episode 内容

每个 episode JSON 包含：

```text
episode_id
split
scene_name
layout_id
state_index
seen_layout_count_before
scene_state.layout_path
start_pose
max_steps
subtasks
```

每个 subtask 包含：

```text
subtask_id
task_type
target_object
target_object_id
target_position
prompt
image_path
language_prompt
success_radius
metadata
```

## 4. 三类任务

- `open_vocab`：开放词汇目标，例如 “Find alarm clock”。
- `image_goal`：使用 `objects_images/` 中的参考图片作为目标。
- `language_goal`：包含语言描述、房间或承载面线索。

## 5. 评测目标

长期能力不是只看单次是否成功，而是看：

- `seen_layout_count_before` 增加后，成功率是否提升。
- 动态物体移动后，agent 是否能降低旧位置权重。
- image-goal/language-goal 是否真正利用多模态信息。
- 候选召回、导航选点和目标确认之间的失败链路在哪里。

## 6. 常见数据路径

```text
benchmark/splits/benchmark_split_longterm_v1.json
benchmark/episodes/longterm_v1/val/<scene>/<layout_id>/*.json
objects_images/
results/layouts/
hm3d/minival/
hm3d/hm3d_val_scene_dataset_config.json
```

