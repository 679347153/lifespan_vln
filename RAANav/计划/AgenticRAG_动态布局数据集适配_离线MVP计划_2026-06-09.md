# Agentic-RAG 与动态布局数据集离线 MVP 适配计划

> 首版日期：2026-06-09  
> 更新日期：2026-06-10  
> 状态：计划存档，暂不实施  
> 目的：根据当前已加入项目的数据集与 benchmark 代码，更新“Agentic-RAG 解决方案如何适配动态家庭布局数据集”的离线 MVP 计划，供后续继续完善与落地。

---

## 1. 目标

将当前项目中的动态家庭布局数据集 `layout JSON + episode JSON + benchmark schemas/evaluator` 适配为 RAANav/AgenticRAG 可消费的长期语义记忆、查询输入、伪导航轨迹和离线评测流程。

本阶段暂不接入 Habitat-Sim 闭环导航，优先完成离线 MVP，用于验证：

- 历史 layout 是否提升目标定位效果
- GMM Top-K 峰值是否接近当前 layout 真值位置
- `seen_layout_count_before` 是否能形成长期探索收益曲线
- Agentic-RAG 的长期记忆、RAG/Agent 融合、GMM 候选点输出是否能与现有 benchmark 格式闭合

---

## 2. 当前已加入的数据集与代码事实

### 2.1 已加入的主要目录

当前项目中已经存在以下关键数据与代码：

```text
results/layouts/
results/scene_info/
results/probabilities/
results/receptacle_queries/
benchmark/
benchmark/episodes/
benchmark/splits/
objects/
objects_images/
hm3d/minival/
scripts/
```

其中与本计划最相关的是：

```text
benchmark/splits/benchmark_split_longterm_v1.json
benchmark/episodes/longterm_v1/val/
results/layouts/<scene>/<batch_id>/manifest.json
results/layouts/<scene>/<batch_id>/layout_*_seed_*.json
results/scene_info/<scene>/<scene>_scene_info.json
results/receptacle_queries/<scene>/<scene>_receptacle_surfaces_all_rooms.json
objects_images/*.webp
```

### 2.2 当前 longterm_v1 数据规模

`benchmark_split_longterm_v1.json` 显示：

- version：`longterm_v1`
- val scenes：
  - `00808-y9hTuugGdiq`
  - `00803-k1cupFYWXJ6`
  - `00802-wcojb4TFT35`
- episode_count：150
- layout_count：30
- 每个 scene 10 个 layout
- `metadata.layout_manifests` 指向 3 个 layout manifest
- `metadata.episodes_root` 为 `benchmark/episodes/longterm_v1`
- `metadata.images_dir` 为 `objects_images`

### 2.3 当前存在多批 layout

同一 scene 下已经存在多个 batch，例如：

```text
results/layouts/00808-y9hTuugGdiq/batch_20260521_051148/
results/layouts/00808-y9hTuugGdiq/batch_20260610_182927/
```

注意：`longterm_v1` episode 中的 `scene_state.layout_path` 指向 `batch_20260521_*`，而新加入的数据里也有 `batch_20260610_*`。因此后续适配必须坚持：

- **评测某个 benchmark version 时，以 split manifest 的 `metadata.layout_manifests` 为权威数据源。**
- 不要混用不同 batch 的 layout，即使 scene 和 layout id 相同。
- 若 episode 的 `scene_state.layout_path` 与 split manifest 中对应 layout 路径不一致，应记录 warning，并优先使用该 episode 所属 version 的 manifest 路径。

---

## 3. 当前匹配判断

数据集与 Agentic-RAG 方案在研究方向上匹配：

- 数据集关注动态家庭物体布局变化。
- Agentic-RAG 关注动态长期寻物和长期语义记忆。
- 数据集中的多时间 layout、`state_index`、`seen_layout_count_before` 能支撑长期记忆实验。
- episode 中的 `open_vocab`、`image_goal`、`language_goal` 可作为多模态目标导航评测入口。
- 当前 benchmark 已有 schema、runner、evaluate、metrics，不需要另起一套完整评测格式。

但二者不能直接对接，主要差异是：

- 数据集提供的是离线生成的真值 layout 和 episode。
- Agentic-RAG 方案默认的是机器人在线感知、建图、融合、查询、失败反馈闭环。
- RAANav 现有 GMM evaluator 更接近“内存地图查询”，而 benchmark evaluator 更接近“读取 trajectory JSONL 判定导航是否成功”。
- 因此需要一个适配层，把 layout/episode 转成 RAANav semantic memory，并把 RAANav 离线查询结果转成 benchmark 兼容 trajectory 或补充诊断结果。

---

## 4. 适配范围

### 4.1 本阶段包含

- 读取现有 `benchmark.schemas.Episode/Subtask/SplitManifest`
- 从 split manifest 解析 version、scene、layout manifest 和 episode root
- layout 转 RAANav semantic memory
- episode subtask 转 RAANav 查询输入
- 根据 `seen_layout_count_before` 构建历史长期记忆
- 从当前 layout 或 subtask 的 `target_position` 提取 GT
- 调用现有 `GMMEvaluator`
- 调用现有 `LightweightNavSim`
- 生成 benchmark 兼容 pseudo trajectory JSONL
- 复用 `benchmark.evaluate` 输出 SR/SPL/exploration curve
- 额外输出 RAANav memory diagnostics，例如 MRA、GHR@3、GHR@5、SSS、min_dist

### 4.2 本阶段不包含

- Habitat-Sim 在线闭环导航
- 相机视锥检测
- 真实视觉 false negative
- 在线 `exist_prob` 贝叶斯更新
- 真正的 image embedding goal 检索
- 真实 geodesic SPL；默认仍使用 benchmark 的 Euclidean fallback

---

## 5. 推荐新增模块

新增目录：

```text
RAANav/AgenticRAG/benchmark_adapter/
```

建议包含：

```text
dataset_index.py
layout_to_floors.py
episode_to_queries.py
memory_builder.py
raanav_pseudo_agent.py
offline_memory_eval.py
```

### 5.1 模块职责

`dataset_index.py`

- 读取 `benchmark/splits/benchmark_split_<version>.json`
- 建立 `scene -> manifest_path`
- 建立 `(scene, layout_id) -> layout_path`
- 校验 episode 引用的 layout 是否属于当前 version

`layout_to_floors.py`

- 将单个 layout JSON 转成 `List[Floor]`
- 将 object 字段转成 RAANav `Object`
- 读取可选 scene_info / receptacle surfaces，补充 room/receptacle 元数据

`episode_to_queries.py`

- 从 `benchmark.schemas.Episode` 和 `Subtask` 生成 RAANav 查询对象
- 保留 `task_type`、`image_path`、`language_prompt`、`metadata.sampled_region_id`、`metadata.target_instance_id`

`memory_builder.py`

- 根据 scene、当前 state_index、`seen_layout_count_before` 选择历史 layouts
- 将历史 layouts 融合为长期 memory floors
- 统计 `N`、`last_update_time`、`R_objs`、`cooccur_stats`

`raanav_pseudo_agent.py`

- 按 episode/subtask 调用 RAANav 查询与轻量搜索
- 生成 benchmark runner/evaluate 兼容的 trajectory JSONL
- 将 RAANav 诊断项写入 trajectory metadata 或旁路 diagnostics JSONL

`offline_memory_eval.py`

- 不走 benchmark trajectory 时，直接批量计算 MRA/GHR/SSS 等记忆指标
- 输出 RAANav 专用诊断文件

### 5.2 核心接口

```python
load_dataset_index(split_manifest_path) -> DatasetIndex
layout_to_floors(layout_path, state_index, scene_info_path=None, surfaces_path=None) -> List[Floor]
build_memory_from_seen_layouts(index, scene_name, layout_id, seen_layout_count_before) -> List[Floor]
episode_to_queries(episode) -> List[QuerySpec]
run_pseudo_agent(episode, memory_floors, evaluator) -> Trajectory
collect_gt_positions(layout_path) -> Dict[str, List[List[float]]]
```

说明：

- `QuerySpec` 是内部轻量结构，不需要替代 `benchmark.schemas.Subtask`。
- 对外仍应兼容 `benchmark.schemas.Episode/Trajectory`。

---

## 6. 字段映射方案

### 6.1 layout object 真实字段

当前 layout object 样例包含：

```text
id
name
model_id
position
rotation
sampled_region_id
target_instance_id
assigned_target_instance_id
placement_target_source
source
placement_y_offset
placement_radius
object_profile
template_collidable
physics_settle
```

其中 `object_profile` 可能包含：

```text
profile_source
placement_class
footprint_x
footprint_z
height
radius
y_offset
```

### 6.2 映射到 RAANav Object

| RAANav / Agentic-RAG 字段 | 数据集来源或生成规则 |
|---|---|
| `obj_id` | `str(object.id)` |
| `obj_id` fallback | 若 `id` 缺失，使用 `model_id`；再缺失则使用 `scene_name:name:index` |
| `label` | 优先由 `model_id` 归一化；否则由 `name` 归一化 |
| `category` | 与 `label` 同值，供规范层引用 |
| `pos_3d` | `object.position` |
| `pos_2d` | `[position[0], position[2]]` |
| `room_id` | `str(sampled_region_id)` |
| `region` | MVP 可用 object bbox 的 2D footprint 多边形近似 |
| `bbox_3d` | 用 `object_profile.footprint_x/footprint_z/height` 或 `radius/y_offset` 近似生成 |
| `last_update_time` | `2026-01-01T00:00:00Z + state_index * 1 day` |
| `stability` | 优先读 `stability_priors.yaml`，缺失时按 `placement_class` 回退 |
| `cooccur_stats` | 由同 room、同 receptacle、近邻关系生成 |
| `R_objs` | 由 `cooccur_stats` 聚合生成 |
| `exist_prob` | 初始为 `1.0` |
| `N` | 该 object identity 在已见 layout 中成功出现次数 |
| `cfd` | MVP 固定为 `1.0`，表示 layout 真值对象 |
| `imgs` | 若存在 `objects_images/<stem>.webp`，记录到 `imgs` |
| `description` | 可记录 `name/model_id/placement_class/receptacle` 摘要 |

### 6.3 label 归一化规则

建议将以下输入：

```text
Alarm Clock 01 4K
alarm_clock_01_4k
Camera_01_4k
```

统一为类似：

```text
alarm_clock_01
camera_01
```

规则：

- 转小写
- 空格转下划线
- 去掉尾部 `_4k`
- 去掉多余非字母数字下划线字符
- 保留实例后缀如 `_01`，因为对象图像与 `objects_images` 使用该 stem

注意：

- benchmark subtask 的 `target_object` 通常是自然名称，例如 `Megaphone 01 4K`。
- `target_object_id` 在样例中是字符串形式，例如 `"3"`。
- layout object 的 `id` 是数字，例如 `3`。
- 适配时必须统一转字符串比较。

---

## 7. Stability 回退规则

优先级：

1. `RAANav/AgenticRAG/config/stability_priors.yaml` 中的 label 先验
2. `object_profile.placement_class`
3. 默认值 `0.5`

建议 placement class 回退：

| placement_class | stability |
|---|---:|
| `floor_only` | 0.75 |
| `small_tabletop` | 0.35 |
| `large_tabletop` | 0.45 |
| `soft_surface` | 0.30 |
| `floor_or_large_surface` | 0.55 |
| `tabletop_or_floor` | 0.50 |
| unknown | 0.50 |

---

## 8. 共现关系规则

共现关系写入现有 `R_objs`，并同步保留到 `cooccur_stats`。

建议规则：

- 同 `sampled_region_id`：房间共现
- 同 `target_instance_id`：目标承载实例共现
- 同 `assigned_target_instance_id`：实际承载实例共现
- 2D 距离小于 2m：邻近共现

建议权重：

| 关系类型 | 说明 | 建议权重 |
|---|---|---:|
| same_room | 同一个 region / room | 1 |
| same_target_receptacle | 同一个 `target_instance_id` | 2 |
| same_assigned_receptacle | 同一个 `assigned_target_instance_id` | 3 |
| near_2d | 2D 距离小于 2m | 2 |

MVP 阶段可以将这些权重统一累加为 `Nr`，并用归一化后的值生成 `Rcfd`。

建议 `cooccur_stats` 结构：

```json
{
  "relations": {
    "other_obj_id": {
      "same_room": 1,
      "same_target_receptacle": 0,
      "same_assigned_receptacle": 1,
      "near_2d": 1,
      "weight_sum": 6
    }
  }
}
```

---

## 9. 查询适配

将 `benchmark.schemas.Subtask` 转成 RAANav 查询输入。

### 9.1 open_vocab

使用：

- `target_object`
- `target_object_id`
- `metadata.model_id`

MVP 推荐查询 label 使用归一化后的 `metadata.model_id` stem；若缺失则使用 `target_object` 归一化。

### 9.2 image_goal

MVP 阶段先转为 label 查询。

同时保留：

- `image_path`
- 原始 `prompt`
- `task_type=image_goal`

当前数据中 image path 示例：

```text
objects_images/antique_ceramic_vase_01.webp
objects_images/brass_vase_03.webp
```

后续可扩展为图像 embedding 检索。

### 9.3 language_goal

使用：

- `target_object`
- `language_prompt`
- `metadata.sampled_region_id`
- `metadata.target_instance_id`

当前语言模板形如：

```text
Go to the Food Apple 01 4K, which is usually placed near receptacle instance 649 in region 13.
```

MVP 阶段仍以目标 label 查询为主，但将 region/receptacle 线索作为 Agent room/receptacle prior 的输入。

---

## 10. 历史记忆构建规则

对某个 episode：

- 当前 scene：`episode.scene_name`
- 当前 layout：`episode.layout_id` 或 `episode.scene_state.layout_id`
- 当前 state：`episode.state_index`
- 已见历史数：`episode.seen_layout_count_before`

历史 layout 选择：

1. 从当前 benchmark version 的 split manifest 找到该 scene 的 layout manifest。
2. 读取 manifest 中 `layouts` 列表。
3. 按 `layout_index` 升序排序。
4. 对 state_index = k 的 episode，历史 memory 使用 `layout_index < k` 且最多前 `seen_layout_count_before` 个 layout。
5. 若 `seen_layout_count_before == 0`，memory 为空或只使用 Agent 先验，不应泄漏当前 layout 真值。
6. 当前 layout 只用于 GT 评测，不得加入查询前 memory。

对象 identity：

- 优先使用 `str(object.id)`。
- 若不同 layout 中同一 `id` 对应的 `model_id` 不一致，记录异常并改用 `model_id`。
- failed_objects 不加入当前可观测 memory，但可作为数据质量统计。

时间：

- MVP 默认 `1 layout_index = 1 day`。
- `last_update_time = 2026-01-01T00:00:00Z + layout_index days`。
- 查询时间可设为当前 state 的同日或下一天，用于时间衰减。

---

## 11. 离线评测流程

### 11.1 RAANav memory diagnostics 流程

单个 episode：

1. 读取 episode JSON，使用 `benchmark.schemas.read_episode`。
2. 通过 split manifest 定位该 scene 的 layout manifest。
3. 构建历史 memory floors。
4. 对每个 subtask 生成查询 label 和 query metadata。
5. 使用 RAANav GMM 查询得到概率峰。
6. 用 subtask 的 `target_position` 作为 GT。
7. 计算 MRA、GHR@3、GHR@5、SSS、found、min_dist。
8. 输出 subtask diagnostics。

### 11.2 benchmark.evaluate 兼容流程

为了复用现有 benchmark evaluator，建议生成 pseudo trajectory JSONL：

- 每个 episode 一行 `Trajectory`
- 每个 subtask 写一条 `SubtaskTrace`
- `final_pose`：
  - 若 `LightweightNavSim` 找到目标，使用命中的候选点或 GT 附近点
  - 若未找到，使用最后访问候选点
- `path_length`：
  - MVP 使用从上一个完成点到访问候选点序列的 Euclidean 距离近似
- `steps`：
  - 使用 `SSS` 或访问候选点数
- `metadata.memory_metrics`：
  - 可填 `dynamic_memory_correct/total`
  - 可额外放 RAANav diagnostics 路径

然后调用：

```text
benchmark.evaluate
```

输出标准：

```text
summary.json
episode_results.jsonl
by_task_type.json
exploration_curve.json
```

---

## 12. GMM 与栅格策略

### 12.1 有 occupancy grid

若后续存在：

```text
occupancy_grid.npy
occupancy_meta.npz
```

则使用现有 free-mask / grid 流程。

### 12.2 当前 MVP 默认：Euclidean canvas

当前加入的数据集中尚未看到 RAANav 所需的 occupancy grid 文件，因此 MVP 默认生成 Euclidean canvas：

- 覆盖当前 scene 所有 layout object 的 x/z 范围
- 加 padding，例如 2m
- resolution 默认 0.1m 或 0.2m
- 全区域视为 free

该 fallback 仅用于：

- 概率峰提取
- Top-K 命中
- 伪导航距离估计

不能声称是真实可导航路径评估。

---

## 13. 输出文件

建议输出到：

```text
benchmark/eval/raanav_<version>/
```

例如：

```text
benchmark/eval/raanav_longterm_v1/
```

### 13.1 benchmark 兼容输出

```text
run.jsonl
summary.json
episode_results.jsonl
by_task_type.json
exploration_curve.json
```

### 13.2 RAANav 诊断输出

```text
memory_diagnostics.jsonl
memory_summary.json
memory_by_task_type.json
memory_exploration_curve.json
```

每条 subtask diagnostics 至少包含：

```json
{
  "episode_id": "...",
  "subtask_id": "...",
  "scene_name": "00808-y9hTuugGdiq",
  "layout_id": "layout_009_seed_51",
  "state_index": 9,
  "seen_layout_count_before": 9,
  "task_type": "open_vocab",
  "target_object": "Megaphone 01 4K",
  "query_label": "megaphone_01",
  "target_object_id": "3",
  "target_position": [3.8829, 2.4904],
  "peaks": [],
  "mra": false,
  "ghr3": false,
  "ghr5": false,
  "sss": 0,
  "found": false,
  "min_dist": null
}
```

---

## 14. 指标

MVP 阶段使用两套指标。

### 14.1 benchmark 原生指标

由 `benchmark.evaluate` 计算：

- episode success rate
- SPL
- average path length
- average shortest path length
- average steps
- by task type success/SPL
- exploration curve by `seen_layout_count_before`

### 14.2 RAANav memory diagnostics

由适配层计算：

- MRA：Top-1 峰值是否命中 GT
- GHR@3：Top-3 峰值是否命中 GT
- GHR@5：Top-5 峰值是否命中 GT
- SSS：找到目标前访问的候选点数量
- found：Top-K 预算内是否找到
- min_dist：Top-1 或最终候选到 GT 的最近距离

按 task type 聚合：

- `open_vocab`
- `image_goal`
- `language_goal`
- `all`

按 `seen_layout_count_before` 聚合：

- MRA curve
- GHR@3 curve
- found curve
- SSS curve
- memory gain

---

## 15. 测试计划

### 15.1 单元测试

- 读取 `benchmark_split_longterm_v1.json` 能解析 3 个 manifest。
- episode 的 `scene_state.layout_path` 能与 split manifest 解析出的 layout 对齐。
- layout object 能完整转成 `Object.to_dict()`。
- 数字 `id` 与字符串 `target_object_id` 能正确匹配。
- `model_id/name` 能稳定归一化到 query label。
- 缺失 `object_profile` 时 `bbox_3d` 生成不崩溃。
- `state_index` 正确映射为虚拟时间戳。
- `placement_class` 能映射到 stability。
- 同 room、同 target receptacle、同 assigned receptacle、近邻关系正确写入 `R_objs`。
- 三类 subtask 都能转成查询输入。

### 15.2 Smoke Test

- 单个 layout 可转成 `List[Floor]`。
- 单个 scene 的 10 个 layout 可构建长期 memory。
- `longterm_v1` 的 150 个 val episodes 可全部读取。
- 无 occupancy grid 时可用 Euclidean canvas 完成评测。
- 能生成 benchmark 兼容 `run.jsonl`。
- `benchmark.evaluate` 能读取 pseudo trajectory 并输出 summary。

### 15.3 验收标准

- 输出格式与当前 `benchmark/evaluate.py` 兼容。
- 每条 RAANav diagnostics 包含 target、task_type、layout_id、seen_layout_count_before、peaks、GT、MRA/GHR/SSS。
- `seen_layout_count_before=0` 和 `>0` 能分别聚合出指标。
- 能生成 benchmark 原生 exploration curve 和 RAANav memory exploration curve。
- 不混用不同 batch 的 layout。

---

## 16. 默认假设

- 每个 layout 间隔默认等价于 1 天。
- 当前 layout 不进入查询前 memory，避免真值泄漏。
- 优先使用 object `id` 做跨 layout 对齐，但统一转字符串。
- 如果 `id` 与 `model_id` 关系异常，则回退到 `model_id`。
- `image_goal` 暂不做图像 embedding，先使用 label 查询。
- 负向 GMM 只在 `LightweightNavSim` 中用候选点 miss 模拟。
- 不在 MVP 阶段更新真实节点的 `exist_prob`。
- benchmark SPL 默认使用 Euclidean fallback，不等价于真实导航 SPL。

---

## 17. 后续可完善方向

- 接入 image goal 的图像 embedding 检索。
- 使用 Habitat-Sim `PathFinder` 替代 Euclidean fallback。
- 将 `language_goal` 中的 receptacle/region 线索纳入 Agent room/receptacle prior。
- 将 RAANav pseudo agent 升级为 `benchmark.runner` 外部 agent adapter。
- 将负向 GMM 从概率场模拟下沉为节点级 `exist_prob` 更新。
- 增加共现负向统计。
- 使用 `objects_images/*.webp` 预构建 image embedding index。
- 设计消融实验：
  - 无 Agent
  - 无时间衰减
  - 无共现关系
  - 无历史 layout
  - 无负向 GMM

---

## 18. 与现有代码的推荐衔接点

优先复用现有模块：

- `benchmark/schemas.py`
- `benchmark/evaluate.py`
- `benchmark/metrics.py`
- `RAANav/AgenticRAG/semantic_map/object.py`
- `RAANav/AgenticRAG/semantic_map/room.py`
- `RAANav/AgenticRAG/semantic_map/floor.py`
- `RAANav/AgenticRAG/eval/gmm_evaluator.py`
- `RAANav/AgenticRAG/eval/nav_sim_lightweight.py`
- `RAANav/AgenticRAG/eval/metrics.py`
- `RAANav/AgenticRAG/config/stability_priors.yaml`

不建议从零重写 GMM 或 benchmark evaluator。当前适配的核心应是：

```text
benchmark split manifest
  -> dataset_index
  -> version-consistent layout history
  -> RAANav semantic memory
  -> existing GMMEvaluator / LightweightNavSim
  -> pseudo trajectory JSONL + memory diagnostics
  -> benchmark.evaluate + RAANav memory summary
```

