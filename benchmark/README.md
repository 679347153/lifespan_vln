# 动态家庭物体导航 Benchmark

本目录用于构建和评测一个面向“同一家庭场景在几周/几个月内发生物体变化”的机器人导航 benchmark。它复用 `batch_generate_layouts.py` 生成的多时间 layout，每个 layout 表示同一 HM3D 场景在某个时间点的物体摆放状态，然后生成类似 ObjectNav 的 episode。

每个 episode 包含 5-10 个连续子任务，子任务类型包括：

- `open_vocab`：开放词汇目标，例如 `Find mug.`
- `image_goal`：开放词汇实例图像目标，直接使用 `objects_images/<object_stem>.*` 中的参考图像
- `language_goal`：语言目标，使用 layout 中的 `sampled_region_id`、`target_instance_id` 等字段生成模板化自然语言描述

主指标以 episode 为单位：只有当 episode 内全部子任务都成功时，该 episode 才算成功。

## 文件结构

- `schemas.py`：统一定义 episode、subtask、trajectory、split manifest 等数据结构。
- `build_episodes.py`：从 layout manifest 和 layout JSON 生成 episode JSON。
- `habitat_adapter.py`：封装 Habitat-Sim/HM3D 相关能力，包括 scene 加载、layout object 加载、导航起点采样、geodesic shortest path 计算。
- `runner.py`：运行入口，按 episode/subtask 顺序调用 agent，输出标准 trajectory JSONL。
- `evaluate.py`：离线评测入口，读取 episode 和 trajectory，计算 SR、SPL、按任务类型指标、探索提升曲线。
- `metrics.py`：基础指标函数和聚合逻辑。
- `README.md`：当前说明文档。

## Episode 数据格式

核心字段如下：

- `episode_id`：episode 唯一 ID。
- `split`：`train`、`val` 或 `test`。
- `scene_name`：HM3D scene 名称，例如 `00808-y9hTuugGdiq`。
- `layout_id`：当前 episode 对应的 layout ID，例如 `layout_000_seed_42`。
- `state_index`：当前 layout 在同一 scene 多时间序列中的顺序。
- `seen_layout_count_before`：运行当前 episode 前 agent 已经见过的 layout 数量，用于计算探索效率提升率。
- `scene_state.layout_path`：源 layout JSON 路径。
- `start_pose`：起点位姿。生成器默认写入占位起点，runner 可在 Habitat 可用时重新采样可导航起点。
- `subtasks`：5-10 个子任务。

每个 subtask 的核心字段：

- `subtask_id`：子任务 ID。
- `task_type`：`open_vocab`、`image_goal` 或 `language_goal`。
- `target_object`：目标物体名称。
- `target_object_id`：layout 中绑定的目标物体 ID。
- `target_position`：目标物体在当前 layout 中的位置。
- `prompt`：给 agent 的任务输入文本。
- `image_path`：`image_goal` 的参考图像路径。
- `language_prompt`：`language_goal` 的自然语言目标。
- `success_radius`：成功半径，默认 `1.2m`。

## 生成 Episode

输入来自 `batch_generate_layouts.py` 的输出：

```text
results/layouts/<scene>/batch_<YYYYmmdd_HHMMSS>/
  manifest.json
  layout_000_seed_42.json
  layout_001_seed_43.json
  ...
```

生成命令示例：

```bash
python -m benchmark.build_episodes \
  --layout-manifest results/layouts/00808-y9hTuugGdiq/batch_YYYYmmdd_HHMMSS/manifest.json \
  --version smoke_v2 \
  --split val \
  --images-dir objects_images \
  --episodes-per-layout 3
```

也可以一次传入多个 scene 的 manifest：

```bash
python -m benchmark.build_episodes \
  --layout-manifest \
    results/layouts/scene_a/batch_x/manifest.json \
    results/layouts/scene_b/batch_y/manifest.json \
  --version longterm_v1 \
  --split train
```

输出位置：

- `benchmark/episodes/<version>/<split>/<scene>/<layout_id>/*.json`
- `benchmark/splits/benchmark_split_<version>.json`

常用参数：

- `--episodes-per-layout`：每个 layout 生成多少个 episode。
- `--min-subtasks` / `--max-subtasks`：每个 episode 的子任务数，必须在 5-10 范围内。
- `--success-radius`：成功半径，默认 `1.2`。
- `--max-steps`：episode 最大步数预算。
- `--seed`：任务生成随机种子。
- `--images-dir`：`image_goal` 参考图像目录。

生成器会尽量平衡三类子任务数量，三类任务数量差不超过 1。`image_goal` 任务必须能在 `--images-dir` 下找到目标物体对应的图片，否则会报错，以避免生成不可执行的图像目标任务。

## 运行 Episode

`runner.py` 负责按 episode 顺序执行 agent，并输出统一 trajectory JSONL。它本身不绑定具体导航策略，只定义 agent adapter 协议：

- `reset(episode) -> None`
- `act(observation, subtask) -> action`
- `on_subtask_done(episode, subtask, trace) -> None`

内置两个 smoke agent：

- `oracle`：直接移动到每个目标位置，用于验证 schema、trajectory 和 evaluator 是否能跑通。
- `noop`：不移动，用于构造失败轨迹和测试评测逻辑。

运行示例：

```bash
python -m benchmark.runner \
  --episodes benchmark/episodes/smoke_v2/val \
  --output benchmark/eval/smoke_v2/run.jsonl \
  --mode oracle \
  --sample-start-pose
```

接入外部 agent：

```bash
python -m benchmark.runner \
  --episodes benchmark/episodes/smoke_v2/val \
  --output benchmark/eval/my_agent/run.jsonl \
  --agent-module my_agent_module:create_agent \
  --agent-id my_agent
```

`create_agent()` 应返回一个实现上述协议的对象。runner 会把当前 pose、episode 和 subtask 传给 agent，并把每个子任务完成后的 `final_pose`、`path_length`、`steps` 写入 trajectory。

trajectory JSONL 每行是一个 episode 的运行结果，包含：

- `episode_id`
- `agent_id`
- `scene_name`
- `layout_id`
- `seen_layout_count_before`
- `steps[*].position/action`
- `subtask_traces[*].final_pose`
- `subtask_traces[*].path_length`
- `subtask_traces[*].steps`

## 离线评测

评测命令：

```bash
python -m benchmark.evaluate \
  --episodes benchmark/episodes/smoke_v2/val \
  --trajectories benchmark/eval/smoke_v2/run.jsonl \
  --output-dir benchmark/eval/smoke_v2
```

输出文件：

- `summary.json`：主指标，包括 episode-level SR、SPL、平均路径长度、平均步数等。
- `episode_results.jsonl`：每个 episode 的详细评测结果。
- `by_task_type.json`：按 `open_vocab`、`image_goal`、`language_goal` 分组的诊断指标。
- `exploration_curve.json`：按 `seen_layout_count_before` 聚合的探索提升曲线。

评测规则：

- evaluator 不信任 agent 自报的 success。
- 对每个 subtask，读取 trajectory 中的完成位姿 `final_pose`，计算它与 `target_position` 的欧氏距离。
- 若距离小于等于 `success_radius`，该 subtask 成功。
- 只有全部 subtask 成功，episode 才成功。
- episode SPL 使用：

```text
SPL = success * shortest_path_sum / max(actual_path_length, shortest_path_sum)
```

其中 `shortest_path_sum` 是从 episode 起点依次到各个 subtask 目标位置的最短路径之和。

## 探索效率提升率

该 benchmark 的长期变化重点通过 `seen_layout_count_before` 表达。对于同一 scene 的 layout 序列：

- 第 0 个 layout 作为 baseline。
- 第 k 个 layout 的 SR/SPL 与 baseline 比较。
- 同时输出 absolute delta 和 percentage gain。

公式：

```text
SR_gain_pct = (SR_k - SR_0) / max(SR_0, 1e-6) * 100
SPL_gain_pct = (SPL_k - SPL_0) / max(SPL_0, 1e-6) * 100
```

对应结果写入 `exploration_curve.json`。

## Habitat-Sim 与 fallback

在完整 Habitat 环境中：

- `HabitatLayoutAdapter` 会通过 `hm3d_paths.resolve_scene_paths` 解析 HM3D scene。
- 可选加载 layout JSON 中的 objects 到 simulator。
- 使用 `habitat_sim.PathFinder` 采样可导航起点。
- 使用 geodesic distance 计算 shortest path。

在没有 Habitat-Sim 的普通 Python 环境中：

- runner 和 evaluator 默认启用 Euclidean fallback。
- 这样仍然可以完成 schema、episode 生成、trajectory 格式和指标聚合的 smoke test。
- 若希望强制使用 Habitat-Sim，可传入 `--no-euclidean-fallback`。

## 推荐 Smoke Test 流程

1. 用 `batch_generate_layouts.py` 生成少量 layout。
2. 用 `benchmark.build_episodes` 生成 `smoke_v2` episode。
3. 用 `benchmark.runner --mode oracle` 生成一份理想轨迹。
4. 用 `benchmark.evaluate` 评测 oracle 轨迹，检查 SR/SPL 是否接近预期。
5. 用 `benchmark.runner --mode noop` 生成失败轨迹，检查 evaluator 是否能正确输出失败结果。

示例：

```bash
python -m benchmark.build_episodes \
  --layout-manifest results/layouts/<scene>/batch_<id>/manifest.json \
  --version smoke_v2 \
  --split val

python -m benchmark.runner \
  --episodes benchmark/episodes/smoke_v2/val \
  --output benchmark/eval/smoke_v2/oracle.jsonl \
  --mode oracle

python -m benchmark.evaluate \
  --episodes benchmark/episodes/smoke_v2/val \
  --trajectories benchmark/eval/smoke_v2/oracle.jsonl \
  --output-dir benchmark/eval/smoke_v2
```
