# Agentic-RAG 动态布局数据集适配离线 MVP 完成报告

> 日期：2026-06-10  
> 对应计划：`RAANav/计划/AgenticRAG_动态布局数据集适配_离线MVP计划_2026-06-09.md`  
> 当前状态：已完成离线 MVP 适配，并通过 `longterm_v1` 全量 val 集合验证。

---

## 1. 完成内容

本次实现了一个不依赖 Habitat-Sim、transformers 或在线视觉检测的离线 MVP 适配层，用于把现有动态布局 benchmark 接入 RAANav/AgenticRAG 长期记忆与轻量搜索评测。

新增内容：

- `hm3d_paths.py`
  - 补齐现有 `benchmark.habitat_adapter` 所需的 `resolve_scene_paths`，使 `benchmark.evaluate` 在 Euclidean fallback 下可正常导入和运行。

- `RAANav/AgenticRAG/benchmark_adapter/`
  - `dataset_index.py`：读取 split manifest，建立 scene/layout manifest/layout path 索引，避免混用不同 batch。
  - `layout_to_floors.py`：将 layout JSON 转为 RAANav `Floor/Room/Object`。
  - `memory_builder.py`：根据 `seen_layout_count_before` 构建历史长期 memory，并生成 `R_objs/cooccur_stats/N/last_update_time`。
  - `episode_to_queries.py`：将 benchmark `Subtask` 转成 RAANav 查询输入。
  - `lightweight_search.py`：纯 Python Top-K 记忆搜索和 MRA/GHR/SSS 计算。
  - `offline_eval.py`：主 CLI，生成 pseudo trajectory JSONL、memory diagnostics，并复用 `benchmark.evaluate` 输出标准指标。

修改内容：

- `RAANav/AgenticRAG/semantic_map/object.py`
  - 向后兼容地增加 `bbox_3d` 字段，用于承接 layout object 的近似 3D bbox。

---

## 2. 运行验证

### 2.1 单 episode 空记忆 smoke test

命令：

```bash
python -m RAANav.AgenticRAG.benchmark_adapter.offline_eval ^
  --split-manifest benchmark/splits/benchmark_split_longterm_v1.json ^
  --split val ^
  --scene 00808-y9hTuugGdiq ^
  --limit 1 ^
  --output-dir benchmark/eval/raanav_smoke ^
  --max-candidates 10
```

结果：

- episodes：1
- benchmark evaluator 正常运行
- missing_episode_count：0
- `seen_layout_count_before=0`，memory 为空，因此 found/MRA 为 0，符合预期。

### 2.2 单 episode 历史记忆 smoke test

命令：

```bash
python -m RAANav.AgenticRAG.benchmark_adapter.offline_eval ^
  --split-manifest benchmark/splits/benchmark_split_longterm_v1.json ^
  --episodes benchmark/episodes/longterm_v1/val/00808-y9hTuugGdiq/layout_009_seed_51/longterm_v1_00808-y9hTuugGdiq_layout_009_seed_51_ep_0045.json ^
  --output-dir benchmark/eval/raanav_smoke_state9 ^
  --max-candidates 10
```

结果摘要：

```json
{
  "episodes": 1,
  "subtasks": 7,
  "found_rate": 0.8571428571428571,
  "mra": 0.5714285714285714,
  "ghr3": 0.7142857142857143,
  "ghr5": 0.8571428571428571,
  "avg_sss": 2.0
}
```

说明：历史 layout memory 已被使用，且能产生有效 Top-K 命中。

### 2.3 longterm_v1 val 全量验证

命令：

```bash
python -m RAANav.AgenticRAG.benchmark_adapter.offline_eval ^
  --split-manifest benchmark/splits/benchmark_split_longterm_v1.json ^
  --split val ^
  --output-dir benchmark/eval/raanav_longterm_v1 ^
  --max-candidates 10
```

全量结果：

```json
{
  "episodes": 150,
  "subtasks": 1136,
  "found_rate": 0.6575704225352113,
  "mra": 0.2632042253521127,
  "ghr3": 0.6056338028169014,
  "ghr5": 0.653169014084507,
  "avg_sss": 2.117957746478873,
  "avg_min_dist": 1.1543617562101227
}
```

benchmark 标准指标：

```json
{
  "episodes": 150,
  "success_rate": 0.15333333333333332,
  "spl": 0.09671868732698946,
  "avg_steps": 16.866666666666667,
  "avg_path_length": 81.72104776,
  "avg_shortest_path_length": 53.66066415508588,
  "dynamic_memory_accuracy": 0.6575704225352113,
  "evaluated_episodes": 150,
  "trajectory_count": 150,
  "missing_episode_count": 0
}
```

输出目录：

```text
benchmark/eval/raanav_longterm_v1/
```

包含：

```text
adapter_warnings.json
by_task_type.json
episode_results.jsonl
exploration_curve.json
memory_by_task_type.json
memory_diagnostics.jsonl
memory_exploration_curve.json
memory_summary.json
run.jsonl
summary.json
```

---

## 3. 如何复现验证

### 3.1 快速 smoke test

```bash
python -m RAANav.AgenticRAG.benchmark_adapter.offline_eval ^
  --split-manifest benchmark/splits/benchmark_split_longterm_v1.json ^
  --split val ^
  --scene 00808-y9hTuugGdiq ^
  --limit 1 ^
  --output-dir benchmark/eval/raanav_smoke ^
  --max-candidates 10
```

查看结果：

```bash
type benchmark\eval\raanav_smoke\summary.json
type benchmark\eval\raanav_smoke\memory_summary.json
```

### 3.2 全量验证

```bash
python -m RAANav.AgenticRAG.benchmark_adapter.offline_eval ^
  --split-manifest benchmark/splits/benchmark_split_longterm_v1.json ^
  --split val ^
  --output-dir benchmark/eval/raanav_longterm_v1 ^
  --max-candidates 10
```

查看核心结果：

```bash
type benchmark\eval\raanav_longterm_v1\summary.json
type benchmark\eval\raanav_longterm_v1\memory_summary.json
type benchmark\eval\raanav_longterm_v1\memory_exploration_curve.json
```

### 3.3 单个指定 episode 验证

```bash
python -m RAANav.AgenticRAG.benchmark_adapter.offline_eval ^
  --split-manifest benchmark/splits/benchmark_split_longterm_v1.json ^
  --episodes benchmark/episodes/longterm_v1/val/00808-y9hTuugGdiq/layout_009_seed_51/longterm_v1_00808-y9hTuugGdiq_layout_009_seed_51_ep_0045.json ^
  --output-dir benchmark/eval/raanav_smoke_state9 ^
  --max-candidates 10
```

---

## 4. 当前 MVP 的边界

当前实现优先保证端到端可运行，因此采用轻量搜索 fallback：

- 不依赖 CLIP image/text embedding。
- 不依赖 Habitat-Sim。
- 不进行真实可导航性检查。
- `image_goal` 当前仍按 label 查询。
- `language_goal` 使用 region/receptacle metadata 作为轻量 prior。
- 负向 GMM 目前体现为 Top-K 顺序访问，没有下沉更新节点级 `exist_prob`。

这些边界不影响当前离线 MVP 的目标：验证 layout history -> semantic memory -> Top-K search -> benchmark evaluator 的闭环。

---

## 5. 下一步建议

优先级建议：

1. 将 `lightweight_search.py` 中的打分替换为正式 `S_rag + S_agent` 融合。
2. 为 `objects_images/*.webp` 建立 image embedding index，支持真正的 `image_goal`。
3. 将 language goal 中的 `region/receptacle` prior 接入 Agent 通道。
4. 生成 occupancy grid 或接入 Habitat PathFinder，使概率峰和 SPL 更接近真实导航。
5. 实现节点级 `exist_prob` 负向反馈和共现负向统计。

