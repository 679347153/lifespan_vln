# Object-node p20v4 验收记录（2026-05-22）

## 用户优先级

本轮确认：object-node 质量门控的核心目的不是把地图过滤得干净，而是提高目标寻找准确率。语义碎片、结构性物体、短 track 可以先保留用于 debug；只有当低质量语义会误导目标查询/GMM 时，才做降权或降级。

配套原则见：`计划/object_node质量门控与找物准确率_2026-05-22.md`。

## 实验目录

```text
/home/adminer/agentRAG/AgenticRAG/RAG_Graph/object_node_verify_20260522_p20v4/
├── habitat_sequence/                         # 80 帧 RGB-D-pose
├── frontend/                                 # v1: masked crop + cosine/0.5 softmax
├── frontend_v2_context_clipscale/             # v2: context crop + CLIP logit_scale
├── raanav_map/                               # v1 adapter 输出
├── raanav_map_v2_context_clipscale/           # v2 adapter 输出，无语义阈值
├── raanav_map_v3_semantic_gate/               # v2 + adapter 低置信语义降级
├── qa/
│   ├── p20v4_qa_summary.md
│   ├── v1_masked_temp05_object_node_contact_sheet.jpg
│   └── v2_context_clipscale_object_node_contact_sheet.jpg
└── logs/
```

## 运行命令

### 1. 导出 Habitat p20v4 序列

```bash
cd /home/adminer/agentRAG/AgenticRAG
/home/adminer/miniconda3/envs/agentrag/bin/python scripts/raanav_frontend/export_habitat_sequence.py \
  --scene-dir /home/adminer/agentRAG/experiment_data/hm3d/val/00814-p53SfW6mjZe \
  --out-dir RAG_Graph/object_node_verify_20260522_p20v4/habitat_sequence \
  --num-positions 20 \
  --n-views 4
```

结果：成功导出 80 帧。Habitat 提示该场景未加载 semantic annotations，本实验只依赖 RGB-D-pose 与前端感知。

### 2. v1 前端

```bash
HF_HUB_OFFLINE=1 /home/adminer/miniconda3/envs/daaam_p0/bin/python scripts/raanav_frontend/run_object_node_frontend.py \
  --data-path RAG_Graph/object_node_verify_20260522_p20v4/habitat_sequence \
  --output-dir RAG_Graph/object_node_verify_20260522_p20v4/frontend \
  --device cuda \
  --without-reid \
  --min-obs-per-track 3 \
  --representatives-per-track 3 \
  --save-vis \
  --vis-interval 10 \
  --record-dashboard \
  --dashboard-save-frames
```

结果：约 19.5 fps，24 tracks，8 object nodes。CLIP label confidence 约 0.02，接近 1/53 均匀分布，top-1 语义基本不可信，且多标成 screen。

### 3. v2 前端

改动：

- representative crop 默认使用 `context`，保留 bbox 内 RGB 上下文；仍可用 `--representative-crop-mode masked` 回退。
- CLIP 概率使用模型训练时的 `logit_scale`，不再用固定 `cosine/0.5`。
- prompt 从 `a photo of a {label}` 改成 `an indoor photo of a {label}`。

```bash
HF_HUB_OFFLINE=1 /home/adminer/miniconda3/envs/daaam_p0/bin/python scripts/raanav_frontend/run_object_node_frontend.py \
  --data-path RAG_Graph/object_node_verify_20260522_p20v4/habitat_sequence \
  --output-dir RAG_Graph/object_node_verify_20260522_p20v4/frontend_v2_context_clipscale \
  --device cuda \
  --without-reid \
  --min-obs-per-track 3 \
  --representatives-per-track 3 \
  --representative-crop-mode context \
  --save-vis \
  --vis-interval 10 \
  --record-dashboard \
  --dashboard-save-frames
```

结果：约 19.4 fps，24 tracks，8 object nodes。top labels 变为 painting/screen/shelf，置信度更有区分度。

## 关键 QA 结论

v2 联系表显示：

- `node_9`：painting，conf 0.319，crop 看起来像墙上画/装饰，合理。
- `node_11`：rug，conf 0.591，crop 是地毯，合理。
- `node_20`：painting，conf 0.221，crop 更像沙发/装饰区域，仍需谨慎。
- `node_8`：raw label 为 microwave，conf 0.102，但 crop 是栏杆/墙饰，并非微波炉。这是会误导目标寻找的典型假阳性。

因此本轮不是“p20v4 已证明找物成功”，而是发现并修复了一个重要失败模式：低置信 CLIP top-1 不应直接成为可查询语义标签。

## Adapter 语义降级

`daaam_to_raanav_map.py` 新增：

```bash
--min-label-confidence 0.2
```

当前策略已修正为默认软门控：低于阈值的标签会：

- `label` 仍保留候选语义，保证目标查询可以把它纳入候选；
- 保留 `raw_label`、`label_confidence`、`label_topk`、`clip_embedding`；
- `semantic_status` 标为 `low_confidence`；
- `cfd` 由观测次数和 label confidence 共同压低，后续由 GMM 评分、stability 和负反馈自然抑制。

如需保守消融，可使用 `--label-gate-mode hard` 把低置信标签暴露为 `unknown`。

v3 转换结果：

```text
labels: unknown=5, painting=2, rug=1
node_8: label=microwave, raw_label=microwave, conf=0.102, semantic_status=low_confidence, cfd低
```

这符合后续修正后的用户要求：不为了干净而过度过滤，让低置信候选仍能进入查询，但由 cfd/stability/负反馈降低其影响。

## 当前判断

1. 实时性：FastSAM + BotSort 前端在 RTX 5060 Ti 上 80 帧约 19 fps，当前方向有实时基础。
2. 结构完整性：Habitat sequence -> frontend summary -> RAANav semantic map/GMM 已跑通。
3. 找物准确率：本轮 p20v4 没有覆盖 chair 等明确目标，不能声称找物成功；但已经验证并修复低置信假 microwave 标签进入目标查询的问题。
4. 下一步需要 target-aware 验收：导出或采样能看到 chair/microwave/refrigerator 的序列，直接检查目标 query 是否指向正确 crop/location。

## 可视化验收文件

```text
RAG_Graph/object_node_verify_20260522_p20v4/qa/v2_context_clipscale_object_node_contact_sheet.jpg
RAG_Graph/object_node_verify_20260522_p20v4/qa/p20v4_qa_summary.md
RAG_Graph/object_node_verify_20260522_p20v4/raanav_map_v3_semantic_gate/object_gmm_heatmap.png
RAG_Graph/object_node_verify_20260522_p20v4/frontend_v2_context_clipscale/dashboard/live.mp4
```
