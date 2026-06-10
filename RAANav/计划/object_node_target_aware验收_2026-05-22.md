# Target-aware object-node 验收记录（2026-05-22）

## 目的

上一轮 p20v4 随机/连续序列没有明确覆盖 chair/microwave/refrigerator，因此本轮加入 target-aware Habitat 导出模式：用旧 Stage 1 地图中的目标候选坐标引导相机到目标附近拍摄，验证 object-node frontend + CLIP top-k + RAANav semantic gate 是否能支持找物。

重要边界：旧 Stage 1 地图的标签来自旧前端，不作为真值，只作为相机采样提示。

## 新增能力

`scripts/raanav_frontend/export_habitat_sequence.py` 新增：

```bash
--trajectory-mode target_views
--target-points-json <json>
--target-camera-radius 1.8
--target-azimuths-deg "0,60,120,180,240,300"
--target-heading-offsets-deg=-20,0,20
--target-max-snap-distance 1.5
```

该模式围绕每个目标候选点生成多个可导航相机位，朝目标拍摄多角度 RGB-D-pose 帧。它是显式验收/调试模式，不改变默认 continuous 导出。

## 实验目录

```text
/home/adminer/agentRAG/AgenticRAG/RAG_Graph/object_node_target_verify_20260522/
├── target_points.json
├── habitat_target_views/
├── frontend/
├── raanav_map_semantic_gate/
├── qa/
│   ├── target_qa_summary.md
│   └── target_object_node_contact_sheet.jpg
└── logs/
```

目标候选来自：

```text
RAG_Graph/sim_verify/stage1_00814_short/00814-p53SfW6mjZe_semantic_map.json
```

包含：3 个 refrigerator、1 个 microwave、1 个 chair 候选。

## 运行结果

### 导出

```bash
/home/adminer/miniconda3/envs/agentrag/bin/python scripts/raanav_frontend/export_habitat_sequence.py \
  --scene-dir /home/adminer/agentRAG/experiment_data/hm3d/val/00814-p53SfW6mjZe \
  --out-dir RAG_Graph/object_node_target_verify_20260522/habitat_target_views \
  --trajectory-mode target_views \
  --target-points-json RAG_Graph/object_node_target_verify_20260522/target_points.json \
  --target-camera-radius 1.8 \
  --target-azimuths-deg "0,60,120,180,240,300" \
  --target-heading-offsets-deg=-20,0,20 \
  --target-max-snap-distance 1.5
```

结果：导出 78 帧。理论最多 90 帧，部分相机位因 navmesh snapping/可达性被跳过。

### 前端

```bash
HF_HUB_OFFLINE=1 /home/adminer/miniconda3/envs/daaam_p0/bin/python scripts/raanav_frontend/run_object_node_frontend.py \
  --data-path RAG_Graph/object_node_target_verify_20260522/habitat_target_views \
  --output-dir RAG_Graph/object_node_target_verify_20260522/frontend \
  --device cuda \
  --without-reid \
  --min-obs-per-track 3 \
  --representatives-per-track 3 \
  --representative-crop-mode context \
  --save-vis \
  --vis-interval 5 \
  --record-dashboard \
  --dashboard-save-frames
```

结果：

```text
78 frames
~17.2 fps
78 tracks
27 labeled tracks
22 object_nodes
```

raw object-node labels：

```text
screen=9, monitor=1, pillow=1, statue=1, towel=2, shower=1,
microwave=1, cabinet=2, book=1, curtain=2, blanket=1
```

经过默认 `--label-gate-mode soft --min-label-confidence 0.2` 转换后，低置信标签仍保留在 `label` 中，但会标记 `semantic_status=low_confidence` 并携带较低 `cfd`。这样目标查询不会被硬过滤，后续由 GMM 评分、stability 与负反馈压制不可靠候选。

soft gate 输出标签分布：

```text
screen=9, towel=2, cabinet=2, curtain=2, monitor=1, pillow=1, statue=1,
shower=1, microwave=1, book=1, blanket=1
```

## 关键 QA 结论

### chair 候选

最近 object nodes 的 crop 主要是乒乓台、窗帘、墙画/装饰。`chair` 没有进入 object-node top-k。一个 track 的 top-k 中出现过 chair，但置信度只有 0.041，crop 也不是清楚的椅子主体。

结论：本轮不能证明 chair 可找。更可能是旧 chair 坐标附近实际可见主体不是椅子，或椅子太小/被遮挡，FastSAM representative 选到了更大的物体。

### microwave 候选

最近 crop 主要是墙画/装饰和柜体，不像微波炉。没有可靠 microwave top-k。旧记录里 microwave 假阳性本就像画框，本轮进一步支持“旧 microwave 标签不可靠”。

结论：microwave 仍是负例/假阳性压力测试，语义门控不能把低置信 microwave 当可查询目标。

### refrigerator 候选

有节点距候选点很近，top-k 包含 refrigerator，但 top-1 是低置信 microwave/cabinet 等。例如：

```text
node_30 dist=0.06m label=microwave conf=0.101 topk=[microwave, dishwasher, towel, refrigerator, counter]
node_34 dist=1.31m label=cabinet conf=0.284 topk=[cabinet, drawer, refrigerator, dishwasher, kitchen cabinet]
```

肉眼看 crop 更像柜体/台面/局部结构，而不是清楚冰箱。

结论：refrigerator 也不能算通过，只能说明目标词偶尔进入 top-k，但视觉证据不足。

## 当前判断

1. target_views 导出能力可用，适合作为后续目标验收工具。
2. 本轮 target-aware 实验没有证明目标寻找准确率足够。
3. 更重要的发现是：旧 Stage 1 开放词汇标签不能作为真值样本集直接评价新前端。它可以引导相机，但结果必须肉眼/GT-lite 复核。
4. 语义软门控方向正确：低置信错误标签保留为可查询候选，但带低 cfd/status，交给评分、stability 和负反馈处理；hard unknown 只作为保守消融。
5. 下一步应该建立一个小型人工确认的 target QA 集：每个目标类别选 5-10 个清楚可见 crop/帧，先验证 CLIP/top-k 与 FastSAM representative 是否能命中，再谈导航找物成功率。

## 可视化

```text
RAG_Graph/object_node_target_verify_20260522/qa/target_object_node_contact_sheet.jpg
RAG_Graph/object_node_target_verify_20260522/qa/target_qa_summary.md
RAG_Graph/object_node_target_verify_20260522/frontend/dashboard/live.mp4
RAG_Graph/object_node_target_verify_20260522/raanav_map_semantic_gate/object_gmm_heatmap.png
```
