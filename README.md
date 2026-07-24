# Agentic-RAG 长期动态家庭导航项目

本项目面向同一家庭环境下的长期具身导航与目标搜索。系统基于 HM3D/Habitat 构建动态物体 layout，生成 long-term benchmark episode，并在真实 Habitat RGB-D 闭环中结合远程 GroundingDINO、MobileSAM、CLIP、SceneMemory、RoomMemory 和 Agentic-RAG 融合评分完成目标搜索。

## 快速入口

推荐先阅读：

- [doc/00_项目总览.md](doc/00_项目总览.md)
- [doc/07_常用命令汇总.md](doc/07_常用命令汇总.md)
- [remote_vision_server/README.md](remote_vision_server/README.md)

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [doc/00_项目总览.md](doc/00_项目总览.md) | 项目目标、系统主线、推荐阅读顺序 |
| [doc/01_工程结构.md](doc/01_工程结构.md) | 顶层目录、核心模块、维护入口 |
| [doc/02_数据集与Benchmark.md](doc/02_数据集与Benchmark.md) | 动态 layout、episode schema、长期评测目标 |
| [doc/03_环境与远程视觉服务.md](doc/03_环境与远程视觉服务.md) | 工作站/服务器分工、SSH tunnel、远程视觉测试 |
| [doc/04_算法流程与导航策略.md](doc/04_算法流程与导航策略.md) | Agentic-RAG 流程、候选评分、room-level planning |
| [doc/05_长期记忆地图.md](doc/05_长期记忆地图.md) | SceneMemory、RoomMemory、object track、负反馈 |
| [doc/06_评测诊断与可视化.md](doc/06_评测诊断与可视化.md) | 指标、诊断、HTML 可视化、搜索视频 |
| [doc/07_常用命令汇总.md](doc/07_常用命令汇总.md) | 常用命令集中汇总，便于复制 |
| [doc/08_输出文件说明与Debug指南.md](doc/08_输出文件说明与Debug指南.md) | 输出文件、失败链路、debug 顺序 |

## 当前主入口

真实长期闭环：

```bash
python -m RAANav.AgenticRAG.benchmark_adapter.run_temporal_habitat_loop
```

诊断：

```bash
python -m RAANav.AgenticRAG.benchmark_adapter.diagnose_temporal_run
```

HTML 可视化：

```bash
python -m RAANav.AgenticRAG.benchmark_adapter.visualize_temporal_run
```

远程视觉服务：

```bash
bash remote_vision_server/run_server.sh
```

更多可复制命令见 [doc/07_常用命令汇总.md](doc/07_常用命令汇总.md)。

