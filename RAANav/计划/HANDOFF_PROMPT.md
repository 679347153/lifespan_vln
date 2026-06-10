# 新窗口交接提示词

请直接复制下面内容给新 AI：

```text
你接手的是 RAANav / AgenticRAG 项目。项目在远端 Ubuntu 机器：
adminer@100.91.224.96:/home/adminer/agentRAG

当前用户的核心要求：不要过度臆断，重要设计必须先和用户商量；可以主动维护文档和实验记录，但不能把未经验证的 AI 产物说成事实。用户懂总体流程八九成，代码细节约三四成，所以你需要帮助他建立代码心智模型，而不是只替他写代码。

项目目标：面向真实变化环境，构建轻量级长期语义地图。机器人反复扫描同一环境，实时/增量更新地图，并用长期记忆、stability、共现关系、CLIP/VLM 语义和 GMM 概率场指导寻找目标物体。最终部署形态是 D435i + Mid360 + Jetson 小车 + 5060 远程算力机器协作。

关键架构理解：
- 参考 DAAAM 的实时前端思想，但不要污染 `参考开源代码/DAAAM` 原仓库。
- DAAAM/HOV-SG/其它开源代码用于借鉴，不是无脑整合。
- AgenticRAG/RAANav 才是本项目核心：轻量语义地图、重复建图、变化合并、GMM 查询、后续导航。
- 真机部署前，先把仿真严格验收，再做 rosbag 离线验证，最后上车。

硬性约束：
1. 不要修改 `参考开源代码/DAAAM` 原始代码；如需改造，复制到 `AgenticRAG/scripts/` 或临时目录验证。
2. 新代码必须有可视化或可验收输出。
3. 长实验前先说明目标、命令、输出目录和验收指标。
4. 不要清理或恢复用户未确认的代码变更；实验输出可以在用户授权后归档。
5. 不要在默认配置里调用外部 Agent/API，除非用户明确要求。

当前最新仿真验收记录：
`/home/adminer/agentRAG/计划/仿真收口验收_2026-05-21.md`

最新干净实验输出：
`/home/adminer/agentRAG/AgenticRAG/RAG_Graph/sim_verify/`

旧混乱输出已归档：
`/home/adminer/agentRAG/AgenticRAG/RAG_Graph/_archive_before_sim_verify_20260521_140831`

最新结论：
- Stage 1 短探索能生成地图、房间、Voronoi 和可视化，但使用 Habitat navmesh 运动，不等价真机。
- `chair` 导航正例成功，GT navmesh 和 `--no-navmesh` 都能发现。
- `microwave` 压力测试失败，原因更像上游开放词汇检测/裁剪假阳性。`microwave_det1_R0_s0030.jpg` 看起来像墙上画框/镜框，而不是微波炉。
- 当前进入真机部署前，最值得优先修的是对象级语义 QA、假阳性降权、重复观测稳定性和无 navmesh 长任务验证。

常用环境：
- Habitat/旧仿真导出：`/home/adminer/miniconda3/envs/agentrag/bin/python`
- DAAAM-style object-node frontend：`/home/adminer/miniconda3/envs/daaam_p0/bin/python`
- `daaam_p0` 已在 RTX 5060 Ti 上修复并验证 CUDA 可用：PyTorch `2.8.0+cu128`、torchvision `0.23.0+cu128`、torchaudio `2.8.0+cu128`、OpenCV `4.11.0`、ultralytics `8.4.47`、boxmot `11.0.8`、open_clip `3.3.0`。
- 环境修复细节和备份位置见：`计划/环境修复记录_2026-05-21.md`

常用工作目录：
`cd /home/adminer/agentRAG/AgenticRAG`

请先读：
1. `计划/仿真收口验收_2026-05-21.md`
2. `计划/当前代码结构总结.md`
3. `计划/RAANav主线前端决策_2026-05-21.md`
4. `计划/对象节点前端问题解答_2026-05-21.md`
5. `计划/环境修复记录_2026-05-21.md`
6. `计划/实时部署/具体实现部分的临时Notes.md`
7. `AgenticRAG/scripts/raanav_frontend/README.md`

最新架构决策：主线已抛弃 GroundingDINO+MobileSAM 和 DAM/VLM 核心依赖，采用 DAAAM-style object-node frontend：FastSAM -> BotSort -> representative assignment -> CLIP object-node embedding/top-k label -> spatial object-node clustering -> RAANav map/GMM。旧前端已迁到 AgenticRAG/scripts/z_legacy/legacy_gdino_frontend/。新入口见 AgenticRAG/scripts/raanav_frontend/README.md。

当前前端注意事项：
- 空检测/空轨迹是合法情况，不能让主线崩溃。
- ReID 是跨遮挡/跨短暂消失恢复同一物体身份；当前 TensorRT 依赖未配好，不作为下一轮验收阻塞项。
- 没有 ReID 时 track 会碎裂，所以 RAANav adapter 应优先消费 `summary.json` 里的 `object_nodes`，再回退 `tracks`。
- 结构性物体如栏杆、门框、固定柜体可以作为长期地图锚点，不要粗暴过滤；只过滤巨大墙/地面平面和低质量 mask。
- 地面 mask 后续适合用于可通行区域、尺度/姿态 sanity check 和代价地图，不进入 object-node 语义检索主体。
- CLIP 默认离线缓存加载；缺模型时才临时开代理下载。
- TensorRT/ReID 仍未修好，`clip_general.engine` 也不存在；当前验收先用 `--without-reid`，不要把 ReID 当作主线阻塞项。
- 不要随便改其它 conda 环境。已知之前曾向 `cow-sm120` 补过若干包，但主线现在应固定使用 `daaam_p0` 跑 DAAAM-style frontend。
- 本地已有两个未 push commit：`415c5c9 Adopt DAAAM-style object node frontend`、`4988557 Document daaam_p0 CUDA environment repair`。之前 push 失败，原因是 GitHub HTTPS 凭据缺失。

下一步建议：跑 20 positions x 4 views 的 Habitat sequence，用 run_object_node_frontend.py 生成 object nodes，再用 daaam_to_raanav_map.py 验证 RAANav map/GMM，并检查 representative crops 与 CLIP top-k label 质量。

建议命令：
1. 用 `agentrag` 导出 Habitat p20v4 RGB-D-pose sequence 到 `/tmp/raanav_habitat_p20v4`。
2. 用 `daaam_p0` 执行：
   `HF_HUB_OFFLINE=1 /home/adminer/miniconda3/envs/daaam_p0/bin/python scripts/raanav_frontend/run_object_node_frontend.py --data-path /tmp/raanav_habitat_p20v4 --output-dir /tmp/raanav_frontend_p20v4 --device cuda --without-reid --save-vis`
3. 再跑 `scripts/daaam/daaam_to_raanav_map.py` 验证 `summary.json` 到 RAANav map/GMM 的转换。
```
