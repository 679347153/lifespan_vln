# Object-node 质量门控与找物准确率优先级（2026-05-22）

## 用户确认的原则

当前阶段 object nodes 的质量门控不是为了把地图做得视觉上干净，也不是优先追求过滤所有语义碎片。最核心的验收目标是：在寻找目标物体的过程中，目标检索与导航指向的准确率足够高。

因此：

- 优先保留可调试信息，尽可能先跑完整链路。
- 语义碎片、结构性物体、短 track 等暂时不是第一优先级问题，只要它们不明显误导目标物查询/GMM。
- 如果明显假阳性会强烈牵引 GMM，例如把画框标成 microwave，则需要降权、待确认或负反馈处理。
- 如果质量门控短期做不完，就先通过可视化和目标查询实验判断系统 差不多能不能行，再决定是否加规则。

## 本轮可验收目标

1. 用 Habitat 20 positions x 4 views 导出 RGB-D-pose 序列。
2. 用 FastSAM + BotSort + CLIP top-k 生成 object nodes。
3. 检查 representative crops、top-k labels、object-node 数量、观测次数分布、GMM 可视化。
4. 优先回答：对 chair / microwave 等目标，检索候选是否肉眼合理，是否存在单个明显假阳性支配 GMM。

## 暂定质量判断顺序

1. 目标查询是否指向真实目标或合理候选。
2. 代表 crop 是否肉眼能支持 top-k 标签。
3. 同一空间位置是否有重复观测支撑。
4. 深度/3D 位置是否明显异常。
5. 语义碎片是否大规模污染目标查询。

## 注意

本文记录的是当前用户优先级，不代表最终算法设计已经定稿。后续如果真实 rosbag 或真机实验显示碎片会影响导航，再把碎片合并/降权作为独立任务处理。


## 2026-05-22 策略修正：默认软门控

用户确认：RAANav 查询链路本来会使用 `cfd`、`stability`、观测次数和后续 GMM 负反馈来处理不可靠目标。因此低置信 CLIP 标签不应默认被硬改成 `unknown`，否则目标查询阶段会完全看不到这个候选。

当前默认策略：

- `label` 保留候选语义，例如低置信 `microwave` 仍写作 `label=microwave`；
- 同时写入 `raw_label`、`label_confidence`、`label_topk`、`semantic_status=low_confidence`；
- `cfd` 按观测次数和 label confidence 计算，低置信候选自然低分；
- 若导航到该处找不到目标，后续由 GMM 负核/负反馈继续压制该位置；
- `--label-gate-mode hard` 仅作为保守消融选项。
