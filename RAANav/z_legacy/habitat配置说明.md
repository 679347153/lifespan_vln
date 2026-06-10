experiment_data/hm3d/
├── hm3d_annotated_basis.scene_dataset_config.json <-- 场景数据集配置文件
└── val/
    └── 00824-Dd4bFSTQ8gi/
        ├── 00824-Dd4bFSTQ8gi.basis.glb          <-- 几何
        ├── 00824-Dd4bFSTQ8gi.semantic.glb       <-- 语义网格
        └── 00824-Dd4bFSTQ8gi.semantic.txt       <-- 语义标签

如果你打开 scene_instances 目录下的 JSON 文件（例如 00824-Dd4bFSTQ8gi.basis.scene_instance.json），你会看到类似下面的结构：JSON"object_instances": [
    {
        "template_name": "chair_model_A",
        "translation": [1.5, 0.2, -3.4],  // 这就是它的三维坐标 (x, y, z)
        "rotation": [0, 0, 0, 1]          // 这是它的旋转四元数 (x, y, z, w)
    }
]
translation：直接控制物体在房间里的具体位置。单位通常是米 (Meters)。rotation：控制物体的朝向（比如椅子面朝哪边）。坐标系注意：根据你之前提供的 scene_dataset_config.json，你的环境配置为 $Z$ 轴向上 ("up": [0, 0, 1])。所以修改 translation 的第三个值，物体就会上下移动。

修改位置的两种方式针对你的 AgenticRAG 研究，你可能会用到以下两种修改方式：
静态修改 想要永久改变房间布局（比如把沙发挪个位置）。直接修改上述的 .scene_instance.json 文件。
动态修改 机器人在导航过程中，物体位置发生了变化（动态环境）。在 Python 代码中调用 sim.set_translation(position, object_id)。

---
semantic.txt 格式：object_id, color_hex, "category", region_id。region_id 就是房间编号（比如0-9，共10个区域）。

https://github.com/matterport/habitat-matterport-3dresearch?tab=readme-ov-file#-downloading-hm3d-v02:~:text=%F0%9F%86%95%20Downloading%20HM3D%20v0.2
压缩包名称,对应文件类型,核心作用
hm3d-val-glb (4G),.basis.glb,视觉外观。包含物体的纹理和照片级的外观，主要用于生成机器人看到的 RGB 图像。
hm3d-val-habitat (3.3G),.basis.navmesh 等,物理与导航。包含用于碰撞检测和路径规划的导航网格（NavMesh）。如果不下载这个，机器人会直接穿墙。
hm3d-val-semantic-annots (2G),.semantic.glb,语义模型。正如你看到的，它把 3D 模型涂成了不同 ID 的色块，用于语义摄像头传感器。
hm3d-val-semantic-configs (40K),.semantic.txt 和 .json,标签与索引。包含你刚才看到的“字典”文件和场景实例配置文件，体积虽小却是连接所有数据的“灵魂”。

---
场景图导出信息，bbox、语义标签、物体 ID、位置坐标等，都是通过 habitat_sim 的 API 从 .basis.glb 和 .semantic.glb 中提取的。你可以在 AgenticRAG/scripts/export_scene_info.py 里找到相关代码。

脚本: export_scene_info.py

已导出文件: 00824-Dd4bFSTQ8gi_scene_info.json

使用方法:
- 导出单个场景
/home/adminer/miniconda3/envs/agentrag/bin/python /home/adminer/agentRAG/AgenticRAG/scripts/export_scene_info.py --scene 00824-Dd4bFSTQ8gi

- 导出全部 20 个场景
/home/adminer/miniconda3/envs/agentrag/bin/python /home/adminer/agentRAG/AgenticRAG/scripts/export_scene_info.py --all

- 自定义数据/输出目录
--data-dir /path/to/data --output-dir /path/to/output

JSON 内容结构（以 00824 为例：393 物体、79 类别、11 房间）:

scene_info — 场景元信息（名称、物体/房间/类别总数、可导航面积 78.69m²）
categories — 按数量排序的类别列表（wall:70, pillow:33, decoration:21...）
rooms — 每个房间的物体列表、包围盒、类别分布
objects — 每个物体的类别、所属房间、3D AABB 和 OBB（中心点+尺寸+旋转）