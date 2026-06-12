# 远程 CLIP Image Embedding 运行流程

本文档说明如何把 CLIP 图像 embedding 放到服务器端运行。修改后的流程是：

```text
工作站 Habitat / 导航脚本
  -> SSH tunnel
  -> 远端 /v1/detect_segment
  -> GroundingDINO 检测
  -> MobileSAM 分割
  -> CLIP 对 mask crop 编码
  -> 返回 bbox / mask / pos_3d 所需 mask / clip_embedding
  -> 工作站只做深度反投影和建图
```

这样工作站不再需要下载或加载 CLIP 图像模型，可以避开你遇到的 `openai/clip-vit-base-patch32` 离线缓存缺失问题。

## 1. 服务器端模型路径

根据你当前服务器目录，CLIP 已经在：

```bash
/home/ma-user/work/zhangWei/mtu3d/data/trans/clip-vit-large-patch14
```

GroundingDINO 的 BERT 文本编码器也在：

```bash
/home/ma-user/work/zhangWei/mtu3d/data/trans/bert-base-uncased
```

建议在服务器端显式导出这些路径：

```bash
cd /home/ma-user/work/zhangWei/mtu3d/data/trans/agent_based_method

export HF_HOME=/home/ma-user/work/zhangWei/mtu3d/data/trans
export TRANSFORMERS_CACHE=/home/ma-user/work/zhangWei/mtu3d/data/trans
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export BERT_BASE_UNCASED_PATH=/home/ma-user/work/zhangWei/mtu3d/data/trans/bert-base-uncased
export REMOTE_VISION_CLIP_MODEL_PATH=/home/ma-user/work/zhangWei/mtu3d/data/trans/clip-vit-large-patch14
```

## 2. 启动远程 Vision 服务

服务器端执行：

```bash
cd /home/ma-user/work/zhangWei/mtu3d/data/trans/agent_based_method

export REMOTE_VISION_HOST=127.0.0.1
export REMOTE_VISION_PORT=8010
export REMOTE_VISION_DEVICE=cuda
export REMOTE_VISION_EAGER_LOAD=1

bash remote_vision_server/run_server.sh
```

说明：

- `REMOTE_VISION_EAGER_LOAD=1` 会在服务启动时预加载 GroundingDINO + MobileSAM。
- CLIP 默认按需加载，第一次请求 `return_clip_embedding=true` 时才加载。
- `run_server.sh` 已经会自动补 PyTorch 的 `torch/lib` 到 `LD_LIBRARY_PATH`，用于解决 `libc10.so` 找不到的问题。

服务器本机检查：

```bash
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8010/v1/models
```

`/health` 中应能看到 `clip_model_path` 指向 `clip-vit-large-patch14`。

## 3. 工作站建立 SSH tunnel

工作站执行：

```bash
cd ~/Downloads/lifespan_vln

export SSHPASS='666666'
export REMOTE_VISION_SSH_HOST=7.216.187.6
export REMOTE_VISION_SSH_PORT=30180
export REMOTE_VISION_SSH_USER=root
export REMOTE_VISION_LOCAL_PORT=50220
export REMOTE_VISION_REMOTE_PORT=8010

bash remote_vision_server/workstation_tunnel.sh
```

新开一个终端检查：

```bash
curl http://127.0.0.1:50220/health
curl http://127.0.0.1:50220/v1/models
```

注意：不要直接在 shell 里输入 `http://127.0.0.1:50220`，那不是命令；要用 `curl`。

## 4. 单张图测试远程 CLIP embedding

在工作站执行：

```bash
python remote_vision_server/test_client.py \
  --base-url http://127.0.0.1:50220 \
  --image objects_images/Camera_01.webp \
  --labels "chair,table,sofa,bed,cabinet,lamp" \
  --return-mask-png \
  --return-clip-embedding
```

成功时，每个 detection 会多出：

```json
"clip_embedding": [ ... ]
```

如果使用 `clip-vit-large-patch14`，embedding 通常是 768 维；如果使用 `clip-vit-base-patch32`，通常是 512 维。关键是同一次实验内保持同一个 CLIP 模型。

## 5. 运行闭环导航

进入 AgenticRAG 目录：

```bash
cd ~/Downloads/lifespan_vln/RAANav/AgenticRAG
```

推荐命令：服务器算图像 CLIP，工作站不加载本地 CLIP。

```bash
python scripts/z_legacy/legacy_gdino_frontend/sim_nav_loop_gdino.py \
  --scene-dir ../../hm3d/minival/00808-y9hTuugGdiq \
  --dataset-config ../../hm3d/hm3d_val_scene_dataset_config.json \
  --target chair \
  --output-dir RAG_Graph/remote_vision_nav_smoke \
  --max-steps 5 \
  --n-views 4 \
  --perception-backend remote \
  --remote-vision-base-url http://127.0.0.1:50220 \
  --remote-clip \
  --remote-clip-model-path /home/ma-user/work/zhangWei/mtu3d/data/trans/clip-vit-large-patch14 \
  --disable-clip
```

这里几个参数的含义：

- `--remote-clip`：请求远端返回每个 detection 的 `clip_embedding`。
- `--remote-clip-model-path`：这是服务器上的路径，不是工作站路径。
- `--disable-clip`：禁止工作站本地加载 CLIP。远程 CLIP 不受影响。

如果你希望工作站后续 GMM 查询仍使用本地 CLIP 文本检索，则不要加 `--disable-clip`，但工作站必须有可加载的 CLIP 模型：

```bash
python scripts/z_legacy/legacy_gdino_frontend/sim_nav_loop_gdino.py \
  --scene-dir ../../hm3d/minival/00808-y9hTuugGdiq \
  --dataset-config ../../hm3d/hm3d_val_scene_dataset_config.json \
  --target chair \
  --output-dir RAG_Graph/remote_vision_nav_full \
  --max-steps 20 \
  --n-views 4 \
  --perception-backend remote \
  --remote-vision-base-url http://127.0.0.1:50220 \
  --remote-clip \
  --remote-clip-model-path /home/ma-user/work/zhangWei/mtu3d/data/trans/clip-vit-large-patch14 \
  --clip-model-path /path/on/workstation/clip-vit-large-patch14
```

## 6. 自动 SSH tunnel 运行方式

也可以不手动开 tunnel，让脚本自动建立：

```bash
python scripts/z_legacy/legacy_gdino_frontend/sim_nav_loop_gdino.py \
  --scene-dir ../../hm3d/minival/00808-y9hTuugGdiq \
  --dataset-config ../../hm3d/hm3d_val_scene_dataset_config.json \
  --target chair \
  --output-dir RAG_Graph/remote_vision_nav_smoke \
  --max-steps 5 \
  --n-views 4 \
  --perception-backend remote \
  --remote-vision-use-ssh-tunnel \
  --remote-vision-ssh-host 7.216.187.6 \
  --remote-vision-ssh-port 30180 \
  --remote-vision-ssh-user root \
  --remote-vision-ssh-password '666666' \
  --remote-vision-remote-port 8010 \
  --remote-clip \
  --remote-clip-model-path /home/ma-user/work/zhangWei/mtu3d/data/trans/clip-vit-large-patch14 \
  --disable-clip
```

工作站需要安装：

```bash
sudo apt install sshpass
```

## 7. 常见问题

### 7.1 仍然在工作站加载 `openai/clip-vit-base-patch32`

说明没有禁用本地 CLIP，或没有打开远程 CLIP。请确认命令里有：

```bash
--remote-clip --disable-clip
```

### 7.2 远端请求 500

先看服务器端日志。常见原因：

- GroundingDINO `_C` 扩展没有成功加载。
- `libc10.so` 不在 `LD_LIBRARY_PATH`。
- `REMOTE_VISION_CLIP_MODEL_PATH` 路径不存在或不是完整 Hugging Face 模型目录。
- 服务器环境设置了离线模式，但 CLIP/BERT 文件不完整。

### 7.3 `/health` 能通，但第一次检测很慢

这是正常的。第一次请求会加载模型和 CUDA kernel，之后会复用内存中的模型。

### 7.4 `clip_embedding` 为空

检查请求是否带了：

```json
"return_clip_embedding": true
```

使用脚本时对应参数是：

```bash
--remote-clip
```

### 7.5 使用 `clip-vit-large-patch14` 后维度变成 768

这是正常的。不要在同一个地图中混用 512 维和 768 维 embedding。正式实验建议固定：

```bash
REMOTE_VISION_CLIP_MODEL_PATH=/home/ma-user/work/zhangWei/mtu3d/data/trans/clip-vit-large-patch14
```

