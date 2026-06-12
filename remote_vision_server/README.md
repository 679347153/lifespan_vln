# 远程 GroundingDINO + MobileSAM 视觉服务器部署流程

> 目标：把 GroundingDINO 和 MobileSAM 放在 GPU 服务器上运行，工作站只通过 SSH tunnel 调用远程视觉服务。调用方式与当前 Qwen/vLLM 流程保持一致：远程服务只监听 `127.0.0.1`，本地通过 SSH 端口转发访问。

当前项目已经接入远程调用：

- 工作站侧兼容入口：`RAANav/AgenticRAG/semantic_map_Create/perception.py`
- 工作站侧远程客户端：`remote_vision_server/client.py`
- 服务器侧模型服务：`remote_vision_server/server.py`
- 旧 GroundingDINO/MobileSAM 导航入口已新增远程参数：`RAANav/AgenticRAG/scripts/z_legacy/legacy_gdino_frontend/sim_nav_loop_gdino.py`

---

## 1. 总体调用链路

```text
工作站 Python 脚本
  -> 通过 SSH 登录远程服务器
  -> 建立本地端口转发 tunnel
  -> 本地访问 http://127.0.0.1:<local_port>/v1/detect_segment
  -> 实际请求被转发到远程服务器 127.0.0.1:8010/v1/detect_segment
  -> 远程 GroundingDINO + MobileSAM 返回 bbox、label、confidence、mask 信息
```

建议端口规划：

```text
Qwen/vLLM 远端端口: 127.0.0.1:8000
Vision 远端端口:   127.0.0.1:8010
Vision 本地端口:   127.0.0.1:50220
```

当前 SSH 信息可沿用已有配置：

```text
SSH host: 7.216.187.6
SSH port: 30180
SSH user: root
远端 Vision 服务地址: 127.0.0.1:8010
```

密码不要写入代码。测试时可以通过环境变量传入：

```bash
export SSHPASS='<server-password>'
```

---

## 2. 服务器端目录准备

在服务器上 clone 项目：

```bash
git clone <your-repo-url> agent_based_method
cd agent_based_method
```

推荐最终结构：

```text
agent_based_method/
  remote_vision_server/
    server.py
    run_server.sh
    requirements_server.txt
  third_party/
    GroundingDINO/
    MobileSAM/
  RAANav/
    AgenticRAG/
      checkpoints/
        groundingdino_swint_ogc.pth
        mobile_sam.pt
```

---

## 3. 服务器端创建 Python 环境

建议使用 Python 3.10：

```bash
conda create -n raanav-vision python=3.10 -y
conda activate raanav-vision

python -m pip install --upgrade pip setuptools wheel
pip install "numpy==1.26.4"
```

安装 PyTorch CUDA 12.8：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

验证 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

安装远程视觉服务依赖：

```bash
pip install -r remote_vision_server/requirements_server.txt
```

---

## 4. 安装 GroundingDINO 和 MobileSAM

```bash
mkdir -p third_party

git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
cd third_party/GroundingDINO
pip install -e .
cd ../..

git clone https://github.com/ChaoningZhang/MobileSAM.git third_party/MobileSAM
cd third_party/MobileSAM
pip install -e .
cd ../..
```

验证导入：

```bash
python -c "import groundingdino; print('groundingdino ok')"
python -c "import mobile_sam; print('mobile_sam ok')"
```

如果启动时出现：

```text
Failed to load custom C++ ops. Running on CPU mode Only!
```

说明 GroundingDINO 的 CUDA/C++ 扩展没有编译成功。建议在服务器上重新编译：

```bash
conda activate raanav-vision
cd /path/to/agent_based_method/third_party/GroundingDINO

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export FORCE_CUDA=1

python setup.py build_ext --inplace
pip install -e .
```

如果服务器没有 `/usr/local/cuda`，先查 CUDA 路径：

```bash
which nvcc
dirname "$(dirname "$(which nvcc)")"
```

然后把 `CUDA_HOME` 设置成该路径。

---

## 5. 下载模型 Checkpoints

```bash
mkdir -p RAANav/AgenticRAG/checkpoints
```

下载 GroundingDINO：

```bash
wget -O RAANav/AgenticRAG/checkpoints/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

下载 MobileSAM：

```bash
wget -O RAANav/AgenticRAG/checkpoints/mobile_sam.pt \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

检查文件：

```bash
ls -lh RAANav/AgenticRAG/checkpoints/
```

应看到：

```text
groundingdino_swint_ogc.pth
mobile_sam.pt
```

---

## 6. 启动服务器端视觉服务

在服务器上执行：

```bash
conda activate raanav-vision
cd /path/to/agent_based_method

export REMOTE_VISION_REPO_ROOT=$PWD
export REMOTE_VISION_DEVICE=cuda
export REMOTE_VISION_HOST=127.0.0.1
export REMOTE_VISION_PORT=8010
export HF_HOME=/path/to/local/hf_or_model_root
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

bash remote_vision_server/run_server.sh
```

如果你的服务器目录和当前示例一致：

```text
/home/ma-user/work/zhangWei/mtu3d/data/trans/
  agent_based_method/
  bert-base-uncased/
  clip-vit-large-patch14/
  dinov2-large/
```

则推荐这样启动：

```bash
conda activate pytorch_cuda12.9
cd /home/ma-user/work/zhangWei/mtu3d/data/trans/agent_based_method

export HF_HOME=/home/ma-user/work/zhangWei/mtu3d/data/trans
export BERT_BASE_UNCASED_PATH=/home/ma-user/work/zhangWei/mtu3d/data/trans/bert-base-uncased
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export REMOTE_VISION_EAGER_LOAD=1

bash remote_vision_server/run_server.sh
```

`run_server.sh` 也会自动尝试寻找：

```text
$REMOTE_VISION_REPO_ROOT/../bert-base-uncased
$HF_HOME/bert-base-uncased
$TRANSFORMERS_CACHE/bert-base-uncased
```

找到后会把 GroundingDINO 的 `text_encoder_type` 改为本地绝对路径，避免联网访问 Hugging Face。

服务默认只监听：

```text
http://127.0.0.1:8010
```

这样外网不能直接访问，只能通过 SSH tunnel 访问，安全性与当前 Qwen/vLLM 调用方式一致。

服务器本机测试：

```bash
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8010/v1/models
```

第一次真正请求 `/v1/detect_segment` 时会加载 GroundingDINO 和 MobileSAM，可能需要几十秒。

如果希望启动时立即加载模型：

```bash
export REMOTE_VISION_EAGER_LOAD=1
bash remote_vision_server/run_server.sh
```

---

## 7. 工作站建立 SSH Tunnel

工作站不需要安装 GroundingDINO、MobileSAM 或下载这两个 checkpoint；这些只放在服务器上。工作站只需要：

```bash
pip install requests pillow
```

工作站需要有 `sshpass`：

```bash
sshpass -V
```

如果没有：

```bash
sudo apt install sshpass
```

建立 Vision tunnel：

```bash
cd /path/to/agent_based_method

export SSHPASS='<server-password>'
export REMOTE_VISION_SSH_HOST=7.216.187.6
export REMOTE_VISION_SSH_PORT=30180
export REMOTE_VISION_SSH_USER=root
export REMOTE_VISION_LOCAL_PORT=50220
export REMOTE_VISION_REMOTE_PORT=8010

bash remote_vision_server/workstation_tunnel.sh
```

如果 tunnel 成功，这个终端会一直停在 SSH 进程中，不会回到 shell prompt。请保持该终端打开，再开第二个终端执行 `curl` 或项目脚本。

tunnel 建立后，工作站访问：

```text
http://127.0.0.1:50220
```

实际会转发到服务器：

```text
127.0.0.1:8010
```

测试：

```bash
curl http://127.0.0.1:50220/health
curl http://127.0.0.1:50220/v1/models
```

如果运行 `bash remote_vision_server/workstation_tunnel.sh` 后马上回到 prompt，说明 SSH tunnel 没有建立成功。打开 debug：

```bash
export REMOTE_VISION_SSH_DEBUG=1
bash remote_vision_server/workstation_tunnel.sh
```

也可以直接检查远端服务是否真的在 `8010`：

```bash
SSHPASS='666666' sshpass -e ssh -p 30180 root@7.216.187.6 \
  "curl -sS http://127.0.0.1:8010/health"
```

---

## 8. 工作站代码自动调用远程模型

### 8.1 方式 A：手动先开 tunnel，再运行项目代码

终端 1，建立 tunnel：

```bash
cd /path/to/agent_based_method

export SSHPASS='<server-password>'
export REMOTE_VISION_SSH_HOST=7.216.187.6
export REMOTE_VISION_SSH_PORT=30180
export REMOTE_VISION_SSH_USER=root
export REMOTE_VISION_LOCAL_PORT=50220
export REMOTE_VISION_REMOTE_PORT=8010

bash remote_vision_server/workstation_tunnel.sh
```

终端 2，运行旧 GroundingDINO/MobileSAM 导航脚本，并指定远程后端：

```bash
cd /path/to/agent_based_method/RAANav/AgenticRAG

python scripts/z_legacy/legacy_gdino_frontend/sim_nav_loop_gdino.py \
  --scene-dir ../../hm3d/minival/00808-y9hTuugGdiq \
  --dataset-config ../../hm3d/hm3d_val_scene_dataset_config.json \
  --target chair \
  --output-dir RAG_Graph/remote_vision_nav_smoke \
  --max-steps 5 \
  --n-views 4 \
  --perception-backend remote \
  --remote-vision-base-url http://127.0.0.1:50220
```

这种方式最稳定，适合长时间实验。tunnel 进程可以单独管理。

### 8.2 方式 B：项目代码自动拉起 SSH tunnel

```bash
cd /path/to/agent_based_method/RAANav/AgenticRAG

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
  --remote-vision-ssh-password '<server-password>' \
  --remote-vision-remote-port 8010
```

自动模式会：

1. 在本地选择一个空闲端口，除非你指定 `--remote-vision-local-port`。
2. 使用 `sshpass + ssh -L` 建立 tunnel。
3. 访问远端 `127.0.0.1:8010`。
4. 运行完成后由 Python 进程清理 tunnel。

也可以完全通过环境变量控制，不改命令行：

```bash
export RAANAV_PERCEPTION_BACKEND=remote
export REMOTE_VISION_USE_SSH_TUNNEL=1
export REMOTE_VISION_SSH_HOST=7.216.187.6
export REMOTE_VISION_SSH_PORT=30180
export REMOTE_VISION_SSH_USER=root
export REMOTE_VISION_SSH_PASSWORD='<server-password>'
export REMOTE_VISION_REMOTE_PORT=8010
```

然后正常运行旧脚本即可。只要代码调用的是：

```python
from semantic_map_Create.perception import get_detector
detector = get_detector()
```

就会自动切换到远程模型。

---

## 9. 工作站测试图像检测与分割

保持 tunnel 进程运行，另开一个终端：

```bash
cd /path/to/agent_based_method

python remote_vision_server/test_client.py \
  --base-url http://127.0.0.1:50220 \
  --image objects_images/Camera_01.webp \
  --labels "camera,chair,table,lamp,cabinet" \
  --box-threshold 0.35 \
  --text-threshold 0.35
```

返回示例：

```json
{
  "model": "GroundingDINO-SwinT-OGC+MobileSAM-vit_t",
  "device": "cuda",
  "elapsed_seconds": 1.234,
  "width": 640,
  "height": 480,
  "detections": [
    {
      "label": "camera",
      "bbox_xyxy": [100.0, 80.0, 220.0, 210.0],
      "confidence": 0.61,
      "mask_area": 8231
    }
  ]
}
```

如果需要返回 mask PNG：

```bash
python remote_vision_server/test_client.py \
  --base-url http://127.0.0.1:50220 \
  --image objects_images/Camera_01.webp \
  --labels "camera,chair,table,lamp,cabinet" \
  --return-mask-png
```

`mask_png_base64` 会明显增大响应体，批量调用时建议默认关闭，只在需要保存 mask 时开启。

---

## 10. HTTP API 说明

### `GET /health`

检查服务是否在线。

### `GET /v1/models`

返回伪 OpenAI-compatible 风格的模型列表，便于和现有 Qwen/vLLM 健康检查习惯保持一致。

### `POST /v1/detect_segment`

请求体：

```json
{
  "image_base64": "data:image/webp;base64,...",
  "labels": ["chair", "table", "lamp"],
  "box_threshold": 0.35,
  "text_threshold": 0.35,
  "return_mask_png": false
}
```

也可以直接传 GroundingDINO prompt：

```json
{
  "image_base64": "data:image/png;base64,...",
  "text_prompt": "chair . table . lamp .",
  "box_threshold": 0.35,
  "text_threshold": 0.35
}
```

服务器本机调试时可以使用远端本地图片路径：

```json
{
  "image_path": "/path/to/image.png",
  "labels": ["chair", "table"]
}
```

---

## 11. 与当前 Qwen/vLLM 流程的关系

现有 Qwen/vLLM：

```text
工作站 127.0.0.1:<local_qwen_port>
  -> SSH tunnel
  -> 服务器 127.0.0.1:8000/v1
```

新增 Vision 服务：

```text
工作站 127.0.0.1:50220
  -> SSH tunnel
  -> 服务器 127.0.0.1:8010/v1/detect_segment
```

两者可以同时运行，只要本地端口和远端端口不同即可：

```text
Qwen:   remote 8000, local 自动空闲端口或指定端口
Vision: remote 8010, local 50220
```

---

## 12. 常见问题

### `sshpass not found`

工作站缺少 `sshpass`：

```bash
sudo apt install sshpass
```

### `SSH tunnel not ready`

检查：

```bash
export SSHPASS='<server-password>'
sshpass -e ssh -p 30180 root@7.216.187.6
```

如果无法进入远程 shell，说明 SSH 地址、端口、用户名、密码或服务器策略有问题。

### `curl http://127.0.0.1:50220/health` 无响应

分别检查：

1. 服务器端 `bash remote_vision_server/run_server.sh` 是否还在运行。
2. 服务器端 `curl http://127.0.0.1:8010/health` 是否正常。
3. 工作站 tunnel 进程是否还在运行。
4. 本地端口 `50220` 是否被其他程序占用。

### 第一次请求很慢

第一次 `/v1/detect_segment` 会加载 GroundingDINO 和 MobileSAM。后续请求会复用同一份模型。

### 启动时反复请求 `https://huggingface.co/bert-base-uncased`

这是 GroundingDINO 的文本编码器没有命中本地 BERT 路径。按以下方式启动：

```bash
cd /home/ma-user/work/zhangWei/mtu3d/data/trans/agent_based_method

export HF_HOME=/home/ma-user/work/zhangWei/mtu3d/data/trans
export BERT_BASE_UNCASED_PATH=/home/ma-user/work/zhangWei/mtu3d/data/trans/bert-base-uncased
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export REMOTE_VISION_EAGER_LOAD=1

bash remote_vision_server/run_server.sh
```

确认 BERT 目录里至少有：

```bash
ls /home/ma-user/work/zhangWei/mtu3d/data/trans/bert-base-uncased/config.json
ls /home/ma-user/work/zhangWei/mtu3d/data/trans/bert-base-uncased/vocab.txt
```

### 工作站仍然尝试加载本地 GroundingDINO

确认以下任一方式已经启用：

```bash
export RAANAV_PERCEPTION_BACKEND=remote
```

或运行脚本时加入：

```bash
--perception-backend remote
```

如果没有设置远程模式，`semantic_map_Create.perception.get_detector()` 会回退到本地 legacy GroundingDINO/MobileSAM。

### 报 `Required model file not found`

检查：

```bash
ls -lh RAANav/AgenticRAG/checkpoints/groundingdino_swint_ogc.pth
ls -lh RAANav/AgenticRAG/checkpoints/mobile_sam.pt
ls -lh third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
```

如路径不同，可以用环境变量覆盖：

```bash
export GDINO_CONFIG=/abs/path/to/GroundingDINO_SwinT_OGC.py
export GDINO_CHECKPOINT=/abs/path/to/groundingdino_swint_ogc.pth
export MOBILESAM_CHECKPOINT=/abs/path/to/mobile_sam.pt
```

### PyTorch 看不到 GPU

检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

必要时重装 CUDA 12.8 wheel：

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## 13. 推荐验收顺序

1. 服务器端 `torch.cuda.is_available()` 为 `True`
2. 服务器端 `import groundingdino` 成功
3. 服务器端 `import mobile_sam` 成功
4. 服务器端 `curl http://127.0.0.1:8010/health` 成功
5. 工作站 SSH tunnel 成功建立
6. 工作站 `curl http://127.0.0.1:50220/health` 成功
7. 工作站 `test_client.py` 能返回 detections
8. 工作站旧导航脚本带 `--perception-backend remote` 能开始运行，并在服务器日志中看到 `/v1/detect_segment` 请求

完成以上步骤后，GroundingDINO + MobileSAM 就已经从工作站本地推理切换为服务器远程推理。
