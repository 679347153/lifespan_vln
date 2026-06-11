# RAANav / Agentic-RAG 在 RTX4000 + CUDA 12.8 环境中的完整构建流程

> 目标：从新机器 `git clone` 项目后，构建可运行离线评测、真实 Habitat-Sim 闭环、CLIP 图像检索、GroundingDINO + MobileSAM 感知模块的环境。

参考源：

- PyTorch 官方安装页支持 CUDA 12.8 选项，并要求 Python 3.10+：<https://pytorch.org/get-started/locally/>
- Habitat-Sim 官方推荐 Conda 安装 `habitat-sim withbullet/headless`：<https://github.com/facebookresearch/habitat-sim>
- GroundingDINO 官方权重下载方式：<https://github.com/IDEA-Research/GroundingDINO>
- MobileSAM 官方安装与 `mobile_sam.pt` 使用方式：<https://github.com/ChaoningZhang/MobileSAM>
- CLIP 模型 `openai/clip-vit-base-patch32`：<https://huggingface.co/openai/clip-vit-base-patch32>

---

## 1. 推荐系统环境

建议使用 Linux，而不是原生 Windows。

推荐：

```text
Ubuntu 20.04 / 22.04
NVIDIA Driver 支持 CUDA 12.8
Python 3.10
Conda / Mamba
RTX4000 GPU
```

检查 GPU：

```bash
nvidia-smi
nvcc --version
```

如果 `nvidia-smi` 能看到 RTX4000，并且 CUDA 版本显示 `12.8` 或兼容版本，即可继续。

---

## 2. Clone 项目

```bash
git clone <your-repo-url> agent_based_method
cd agent_based_method
```

建议之后所有命令都在项目根目录执行。

项目根目录应类似：

```text
agent_based_method/
  RAANav/
  benchmark/
  hm3d_paths.py
  objects/
  objects_images/
  hm3d/
  results/
```

注意：`hm3d/`、`objects/`、`objects_images/`、`results/`、大模型权重通常不会随 Git 一起上传，需要单独复制或下载。

---

## 3. 创建 Conda 环境

```bash
conda create -n raanav python=3.10 -y
conda activate raanav

python -m pip install --upgrade pip setuptools wheel
```

建议固定 `numpy<2`，避免 Habitat-Sim、OpenCV、部分 CUDA 扩展出现 ABI 问题：

```bash
pip install "numpy==1.26.4"
```

---

## 4. 安装 PyTorch CUDA 12.8

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

期望看到：

```text
True
NVIDIA RTX ...
```

---

## 5. 安装项目基础依赖

```bash
pip install -r RAANav/AgenticRAG/requirements.txt
```

再补充安装闭环评测、图像检索、感知模块常用依赖：

```bash
pip install \
  opencv-python \
  pillow \
  scipy \
  scikit-image \
  scikit-learn \
  tqdm \
  requests \
  huggingface_hub \
  safetensors \
  shapely
```

---

## 6. 安装 Habitat-Sim

真实闭环导航需要 Habitat-Sim。官方推荐通过 Conda 安装。

有显示器环境：

```bash
conda install habitat-sim withbullet -c conda-forge -c aihabitat -y
```

服务器 / headless 环境：

```bash
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat -y
```

验证：

```bash
python -c "import habitat_sim; print('habitat_sim ok')"
```

如果这一步失败，项目仍可用欧氏距离 fallback 跑离线闭环，但不能算真实 Habitat-Sim 闭环。

---

## 7. 安装 GroundingDINO

```bash
mkdir -p third_party
git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
cd third_party/GroundingDINO
pip install -e .
cd ../..
```

验证：

```bash
python -c "import groundingdino; print('groundingdino ok')"
```

---

## 8. 安装 MobileSAM

```bash
git clone https://github.com/ChaoningZhang/MobileSAM.git third_party/MobileSAM
cd third_party/MobileSAM
pip install -e .
cd ../..
```

验证：

```bash
python -c "import mobile_sam; print('mobile_sam ok')"
```

---

## 9. 下载模型 Checkpoints

创建权重目录：

```bash
mkdir -p RAANav/AgenticRAG/checkpoints
```

### 9.1 GroundingDINO 权重

```bash
wget -O RAANav/AgenticRAG/checkpoints/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

期望文件：

```text
RAANav/AgenticRAG/checkpoints/groundingdino_swint_ogc.pth
```

### 9.2 MobileSAM 权重

```bash
wget -O RAANav/AgenticRAG/checkpoints/mobile_sam.pt \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

期望文件：

```text
RAANav/AgenticRAG/checkpoints/mobile_sam.pt
```

### 9.3 CLIP 权重

项目使用 Hugging Face 的：

```text
openai/clip-vit-base-patch32
```

提前下载到本地缓存：

```bash
python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); print('clip downloaded')"
```

如果希望缓存放在项目目录：

```bash
export HF_HOME=$PWD/.hf_cache
```

如果后续离线运行：

```bash
export HF_HUB_OFFLINE=1
```

---

## 10. 准备数据目录

Git clone 后通常不会自动包含大数据。需要从原机器复制这些目录：

```text
hm3d/
objects/
objects_images/
results/layouts/
benchmark/episodes/
benchmark/splits/
```

推荐结构：

```text
agent_based_method/
  hm3d/
    minival/
      00808-y9hTuugGdiq/
        *.basis.glb
        *.basis.navmesh
        *.semantic.glb
        *.semantic.txt

  objects/
    ...

  objects_images/
    ...

  results/
    layouts/
      00808-y9hTuugGdiq/
        batch_xxx/
          manifest.json
          layout_*.json

  benchmark/
    episodes/
      longterm_v1/
        val/
          ...
    splits/
      benchmark_split_longterm_v1.json
```

从原机器复制示例：

```bash
rsync -av --progress \
  hm3d objects objects_images results benchmark/episodes benchmark/splits \
  user@target-machine:/path/to/agent_based_method/
```

---

## 11. 设置环境变量

每次进入环境后执行：

```bash
conda activate raanav
cd /path/to/agent_based_method

export PYTHONPATH=$PWD:$PWD/RAANav/AgenticRAG:$PWD/third_party/GroundingDINO:$PWD/third_party/MobileSAM:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$PWD/.hf_cache
```

如果使用需要 LLM 的 Agent/RAG 评分模块，再设置 API Key：

```bash
export QWEN_API_KEY=<your-api-key>
```

---

## 12. 最小导入验证

```bash
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
python -c "import habitat_sim; print('habitat_sim ok')"
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); print('clip ok')"
python -c "import groundingdino; import mobile_sam; print('vision models ok')"
```

---

## 13. 运行闭环 Smoke Test

### 13.1 不强制 Habitat-Sim，允许 fallback

```bash
python -m RAANav.AgenticRAG.benchmark_adapter.run_closed_loop \
  --split-manifest benchmark/splits/benchmark_split_longterm_v1.json \
  --split val \
  --scene 00808-y9hTuugGdiq \
  --limit 1 \
  --output-dir benchmark/eval/env_smoke_fallback \
  --max-candidates 10 \
  --max-rounds 5
```

### 13.2 强制真实 Habitat-Sim 闭环

```bash
python -m RAANav.AgenticRAG.benchmark_adapter.run_closed_loop \
  --split-manifest benchmark/splits/benchmark_split_longterm_v1.json \
  --split val \
  --scene 00808-y9hTuugGdiq \
  --limit 1 \
  --output-dir benchmark/eval/env_smoke_habitat \
  --max-candidates 10 \
  --max-rounds 5 \
  --load-layout-objects \
  --no-euclidean-fallback
```

如果这条命令成功，说明 Habitat-Sim、HM3D 路径、layout object 加载、Agentic-RAG 闭环都已经接通。

---

## 14. 运行完整验证

```bash
python -m RAANav.AgenticRAG.benchmark_adapter.run_closed_loop \
  --split-manifest benchmark/splits/benchmark_split_longterm_v1.json \
  --split val \
  --output-dir benchmark/eval/raanav_closed_loop_longterm_v1_habitat \
  --max-candidates 10 \
  --max-rounds 5 \
  --load-layout-objects \
  --no-euclidean-fallback
```

输出目录中应生成：

```text
benchmark/eval/raanav_closed_loop_longterm_v1_habitat/
  run.jsonl
  memory_diagnostics.jsonl
  candidate_traces.jsonl
  failure_feedback.jsonl
  memory_summary.json
  memory_by_task_type.json
  memory_exploration_curve.json
  summary.json
  by_task_type.json
  episode_results.jsonl
```

---

## 15. 常见问题

### PyTorch 显示 `cuda.is_available() == False`

重新安装 CUDA 12.8 wheel：

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

同时确认：

```bash
nvidia-smi
```

### `habitat_sim` 导入失败

优先使用 Linux + Conda 安装：

```bash
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat -y
```

原生 Windows 不推荐跑 Habitat-Sim。

### CLIP 下载失败

可提前手动下载：

```bash
huggingface-cli download openai/clip-vit-base-patch32
```

网络不稳定时可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 显存不足

先跑小规模验证：

```bash
--limit 1
--max-candidates 5
--max-rounds 3
```

确认流程通了以后，再扩大到完整验证。

---

## 16. 推荐最终验收顺序

1. `torch.cuda.is_available()` 为 `True`
2. `import habitat_sim` 成功
3. CLIP 权重能加载
4. GroundingDINO / MobileSAM 能导入
5. fallback smoke test 成功
6. `--no-euclidean-fallback` Habitat smoke test 成功
7. 完整 `longterm_v1 val` 闭环评测成功

完成以上 7 步后，这台 RTX4000 + CUDA 12.8 机器就具备完整运行当前项目的环境。
