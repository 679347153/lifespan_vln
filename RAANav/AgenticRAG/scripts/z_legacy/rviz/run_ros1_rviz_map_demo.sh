#!/usr/bin/env bash
set -euo pipefail

MAP_JSON="${1:-/home/adminer/agentRAG/AgenticRAG/RAG_Graph/map_now.json}"
FULL_PCD="${2:-/home/adminer/agentRAG/HOV-SG/output_min/hm3dsem/00824-Dd4bFSTQ8gi/full_pcd.ply}"
RVIZ_CFG="${3:-/home/adminer/agentRAG/AgenticRAG/scripts/rviz/agentrag_map.rviz}"

if [[ -f /opt/ros/noetic/setup.bash ]]; then
  source /opt/ros/noetic/setup.bash
fi

echo "[INFO] MAP_JSON=$MAP_JSON"
echo "[INFO] FULL_PCD=$FULL_PCD"

if ! command -v roscore >/dev/null 2>&1; then
  echo "[ERROR] 未找到 roscore，请先安装/配置 ROS1 (noetic)."
  exit 1
fi

if ! command -v rviz >/dev/null 2>&1; then
  echo "[ERROR] 未找到 rviz，请先安装 rviz (ROS1)."
  exit 1
fi

# 启动 roscore（若未运行）
if ! (rostopic list >/dev/null 2>&1); then
  echo "[INFO] 启动 roscore..."
  roscore >/tmp/agentrag_roscore.log 2>&1 &
  sleep 2
fi

# 启动发布节点
python3 /home/adminer/agentRAG/AgenticRAG/scripts/ros1_rviz_visualize_map.py \
  --map_json "$MAP_JSON" \
  --full_pcd "$FULL_PCD" \
  --frame_id map \
  --rate 1.0 >/tmp/agentrag_map_vis.log 2>&1 &
VIS_PID=$!

sleep 1

echo "[INFO] 启动 RViz..."
rviz -d "$RVIZ_CFG"

# 退出 RViz 后清理发布进程
kill "$VIS_PID" >/dev/null 2>&1 || true
