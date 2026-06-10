#!/usr/bin/env python3
"""Export a ROS1 bag to DAAAM ImageSequenceDataset format.

Expected output structure:
  out_dir/
    rgb/000000.png
    depth/000000.png
    pose/poses.txt  (optional)
    camera_info.json

Notes:
- This script requires a ROS1 environment with rosbag and cv_bridge installed.
- Use --pose-topic to extract odometry/nav_msgs/Odometry messages into pose/poses.txt.
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

import rosbag
from cv_bridge import CvBridge
import cv2


def odom_to_4x4(odom_msg) -> np.ndarray:
    """Convert a nav_msgs/Odometry message to a 4x4 transformation matrix.

    Returns world_T_camera (camera pose in the odom/world frame).
    """
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    p = odom_msg.pose.pose.position
    q = odom_msg.pose.pose.orientation
    T[:3, 3] = [p.x, p.y, p.z]
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return T


def matrix_to_flat_str(T: np.ndarray) -> str:
    """Convert 4x4 matrix to a space-separated string of 16 floats (row-major)."""
    return " ".join(f"{v:.8f}" for v in T.flatten())


def load_odom_messages(bag_path: Path, pose_topic: str) -> List:
    """Load all odometry messages from a bag file, sorted by time."""
    msgs = []
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, msg, t in bag.read_messages(topics=[pose_topic]):
            msgs.append((t.to_sec(), msg))
    msgs.sort(key=lambda x: x[0])
    return msgs


def find_nearest_odom(odom_msgs, target_time, hint_idx):
    """Find the odometry message closest to target_time, starting from hint_idx."""
    best_idx = hint_idx
    best_dt = abs(odom_msgs[hint_idx][0] - target_time)
    search_range = 5
    start = max(0, hint_idx - search_range)
    end = min(len(odom_msgs), hint_idx + search_range + 1)
    for idx in range(start, end):
        dt = abs(odom_msgs[idx][0] - target_time)
        if dt < best_dt:
            best_dt = dt
            best_idx = idx
    return best_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ROS1 bag to image sequence dataset")
    parser.add_argument("--bag", required=True, help="Path to ROS1 .bag file")
    parser.add_argument("--out-dir", required=True, help="Output dataset root directory")
    parser.add_argument("--rgb-topic", required=True, help="RGB image topic")
    parser.add_argument("--depth-topic", required=True, help="Depth image topic")
    parser.add_argument("--camera-info-topic", required=True, help="CameraInfo topic")
    parser.add_argument("--pose-topic", type=str, default=None, help="Odometry topic for pose extraction (nav_msgs/Odometry)")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Depth scale to meters (e.g., 1000 for mm)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to export")
    parser.add_argument("--stride", type=int, default=1, help="Export every Nth RGB frame")
    parser.add_argument("--sync-tolerance", type=float, default=0.02, help="Max time delta (sec) for RGB-depth pairing")
    parser.add_argument("--poses-txt", type=str, default=None, help="Optional pre-built poses.txt to copy into pose/ directory")
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> Tuple[Path, Path, Path]:
    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    pose_dir = out_dir / "pose"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    return rgb_dir, depth_dir, pose_dir


def save_camera_info(out_dir: Path, msg) -> None:
    camera_info = {
        "width": msg.width,
        "height": msg.height,
        "intrinsics": [
            [msg.K[0], msg.K[1], msg.K[2]],
            [msg.K[3], msg.K[4], msg.K[5]],
            [msg.K[6], msg.K[7], msg.K[8]],
        ],
        "distortion": list(msg.D),
    }
    with open(out_dir / "camera_info.json", "w", encoding="utf-8") as f:
        json.dump(camera_info, f, indent=2)


def find_nearest_depth(depth_msgs, depth_times, target_time, start_idx) -> Tuple[Optional[int], float]:
    idx = start_idx
    best_idx = None
    best_dt = float("inf")
    while idx < len(depth_times) and depth_times[idx] < target_time:
        dt = abs(depth_times[idx] - target_time)
        if dt < best_dt:
            best_dt = dt
            best_idx = idx
        idx += 1
    if idx < len(depth_times):
        dt = abs(depth_times[idx] - target_time)
        if dt < best_dt:
            best_dt = dt
            best_idx = idx
    return best_idx, best_dt


def main() -> None:
    args = parse_args()
    bag_path = Path(args.bag)
    out_dir = Path(args.out_dir)

    if not bag_path.exists():
        raise FileNotFoundError(f"Bag not found: {bag_path}")

    rgb_dir, depth_dir, pose_dir = ensure_dirs(out_dir)
    bridge = CvBridge()

    # Load camera info (first message)
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[args.camera_info_topic]):
            save_camera_info(out_dir, msg)
            break

    # Pre-load odometry messages if requested
    odom_msgs = None
    if args.pose_topic:
        odom_msgs = load_odom_messages(bag_path, args.pose_topic)
        print(f"Loaded {len(odom_msgs)} odometry messages from {args.pose_topic}")

    # Pre-load depth message timestamps for fast pairing
    depth_msgs = []
    depth_times = []
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, msg, t in bag.read_messages(topics=[args.depth_topic]):
            depth_msgs.append(msg)
            depth_times.append(t.to_sec())

    if not depth_msgs:
        raise RuntimeError("No depth messages found for topic: " + args.depth_topic)

    # Export RGB + depth + poses
    pose_lines = []
    frame_idx = 0
    depth_idx_hint = 0
    odom_idx_hint = 0
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, msg, t in bag.read_messages(topics=[args.rgb_topic]):
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            if (frame_idx % args.stride) != 0:
                frame_idx += 1
                continue

            rgb_time = t.to_sec()
            depth_idx, dt = find_nearest_depth(depth_msgs, depth_times, rgb_time, depth_idx_hint)
            if depth_idx is None or dt > args.sync_tolerance:
                frame_idx += 1
                continue

            depth_idx_hint = max(depth_idx - 1, 0)

            # Snap odometry to this frame time
            if odom_msgs is not None:
                odom_idx_hint = find_nearest_odom(odom_msgs, rgb_time, odom_idx_hint)
                odom_time, odom_msg = odom_msgs[odom_idx_hint]
                T = odom_to_4x4(odom_msg)
                pose_lines.append(matrix_to_flat_str(T))

            rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            depth_msg = depth_msgs[depth_idx]
            depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            # Normalize depth to 16UC1 if needed
            if depth.dtype != "uint16":
                depth = (depth * args.depth_scale).astype("uint16")

            rgb_path = rgb_dir / f"{frame_idx:06d}.png"
            depth_path = depth_dir / f"{frame_idx:06d}.png"
            cv2.imwrite(str(rgb_path), rgb)
            cv2.imwrite(str(depth_path), depth)

            frame_idx += 1

    # Write poses.txt
    if pose_lines:
        pose_dir.mkdir(parents=True, exist_ok=True)
        pose_path = pose_dir / "poses.txt"
        with open(pose_path, "w", encoding="utf-8") as f:
            f.write("\n".join(pose_lines) + "\n")
        print(f"Wrote {len(pose_lines)} poses to {pose_path}")

    # Optional external poses.txt
    if args.poses_txt and not pose_lines:
        pose_dir.mkdir(parents=True, exist_ok=True)
        dst = pose_dir / "poses.txt"
        with open(args.poses_txt, "r", encoding="utf-8") as f_src:
            with open(dst, "w", encoding="utf-8") as f_dst:
                f_dst.write(f_src.read())

    print(f"Export complete. Frames written: {frame_idx}")


if __name__ == "__main__":
    main()
