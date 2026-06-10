#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _try_import_ros():
    try:
        import rospy  # type: ignore
        from std_msgs.msg import Header, ColorRGBA  # type: ignore
        from geometry_msgs.msg import Point  # type: ignore
        from visualization_msgs.msg import Marker, MarkerArray  # type: ignore
        from sensor_msgs.msg import PointCloud2  # type: ignore
        from sensor_msgs import point_cloud2  # type: ignore
        return rospy, Header, ColorRGBA, Point, Marker, MarkerArray, PointCloud2, point_cloud2
    except Exception as exc:
        raise RuntimeError(
            "ROS1 Python 依赖不可用。请先 source ROS1 环境，例如: source /opt/ros/noetic/setup.bash"
        ) from exc


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"地图顶层应为 list: {path}")
    return data


def _room_z_center(floor: Dict[str, Any]) -> float:
    zr = floor.get('z_range') or {}
    z_min = float(zr.get('z_min', 0.0))
    z_max = float(zr.get('z_max', 0.0))
    return (z_min + z_max) * 0.5


def _poly_to_points(poly: List[Dict[str, Any]], z: float, Point):
    pts = []
    for p in poly:
        pt = Point()
        pt.x = float(p.get('x', 0.0))
        pt.y = float(p.get('y', 0.0))
        pt.z = float(z)
        pts.append(pt)
    if pts:
        pts.append(pts[0])
    return pts


def _marker_line(marker_id: int, ns: str, frame_id: str, pts, color, scale: float, Marker):
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = ns
    m.id = marker_id
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = scale
    m.color = color
    m.points = pts
    m.pose.orientation.w = 1.0
    return m


def _marker_text(marker_id: int, ns: str, frame_id: str, x: float, y: float, z: float, text: str, color, Marker):
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = ns
    m.id = marker_id
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x = float(x)
    m.pose.position.y = float(y)
    m.pose.position.z = float(z)
    m.pose.orientation.w = 1.0
    m.scale.z = 0.25
    m.color = color
    m.text = text
    return m


def _poly_centroid(poly: List[Dict[str, Any]]) -> Tuple[float, float]:
    if not poly:
        return 0.0, 0.0
    x = sum(float(p.get('x', 0.0)) for p in poly) / len(poly)
    y = sum(float(p.get('y', 0.0)) for p in poly) / len(poly)
    return x, y


def _load_pcd_xyz(full_pcd: Path) -> Optional[List[Tuple[float, float, float]]]:
    if not full_pcd.exists():
        return None
    try:
        import open3d as o3d  # type: ignore
        import numpy as np  # type: ignore

        pcd = o3d.io.read_point_cloud(str(full_pcd))
        arr = np.asarray(pcd.points)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return None
        # 限流，避免 RViz 卡顿
        if arr.shape[0] > 120000:
            idx = np.random.choice(arr.shape[0], size=120000, replace=False)
            arr = arr[idx]
        return [(float(x), float(y), float(z)) for x, y, z in arr[:, :3]]
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description='ROS1 RViz 可视化 AgenticRAG map')
    parser.add_argument('--map_json', type=str, required=True)
    parser.add_argument('--frame_id', type=str, default='map')
    parser.add_argument('--full_pcd', type=str, default='')
    parser.add_argument('--rate', type=float, default=1.0)
    args = parser.parse_args()

    rospy, Header, ColorRGBA, Point, Marker, MarkerArray, PointCloud2, point_cloud2 = _try_import_ros()

    floors = _load_json(Path(args.map_json))

    rospy.init_node('agentrag_map_visualizer', anonymous=True)
    pub_rooms = rospy.Publisher('/agentrag/rooms', MarkerArray, queue_size=1, latch=True)
    pub_objs = rospy.Publisher('/agentrag/objects', MarkerArray, queue_size=1, latch=True)
    pub_pcd = rospy.Publisher('/agentrag/full_pcd', PointCloud2, queue_size=1, latch=True)

    rooms_arr = MarkerArray()
    objs_arr = MarkerArray()

    c_room = ColorRGBA(0.2, 0.8, 1.0, 0.95)
    c_obj = ColorRGBA(1.0, 0.35, 0.35, 0.95)
    c_txt = ColorRGBA(1.0, 1.0, 0.9, 0.95)

    rid = 0
    oid = 0
    tid = 100000

    for floor in floors:
        zc = _room_z_center(floor)
        for room in floor.get('rooms', []):
            rpoly = room.get('Region') or room.get('region') or []
            rpts = _poly_to_points(rpoly, zc, Point)
            if rpts:
                rooms_arr.markers.append(_marker_line(rid, 'room_boundary', args.frame_id, rpts, c_room, 0.06, Marker))
                rid += 1
                cx, cy = _poly_centroid(rpoly)
                rooms_arr.markers.append(_marker_text(tid, 'room_label', args.frame_id, cx, cy, zc + 0.25, room.get('room_id', 'room'), c_txt, Marker))
                tid += 1

            for obj in room.get('objects', []):
                opoly = obj.get('region') or obj.get('Region') or []
                opts = _poly_to_points(opoly, zc + 0.02, Point)
                if opts:
                    objs_arr.markers.append(_marker_line(oid, 'obj_region', args.frame_id, opts, c_obj, 0.04, Marker))
                    oid += 1
                    cx, cy = _poly_centroid(opoly)
                    label = f"{obj.get('obj_id', 'obj')}"
                    objs_arr.markers.append(_marker_text(tid, 'obj_label', args.frame_id, cx, cy, zc + 0.12, label, c_txt, Marker))
                    tid += 1

    points_xyz = _load_pcd_xyz(Path(args.full_pcd)) if args.full_pcd else None

    rate = rospy.Rate(max(0.2, args.rate))
    while not rospy.is_shutdown():
        now = rospy.Time.now()
        for m in rooms_arr.markers:
            m.header.stamp = now
        for m in objs_arr.markers:
            m.header.stamp = now
        pub_rooms.publish(rooms_arr)
        pub_objs.publish(objs_arr)

        if points_xyz is not None:
            header = Header()
            header.stamp = now
            header.frame_id = args.frame_id
            cloud = point_cloud2.create_cloud_xyz32(header, points_xyz)
            pub_pcd.publish(cloud)

        rate.sleep()


if __name__ == '__main__':
    main()
