"""开放词汇感知模块 — GroundingDINO + MobileSAM.

替代 GT 语义传感器, 实现:
  1. GroundingDINO: 文本引导的目标检测 (bbox + 置信度 + 短语)
  2. MobileSAM: 从 bbox 生成精确分割 mask
  3. 深度反投影: mask 中心 + 深度 → 3D 世界坐标

参考:
  - AutoX-SemMap/robokit/perception.py (GroundingDINO + MobileSAM pipeline)
  - BeliefMapNav (GroundingDINO + CLIP)
  - DovSG (RAM → GroundingDINO → SAM2 → CLIP)

用法:
    detector = OpenVocabDetector()   # 自动加载模型, ~2GB VRAM
    detections = detector.detect(rgb_frame, "chair . table . microwave")
    # detections: [{label, bbox_xyxy, confidence, mask}]

    # 带深度反投影:
    detections_3d = detector.detect_with_depth(
        rgb_frame, depth_frame, "chair . table",
        intrinsics, agent_pos, agent_heading_deg,
    )
    # 增加 pos_3d, pos_2d 字段
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

_PROJ_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = _PROJ_ROOT.parents[1]
for _path in [str(_PROJ_ROOT), str(_REPO_ROOT)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# GroundingDINO 配置路径 (安装在 ~/agentRAG/models/GroundingDINO)
_GDINO_CFG = os.path.expanduser("~/agentRAG/models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
_GDINO_CKPT = os.path.join(str(_PROJ_ROOT), "checkpoints", "groundingdino_swint_ogc.pth")
_MOBILESAM_CKPT = os.path.join(str(_PROJ_ROOT), "checkpoints", "mobile_sam.pt")


class OpenVocabDetector:
    """GroundingDINO + MobileSAM 开放词汇检测器.

    单例式设计: 模型加载一次, 之后复用.
    """

    def __init__(
        self,
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.35,
        gdino_cfg: str = _GDINO_CFG,
        gdino_ckpt: str = _GDINO_CKPT,
        mobilesam_ckpt: str = _MOBILESAM_CKPT,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # 加载 GroundingDINO
        print(f"[Perception] 加载 GroundingDINO ...")
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.models import build_model
        from groundingdino.util.utils import clean_state_dict

        args = SLConfig.fromfile(gdino_cfg)
        args.device = device
        self.gdino_model = build_model(args)
        ckpt = torch.load(gdino_ckpt, map_location="cpu")
        self.gdino_model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
        self.gdino_model.eval()
        self.gdino_model.to(device)
        print(f"[Perception] GroundingDINO 加载完成")

        # 加载 MobileSAM
        print(f"[Perception] 加载 MobileSAM ...")
        from mobile_sam import sam_model_registry, SamPredictor

        sam = sam_model_registry["vit_t"](checkpoint=mobilesam_ckpt)
        sam.to(device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)
        print(f"[Perception] MobileSAM 加载完成")

    def detect(
        self,
        rgb: np.ndarray,
        text_prompt: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """检测 RGB 图像中与文本匹配的物体.

        Args:
            rgb: (H, W, 3) uint8, RGB 格式
            text_prompt: 检测文本, 用 " . " 分隔多类别, 如 "chair . table . microwave"
            box_threshold: bbox 置信度阈值
            text_threshold: 文本匹配阈值

        Returns:
            list of {label, bbox_xyxy, confidence, mask}
            - bbox_xyxy: [x1, y1, x2, y2] 像素坐标
            - mask: (H, W) bool
        """
        box_thr = box_threshold or self.box_threshold
        text_thr = text_threshold or self.text_threshold

        # --- GroundingDINO 检测 ---
        from groundingdino.util.inference import predict
        import groundingdino.datasets.transforms as T

        # 预处理: RGB numpy → PIL → tensor
        from PIL import Image
        pil_image = Image.fromarray(rgb)
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(pil_image, None)

        # 推理
        with torch.no_grad():
            boxes, logits, phrases = predict(
                self.gdino_model,
                image_tensor,
                text_prompt,
                box_thr,
                text_thr,
                device=self.device,
            )

        if len(boxes) == 0:
            return []

        # boxes: (N, 4) cxcywh normalized → xyxy pixel
        H, W = rgb.shape[:2]
        boxes_xyxy = self._cxcywh_to_xyxy(boxes, W, H)

        # --- MobileSAM 分割 ---
        self.sam_predictor.set_image(rgb)

        # 转换 bbox 格式给 SAM
        boxes_torch = torch.tensor(boxes_xyxy, dtype=torch.float32, device=self.device)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_torch, rgb.shape[:2]
        )

        with torch.no_grad():
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
        # masks: (N, 1, H, W) bool
        masks_np = masks.cpu().numpy()[:, 0, :, :]  # (N, H, W)

        # 组装结果
        detections = []
        for i in range(len(boxes)):
            detections.append({
                "label": phrases[i].strip().lower(),
                "bbox_xyxy": boxes_xyxy[i].tolist(),
                "confidence": float(logits[i]),
                "mask": masks_np[i],  # (H, W) bool
            })

        return detections

    def detect_with_depth(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        text_prompt: str,
        intrinsics: Any,
        agent_pos: np.ndarray,
        agent_heading_deg: float,
        max_depth: float = 5.0,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """检测 + 深度反投影获取 3D 世界坐标.

        在 detect() 基础上, 利用 mask 区域的中位深度值 + 相机内参
        反投影到世界坐标系.

        额外返回:
            pos_3d: [wx, wy, wz] 世界坐标
            pos_2d: {"x": wx, "y": wz} 2D 俯视坐标
            depth_median: mask 区域的中位深度 (米)
        """
        detections = self.detect(rgb, text_prompt, box_threshold, text_threshold)
        if not detections:
            return []

        from semantic_map_Create.occupancy_grid import (
            depth_to_local_pointcloud, camera_to_world,
        )

        for det in detections:
            mask = det["mask"]  # (H, W) bool
            bbox = det["bbox_xyxy"]  # [x1, y1, x2, y2]

            # 从 mask 区域取深度中位数 (更鲁棒)
            mask_depths = depth[mask]
            valid_depths = mask_depths[(mask_depths > 0) & (mask_depths < max_depth)]

            if len(valid_depths) == 0:
                # 退回到 bbox 中心
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                cx = min(max(cx, 0), depth.shape[1] - 1)
                cy = min(max(cy, 0), depth.shape[0] - 1)
                d = float(depth[cy, cx])
                if d <= 0 or d >= max_depth:
                    det["pos_3d"] = None
                    det["pos_2d"] = None
                    det["depth_median"] = None
                    continue
                valid_depths = np.array([d])

            median_d = float(np.median(valid_depths))

            # mask 中心像素
            ys, xs = np.where(mask)
            cu = float(np.mean(xs))
            cv = float(np.mean(ys))

            # 相机坐标系: 单点反投影
            cam_x = (cu - intrinsics.cx) * median_d / intrinsics.fx
            cam_y = (cv - intrinsics.cy) * median_d / intrinsics.fy
            cam_z = median_d
            cam_point = np.array([[cam_x, cam_y, cam_z]], dtype=np.float32)

            # 世界坐标系
            world_point = camera_to_world(cam_point, agent_pos, agent_heading_deg)
            wx, wy, wz = float(world_point[0, 0]), float(world_point[0, 1]), float(world_point[0, 2])

            det["pos_3d"] = [wx, wy, wz]
            det["pos_2d"] = {"x": wx, "y": wz}
            det["depth_median"] = median_d

        return detections

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor, W: int, H: int) -> np.ndarray:
        """cxcywh normalized [0,1] → xyxy pixel coords."""
        boxes_np = boxes.cpu().numpy()
        cx, cy, w, h = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3]
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H
        x1 = np.clip(x1, 0, W)
        y1 = np.clip(y1, 0, H)
        x2 = np.clip(x2, 0, W)
        y2 = np.clip(y2, 0, H)
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def build_text_prompt(labels: List[str]) -> str:
        """构建 GroundingDINO 的文本提示.

        格式: "label1 . label2 . label3"
        """
        return " . ".join(labels)


# ---------------------------------------------------------------------------
# 全局单例, 避免重复加载模型
# ---------------------------------------------------------------------------
_singleton: Optional[OpenVocabDetector] = None


def get_detector(**kwargs) -> OpenVocabDetector:
    """获取全局检测器实例 (单例)."""
    global _singleton
    if _singleton is None:
        _singleton = OpenVocabDetector(**kwargs)
    return _singleton
