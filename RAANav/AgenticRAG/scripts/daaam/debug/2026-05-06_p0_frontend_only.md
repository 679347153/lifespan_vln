# P0 Debug Record — DAAAM Frontend-Only Pipeline

## Date: 2026-05-06

## Step 1: Environment Setup

### Conda Environment
- **Name:** daaam_p0
- **Python:** 3.10
- **Key packages:** torch 2.5.0, torchvision 0.20.0, ultralytics 8.4.47, boxmot 11.0.8, opencv-python 4.13.0, numpy 1.26.4

### torch iJIT Workaround
- **Problem:** torch from conda env references iJIT symbols (Intel VTune) that don't exist on this system
- **Solution:** Created stub shared library at `/tmp/libvitjot_stub.so` with 3 symbols:
  - `iJIT_NotifyEvent`
  - `iJIT_IsProfilingActive`
  - `iJIT_GetNewMethodID`
- **Usage:** `LD_PRELOAD=/tmp/libvitjot_stub.so` before any python command

## Step 2: Model Checkpoint
- **Model:** FastSAM-s.pt (ultralytics, downloaded via ultralytics API)
- **Location:** `/home/adminer/agentRAG/参考开源代码/DAAAM/checkpoints/fastsam/FastSAM-s.pt`
- **Device:** CPU (CUDA unavailable on server)
- **Inference time:** ~150ms per 640×480 frame on CPU

## Step 3: Dataset Conversion
- **Source:** `/home/adminer/ClioDocker/clio_dataset/apartment/apartment.bag` (1.7 GB)
- **Script:** `AgenticRAG/scripts/ros1/bag_to_image_sequence.py` (run with system Python for ROS1 rosbag)
- **Output:** `/tmp/daaam_apartment/` (1168 RGB + 1168 depth frames)
- **Topics:** `/dominic/forward/color/image_raw` (RGB), `/dominic/forward/depth/image_rect_raw` (depth)
- **No camera_info** available — intrinsics estimated as fx=fy=640, cx=320, cy=240 (synthetic)
- **No pose** available — 3D positions are in camera frame only

## Step 4: Frontend-Only Pipeline
- **Script:** `AgenticRAG/scripts/daaam/run_frontend_only_pipeline.py`
- **What it does:**
  1. Load ImageSequenceDataset
  2. Initialize SegmentationService (FastSAM) + TrackingService (BotSort without ReID)
  3. For each frame: segment → track → compute 3D positions → save JSON
  4. Save per-frame visualizations every N frames

## Step 5: Results (30-frame test)
- **FPS:** ~5.2 on CPU
- **Detections/frame:** ~43-70 (FastSAM)
- **Tracks/frame:** ~27-41 (BotSort)
- **Tracks with 3D positions:** 13-29 per frame
- **Unique tracks across 30 frames:** 54
- **Depth values:** Valid (median depth range ~0.5-5m after dividing raw mm by 1000)

## Blockers Resolved
1. ~~torch iJIT symbols~~ → LD_PRELOAD stub
2. ~~torchvision/torch incompatible~~ → torchvision 0.20.0
3. ~~camera_intrinsics attribute error~~ → extract from camera_info dict
4. ~~depth scale wrong~~ → use --depth-scale 1000

## Remaining Issues
1. Camera intrinsics are synthetic (fx=fy=640) — real D435i intrinsics should be used for accurate 3D positions
2. No poses available — all 3D positions are in camera frame, not world frame
3. CPU-only inference is slow (~5 fps) — Jetson Orin with TensorRT would be much faster
4. No ReID for tracking — tracks may switch IDs on occlusion
