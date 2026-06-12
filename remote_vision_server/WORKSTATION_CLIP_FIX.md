# Workstation CLIP Loading Fix

Update: the remote vision server can now compute CLIP image embeddings on the server side. For the preferred remote workflow, see `remote_vision_server/REMOTE_CLIP_EMBEDDING.md` and run the navigation script with `--remote-clip --disable-clip`.

In the older workflow, the remote vision server only ran GroundingDINO + MobileSAM, and the legacy navigation script computed CLIP embeddings on the workstation after receiving detections.

If the workstation is offline and does not have `openai/clip-vit-base-patch32` cached, you may see:

```text
Cannot find the requested files in the disk cache and outgoing traffic has been disabled
```

## Fast Smoke Test: Disable Local CLIP

Use this first to verify the remote detection + navigation loop:

```bash
cd ~/Downloads/lifespan_vln/RAANav/AgenticRAG

python scripts/z_legacy/legacy_gdino_frontend/sim_nav_loop_gdino.py \
  --scene-dir ../../hm3d/minival/00808-y9hTuugGdiq \
  --dataset-config ../../hm3d/hm3d_val_scene_dataset_config.json \
  --target chair \
  --output-dir RAG_Graph/remote_vision_nav_smoke \
  --max-steps 5 \
  --n-views 4 \
  --perception-backend remote \
  --remote-vision-base-url http://127.0.0.1:50220 \
  --disable-clip
```

This keeps detections and depth back-projection enabled, but writes empty `clip_embedding` fields.

## Use A Local CLIP Directory

If you have a local Hugging Face CLIP directory, pass it explicitly:

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
  --clip-model-path /path/to/clip-vit-large-patch14
```

The server currently has:

```text
/home/ma-user/work/zhangWei/mtu3d/data/trans/clip-vit-large-patch14
```

That directory must also exist on the workstation if the workstation is doing CLIP embedding locally. You can copy it from the server or mount it.

## Environment Variables

Equivalent environment variables:

```bash
export RAANAV_DISABLE_CLIP=1
export RAANAV_CLIP_MODEL_PATH=/path/to/clip-vit-large-patch14
export RAANAV_CLIP_MODEL=openai/clip-vit-base-patch32
export RAANAV_CLIP_LOCAL_FILES_ONLY=1
```

`RAANAV_CLIP_LOCAL_FILES_ONLY=1` is the default when `HF_HUB_OFFLINE=1` or `TRANSFORMERS_OFFLINE=1`.

## Allow Online Download

If the workstation can access Hugging Face or `hf-mirror.com`, you can allow online loading:

```bash
python scripts/z_legacy/legacy_gdino_frontend/sim_nav_loop_gdino.py \
  ... \
  --clip-online
```

For reproducible offline runs, prefer `--disable-clip` for smoke tests or `--clip-model-path` for full runs.
