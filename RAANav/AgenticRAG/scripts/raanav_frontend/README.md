# RAANav Object-Node Frontend

This is the main perception frontend after retiring the old GroundingDINO + MobileSAM Habitat path.

## Architecture

```text
RGB-D-pose source
  -> FastSAM segmentation
  -> BotSort tracking
  -> representative observation assignment per track
  -> CLIP object-node embedding + top-k labels
  -> RAANav semantic map / GMM
```

The core path intentionally does **not** depend on DAM/VLM descriptions. RAANav learns from object nodes: stable track IDs, positions, CLIP embeddings, top-k semantic candidates, observation counts, timestamps, crops, and later co-occurrence statistics.

## Why This Replaces the Old Simulation Frontend

The retired simulation scripts used:

```text
GroundingDINO + MobileSAM -> single-frame detections -> CLIP embedding/query -> GMM
```

That path was useful during early exploration, but it is no longer the project direction. It is now stored under:

```text
scripts/z_legacy/legacy_gdino_frontend/
```

The main line should use this frontend for Habitat exports, rosbag exports, and online ROS sources.

## Scripts

### 1. Export Habitat RGB-D-pose Sequence

Run in `agentrag` environment because it owns Habitat:

```bash
cd /home/adminer/agentRAG/AgenticRAG
/home/adminer/miniconda3/envs/agentrag/bin/python scripts/raanav_frontend/export_habitat_sequence.py \
  --scene-dir /home/adminer/agentRAG/experiment_data/hm3d/val/00814-p53SfW6mjZe \
  --out-dir /tmp/raanav_habitat_seq \
  --num-positions 20 \
  --n-views 4
```

Output format is DAAAM `ImageSequenceDataset` compatible:

```text
rgb/*.png
depth/*.png
pose/poses.txt
camera_info.json
metadata.json
```


### Target-aware Habitat Validation

Use this only for validation/debugging. It renders camera poses around prior-map or manually curated target candidate positions; those labels are sampling hints, not ground truth.

```bash
/home/adminer/miniconda3/envs/agentrag/bin/python scripts/raanav_frontend/export_habitat_sequence.py \
  --scene-dir /home/adminer/agentRAG/experiment_data/hm3d/val/00814-p53SfW6mjZe \
  --out-dir /tmp/raanav_target_views \
  --trajectory-mode target_views \
  --target-points-json /tmp/target_points.json \
  --target-camera-radius 1.8 \
  --target-azimuths-deg "0,60,120,180,240,300" \
  --target-heading-offsets-deg=-20,0,20
```

The output is the same ImageSequenceDataset format as the continuous exporter. Always inspect crops/frames before treating target labels as success evidence.

### 2. Run Object-Node Frontend

Run in `daaam_p0` environment because it owns FastSAM/BotSort/open_clip:

```bash
cd /home/adminer/agentRAG/AgenticRAG
/home/adminer/miniconda3/envs/daaam_p0/bin/python scripts/raanav_frontend/run_object_node_frontend.py \
  --data-path /tmp/raanav_habitat_seq \
  --output-dir /tmp/raanav_object_nodes \
  --device cuda \
  --min-obs-per-track 3 \
  --representatives-per-track 3 \
  --representative-crop-mode context \
  --save-vis
```

For CPU smoke tests, use `--device cpu --without-reid --max-frames 1`.

Notes:

- `smoke test` means a very small run that only checks whether the whole pipeline can start, exchange data, and write outputs. It is not a quality proof.
- Empty detections/tracks are normal in real sequences. The entry script guards this case so a frame with no reliable mask does not crash the whole run.
- ReID means re-identification: when an object disappears and appears again, the tracker tries to decide whether it is the same object. If ReID is disabled or broken, short fragmented tracks are expected, so the frontend also builds spatial object nodes from 3D observations.
- Large structural things such as railings can remain useful map nodes. The default mask filter removes mostly huge edge-touching planes and poor masks, not all structural objects.


Labeling notes:

- `--representative-crop-mode context` is the default because CLIP labels are usually more stable when the crop keeps local RGB context around the mask. Use `masked` only when you explicitly want black-background masked crops for ablation/debug.
- CLIP labels are zero-shot hints, not ground truth. Downstream RAANav conversion should keep low-confidence labels as debug metadata instead of treating them as confirmed target labels.

Output:

```text
summary.json          # track-level object nodes, consumable by daaam_to_raanav_map.py
all_frames.json       # per-frame tracks and timing
labels/*.json         # per-frame JSON
representatives/      # selected masked crops per track
vis/*.png             # optional track visualization
```

### 3. Convert To RAANav Map/GMM

```bash
cd /home/adminer/agentRAG/AgenticRAG
/home/adminer/miniconda3/envs/agentrag/bin/python scripts/daaam/daaam_to_raanav_map.py \
  --daaam-summary /tmp/raanav_object_nodes/summary.json \
  --output-dir /tmp/raanav_map_from_object_nodes \
  --sigma 1.0 \
  --min-label-confidence 0.2 \
  --label-gate-mode soft
```


`--min-label-confidence` marks zero-shot labels as `confirmed` or `low_confidence`. In the default soft gate, weak labels remain queryable but carry low `cfd`/status metadata so RAANav scoring, stability, and negative feedback can suppress unreliable targets. Use `--label-gate-mode hard` only for conservative ablations that should expose weak labels as `unknown`.

## Current Smoke Test

Completed on 2026-05-21:

```bash
/home/adminer/miniconda3/envs/agentrag/bin/python scripts/raanav_frontend/export_habitat_sequence.py \
  --scene-dir /home/adminer/agentRAG/experiment_data/hm3d/val/00814-p53SfW6mjZe \
  --out-dir /tmp/raanav_habitat_smoke \
  --num-positions 1 --n-views 1

/home/adminer/miniconda3/envs/daaam_p0/bin/python scripts/raanav_frontend/run_object_node_frontend.py \
  --data-path /tmp/raanav_habitat_smoke \
  --output-dir /tmp/raanav_frontend_smoke \
  --device cpu --max-frames 1 --without-reid --save-vis
```

Result: 1 frame exported, FastSAM/BotSort ran, 19 tracks and 17 3D positions produced. `labeled=0` is expected for a 1-frame smoke test because `min_obs_per_track=3`.

## Next Validation

Run a real validation sequence with at least 20 positions x 4 views, then inspect:

- track count and position stability;
- representative crops per track;
- CLIP top-k label quality;
- conversion through `daaam_to_raanav_map.py`;
- whether single-frame false positives are filtered by `min_obs_per_track` and representative assignment.

## Offline Model Loading And Proxy

The workstation already has model weights in local caches, but some libraries still issue HuggingFace metadata `HEAD` requests to check refs or file freshness. That can stall experiments when the network is poor. The CLIP labeler therefore defaults to `HF_HUB_OFFLINE=1`.

If a model is missing from cache, disable offline mode only for that setup run and use the machine proxy explicitly, for example:

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HF_HUB_OFFLINE=0
```

Verified on 2026-05-21: command-line proxy is listening on `127.0.0.1:7890`; direct HuggingFace access timed out, proxy access succeeded. After the model downloads successfully, return to offline mode for repeatable experiments.

## Current Engineering Choices

- Core map units are object nodes, not DAM/VLM text descriptions.
- Track-level summaries are kept for debugging, but RAANav map conversion should prefer spatial `object_nodes` when present.
- Spatial object-node clustering is a robustness layer over tracking: nearby high-quality 3D observations can merge into one node even if BotSort created several short tracks.
- We do not introduce dense world models or per-point CLIP maps in the current main path because they increase memory/compute and are not necessary for the immediate object-node map goal.
