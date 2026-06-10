"""Retired GroundingDINO + MobileSAM frontend.

The implementation was moved to:
  scripts/z_legacy/legacy_gdino_frontend/perception_gdino_mobilesam.py

The RAANav main line now uses a DAAAM-style object-node frontend based on
segmentation, tracking, assignment, and CLIP object-node embeddings.
"""

raise ImportError(
    "semantic_map_Create.perception was retired. "
    "Use scripts/raanav_frontend/run_object_node_frontend.py. "
    "Legacy implementation: scripts/z_legacy/legacy_gdino_frontend/perception_gdino_mobilesam.py"
)
