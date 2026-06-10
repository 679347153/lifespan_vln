#!/usr/bin/env python3
"""Legacy entry point removed from the RAANav main line.

The old Habitat exploration script used GroundingDINO + MobileSAM as its
visual frontend. That path was retired after adopting the DAAAM-style object
node frontend.

Use:
  scripts/raanav_frontend/export_habitat_sequence.py
  scripts/raanav_frontend/run_object_node_frontend.py
  scripts/daaam/daaam_to_raanav_map.py

Legacy copy:
  scripts/z_legacy/legacy_gdino_frontend/deep_exploration_gdino.py
"""

raise SystemExit(
    "scripts/deep_exploration.py is retired. "
    "Use scripts/raanav_frontend/export_habitat_sequence.py + "
    "scripts/raanav_frontend/run_object_node_frontend.py. "
    "Legacy copy: scripts/z_legacy/legacy_gdino_frontend/deep_exploration_gdino.py"
)
