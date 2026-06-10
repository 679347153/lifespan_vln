#!/usr/bin/env python3
"""Legacy entry point removed from the RAANav main line.

The old navigation loop used GroundingDINO + MobileSAM for online object
perception. It is now kept only for historical comparison.

Current main line:
  DAAAM-style object-node frontend -> RAANav map/GMM -> new ROS/navigation bridge.

Legacy copy:
  scripts/z_legacy/legacy_gdino_frontend/sim_nav_loop_gdino.py
"""

raise SystemExit(
    "scripts/sim_nav_loop.py is retired from the main line. "
    "Use the DAAAM-style frontend under scripts/raanav_frontend/. "
    "Legacy copy: scripts/z_legacy/legacy_gdino_frontend/sim_nav_loop_gdino.py"
)
