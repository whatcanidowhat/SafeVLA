"""Manual NVIDIA integration only; formal control/CI approvals remain unclaimed."""
import os

if os.environ.get("UNMASK_NVIDIA_MANUAL") == "ACKNOWLEDGED_MANUAL_NVIDIA_RUN":
    from diagnostics.actor_memory_unmask import online_gate
    from manual_gate import authorize

    online_gate.authorize = authorize
