import os
import hashlib
from pathlib import Path

from diagnostics.actor_logits import evaluate
from diagnostics.actor_logits import recovery003 as recovery
from diagnostics.actor_logits.storage import read_json, require

output = Path("/nvme2/user/qyy/SafeVLA_manual_on200_20261015/ON200")

def manual_before_model(args, tasks, manifest_bytes):
    require(args.actor_reset == "on", "Only ON200 permitted")
    require(args.output.resolve() == output, "Unexpected output")
    require(not output.exists(), "Output already exists")

    proof = recovery.verify_off()
    old = read_json(recovery.OFF / "RUN_MANIFEST.json")

    require(tasks == old["tasks"], "TaskSpecs mismatch")
    require(
        hashlib.sha256(manifest_bytes).hexdigest() == recovery.MANIFEST_SHA,
        "Task manifest mismatch"
    )

    _, identity = evaluate.code_identity()
    recovery.science_parity(
        old["provenance"]["source_sha256"],
        identity["source_sha256"]
    )

    print("OFF200 VERIFIED; MANUAL ON200 STARTING", flush=True)

    return {
        "execution_mode": "MANUAL_OUTSIDE_RESEARCH_LOOP",
        "claim_commit": None,
        "approval_commit": None,
        "OFF_reference_result_commit": recovery.OLD_RESULT,
        "OFF_verified": proof["status"],
        "gpu_logger_equivalence": "NOT_RUN_PI_WAIVED"
    }

recovery.before_model = manual_before_model

evaluate.main([
    "--actor-reset", "on",
    "--output", str(output),
    "--checkpoint",
    "/home/amax/public/datasets/qyy/checkpoints/safe_objnav.pt",
    "--physical-gpu", os.environ["MANUAL_GPU"],
    "--num-workers", "1",
    "--seed", "123",
    "--shuffle",
    "--greedy", "false",
    "--test-augmentation",
    "--episodes", "200",
    "--execute-authorized",
    "--pi-waive-gpu-equivalence"
])
