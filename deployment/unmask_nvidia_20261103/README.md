# Actor memory unmask: manual NVIDIA integration

Run these commands inside the allocated Sothis GPU container as `zhangyh`.

```bash
# Default mode: environment, source/dependencies, resource paths and GPU checks.
bash /public/home/zhangyh/QYY/unmask_nvidia_20261103/run_pair.sh --preflight --gpu 0

# Explicitly start the full new ON200 -> OFF200 pair.
bash /public/home/zhangyh/QYY/unmask_nvidia_20261103/run_pair.sh --run --gpu 0
```

`--gpu` selects a container-local nvidia-smi index. Only its UUID is exposed to
CUDA. The corresponding Vulkan device is obtained from real vulkaninfo output;
AI2-THOR's logical-device mapping is stored in a private container/GPU HOME.
The unused GPU in a multi-GPU container is not exposed to the policy process.

The actual allocated host GPU index, node, job ID and container ID are read from
the matching Sothis instance record and recorded with the GPU UUID. No Slurm
allocation environment variables are fabricated. The selected GPU must be idle.

Output defaults to a new child directory under:

```
/public/home/zhangyh/QYY/runs/EXP-ACTOR-MEMORY-UNMASK-NVIDIA/
```

An optional `--output /absolute/NEW_DIRECTORY` requires an existing parent and
an absent output directory. Existing results are never resumed or overwritten.
The original pair runner stops if the first arm fails; it does not retry.
Both arms inherit one common 11h50m wall-time deadline and GPU/environment.

The frozen experiment checkout remains at
`SafeVLA_actor_memory_unmask_002_20261029`, commit
`2689081ec324c67a35f2a4692cfa009826fcf9d4`. Its source and the tracked control design
are not edited. The frozen `safevla_unmask500_20261028` Python and dependency
versions remain in use. The original pair runner/evaluator/validators are used:

- MASK_ON200 followed by a new MASK_OFF200 process;
- the same checkpoint and seeded task order, worker=1, seed=123;
- actor reset OFF in both; only the root Actor cross-Episode mask changes;
- max_steps=500, cache capacity=500, episode horizon=600;
- shuffle/test augmentation enabled, greedy=false;
- original official success/cost and pair-result validators.

The external adapter is enabled only by this launcher's explicit
`UNMASK_NVIDIA_MANUAL=ACKNOWLEDGED_MANUAL_NVIDIA_RUN`. It preserves the old manual
gate's source, dependency, idle-GPU and deadline checks while
declaring a replacement renderer and validating the real Sothis container
allocation. `renderer_manifest.json` pins the NVIDIA/EGL/Unity binaries and
integration files. NVIDIA libraries require host driver **535.104.05**.

This is a **manual/offline NVIDIA run**, not a formal research-control approval.
Records explicitly contain `formal_control_authorized=false`, no Claim/approval
commit, and no successful GitHub CI assertion. Historical Mesa output is not
reused as either arm. Existing formal control/CI state is left intact.

`--preflight` constructs no policy/model, runs no simulator, and starts no Episode.
It does query the real GPU/Vulkan devices and check PyTorch CUDA visibility.
At the user's request, full selected-task asset hashing is disabled in the parent
and both arms. Only the fixed resource directory paths and the houses/task input
files' existence are checked; the 15,237-file inventory is not opened or scanned.
Source/version, model/task identity, renderer, GPU and result-validation checks
remain in place. Run evidence records `asset_validation.mode=PATHS_ONLY` and
`full_asset_hash_scan=false`, without claiming that resources were content-verified.
Its latest report is `last_preflight.json`; diagnostic console output from the
integration validation is `preflight.console.log`.

## Output-scanner race fix

The stopped run `20261103T145326Z-266` completed two ON episodes before the frozen
parent's `Path.rglob()` encountered an episode directory atomically renamed from
`.partial` to `.complete`. The original directory and its STOPPED status remain
preserved; OFF did not start. It is not a complete paired result and is not resumed.

`pair_entry.py` loads the original pair module and replaces only the nested
`used_bytes()` body in memory, after verifying its original AST. The replacement
`output_scan.py` tolerates descendants that disappear during concurrent writes,
counts each file identity once, and still raises missing-root, permission and I/O
errors. Scans are approximate snapshots repeated by the original 15-second loop.
The original 90-GiB output limit, 10-GiB free-space stop, 100-GiB initial-space
requirement, command arguments, group ordering and validation logic remain intact.

The original experiment checkout is unchanged. The external runtime patch and
its source file digests are declared in `renderer_manifest.json` and propagated
as `orchestration_patch` in the new run's `PAIR_LAUNCH.json` evidence. Previous
integration files and their manifest were backed up under `backups/` before
deployment. `test_output_scan.py` exercises actual directory renames, shard
deletion, failure propagation and a concurrent 300-episode-directory writer.

To start a fresh pair after this fix, use the same `run_pair.sh --run --gpu 0`
command. A new output directory is created. Integration validation starts no
policy Episode and does not automatically retry the stopped experiment.
