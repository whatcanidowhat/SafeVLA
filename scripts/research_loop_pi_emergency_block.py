#!/usr/bin/env python3
"""PI emergency BLOCKED publication for an archived real P2 ndarray failure. No GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from validate_research_loop import STATE, View, audit_tree, validate_history, validate_transition

CYCLE = "actor-reset-off200-on200-20261012"
EXPERIMENT = "EXP-ACTOR-RESET-OFF200-ON200-001"
SOURCE_SHA = "33ffa28000522e3715a0920bf3679c27e79b423e"


def check(ok, message):
    if not ok:
        raise ValueError(message)


def git(root, *args):
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                       timeout=90, env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    check(p.returncode == 0, "Git operation failed: " + args[0])
    return p.stdout.strip()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    contents = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n" \
        if isinstance(value, (dict, list)) else value
    check(len(contents.encode("utf8")) < 1024 * 1024, "Git artifact exceeds limit")
    Path(path).write_text(contents, encoding="utf8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--control-root", type=Path,
                   default=Path("/nvme2/user/qyy/SafeVLA_loop_control"))
    a = p.parse_args()
    root, archive = a.control_root.resolve(), a.archive.resolve()
    source = archive / "original_results"
    off = source / "OFF200"
    error_file = off / "ERROR.json"
    check(error_file.is_file() and (off / "RUN_MANIFEST.json").is_file()
          and (off / "TASK_MANIFEST.csv").is_file()
          and (off / "episodes" / "0000.partial").is_dir()
          and (archive / "source_commit.txt").is_file()
          and (archive / "FILE_SHA256.txt").is_file(), "Incomplete failure archive")
    check((archive / "source_commit.txt").read_text().strip() == SOURCE_SHA,
          "Source SHA differs from frozen P2")
    check(not (source / "ON200").exists(), "ON200 exists: not the reviewed case")
    e, r = read(error_file), read(off / "RUN_MANIFEST.json")
    check(e.get("location") == "end_episode"
          and e.get("error") == "Unsupported JSON type: ndarray"
          and e.get("episode_order") == 0 and e.get("global_step") == 27,
          "Error evidence does not match the actual reviewed first-episode failure")
    check(r.get("status") == "INCOMPLETE" and r.get("episodes_completed", 0) == 0,
          "Original run reports completed episodes; review manually")
    check(not (off / "episodes" / "0000.complete").exists(), "Episode 0 complete")
    ps = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True, timeout=30)
    for line in ps.splitlines():
        bits = line.strip().split(None, 1)
        if bits and bits[0].isdigit() and int(bits[0]) != os.getpid():
            cmd = bits[1] if len(bits) == 2 else ""
            if ("diagnostics.actor_logits.evaluate" in cmd
                or ("bash " in cmd and "diagnostics/actor_logits/run_p2_full400.sh" in cmd)):
                raise ValueError("Live P2 evaluator or launcher may exist; STOP")
    check(git(root, "branch", "--show-current") == "research-loop", "Wrong control branch")
    check(not git(root, "status", "--porcelain"), "Dirty control checkout")
    old_sha = git(root, "rev-parse", "HEAD")
    remote = git(root, "ls-remote", "--heads", "origin", "refs/heads/research-loop").split()
    check(remote and remote[0] == old_sha, "Control checkout is not remote HEAD")
    old = read(root / STATE)
    check(old["status"] == "CODEX_RUNNING" and old["next_actor"] == "CODEX"
          and old["cycle_id"] == CYCLE and old["experiment_id"] == EXPERIMENT
          and old["claim_id"] and old["instruction_commit"], "Not active claimed P2")
    base = root / "research" / "handoffs" / CYCLE
    check(not base.exists(), "Existing handoff: refusing overwrite")

    # cp -a faithfully preserves symlinks. Record link texts as evidence, but
    # never follow links into checkpoint caches or folders outside the archive.
    archived = []
    symlinks = []
    for f in sorted(source.rglob("*")):
        relative = f.relative_to(source).as_posix()
        if f.is_symlink():
            text = os.readlink(f)
            symlinks.append({"path": relative,
                             "link_text_sha256": hashlib.sha256(os.fsencode(text)).hexdigest(),
                             "link_text_length": len(os.fsencode(text)),
                             "followed": False})
        elif f.is_file():
            archived.append((relative, f.stat().st_size, digest(f)))
        else:
            check(f.is_dir(), "Unsupported filesystem node in archive")
    check(archived, "No evidence files")
    for needed in (source, off, off / "episodes", off / "episodes" / "0000.partial",
                   error_file, off / "RUN_MANIFEST.json", off / "TASK_MANIFEST.csv"):
        check(not needed.is_symlink(), "Required failure evidence is a symlink")
    start_count = len(list((off / "episodes").glob("*.partial")))
    start_count += len(list((off / "episodes").glob("*.complete")))
    check(start_count == 1, "Unexpected number of started episodes")
    now = datetime.now(timezone.utc).isoformat()
    original_error_sha = digest(error_file)
    archival_index_sha = digest(archive / "FILE_SHA256.txt")
    actual_gpu = source / "P2_LAUNCH_CONTEXT.json"
    gpu_report = {"status": "SOURCE_UNAVAILABLE", "note": "No original launch metadata"}
    if actual_gpu.is_file():
        gpu_report = {"status": "ORIGINAL_ARCHIVED", "sha256": digest(actual_gpu),
                      "record": read(actual_gpu)}
    manifest = {
        "status": "BLOCKED", "handoff_actor": "PI",
        "closure_reason": "CODEX_UNAVAILABLE_AND_RUNTIME_ERROR",
        "cycle_id": CYCLE, "experiment_id": EXPERIMENT,
        "claim_id": old["claim_id"], "original_claim_id": old["claim_id"],
        "instruction_commit": old["instruction_commit"], "source_sha": SOURCE_SHA,
        "gpu_count": 1, "episodes_started": 1, "episodes_completed": 0,
        "failure_location": "end_episode", "failure_type": "TypeError",
        "failure_message": e["error"], "global_step": 27,
        "on200_started": False, "gpu_logger_equivalence": "NOT_RUN_PI_WAIVED",
        "command": ["bash", "diagnostics/actor_logits/run_p2_full400.sh"],
        "archive_path": str(archive), "archive_files": len(archived),
        "archive_symlink_count": len(symlinks),
        "archive_file_list_sha256": archival_index_sha,
        "error_json_sha256": original_error_sha,
        "source_run_manifest_sha256": digest(off / "RUN_MANIFEST.json"),
        "recorded_at_utc": now, "actual_SR_cost_results": None,
    }
    frozen = read(root / "research/NEXT_EXPERIMENT.json")
    check(frozen["cycle_id"] == CYCLE and frozen["status"] == "APPROVED", "Frozen design changed")
    base.mkdir(parents=True)
    files = {
        "RUN_MANIFEST.json": manifest,
        "RESULT_SUMMARY.md": "# BLOCKED: P2 Logger ndarray metric serialization\n\n"
            "OFF200 Episode 0 started and produced 27 decisions but did not finish logging. "
            "TypeError Unsupported JSON type: ndarray at end_episode. No validated Episode. "
            "ON200 not started. SR and official Cost differences unavailable.\n",
        "REVIEW_NOTES.md": "# PI emergency BLOCKED handoff\n\n"
            "Codex 5-hour quota exhausted. PI publishes this terminal status honestly, "
            "preserving CODEX claim. The archive is on server, and no retry is authorized. "
            "Review this BLOCKED entry then acknowledge before new cycle.\n",
        "FROZEN_PROTOCOL.md": "# Frozen failed protocol\n\n"
            "OFF200 vs root Actor-only reset ON200, same 200 TaskSpecs, one worker, "
            "seed123, greedy false, original metrics and full logits, no Critic reset.\n",
        "LOGGER_WAIVER.md": "# Original PI waiver\n\n"
            "GPU_EQUIVALENCE=NOT_RUN_PI_WAIVED. This is NOT a PASS.\n",
        "OFF_ON_RESULTS.md": "# No paired results\n\n"
            "OFF200 incomplete; 0 valid Episodes; ON200 NOT_STARTED. "
            "Delta SR, Delta Cost and logits comparison NOT_AVAILABLE.\n",
        "VALIDATION_SUMMARY.md": "# Failure evidence\n\n"
            "Episode 0 partial, ERROR.json and INCOMPLETE source run manifest checked. "
            "Full 200+200 and sample-level integrity validation NOT_RUN.\n",
        "GPU_QUEUE_REPORT.json": gpu_report,
        "EVIDENCE_INDEX.json": {
            "archive_path": str(archive), "error_sha256": original_error_sha,
            "archive_file_list_sha256": archival_index_sha,
            "archived_files": len(archived), "archived_bytes": sum(v[1] for v in archived),
            "first_100_files": [{"path": n, "size": v, "sha256": h}
                                for n, v, h in sorted(archived)[:100]],
            "omitted_count": max(0, len(archived) - 100),
            "symlink_policy": "Link text recorded without dereference; target content not inspected",
            "archived_symlink_count": len(symlinks),
            "first_100_symlinks": symlinks[:100],
            "omitted_symlinks": max(0, len(symlinks) - 100),
        },
    }
    for name, val in files.items():
        write(base / name, val)
    shutil.copyfile(off / "TASK_MANIFEST.csv", base / "TASK_MANIFEST.csv")
    shutil.copyfile(error_file, base / "ORIGINAL_ERROR.json")
    nxt = dict(old)
    nxt.update(status="BLOCKED", next_actor="PI",
               updated_by="PI", state_version=old["state_version"] + 1,
               updated_at_utc=now)
    write(root / STATE, nxt)
    artifacts = []
    for f in sorted(base.iterdir()):
        rel = f.relative_to(root).as_posix()
        artifacts.append({"name": f.name, "server_path": str(f),
                          "git_path": rel, "sha256": digest(f),
                          "size": f.stat().st_size, "required_for_PI_review": True})
    write(base / "ARTIFACT_INDEX.json", {"schema_version": "1.0", "artifacts": artifacts})
    check(set(old["required_outputs"]).issubset({f.relative_to(root).as_posix()
                                                 for f in base.iterdir()}),
          "Missing required handoff files")
    validate_transition(View(root, old_sha), View(root), old_sha)
    audit_tree(View(root))
    git(root, "add", "--", STATE, str(base.relative_to(root)))
    git(root, "commit", "-m", "PI emergency BLOCKED: original P2 ndarray logger failure")
    this_sha = git(root, "rev-parse", "HEAD")
    validate_history(root)
    remote = git(root, "ls-remote", "--heads", "origin", "refs/heads/research-loop").split()
    check(remote and remote[0] == old_sha, "Remote advanced; STOP without retry")
    git(root, "push", "--atomic", "origin", "HEAD:refs/heads/research-loop")
    remote = git(root, "ls-remote", "--heads", "origin", "refs/heads/research-loop").split()
    check(remote and remote[0] == this_sha, "Push status uncertain; STOP without retry")
    print(json.dumps({"status": "PI_BLOCKED_PUBLISHED", "terminal_commit": this_sha,
                      "claim_id": old["claim_id"],
                      "next": "Await terminal CI SUCCESS, then PI acknowledgement and new cycle"},
                     indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("PI_BLOCKED_PUBLISH_FAILED: " + type(exc).__name__ + ": "
              + str(exc)[:200] + "; STOP (no experiment)", file=sys.stderr)
        sys.exit(1)
