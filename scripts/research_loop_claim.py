#!/usr/bin/env python3
"""Publish one claim on research-loop. This program never executes experiments."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from validate_research_loop import (Invalid, STATE, View, audit_tree, require, utc,
                                    validate_history, validate_snapshot, validate_transition)


def command(root, *args):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    try:
        p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=90, env=env)
    except subprocess.TimeoutExpired:
        raise Invalid("GIT_TRANSPORT_TIMEOUT; no execution authorized")
    if p.returncode:
        raise Invalid("GIT_OPERATION_FAILED: " + args[0] + "; credential-bearing output suppressed")
    return p.stdout.decode().strip()


def fetch(root):
    # No '+' refspec and no force; rewind is an error.
    command(root, "fetch", "--no-tags", "origin", "refs/heads/research-loop:refs/remotes/origin/research-loop")
    return command(root, "rev-parse", "refs/remotes/origin/research-loop")


def remote_head(root):
    output = command(root, "ls-remote", "--heads", "origin", "refs/heads/research-loop")
    require(len(output.splitlines()) == 1, "REMOTE_BRANCH_MISSING")
    return output.split()[0]


def claim(root):
    root = Path(root).resolve()
    require(command(root, "branch", "--show-current") == "research-loop", "WRONG_CONTROL_BRANCH")
    require(not command(root, "status", "--porcelain"), "DIRTY_CONTROL_WORKTREE")
    require(not (root / "training/online/online_eval.py").exists(), "RUNTIME_WORKTREE_FORBIDDEN")
    lock = Path(command(root, "rev-parse", "--path-format=absolute", "--git-path", "research-loop-claim.lock"))
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise Invalid("LOCAL_CLAIM_LOCKED; do not delete another executor's lock")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()) + "\n")
        approval = fetch(root)
        require(command(root, "rev-parse", "HEAD") == approval, "STALE_LOCAL_HEAD; stop, review and update separately")
        validate_history(root)
        v = View(root)
        audit_tree(v)
        old = validate_snapshot(v)
        require(old["status"] == "APPROVED_FOR_CODEX" and old["next_actor"] == "CODEX", "NOT_APPROVED_FOR_CODEX")
        require(utc(old["authorization"]["expires_at_utc"]) > datetime.now(timezone.utc), "APPROVAL_EXPIRED")
        require(remote_head(root) == approval, "STALE_CLAIM; remote advanced before claim")
        state = dict(old)
        state.update(status="CODEX_RUNNING", next_actor="CODEX",
                     instruction_commit=approval, state_version=old["state_version"] + 1,
                     claim_id=uuid.uuid4().hex, updated_by="CODEX",
                     updated_at_utc=datetime.now(timezone.utc).isoformat())
        (root / STATE).write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        validate_transition(View(root, approval), View(root), approval)
        audit_tree(View(root))
        command(root, "add", "--", STATE)
        command(root, "commit", "-m", "Claim " + state["experiment_id"] + " for cycle " + state["cycle_id"])
        claimed = command(root, "rev-parse", "HEAD")
        validate_history(root)
        try:
            # Sibling claims cannot both fast-forward the same remote branch.
            command(root, "push", "--atomic", "origin", "HEAD:refs/heads/research-loop")
        except Invalid:
            try:
                observed = fetch(root)
                reason = "STALE_CLAIM" if observed != approval else "CLAIM_PUSH_FAILED_OR_UNCERTAIN"
            except Invalid:
                reason = "CLAIM_PUSH_UNCERTAIN"
            raise Invalid(reason + "; local claim retained, no overwrite/retry/execution")
        require(remote_head(root) == claimed, "STALE_CLAIM_AFTER_PUSH; no execution")
        print(json.dumps({"status": "CLAIM_PUBLISHED", "claim_commit": claimed,
                          "instruction_commit": approval, "cycle_id": state["cycle_id"],
                          "experiment_id": state["experiment_id"],
                          "next": "Wait for this commit's control CI, verify remote is still this claim, then only execute the approved task."}))
        return claimed
    finally:
        lock.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    a = p.parse_args()
    try:
        claim(a.root)
        return 0
    except (Invalid, OSError, ValueError):
        e = sys.exc_info()[1]
        print(str(e) if isinstance(e, Invalid) else "CLAIM_FAILED (details suppressed); STOP", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

