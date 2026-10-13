#!/usr/bin/env python3
"""PI acknowledgement of an evidence-backed terminal BLOCKED claim; NO execution."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from validate_research_loop import STATE, View, validate_history, validate_transition, terminal_state_entry_commit


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True,
                            timeout=90, env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    require(result.returncode == 0, "Git failure at "+args[0])
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-terminal-commit", required=True,
                        help="Actual terminal BLOCKED state entry commit previously reviewed by PI")
    parser.add_argument("--control-root", type=Path,
                        default=Path("/nvme2/user/qyy/SafeVLA_loop_control"))
    args = parser.parse_args()
    root = args.control_root.resolve()
    require(git(root, "branch", "--show-current") == "research-loop", "Wrong control branch")
    require(not git(root, "status", "--porcelain"), "Dirty control worktree")
    parent = git(root, "rev-parse", "HEAD")
    remote = git(root, "ls-remote", "--heads", "origin", "refs/heads/research-loop").split()
    require(remote and remote[0] == parent, "Stale control checkout")
    old = json.loads((root / STATE).read_text())
    require(old["status"] == "BLOCKED" and old["next_actor"] == "PI"
            and old["updated_by"] == "PI"
            and old["cycle_id"] == "actor-reset-off200-on200-20261012"
            and old["claim_id"], "Not the PI-recovered P2 BLOCKED claim")
    actual_entry = terminal_state_entry_commit(root, parent)
    require(args.reviewed_terminal_commit == actual_entry,
            "Review SHA is not the exact terminal BLOCKED state entry commit")
    prefix = root / "research" / "handoffs" / old["cycle_id"]
    manifest = json.loads((prefix / "RUN_MANIFEST.json").read_text())
    require(manifest["status"] == "BLOCKED"
            and manifest["claim_id"] == old["claim_id"]
            and manifest["handoff_actor"] == "PI"
            and manifest["closure_reason"] == "CODEX_UNAVAILABLE_AND_RUNTIME_ERROR",
            "Missing real PI handoff")
    state = dict(old)
    state.update(status="PI_REVIEW", next_actor="PI",
                 reviewed_result_commit=actual_entry, state_version=old["state_version"]+1,
                 updated_by="PI", updated_at_utc=datetime.now(timezone.utc).isoformat())
    state["authorization"] = dict(old["authorization"], status="NOT_AUTHORIZED",
                                  approved_by=None, approved_at_utc=None,
                                  expires_at_utc=None)
    (root / STATE).write_text(json.dumps(state, ensure_ascii=False, indent=2)+"\n")
    dp = root / "research" / "NEXT_EXPERIMENT.json"
    design = json.loads(dp.read_text())
    require(design["cycle_id"] == state["cycle_id"] and design["status"] == "APPROVED",
            "Old design not approved")
    design["status"] = "DRAFT"
    dp.write_text(json.dumps(design, ensure_ascii=False, indent=2)+"\n")
    md = root / "research" / "NEXT_EXPERIMENT.md"
    content = md.read_text(encoding="utf8")
    require(content.count("Status: APPROVED") == 1, "Unexpected markdown design status")
    md.write_text(content.replace("Status: APPROVED", "Status: DRAFT", 1), encoding="utf8")
    validate_transition(View(root, parent), View(root), parent)
    git(root, "add", "--", STATE, "research/NEXT_EXPERIMENT.json", "research/NEXT_EXPERIMENT.md")
    git(root, "commit", "-m", "PI acknowledge real P2 BLOCKED handoff; clear execution authorization")
    nxt = git(root, "rev-parse", "HEAD")
    validate_history(root)
    remote = git(root, "ls-remote", "--heads", "origin", "refs/heads/research-loop").split()
    require(remote and remote[0] == parent, "Control remote advanced, STOP")
    git(root, "push", "--atomic", "origin", "HEAD:refs/heads/research-loop")
    remote = git(root, "ls-remote", "--heads", "origin", "refs/heads/research-loop").split()
    require(remote and remote[0] == nxt, "Cannot verify push; STOP")
    print(json.dumps({"status":"PI_REVIEW_PUBLISHED",
                      "ack_commit":nxt, "reviewed_terminal":actual_entry,
                      "next":"Wait for ACK commit CI SUCCESS; new cycle must be separately designed and approved."},
                     indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("PI_ACK_FAILED "+type(exc).__name__+": "+str(exc)[:220]+"; STOP", file=sys.stderr)
        sys.exit(1)
