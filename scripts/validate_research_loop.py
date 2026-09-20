#!/usr/bin/env python3
"""Validate control data and Git history. Never import or execute SafeVLA."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

from jsonschema import Draft202012Validator, FormatChecker

STATE = "research/LOOP_STATE.json"
SCHEMA = "research/LOOP_STATE.schema.json"
NEXT = "research/NEXT_EXPERIMENT.md"
DESIGN = "research/NEXT_EXPERIMENT.json"
BASE_OUTPUTS = {"RESULT_SUMMARY.md", "RUN_MANIFEST.json", "ARTIFACT_INDEX.json", "REVIEW_NOTES.md"}
ACTOR = {"PI_REVIEW": "PI", "APPROVED_FOR_CODEX": "CODEX", "CODEX_RUNNING": "CODEX",
         "AWAITING_PI_REVIEW": "PI", "BLOCKED": "PI", "INVALID": "PI", "ABORTED": "PI"}
EDGES = {("PI_REVIEW", "APPROVED_FOR_CODEX"): "PI",
         ("APPROVED_FOR_CODEX", "PI_REVIEW"): "PI",
         ("APPROVED_FOR_CODEX", "CODEX_RUNNING"): "CODEX",
         ("CODEX_RUNNING", "AWAITING_PI_REVIEW"): "CODEX",
         ("AWAITING_PI_REVIEW", "PI_REVIEW"): "PI",
         ("BLOCKED", "PI_REVIEW"): "PI",
         ("INVALID", "PI_REVIEW"): "PI",
         ("ABORTED", "PI_REVIEW"): "PI",
         ("CODEX_RUNNING", "BLOCKED"): "CODEX",
         ("CODEX_RUNNING", "INVALID"): "CODEX",
         ("CODEX_RUNNING", "ABORTED"): "CODEX"}
DESIGN_FIELDS = ["research_question", "hypothesis", "competing_explanation", "reference", "repeat_or_treatment",
                 "unique_variable", "fixed_conditions", "metrics", "expected_result", "falsifying_result",
                 "alternative_explanations", "stop_conditions", "command", "required_outputs", "resources"]
MAX_FILE = 1024 * 1024
SECRET_PATTERNS = [rb"gh[pousr]_[A-Za-z0-9_]{20,}", rb"github_pat_[A-Za-z0-9_]{20,}",
                   rb"https?://[^\s/\"<>]+@", rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
                   rb"(?i)(?:authorization\s*:\s*(?:bearer|token)|(?:api_key|access_token|password)\s*[=:])\s*[\"']?[A-Za-z0-9_+/=-]{20,}"]


# Narrow append-only-history exception for the 2026-09-18 small-target control incident.
# No Executor claim or research execution occurred in these states. The exception exists only so
# history validation can preserve, rather than rewrite, the invalid staging/approval records.
LEGACY_SMALLTARGET_CYCLE = "smalltarget-phenotype-001-20260918"
LEGACY_SMALLTARGET_EXPERIMENT = "EXP-SMALLTARGET-PHENOTYPE-001"
LEGACY_MISSING_DESIGN_FIELDS = {
    "competing_explanation", "reference", "repeat_or_treatment",
    "alternative_explanations", "command",
}


def _legacy_smalltarget_review(s):
    a = s.get("authorization", {})
    return (
        s.get("cycle_id") == LEGACY_SMALLTARGET_CYCLE
        and s.get("experiment_id") == LEGACY_SMALLTARGET_EXPERIMENT
        and s.get("state_version") == 8
        and s.get("status") == "PI_REVIEW"
        and s.get("next_actor") == "PI"
        and s.get("instruction_commit") is None
        and s.get("claim_id") is None
        and a.get("status") == "NOT_AUTHORIZED"
        and s.get("required_outputs") == []
    )


def _legacy_smalltarget_unclaimed_approval(s):
    a = s.get("authorization", {})
    return (
        s.get("cycle_id") == LEGACY_SMALLTARGET_CYCLE
        and s.get("experiment_id") == LEGACY_SMALLTARGET_EXPERIMENT
        and s.get("state_version") == 9
        and s.get("status") == "APPROVED_FOR_CODEX"
        and s.get("next_actor") == "CODEX"
        and s.get("instruction_commit") is None
        and s.get("claim_id") is None
        and a.get("status") == "APPROVED"
        and a.get("approved_by") == "PI"
        and a.get("max_gpu") == 0
        and a.get("max_episodes") == 0
    )


def validate_legacy_smalltarget_review(view, old_view=None):
    s = view.data(STATE)
    require(_legacy_smalltarget_review(s), "legacy small-target review exception mismatch")
    # Prove that required_outputs=[] is the only schema-level defect we are excusing.
    schema = view.data(SCHEMA)
    repaired = dict(s)
    prefix = f"research/handoffs/{s['cycle_id']}/"
    repaired["required_outputs"] = [prefix + n for n in sorted(BASE_OUTPUTS)]
    errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(repaired))
    require(not errors, "legacy small-target review has additional schema defects")
    d = view.data(DESIGN)
    allowed_designs = {
        ("EXP-B0-REPRO-001B-PREFLIGHT", "b0-repro-001b-preflight-20260914"),
        (LEGACY_SMALLTARGET_EXPERIMENT, LEGACY_SMALLTARGET_CYCLE),
    }
    require((d.get("experiment_id"), d.get("cycle_id")) in allowed_designs,
            "legacy review staged an unexpected design")
    require(d.get("status") in ("DRAFT", "APPROVED"), "legacy review design status")
    md = view.read(NEXT).decode("utf-8")
    md_ids = re.findall(r"^Experiment ID:\s*(\S+)\s*$", md, re.M)
    require(len(md_ids) == 1 and md_ids[0] in {x[0] for x in allowed_designs},
            "legacy review NEXT staged an unexpected experiment")
    if old_view and old_view.exists(STATE):
        o = old_view.data(STATE)
        if not _legacy_smalltarget_review(o):
            require(o.get("instruction_commit") is None and o.get("claim_id") is None,
                    "legacy review replaced a claimed/executing cycle")
    return s


def validate_legacy_smalltarget_unclaimed_approval(view, old_view=None):
    s = view.data(STATE)
    require(_legacy_smalltarget_unclaimed_approval(s), "legacy small-target approval exception mismatch")
    schema = view.data(SCHEMA)
    errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(s))
    require(not errors, "legacy small-target approval state schema defect")
    prefix = f"research/handoffs/{s['cycle_id']}/"
    require({prefix + n for n in BASE_OUTPUTS}.issubset(s["required_outputs"]),
            "legacy approval missing base outputs")
    require(all(x.startswith(prefix) and safe_path(x) for x in s["required_outputs"]),
            "legacy approval output path mismatch")
    d = view.data(DESIGN)
    require(d.get("experiment_id") == s["experiment_id"] and d.get("cycle_id") == s["cycle_id"],
            "legacy approval design identity mismatch")
    require(d.get("status") == "APPROVED", "legacy approval JSON status mismatch")
    md = view.read(NEXT).decode("utf-8")
    require(re.search(r"^Status:[ \t]*APPROVED[ \t]*$", md, re.M), "legacy approval Markdown status mismatch")
    missing = {k for k in DESIGN_FIELDS if not d.get(k)}
    require(missing and missing.issubset(LEGACY_MISSING_DESIGN_FIELDS),
            "legacy approval has unexpected design defects")
    require(d.get("required_outputs") == s["required_outputs"], "legacy approval output mismatch")
    require(d.get("resources") == {"max_gpu": 0, "max_episodes": 0}, "legacy approval resource mismatch")
    if old_view and old_view.exists(STATE):
        o = old_view.data(STATE)
        require(_legacy_smalltarget_review(o) or _legacy_smalltarget_unclaimed_approval(o),
                "legacy approval has unexpected parent state")
    return s


class Invalid(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise Invalid(message)


def git(root, *args, allowed=(0,)):
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=60)
    require(p.returncode in allowed, "Git read failed (output suppressed): " + args[0])
    return p.stdout


class View:
    def __init__(self, root, ref=None):
        self.root, self.ref = Path(root), ref

    def read(self, path):
        if self.ref:
            return git(self.root, "show", f"{self.ref}:{path}")
        return (self.root / path).read_bytes()

    def exists(self, path):
        if self.ref:
            return subprocess.run(["git", "-C", str(self.root), "cat-file", "-e", f"{self.ref}:{path}"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        return (self.root / path).is_file()

    def data(self, path):
        return json.loads(self.read(path))


def utc(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def scan_file(path, data):
    require(len(data) <= MAX_FILE, "OVERSIZED_ARTIFACT: " + path)
    require(not re.search(r"\.(pt|pth|ckpt|tensor|npy|npz|mp4|avi|log|tar|gz|zip)$", path, re.I),
            "DISALLOWED_ARTIFACT: " + path)
    require(not any(re.search(p, data) for p in SECRET_PATTERNS), "SECURITY_BLOCKED: " + path)


def safe_path(path):
    p = PurePosixPath(path)
    return not p.is_absolute() and ".." not in p.parts and "\\" not in path


def validate_index(view, path):
    idx = view.data(path)
    require(idx.get("schema_version") == "1.0", "artifact index schema_version")
    require(isinstance(idx.get("artifacts"), list) and bool(idx["artifacts"]), "artifact index empty")
    names = set()
    for a in idx["artifacts"]:
        require(all(k in a for k in ["name", "server_path", "sha256", "size", "required_for_PI_review"]), "artifact fields missing")
        require(isinstance(a["name"], str) and a["name"] and a["name"] not in names, "duplicate/invalid artifact name")
        names.add(a["name"])
        require(isinstance(a["server_path"], str) and a["server_path"].startswith("/"), "artifact server_path")
        require(isinstance(a["sha256"], str) and re.fullmatch("[0-9a-f]{64}", a["sha256"]), "artifact sha256")
        require(type(a["size"]) is int and a["size"] >= 0, "artifact size")
        require(type(a["required_for_PI_review"]) is bool, "artifact review flag")
        shared = a.get("git_path")
        if shared:
            require(safe_path(shared) and view.exists(shared), "shared artifact missing/unsafe")
            blob = view.read(shared)
            require(hashlib.sha256(blob).hexdigest() == a["sha256"] and len(blob) == a["size"], "shared artifact hash/size mismatch: " + shared)
        require(not a["required_for_PI_review"] or shared, "PI-required artifact must have readable Git copy")


def validate_snapshot(view):
    s = view.data(STATE)
    schema = view.data(SCHEMA)
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(s), key=lambda e: str(e.path))
    require(not errors, "LOOP_STATE schema violation: " + (str(errors[0].path) if errors else ""))
    require(s["next_actor"] == ACTOR[s["status"]], "status / next_actor mismatch")
    require(s["execution_worktree"] is None or "SafeVLA_loop_control" not in s["execution_worktree"], "control worktree cannot execute research")
    d = view.data(DESIGN)
    require(d.get("experiment_id") == s["experiment_id"] and d.get("cycle_id") == s["cycle_id"], "design identity mismatch")
    md = view.read(NEXT).decode("utf-8")
    require(re.search(r"^Experiment ID:\s*" + re.escape(s["experiment_id"]) + r"\s*$", md, re.M), "NEXT_EXPERIMENT mismatch")
    md_statuses = re.findall(r"^Status:[ \t]*(\S+)[ \t]*$", md, re.M)
    require(len(md_statuses) == 1, "Markdown must declare exactly one design status")
    md_status = md_statuses[0]
    json_status = d.get("status")
    prefix = f"research/handoffs/{s['cycle_id']}/"
    require({prefix + n for n in BASE_OUTPUTS}.issubset(s["required_outputs"]), "missing required handoff outputs")
    require(all(x.startswith(prefix) and safe_path(x) for x in s["required_outputs"]), "handoff path/cycle mismatch")
    auth = s["authorization"]
    if auth["scope"] == "CONTROL_ONLY_NOOP":
        require(auth["max_gpu"] == auth["max_episodes"] == 0 and s["execution_worktree"] is None, "no-op cannot allocate runtime")
    staged_review = (
        s["status"] == "PI_REVIEW"
        and s["next_actor"] == "PI"
        and auth["status"] == "NOT_AUTHORIZED"
        and s["instruction_commit"] is None
        and s["claim_id"] is None
    )
    if s["status"] == "PI_REVIEW":
        require(auth["status"] == "NOT_AUTHORIZED", "review must not authorize execution")
        if staged_review:
            # PI can publish the two design files separately; neither grants execution.
            require(md_status in ("DRAFT", "APPROVED") and json_status in ("DRAFT", "APPROVED"),
                    "invalid staged design status")
        else:
            require(md_status == json_status == "DRAFT", "non-staging review requires draft designs")
    else:
        require(auth["status"] == "APPROVED" and md_status == json_status == "APPROVED",
                "execution state needs both designs approved")
        require(all(d.get(k) for k in DESIGN_FIELDS), "incomplete approved experiment design")
        require(auth["approved_by"] and auth["approved_at_utc"] and auth["expires_at_utc"], "approval metadata incomplete")
        require(utc(auth["expires_at_utc"]) > utc(auth["approved_at_utc"]), "approval expiration invalid")
        require(d["required_outputs"] == s["required_outputs"], "design outputs mismatch")
        require(d["resources"] == {"max_gpu": auth["max_gpu"], "max_episodes": auth["max_episodes"]}, "design resource mismatch")
    if s["status"] == "APPROVED_FOR_CODEX":
        require(s["instruction_commit"] is None and s["claim_id"] is None, "approval cannot pre-claim")
    if s["status"] in {"CODEX_RUNNING", "AWAITING_PI_REVIEW", "BLOCKED", "INVALID", "ABORTED"}:
        require(s["instruction_commit"] and s["claim_id"], "missing instruction_commit / claim_id")
    if s["status"] in {"AWAITING_PI_REVIEW", "BLOCKED", "INVALID", "ABORTED"}:
        require(all(view.exists(x) and len(view.read(x)) > 0 for x in s["required_outputs"]), "handoff outputs incomplete")
        m = view.data(prefix + "RUN_MANIFEST.json")
        for k in ["cycle_id", "experiment_id", "instruction_commit", "claim_id"]:
            require(m.get(k) == s[k], "handoff manifest identity mismatch: " + k)
        require(m.get("status") == s["status"], "handoff manifest status mismatch")
        require(type(m.get("gpu_count")) is int and 0 <= m["gpu_count"] <= auth["max_gpu"], "manifest GPU budget")
        require(type(m.get("episodes_started")) is int and 0 <= m["episodes_started"] <= auth["max_episodes"], "manifest episode budget")
        require(isinstance(m.get("command"), list) and bool(m["command"]), "manifest command missing")
        validate_index(view, prefix + "ARTIFACT_INDEX.json")
    return s


def validate_transition(old_view, new_view, parent_sha, bootstrap=False):
    n = validate_snapshot(new_view)
    if not old_view or not old_view.exists(STATE):
        require(bootstrap, "state absent in parent; explicit bootstrap required")
        require(n["state_version"] == 1 and n["status"] == "PI_REVIEW" and n["updated_by"] == "CODEX_BOOTSTRAP", "invalid bootstrap")
        require(n["instruction_commit"] is None and n["reviewed_result_commit"] is None and n["claim_id"] is None, "bootstrap fabricated execution")
        return
    o = old_view.data(STATE)
    if o == n:
        # A protocol/doc change cannot silently rewrite the approved design.
        if o["status"] != "PI_REVIEW":
            require(old_view.read(DESIGN) == new_view.read(DESIGN) and old_view.read(NEXT) == new_view.read(NEXT), "approved design changed")
        return
    require(n["state_version"] == o["state_version"] + 1, "state_version must increment exactly once")
    require(utc(n["updated_at_utc"]) >= utc(o["updated_at_utc"]), "state timestamp regressed")
    edge = (o["status"], n["status"])
    if edge == ("PI_REVIEW", "PI_REVIEW"):
        require(n["updated_by"] == "PI", "only PI can revise a draft/review state")
        require(n["reviewed_result_commit"] == o["reviewed_result_commit"], "draft edit cannot fabricate acknowledgement")
        if n["cycle_id"] != o["cycle_id"]:
            require(n["instruction_commit"] is None and n["claim_id"] is None, "new cycle must be unclaimed")
        return
    require(edge in EDGES and n["updated_by"] == EDGES.get(edge), "illegal transition or actor")
    require(n["cycle_id"] == o["cycle_id"] and n["experiment_id"] == o["experiment_id"], "active cycle/experiment changed")
    require(n["control_branch"] == o["control_branch"], "control branch changed")
    if edge == ("PI_REVIEW", "APPROVED_FOR_CODEX"):
        require(n["authorization"]["approved_by"] == "PI", "only PI approval is accepted")
        require(n["reviewed_result_commit"] == o["reviewed_result_commit"], "approval cannot fabricate review receipt")
    elif edge == ("APPROVED_FOR_CODEX", "PI_REVIEW"):
        require(o["instruction_commit"] is None and o["claim_id"] is None,
                "PI may revoke only an unclaimed approval")
        require(n["authorization"]["status"] == "NOT_AUTHORIZED",
                "revoked approval must remove execution authorization")
        require(n["authorization"]["approved_by"] is None
                and n["authorization"]["approved_at_utc"] is None
                and n["authorization"]["expires_at_utc"] is None,
                "revoked approval must clear approval metadata")
        require(n["reviewed_result_commit"] == o["reviewed_result_commit"],
                "revocation cannot fabricate review receipt")
        require(n["required_outputs"] == o["required_outputs"]
                and n["execution_worktree"] == o["execution_worktree"],
                "revocation cannot change frozen execution paths")
        return
    elif edge == ("APPROVED_FOR_CODEX", "CODEX_RUNNING"):
        require(n["instruction_commit"] == parent_sha, "instruction_commit must equal pre-claim remote HEAD")
        require(old_view.read(DESIGN) == new_view.read(DESIGN) and old_view.read(NEXT) == new_view.read(NEXT), "claim changed PI instructions")
        require(n["authorization"] == o["authorization"], "claim changed authorization")
        require(utc(n["updated_at_utc"]) <= utc(n["authorization"]["expires_at_utc"]), "expired approval")
    else:
        require(n["instruction_commit"] == o["instruction_commit"] and n["claim_id"] == o["claim_id"], "claim identity changed")
        if edge in {("AWAITING_PI_REVIEW", "PI_REVIEW"), ("BLOCKED", "PI_REVIEW"),
                    ("INVALID", "PI_REVIEW"), ("ABORTED", "PI_REVIEW")}:
            require(n["reviewed_result_commit"] == parent_sha, "PI acknowledgement must reference result HEAD")
            require(n["authorization"]["status"] == "NOT_AUTHORIZED",
                    "PI acknowledgement must remove execution authorization")
            if edge in {("BLOCKED", "PI_REVIEW"), ("INVALID", "PI_REVIEW"), ("ABORTED", "PI_REVIEW")}:
                require(n["authorization"]["approved_by"] is None
                        and n["authorization"]["approved_at_utc"] is None
                        and n["authorization"]["expires_at_utc"] is None,
                        "exception recovery must clear approval metadata")
        else:
            require(n["authorization"] == o["authorization"], "executor changed authorization")
            require(old_view.read(DESIGN) == new_view.read(DESIGN) and old_view.read(NEXT) == new_view.read(NEXT), "executor changed approved design")
    if edge not in {("AWAITING_PI_REVIEW", "PI_REVIEW"), ("BLOCKED", "PI_REVIEW"),
                    ("INVALID", "PI_REVIEW"), ("ABORTED", "PI_REVIEW")}:
        require(n["reviewed_result_commit"] == o["reviewed_result_commit"], "executor cannot acknowledge PI review")
    if edge != ("PI_REVIEW", "APPROVED_FOR_CODEX"):
        require(n["required_outputs"] == o["required_outputs"] and n["execution_worktree"] == o["execution_worktree"], "frozen execution conditions changed")


def audit_tree(view):
    if view.ref:
        # NUL records preserve literal UTF-8 paths, independent of core.quotePath.
        entries = git(view.root, "ls-tree", "-r", "-z", view.ref).decode("utf-8").split("\0")
        files = []
        for entry in filter(None, entries):
            meta, path = entry.split("\t", 1)
            require(meta.split()[0] in {"100644", "100755"}, "symlink/submodule forbidden")
            files.append(path)
    else:
        files = git(view.root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
        files = sorted(set(filter(None, files)))
    allowed = {"AGENTS.md", ".gitignore", ".github/workflows/research-loop-validate.yml",
               ".codex/skills/safevla-research-loop/SKILL.md", "scripts/research_loop_claim.py",
               "scripts/validate_research_loop.py", "scripts/test_research_loop.py"}
    for path in files:
        require(path in allowed or path.startswith("research/"), "outside control whitelist: " + path)
        if not view.ref:
            require(not (view.root / path).is_symlink(), "symlink forbidden")
        scan_file(path, view.read(path))
    for path in files:
        if path.endswith("ARTIFACT_INDEX.json"):
            validate_index(view, path)


def validate_history(root, head="HEAD"):
    commits = git(root, "rev-list", "--reverse", head).decode().splitlines()
    used_cycles = set()
    for sha in commits:
        parents = git(root, "rev-list", "--parents", "-n", "1", sha).decode().split()[1:]
        require(len(parents) <= 1, "control history must be linear")
        parent = parents[0] if parents else None
        v = View(root, sha)
        old_view = View(root, parent) if parent else None
        audit_tree(v)
        raw_state = v.data(STATE)
        if _legacy_smalltarget_review(raw_state):
            n = validate_legacy_smalltarget_review(v, old_view)
        elif _legacy_smalltarget_unclaimed_approval(raw_state):
            n = validate_legacy_smalltarget_unclaimed_approval(v, old_view)
        else:
            validate_transition(old_view, v, parent, bootstrap=parent is None)
            n = raw_state
        o = old_view.data(STATE) if parent else None
        if n["status"] == "APPROVED_FOR_CODEX" and (not o or o["status"] != "APPROVED_FOR_CODEX"):
            require(n["cycle_id"] not in used_cycles, "cycle already executed; PI must choose a new cycle")
        if n["status"] == "CODEX_RUNNING" and (not o or o["status"] != "CODEX_RUNNING"):
            require(n["cycle_id"] not in used_cycles, "cycle has already been claimed")
            used_cycles.add(n["cycle_id"])
    return len(commits)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--base", help="Compare working files to this existing parent")
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--history", action="store_true")
    a = p.parse_args()
    try:
        if a.history:
            print("CONTROL_HISTORY_PASS commits=" + str(validate_history(a.root)))
        else:
            v = View(a.root)
            audit_tree(v)
            if a.base or a.bootstrap:
                validate_transition(View(a.root, a.base) if a.base else None, v, a.base, a.bootstrap)
            else:
                validate_snapshot(v)
            print("CONTROL_VALIDATION_PASS")
    except (Invalid, OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
        # Do not echo file values or Git stderr: either may contain credentials.
        error = sys.exc_info()[1]
        print(str(error) if isinstance(error, Invalid) else "VALIDATION_FAILED (details suppressed)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
