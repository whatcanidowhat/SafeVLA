#!/usr/bin/env python3
"""Control-protocol tests using synthetic EXP-TEST-* tasks and temporary Git repos."""
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import research_loop_claim as claim_module
from validate_research_loop import (Invalid, View, STATE, validate_snapshot,
                                    validate_transition, validate_history, validate_index, scan_file, audit_tree)

SOURCE = Path(__file__).resolve().parents[1]


def git(root, *args):
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError("fixture git failed: " + args[0])
    return p.stdout.strip()


def save(root, path, data):
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n" if not isinstance(data, str) else data)


def initial():
    return {
        "schema_version": "1.0", "state_version": 1, "cycle_id": "test-cycle-1",
        "experiment_id": "EXP-TEST-HANDOFF-001", "status": "PI_REVIEW", "next_actor": "PI",
        "control_branch": "research-loop", "execution_worktree": None, "instruction_commit": None,
        "reviewed_result_commit": None, "claim_id": None,
        "authorization": {"status": "NOT_AUTHORIZED", "approved_by": None, "approved_at_utc": None,
                          "expires_at_utc": None, "scope": "CONTROL_ONLY_NOOP", "max_gpu": 0, "max_episodes": 0},
        "required_outputs": ["research/handoffs/test-cycle-1/" + n for n in
                             ["RESULT_SUMMARY.md", "RUN_MANIFEST.json", "ARTIFACT_INDEX.json", "REVIEW_NOTES.md"]],
        "updated_by": "CODEX_BOOTSTRAP", "updated_at_utc": "2026-09-13T00:00:00+00:00"
    }


def design(s, status="DRAFT"):
    from validate_research_loop import DESIGN_FIELDS
    d = {k: "Synthetic protocol test only" for k in DESIGN_FIELDS}
    d.update(experiment_id=s["experiment_id"], cycle_id=s["cycle_id"], status=status,
             required_outputs=s["required_outputs"], resources={"max_gpu": 0, "max_episodes": 0},
             command=["python3", "-c", "print('unit fixture only')"])
    return d


def install(root, s, status="DRAFT"):
    save(root, STATE, s)
    save(root, "research/NEXT_EXPERIMENT.json", design(s, status))
    save(root, "research/NEXT_EXPERIMENT.md", f"Experiment ID: {s['experiment_id']}\nStatus: {status}\n")


def fixture(root, approved=False):
    root.mkdir()
    git(root, "init", "-b", "research-loop")
    git(root, "config", "user.name", "Control Unit Test")
    git(root, "config", "user.email", "control-test@example.invalid")
    (root / "research").mkdir()
    shutil.copy2(SOURCE / "research/LOOP_STATE.schema.json", root / "research/LOOP_STATE.schema.json")
    s = initial()
    install(root, s)
    git(root, "add", ".")
    git(root, "commit", "-m", "Synthetic bootstrap")
    if approved:
        s.update(status="APPROVED_FOR_CODEX", next_actor="CODEX", state_version=2, updated_by="PI")
        s["authorization"].update(status="APPROVED", approved_by="PI",
                                 approved_at_utc="2026-09-13T00:00:00+00:00",
                                 expires_at_utc=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat())
        install(root, s, "APPROVED")
        git(root, "add", ".")
        git(root, "commit", "-m", "Synthetic PI approval")
    return s


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "control"
        self.s = fixture(self.root, approved=True)
        self.a = git(self.root, "rev-parse", "HEAD")

    def tearDown(self):
        self.temp.cleanup()

    def running(self):
        s = copy.deepcopy(self.s)
        s.update(status="CODEX_RUNNING", state_version=3, instruction_commit=self.a,
                 claim_id="syntheticclaim1", updated_by="CODEX", updated_at_utc=datetime.now(timezone.utc).isoformat())
        save(self.root, STATE, s)
        return s

    def test_bootstrap_and_approval_history(self):
        self.assertEqual(validate_history(self.root), 2)


    def test_history_preserves_non_ascii_paths(self):
        path = "research/\u5206\u6790 output.md"
        save(self.root, path, "Synthetic UTF-8 filename fixture")
        git(self.root, "add", path)
        git(self.root, "commit", "-m", "Synthetic non-ASCII artifact")
        for quote in ("true", "false"):
            with self.subTest(quote_path=quote):
                git(self.root, "config", "core.quotePath", quote)
                self.assertEqual(validate_history(self.root), 3)

    def test_non_ascii_path_still_enforces_whitelist(self):
        path = "outside/\u5206\u6790.md"
        save(self.root, path, "Synthetic forbidden path")
        git(self.root, "add", path)
        git(self.root, "commit", "-m", "Synthetic whitelist violation")
        with self.assertRaisesRegex(Invalid, "outside control whitelist"):
            audit_tree(View(self.root, "HEAD"))

    def test_non_ascii_path_still_enforces_artifact_rules(self):
        path = "research/\u5206\u6790.pt"
        save(self.root, path, "Synthetic forbidden extension")
        git(self.root, "add", path)
        git(self.root, "commit", "-m", "Synthetic artifact violation")
        with self.assertRaisesRegex(Invalid, "DISALLOWED_ARTIFACT"):
            audit_tree(View(self.root, "HEAD"))

    def test_pi_review_allows_staged_design_commits(self):
        root = Path(self.temp.name) / "pi-staging"
        s = fixture(root)
        original_state = (root / STATE).read_bytes()
        # All four design combinations are inert while every staging guard holds.
        for md_status, json_status in [("DRAFT", "DRAFT"), ("APPROVED", "DRAFT"),
                                       ("DRAFT", "APPROVED"), ("APPROVED", "APPROVED")]:
            with self.subTest(markdown=md_status, json=json_status):
                save(root, "research/NEXT_EXPERIMENT.json", design(s, json_status))
                save(root, "research/NEXT_EXPERIMENT.md",
                     f"Experiment ID: {s['experiment_id']}\nStatus: {md_status}\n")
                checked = validate_snapshot(View(root))
                self.assertEqual(checked["status"], "PI_REVIEW")
                self.assertEqual(checked["authorization"]["status"], "NOT_AUTHORIZED")
                self.assertEqual((root / STATE).read_bytes(), original_state)

        # Relaxation is unavailable if any eligibility guard is absent.
        for change in [{"next_actor": "CODEX"}, {"instruction_commit": "a" * 40},
                       {"claim_id": "syntheticclaim1"}, {"authorization": dict(s["authorization"], status="APPROVED")}]:
            with self.subTest(ineligible=change):
                save(root, STATE, dict(s, **change))
                with self.assertRaises(Invalid):
                    validate_snapshot(View(root))
        save(root, STATE, s)

        # Reproduce the actual four-commit order without changing LOOP_STATE
        # in either intermediate design commit.
        install(root, s)
        save(root, "research/NEXT_EXPERIMENT.json", design(s, "APPROVED"))
        git(root, "add", "research/NEXT_EXPERIMENT.json")
        git(root, "commit", "-m", "PI stages JSON approval")
        self.assertEqual(validate_history(root), 2)
        save(root, "research/NEXT_EXPERIMENT.md",
             f"Experiment ID: {s['experiment_id']}\nStatus: APPROVED\n")
        git(root, "add", "research/NEXT_EXPERIMENT.md")
        git(root, "commit", "-m", "PI stages Markdown approval")
        self.assertEqual(validate_history(root), 3)
        self.assertEqual((root / STATE).read_bytes(), original_state)
        s.update(status="APPROVED_FOR_CODEX", next_actor="CODEX", state_version=2, updated_by="PI")
        s["authorization"].update(status="APPROVED", approved_by="PI",
                                 approved_at_utc="2026-09-13T00:00:00+00:00",
                                 expires_at_utc=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat())
        save(root, STATE, s)
        git(root, "add", STATE)
        git(root, "commit", "-m", "PI publishes final LOOP_STATE authorization")
        self.assertEqual(validate_history(root), 4)

    def test_pi_can_revoke_unclaimed_approval(self):
        old = copy.deepcopy(self.s)
        n = copy.deepcopy(old)
        n.update(status="PI_REVIEW", next_actor="PI", state_version=3, updated_by="PI",
                 updated_at_utc=datetime.now(timezone.utc).isoformat())
        n["authorization"] = {
            "status": "NOT_AUTHORIZED", "approved_by": None, "approved_at_utc": None,
            "expires_at_utc": None, "scope": old["authorization"]["scope"],
            "max_gpu": old["authorization"]["max_gpu"], "max_episodes": old["authorization"]["max_episodes"],
        }
        install(self.root, n, "APPROVED")
        checked = validate_transition(View(self.root, self.a), View(self.root), self.a)
        self.assertIsNone(checked)

        # Once claimed/running, PI cannot use the pre-claim revocation edge.
        install(self.root, old, "APPROVED")
        running = copy.deepcopy(old)
        running.update(status="CODEX_RUNNING", next_actor="CODEX", state_version=3,
                       instruction_commit=self.a, claim_id="syntheticclaim1",
                       updated_by="CODEX", updated_at_utc=datetime.now(timezone.utc).isoformat())
        save(self.root, STATE, running)
        revoked = copy.deepcopy(running)
        revoked.update(status="PI_REVIEW", next_actor="PI", state_version=4, updated_by="PI")
        revoked["authorization"] = dict(n["authorization"])
        save(self.root, STATE, revoked)
        with self.assertRaises(Invalid):
            validate_transition(View(self.root, self.a), View(self.root), self.a)

    def test_executable_state_rejects_design_mismatch(self):
        validate_snapshot(View(self.root))  # APPROVED / APPROVED is still valid.
        for md_status, json_status in [("DRAFT", "APPROVED"), ("APPROVED", "DRAFT"),
                                       ("DRAFT", "DRAFT"), ("READY", "APPROVED"),
                                       ("APPROVED", None)]:
            with self.subTest(markdown=md_status, json=json_status):
                save(self.root, "research/NEXT_EXPERIMENT.json", design(self.s, json_status))
                save(self.root, "research/NEXT_EXPERIMENT.md",
                     f"Experiment ID: {self.s['experiment_id']}\nStatus: {md_status}\n")
                with self.assertRaises(Invalid):
                    validate_snapshot(View(self.root))
        install(self.root, self.s, "APPROVED")
        unauthorized = copy.deepcopy(self.s)
        unauthorized["authorization"]["status"] = "NOT_AUTHORIZED"
        save(self.root, STATE, unauthorized)
        with self.assertRaises(Invalid):
            validate_snapshot(View(self.root))

    def test_claim_binding(self):
        self.running()
        validate_transition(View(self.root, self.a), View(self.root), self.a)
        with self.assertRaises(Invalid):
            validate_transition(View(self.root, self.a), View(self.root), "0" * 40)

    def test_schema_and_snapshot_rejections(self):
        for change in [{"next_actor": "PI"}, {"experiment_id": "EXP-WRONG"},
                       {"status": "READY"}, {"extra": True}, {"required_outputs": []},
                       {"execution_worktree": "/nvme2/user/qyy/SafeVLA_loop_control"}]:
            with self.subTest(change=change):
                s = copy.deepcopy(self.s)
                s.update(change)
                save(self.root, STATE, s)
                with self.assertRaises(Invalid):
                    validate_snapshot(View(self.root))

    def test_transition_rejections(self):
        for change in [{"state_version": 8}, {"cycle_id": "other-cycle"},
                       {"updated_by": "PI"}, {"instruction_commit": None},
                       {"instruction_commit": "f" * 40}, {"claim_id": None}]:
            with self.subTest(change=change):
                s = self.running()
                s.update(change)
                save(self.root, STATE, s)
                with self.assertRaises(Invalid):
                    validate_transition(View(self.root, self.a), View(self.root), self.a)

    def test_expired_claim(self):
        s = self.running()
        s["authorization"]["expires_at_utc"] = "2026-09-13T00:01:00+00:00"
        save(self.root, STATE, s)
        with self.assertRaises(Invalid):
            validate_transition(View(self.root, self.a), View(self.root), self.a)

    def test_missing_design(self):
        save(self.root, "research/NEXT_EXPERIMENT.json", {"experiment_id": self.s["experiment_id"],
             "cycle_id": self.s["cycle_id"], "status": "APPROVED"})
        with self.assertRaises(Invalid):
            validate_snapshot(View(self.root))

    def test_outputs_and_illegal_self_approval(self):
        s = self.running()
        git(self.root, "add", ".")
        git(self.root, "commit", "-m", "Synthetic claim")
        b = git(self.root, "rev-parse", "HEAD")
        for state in ["AWAITING_PI_REVIEW", "BLOCKED", "INVALID", "ABORTED", "APPROVED_FOR_CODEX"]:
            with self.subTest(status=state):
                n = copy.deepcopy(s)
                n.update(status=state, next_actor="CODEX" if state == "APPROVED_FOR_CODEX" else "PI", state_version=4)
                save(self.root, STATE, n)
                with self.assertRaises(Invalid):
                    validate_transition(View(self.root, b), View(self.root), b)

    def test_full_synthetic_result_ack(self):
        s = self.running()
        git(self.root, "add", "."); git(self.root, "commit", "-m", "Synthetic claim")
        b = git(self.root, "rev-parse", "HEAD")
        prefix = "research/handoffs/test-cycle-1/"
        summary = "Synthetic unit test result; no research execution.\n"
        save(self.root, prefix + "RESULT_SUMMARY.md", summary)
        save(self.root, prefix + "REVIEW_NOTES.md", "Synthetic review notes\n")
        import hashlib
        save(self.root, prefix + "ARTIFACT_INDEX.json", {"schema_version": "1.0", "artifacts": [{
            "name": "summary", "server_path": "/synthetic/RESULT_SUMMARY.md",
            "git_path": prefix + "RESULT_SUMMARY.md", "sha256": hashlib.sha256(summary.encode()).hexdigest(),
            "size": len(summary.encode()), "required_for_PI_review": True}]})
        s.update(status="AWAITING_PI_REVIEW", next_actor="PI", state_version=4)
        save(self.root, STATE, s)
        save(self.root, prefix + "RUN_MANIFEST.json", {k: s[k] for k in
             ["cycle_id", "experiment_id", "instruction_commit", "claim_id", "status"]} |
             {"gpu_count": 0, "episodes_started": 0, "command": ["synthetic-no-op"]})
        validate_transition(View(self.root, b), View(self.root), b)
        git(self.root, "add", "."); git(self.root, "commit", "-m", "Synthetic result")
        c = git(self.root, "rev-parse", "HEAD")
        s.update(status="PI_REVIEW", next_actor="PI", state_version=5, updated_by="PI", reviewed_result_commit=c)
        s["authorization"].update(status="NOT_AUTHORIZED", approved_by=None,
                                  approved_at_utc=None, expires_at_utc=None)
        install(self.root, s)
        validate_transition(View(self.root, c), View(self.root), c)
        s["reviewed_result_commit"] = b
        save(self.root, STATE, s)
        with self.assertRaises(Invalid):
            validate_transition(View(self.root, c), View(self.root), c)

    def test_pi_acknowledges_blocked_handoff(self):
        s = self.running()
        git(self.root, "add", "."); git(self.root, "commit", "-m", "Synthetic claim")
        b = git(self.root, "rev-parse", "HEAD")
        prefix = "research/handoffs/test-cycle-1/"
        summary = "Synthetic blocked result; no research execution.\n"
        save(self.root, prefix + "RESULT_SUMMARY.md", summary)
        save(self.root, prefix + "REVIEW_NOTES.md", "Synthetic blocked review notes\n")
        import hashlib
        save(self.root, prefix + "ARTIFACT_INDEX.json", {"schema_version": "1.0", "artifacts": [{
            "name": "summary", "server_path": "/synthetic/RESULT_SUMMARY.md",
            "git_path": prefix + "RESULT_SUMMARY.md", "sha256": hashlib.sha256(summary.encode()).hexdigest(),
            "size": len(summary.encode()), "required_for_PI_review": True}]})
        s.update(status="BLOCKED", next_actor="PI", state_version=4)
        save(self.root, STATE, s)
        save(self.root, prefix + "RUN_MANIFEST.json", {k: s[k] for k in
             ["cycle_id", "experiment_id", "instruction_commit", "claim_id", "status"]} |
             {"gpu_count": 0, "episodes_started": 0, "command": ["synthetic-blocked"]})
        validate_transition(View(self.root, b), View(self.root), b)
        git(self.root, "add", "."); git(self.root, "commit", "-m", "Synthetic blocked result")
        c = git(self.root, "rev-parse", "HEAD")

        ack = copy.deepcopy(s)
        ack.update(status="PI_REVIEW", next_actor="PI", state_version=5, updated_by="PI",
                   reviewed_result_commit=c, updated_at_utc=datetime.now(timezone.utc).isoformat())
        ack["authorization"] = {
            "status": "NOT_AUTHORIZED", "approved_by": None, "approved_at_utc": None,
            "expires_at_utc": None, "scope": s["authorization"]["scope"],
            "max_gpu": s["authorization"]["max_gpu"], "max_episodes": s["authorization"]["max_episodes"],
        }
        install(self.root, ack, "DRAFT")
        validate_transition(View(self.root, c), View(self.root), c)

        # Direct exception-state re-approval remains forbidden.
        install(self.root, s, "APPROVED")
        bad = copy.deepcopy(s)
        bad.update(status="APPROVED_FOR_CODEX", next_actor="CODEX", state_version=5, updated_by="PI")
        save(self.root, STATE, bad)
        with self.assertRaises(Invalid):
            validate_transition(View(self.root, c), View(self.root), c)

    def test_secret_and_size(self):
        for blob in [(b"gh" + b"p_" + b"a" * 30), (b"https://" + b"user:credential@example.invalid/x"),
                     b"a" * (1024 * 1024 + 1)]:
            with self.assertRaises(Invalid):
                scan_file("research/test.txt", blob)

    def test_artifact_hash_mismatch(self):
        save(self.root, "research/test.txt", "small")
        save(self.root, "research/ARTIFACT_INDEX.json", {"schema_version": "1.0", "artifacts": [{
            "name": "x", "server_path": "/tmp/x", "sha256": "0" * 64, "size": 5,
            "required_for_PI_review": True, "git_path": "research/test.txt"}]})
        with self.assertRaises(Invalid):
            validate_index(View(self.root), "research/ARTIFACT_INDEX.json")

    def clones(self):
        bare = Path(self.temp.name) / "origin.git"
        git(self.root, "clone", "--bare", str(self.root), str(bare))
        clones = []
        for name in ["executor-a", "executor-b"]:
            target = Path(self.temp.name) / name
            git(self.root, "clone", str(bare), str(target))
            git(target, "config", "user.name", "Control Unit Test")
            git(target, "config", "user.email", "control-test@example.invalid")
            clones.append(target)
        return clones

    def test_claim_and_second_executor_stops(self):
        a, b = self.clones()
        with contextlib.redirect_stdout(io.StringIO()):
            claim_module.claim(a)
        with self.assertRaisesRegex(Invalid, "STALE_LOCAL_HEAD"):
            claim_module.claim(b)
        self.assertEqual(View(a).data(STATE)["instruction_commit"], self.a)

    def test_push_race_stops_without_overwrite(self):
        a, b = self.clones()
        real = claim_module.command
        raced = [False]
        def race(root, *args):
            if Path(root) == a and args[0] == "push" and not raced[0]:
                raced[0] = True
                with contextlib.redirect_stdout(io.StringIO()):
                    claim_module.claim(b)
            return real(root, *args)
        with patch.object(claim_module, "command", side_effect=race):
            with self.assertRaisesRegex(Invalid, "STALE_CLAIM"):
                claim_module.claim(a)
        self.assertEqual(claim_module.remote_head(b), git(b, "rev-parse", "HEAD"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

