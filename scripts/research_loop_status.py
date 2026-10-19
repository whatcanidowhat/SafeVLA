#!/usr/bin/env python3
"""Observe a pinned control snapshot. Exit 0 means observation, never execution."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from jsonschema import Draft202012Validator, FormatChecker
from research_loop_readonly import GitHub, Parser, Stop, emit, guarded, positive, report, require
from validate_research_loop import ACTOR, BASE_OUTPUTS, DESIGN_FIELDS, safe_path

ROOT = Path(__file__).resolve().parents[1]
PATHS = ('research/LOOP_STATE.json', 'research/NEXT_EXPERIMENT.json', 'research/NEXT_EXPERIMENT.md')


def assess(state, design, markdown, now, expected_cycle=None, expected_claim=None):
    schema = json.loads((ROOT / 'research/LOOP_STATE.schema.json').read_text(encoding='utf-8'))
    require(not list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(state)),
            'STATE_SCHEMA_CONFLICT')
    require(state['next_actor'] == ACTOR[state['status']], 'ACTOR_CONFLICT')
    require(all(design.get(k) == state[k] for k in ('cycle_id', 'experiment_id')), 'DESIGN_IDENTITY_CONFLICT')
    require(re.findall(r'^Experiment ID:[ \t]*(\S+)[ \t]*$', markdown, re.M) == [state['experiment_id']],
            'MARKDOWN_IDENTITY_CONFLICT')
    statuses = re.findall(r'^Status:[ \t]*(\S+)[ \t]*$', markdown, re.M)
    require(len(statuses) == 1, 'MARKDOWN_STATUS_MISSING')
    status, auth = state['status'], state['authorization']
    require(state['execution_worktree'] is None or 'SafeVLA_loop_control' not in state['execution_worktree'],
            'CONTROL_WORKTREE_FORBIDDEN')
    prefix = 'research/handoffs/' + state['cycle_id'] + '/'
    require({prefix + name for name in BASE_OUTPUTS}.issubset(state['required_outputs']) and
            all(p.startswith(prefix) and safe_path(p) for p in state['required_outputs']),
            'OUTPUT_PATH_CONFLICT')
    if auth['scope'] == 'CONTROL_ONLY_NOOP':
        require(auth['max_gpu'] == auth['max_episodes'] == 0 and state['execution_worktree'] is None,
                'NOOP_RESOURCE_CONFLICT')
    if expected_cycle is not None:
        require(state['cycle_id'] == expected_cycle, 'STALE_CYCLE')
    if expected_claim is not None:
        require(state['claim_id'] == expected_claim, 'STALE_CLAIM')
    if status == 'PI_REVIEW':
        require(auth['status'] == 'NOT_AUTHORIZED', 'REVIEW_AUTH_CONFLICT')
        staged = state['instruction_commit'] is None and state['claim_id'] is None
        allowed = ('DRAFT', 'APPROVED') if staged else ('DRAFT',)
        require(statuses[0] in allowed and design.get('status') in allowed, 'REVIEW_DESIGN_CONFLICT')
        return 'BLOCKED', 'PI_REVIEW_NOT_EXECUTABLE'
    require(auth['status'] == 'APPROVED' and design.get('status') == statuses[0] == 'APPROVED',
            'APPROVAL_DESIGN_CONFLICT')
    require(all(design.get(k) for k in DESIGN_FIELDS), 'INCOMPLETE_DESIGN')
    require(design.get('resources') == {k: auth[k] for k in ('max_gpu', 'max_episodes')} and
            design.get('required_outputs') == state['required_outputs'], 'DESIGN_SCOPE_CONFLICT')
    require(auth['approved_by'] == 'PI', 'APPROVER_CONFLICT')
    approved = datetime.fromisoformat(auth['approved_at_utc'].replace('Z', '+00:00'))
    expiry = datetime.fromisoformat(auth['expires_at_utc'].replace('Z', '+00:00'))
    require(approved <= now and approved < expiry, 'APPROVAL_TIME_CONFLICT')
    is_approval = status.startswith('APPROVED_FOR_')
    require((state['claim_id'] is None and state['instruction_commit'] is None) if is_approval
            else bool(state['claim_id'] and state['instruction_commit']), 'CLAIM_BINDING_CONFLICT')
    if status in ('BLOCKED', 'INVALID', 'ABORTED', 'AWAITING_PI_REVIEW'):
        return 'BLOCKED', 'TERMINAL_OR_REVIEW_STATE; HISTORICAL_CLAIM_NOT_REUSABLE'
    if now >= expiry:
        return 'BLOCKED', 'APPROVAL_EXPIRED'
    if not is_approval:
        return 'BLOCKED', 'EXISTING_CLAIM; NO_AUTOMATIC_RESUME'
    return 'OBSERVED', 'APPROVAL_OBSERVED_ONLY; CI_CLAIM_RUNTIME_GATES_NOT_CHECKED'


def query(api=None, root=ROOT, local=False, now=None, expected_cycle=None, expected_claim=None):
    evidence = {'repository': 'whatcanidowhat/SafeVLA', 'branch': 'research-loop',
                'scope': 'STATE_AND_DESIGN_ONLY; NOT_FULL_HISTORY_OR_ARTIFACT_AUDIT'}
    try:
        if local:
            def git(*args):
                return subprocess.check_output(['git', '-C', str(root), *args],
                                               stderr=subprocess.DEVNULL, timeout=10)
            revision = git('rev-parse', 'HEAD').decode().strip()
            raw = [git('show', revision + ':' + p) for p in PATHS]
            evidence['remote'] = 'REMOTE_UNVERIFIED'
        else:
            revision = api.head('research-loop')
            raw = [api.file(revision, p) for p in PATHS]
            require(api.head('research-loop') == revision, 'REMOTE_HEAD_CHANGED')
            evidence['remote'] = 'VERIFIED_AT_QUERY'
        evidence['sha'] = revision
        state, design = json.loads(raw[0]), json.loads(raw[1])
        verdict, reason = assess(state, design, raw[2].decode('utf-8'),
                                 now or datetime.now(timezone.utc), expected_cycle, expected_claim)
        evidence.update({k: state[k] for k in ('status', 'next_actor', 'experiment_id', 'cycle_id',
                                             'claim_id', 'instruction_commit')})
        evidence['authorization'] = state['authorization']
        if local:
            return report('REMOTE_UNVERIFIED', evidence, 'LOCAL_COMMITTED_SNAPSHOT_ONLY; ' + reason)
        return report(verdict, evidence, reason, 'Report observation; follow existing PI protocol separately')
    except Stop as e:
        evidence['error_kind'] = e.verdict
        return report('UNKNOWN', evidence, e.reason)


def main(argv=None):
    def run():
        p = Parser(description=__doc__)
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument('--local', action='store_true')
        group.add_argument('--remote-readonly', action='store_true')
        p.add_argument('--root', type=Path, default=ROOT)
        p.add_argument('--timeout', type=positive, default=60)
        p.add_argument('--expected-cycle')
        p.add_argument('--expected-claim')
        a = p.parse_args(argv)
        return query(None if a.local else GitHub(a.timeout), root=a.root, local=a.local,
                     expected_cycle=a.expected_cycle, expected_claim=a.expected_claim)
    return emit(guarded(run))


if __name__ == '__main__':
    raise SystemExit(main())
