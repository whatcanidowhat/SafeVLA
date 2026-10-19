"""Offline adversarial tests. No real credentials, GitHub writes or experiments."""
import base64
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error

import research_loop_ci_wait as ci
import research_loop_status as status
from research_loop_readonly import API, GitHub, NoRedirect, Stop, emit, guarded, report
from validate_research_loop import Invalid, View, audit_tree
from test_research_loop import initial, design

SHA = 'a' * 40
OTHER = 'b' * 40
NOW = datetime(2026, 10, 17, tzinfo=timezone.utc)
WF = {'id': 123, 'name': 'research-loop-validate', 'path': '.github/workflows/research-loop-validate.yml'}


def run_state(state='completed', conclusion='success', attempt=1):
    return {'id': 7, 'head_sha': SHA, 'head_branch': 'research-loop', 'event': 'push',
            'workflow_id': 123, 'name': WF['name'], 'path': WF['path'],
            'run_attempt': attempt, 'status': state, 'conclusion': conclusion}


class Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class CIAPI:
    def __init__(self, runs=None, head=SHA, count=1, error=None):
        self.runs = copy.deepcopy(runs or [run_state()])
        self.remote = head
        self.count, self.error = count, error
        self.index, self.calls = 0, []
        self.detail_patch = {}
        self.head_reads = 0
        self.move_at = None

    def head(self, branch):
        self.calls.append(('HEAD', branch))
        self.head_reads += 1
        return OTHER if self.move_at == self.head_reads else self.remote

    def get(self, path):
        self.calls.append(('GET', path))
        if self.error:
            raise self.error
        if '/runs?' in path:
            n = min(self.index, len(self.runs) - 1)
            self.current = copy.deepcopy(self.runs[n])
            self.index += 1
            return {'total_count': self.count, 'workflow_runs': [self.current] * self.count}
        if path.startswith('/actions/runs/'):
            return {**self.current, **self.detail_patch}
        return copy.deepcopy(WF)


def wait(api, **kwargs):
    clock = Clock()
    return ci.wait_ci(api, SHA, 'research-loop-validate.yml', 'push',
                      timeout=kwargs.pop('timeout', 30), interval=10, clock=clock,
                      sleep=clock.sleep, **kwargs)


def snapshot(state='PI_REVIEW'):
    s = initial()
    s['updated_by'] = 'PI'
    if state != 'PI_REVIEW':
        s.update(status=state, next_actor=status.ACTOR[state])
        s['authorization'].update(status='APPROVED', approved_by='PI',
                                  approved_at_utc='2026-10-16T00:00:00Z',
                                  expires_at_utc='2026-10-18T00:00:00Z')
        if not state.startswith('APPROVED_FOR_'):
            s.update(claim_id='old-claim-123', instruction_commit=OTHER)
    d = design(s, 'DRAFT' if state == 'PI_REVIEW' else 'APPROVED')
    md = f"Experiment ID: {s['experiment_id']}\nStatus: {d['status']}\n"
    return s, d, md


class StateAPI:
    def __init__(self, state='PI_REVIEW'):
        self.data = snapshot(state)
        self.refs, self.heads = [], 0
        self.move = False

    def head(self, branch):
        self.heads += 1
        return OTHER if self.move and self.heads == 2 else SHA

    def file(self, revision, path):
        self.refs.append((revision, path))
        i = status.PATHS.index(path)
        return (self.data[i] if i == 2 else json.dumps(self.data[i])).encode()


class StateTests(unittest.TestCase):
    def test_future_approval_and_scope_conflicts(self):
        for change in (lambda s: s['authorization'].update(approved_at_utc='2026-10-17T01:00:00Z'),
                       lambda s: s.update(execution_worktree='/x/SafeVLA_loop_control'),
                       lambda s: s['authorization'].update(max_gpu=1),
                       lambda s: s['required_outputs'].pop(),
                       lambda s: s['authorization'].update(approved_by='CODEX')):
            api = StateAPI('APPROVED_FOR_CODEX')
            change(api.data[0])
            self.assertEqual(status.query(api, now=NOW)['VERDICT'], 'UNKNOWN')

    def test_missing_state_or_network_failure(self):
        api = StateAPI()
        with patch.object(api, 'file', side_effect=Stop('NETWORK_ERROR', 'TLS')):
            result = status.query(api, now=NOW)
            self.assertEqual(result['VERDICT'], 'UNKNOWN')
            self.assertEqual(result['EVIDENCE']['error_kind'], 'NETWORK_ERROR')
        with patch.object(api, 'file', return_value=b'not-json'):
            self.assertEqual(guarded(lambda: status.query(api, now=NOW))['VERDICT'], 'UNKNOWN')

    def test_all_states_never_grant_execution(self):
        for name in status.ACTOR:
            with self.subTest(state=name):
                result = status.query(StateAPI(name), now=NOW)
                self.assertFalse(result['EVIDENCE']['CAN_EXECUTE'])
                self.assertEqual(result['VERDICT'], 'OBSERVED' if name.startswith('APPROVED_FOR_') else 'BLOCKED')

    def test_expired_approval(self):
        api = StateAPI('APPROVED_FOR_CODEX')
        api.data[0]['authorization']['expires_at_utc'] = '2026-10-16T01:00:00Z'
        result = status.query(api, now=NOW)
        self.assertEqual(result['BLOCKER'], 'APPROVAL_EXPIRED')

    def test_conflicts_and_missing_evidence(self):
        mutations = [lambda s, d: s.update(next_actor='CODEX'),
                     lambda s, d: s.update(status='NOT_A_STATE'),
                     lambda s, d: d.update(cycle_id='another-cycle'),
                     lambda s, d: d.pop('status'),
                     lambda s, d: s['authorization'].update(status='APPROVED')]
        for change in mutations:
            api = StateAPI()
            change(api.data[0], api.data[1])
            self.assertEqual(status.query(api, now=NOW)['VERDICT'], 'UNKNOWN')

    def test_claim_is_not_reusable(self):
        for name in ('BLOCKED', 'CODEX_RUNNING', 'OPERATOR_RUNNING'):
            api = StateAPI(name)
            result = status.query(api, now=NOW)
            self.assertEqual(result['VERDICT'], 'BLOCKED')
            result = status.query(api, now=NOW, expected_claim='different')
            self.assertEqual(result['BLOCKER'], 'STALE_CLAIM')

    def test_approval_cannot_contain_old_claim(self):
        api = StateAPI('APPROVED_FOR_CODEX')
        api.data[0].update(claim_id='old-claim-123', instruction_commit=OTHER)
        self.assertEqual(status.query(api, now=NOW)['BLOCKER'], 'CLAIM_BINDING_CONFLICT')

    def test_remote_moves_and_revision_pinning(self):
        api = StateAPI()
        api.move = True
        self.assertEqual(status.query(api, now=NOW)['BLOCKER'], 'REMOTE_HEAD_CHANGED')
        self.assertTrue(all(ref == SHA for ref, _ in api.refs))

    def test_repeated_queries_recheck_without_mutation(self):
        api = StateAPI('BLOCKED')
        before = copy.deepcopy(api.data)
        one, two = status.query(api, now=NOW), status.query(api, now=NOW)
        self.assertEqual(one, two)
        self.assertEqual(before, api.data)
        self.assertEqual(api.heads, 4)

    def test_local_uses_committed_snapshot_never_network(self):
        s, d, md = snapshot('APPROVED_FOR_CODEX')
        values = [SHA.encode(), json.dumps(s).encode(), json.dumps(d).encode(), md.encode()]
        with patch.object(status.subprocess, 'check_output', side_effect=values) as command:
            result = status.query(local=True, now=NOW)
        self.assertEqual(result['VERDICT'], 'REMOTE_UNVERIFIED')
        self.assertFalse(result['EVIDENCE']['CAN_EXECUTE'])
        self.assertEqual([c.args[0][3] for c in command.call_args_list], ['rev-parse', 'show', 'show', 'show'])


class CITests(unittest.TestCase):
    def test_pending_then_success(self):
        api = CIAPI([run_state('queued', None), run_state('in_progress', None), run_state()])
        result = wait(api)
        self.assertEqual(result['VERDICT'], 'SUCCESS')
        self.assertEqual(result['EVIDENCE']['polls'], 3)
        self.assertFalse(result['EVIDENCE']['CAN_EXECUTE'])

    def test_terminal_outcomes(self):
        for conclusion, verdict in [('failure', 'FAILURE'), ('cancelled', 'CANCELLED'),
                                    ('timed_out', 'TIMED_OUT'), ('neutral', 'UNKNOWN'),
                                    ('skipped', 'UNKNOWN'), (None, 'UNKNOWN')]:
            with self.subTest(conclusion=conclusion):
                self.assertEqual(wait(CIAPI([run_state(conclusion=conclusion)]))['VERDICT'], verdict)

    def test_identity_errors(self):
        for field, value in [('head_sha', OTHER), ('head_branch', 'other'), ('event', 'pull_request'),
                             ('workflow_id', 999), ('name', 'another'), ('path', 'other.yml'), ('id', 999)]:
            api = CIAPI()
            api.detail_patch[field] = value
            with self.subTest(field=field):
                self.assertEqual(wait(api)['VERDICT'], 'UNKNOWN')

    def test_wrong_remote_and_movement(self):
        self.assertEqual(wait(CIAPI(head=OTHER))['BLOCKER'], 'REMOTE_HEAD_MISMATCH')
        api = CIAPI()
        api.move_at = 3
        self.assertEqual(wait(api)['BLOCKER'], 'REMOTE_HEAD_CHANGED')

    def test_no_run_or_multiple_runs(self):
        self.assertEqual(wait(CIAPI(count=0))['VERDICT'], 'NOT_FOUND')
        self.assertEqual(wait(CIAPI(count=2))['BLOCKER'], 'MULTIPLE_MATCHING_RUNS')

    def test_rerun_requires_explicit_current_attempt(self):
        self.assertEqual(wait(CIAPI([run_state(attempt=2)]))['VERDICT'], 'UNKNOWN')
        self.assertEqual(wait(CIAPI([run_state(attempt=2)]), attempt=2)['VERDICT'], 'SUCCESS')
        self.assertEqual(wait(CIAPI([run_state(attempt=2)]), attempt=1)['VERDICT'], 'UNKNOWN')

    def test_attempt_changes_during_poll_or_between_list_and_detail(self):
        api = CIAPI([run_state('queued', None), run_state(attempt=2)])
        self.assertEqual(wait(api)['VERDICT'], 'UNKNOWN')
        api = CIAPI()
        api.detail_patch['run_attempt'] = 2
        self.assertEqual(wait(api)['BLOCKER'], 'ATTEMPT_RACE_OR_MISSING')

    def test_new_run_replaces_old_run(self):
        new = run_state()
        new['id'] = 8
        self.assertEqual(wait(CIAPI([run_state('queued', None), new]))['BLOCKER'], 'RUN_OR_ATTEMPT_CHANGED')

    def test_timeout_is_bounded(self):
        result = wait(CIAPI([run_state('queued', None)]), timeout=25)
        self.assertEqual(result['VERDICT'], 'TIMED_OUT')
        self.assertEqual(result['EVIDENCE']['polls'], 3)

    def test_errors_never_retry(self):
        for verdict in ('NETWORK_ERROR', 'PERMISSION_DENIED', 'RATE_LIMITED', 'NOT_FOUND'):
            api = CIAPI(error=Stop(verdict, 'SYNTHETIC'))
            result = wait(api)
            self.assertEqual(result['VERDICT'], verdict)
            self.assertEqual(len([c for c in api.calls if c[0] == 'GET']), 1)

    def test_incomplete_pagination(self):
        api = CIAPI()
        original = api.get
        def truncated(path):
            value = original(path)
            if '/runs?' in path:
                value['total_count'] = 101
            return value
        api.get = truncated
        self.assertEqual(wait(api)['BLOCKER'], 'INCOMPLETE_RUN_LIST_OR_PAGINATION')

    def test_repeated_success_has_no_side_effects(self):
        api = CIAPI()
        self.assertEqual(wait(api)['VERDICT'], 'SUCCESS')
        self.assertEqual(wait(api)['VERDICT'], 'SUCCESS')
        self.assertTrue(all(method in ('HEAD', 'GET') for method, _ in api.calls))

    def test_cli_emits_one_json_and_exit_code(self):
        api = CIAPI([run_state(conclusion='cancelled')])
        out = io.StringIO()
        with patch.object(ci, 'GitHub', return_value=api), contextlib.redirect_stdout(out):
            code = ci.main(['--sha', SHA, '--workflow', '123', '--event', 'push'])
        self.assertEqual(code, 4)
        self.assertEqual(len(out.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(out.getvalue())['VERDICT'], 'CANCELLED')

    def test_invalid_cli_is_structured(self):
        for args in ([], ['--sha', SHA, '--workflow', '123', '--event', 'push', '--timeout', 'nan']):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = ci.main(args)
            self.assertEqual(code, 64)
            self.assertEqual(json.loads(out.getvalue())['VERDICT'], 'INVALID_INPUT')


class Response(io.BytesIO):
    def __init__(self, content, url):
        super().__init__(json.dumps(content).encode())
        self.url = url

    def geturl(self):
        return self.url


class TransportTests(unittest.TestCase):
    def test_stalled_dns_is_bounded(self):
        release = threading.Event()
        class Opener:
            def open(self, req, timeout):
                release.wait(2)
                raise TimeoutError
        start = time.monotonic()
        try:
            result = guarded(lambda: GitHub(timeout=0.03, opener=Opener()).get('/x'))
            self.assertEqual(result['VERDICT'], 'TIMED_OUT')
            self.assertLess(time.monotonic() - start, 0.5)
        finally:
            release.set()

    def test_invalid_json_and_redirect_origin_are_not_success(self):
        class Opener:
            def open(self, req, timeout):
                return Response({}, 'https://other.invalid')
        self.assertEqual(guarded(lambda: GitHub(opener=Opener()).get('/x'))['VERDICT'], 'UNKNOWN')

    def test_tls_context_verifies_certificates(self):
        import ssl
        import research_loop_readonly as shared
        original = ssl.create_default_context
        contexts = []
        def context():
            ctx = original()
            contexts.append(ctx)
            return ctx
        with patch.object(shared.ssl, 'create_default_context', side_effect=context):
            GitHub()
        self.assertEqual(contexts[0].verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(contexts[0].check_hostname)

    def test_output_size_fail_closed(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = emit(report('SUCCESS', {'long': 'x' * 3000}))
        self.assertEqual(code, 2)
        self.assertLess(len(out.getvalue().encode()), 2048)
        self.assertFalse(json.loads(out.getvalue())['EVIDENCE']['CAN_EXECUTE'])

    def test_exact_whitelist_does_not_allow_other_scripts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            subprocess.run(['git', 'init', '-q', str(root)], check=True, capture_output=True)
            (root / 'scripts').mkdir()
            for name in ('research_loop_status.py', 'research_loop_ci_wait.py',
                         'research_loop_readonly.py', 'test_research_loop_readonly.py'):
                (root / 'scripts' / name).write_text('# fixture\n', encoding='utf-8')
            audit_tree(View(root))
            forbidden = root / 'scripts/unauthorized.py'
            forbidden.write_text('# fixture\n')
            with self.assertRaisesRegex(Invalid, 'outside control whitelist'):
                audit_tree(View(root))
            forbidden.unlink()
            (root / 'scripts/research_loop_status.py').write_bytes(b'x' * (1024 * 1024 + 1))
            with self.assertRaisesRegex(Invalid, 'OVERSIZED_ARTIFACT'):
                audit_tree(View(root))

    def test_get_only_fixed_host_and_blob_hash(self):
        raw = b'hello\n'
        value = {'type': 'file', 'path': 'research/example.md', 'encoding': 'base64',
                 'content': base64.b64encode(raw).decode(), 'size': len(raw),
                 'sha': hashlib.sha1(b'blob 6\0' + raw).hexdigest()}
        class Opener:
            def open(self, req, timeout):
                self.request = req
                return Response(value, req.full_url)
        opener = Opener()
        self.assertEqual(GitHub(opener=opener).file(SHA, 'research/example.md'), raw)
        self.assertEqual(opener.request.get_method(), 'GET')
        self.assertTrue(opener.request.full_url.startswith(API + '/contents/'))
        value['sha'] = OTHER
        with self.assertRaisesRegex(Stop, 'FILE_BLOB_MISMATCH'):
            GitHub(opener=opener).file(SHA, 'research/example.md')

    def test_http_errors_sanitized_and_not_retried(self):
        for code, headers, verdict in [(401, {}, 'PERMISSION_DENIED'), (403, {}, 'PERMISSION_DENIED'),
                                      (403, {'X-RateLimit-Remaining': '0'}, 'RATE_LIMITED'),
                                      (429, {}, 'RATE_LIMITED'), (404, {}, 'NOT_FOUND'),
                                      (503, {}, 'NETWORK_ERROR')]:
            class Opener:
                calls = 0
                def open(self, req, timeout):
                    self.calls += 1
                    raise urllib.error.HTTPError(req.full_url, code, 'SENSITIVE_SENTINEL', headers, None)
            opener = Opener()
            result = guarded(lambda: GitHub(opener=opener).get('/actions/runs/7'))
            self.assertEqual(result['VERDICT'], verdict)
            self.assertNotIn('SENSITIVE_SENTINEL', json.dumps(result))
            self.assertEqual(opener.calls, 1)

    def test_tls_proxy_timeout_fail_closed(self):
        for error in (urllib.error.URLError('TLS certificate invalid'), TimeoutError('secret proxy')):
            class Opener:
                def open(self, req, timeout):
                    raise error
            self.assertEqual(guarded(lambda: GitHub(opener=Opener()).get('/x'))['VERDICT'], 'NETWORK_ERROR')

    def test_redirect_refused(self):
        with self.assertRaises(Stop):
            NoRedirect().redirect_request(None, None, 302, None, None, 'https://other.invalid')


def benchmark():
    """Five offline control tasks; actual LLM/token fields deliberately unmeasured."""
    scenarios = {'immediate_success': [run_state()],
                 'queued_success': [run_state('queued', None), run_state('in_progress', None), run_state()],
                 'failure': [run_state('queued', None), run_state(conclusion='failure')],
                 'cancelled': [run_state(conclusion='cancelled')],
                 'timeout': [run_state('queued', None)]}
    rows = []
    for name, states in scenarios.items():
        api = CIAPI(states)
        start = time.perf_counter()
        result = wait(api)
        elapsed = time.perf_counter() - start
        polls = result['EVIDENCE'].get('polls', 0)
        rows.append({'task': name, 'verdict': result['VERDICT'],
                     'synthetic_polls': polls, 'synthetic_api_reads': len(api.calls),
                     'python_harness_wall_seconds': elapsed,
                     'modeled_before_llm_tool_calls': polls, 'modeled_after_llm_tool_calls': 1,
                     'modeled_before_wait_interactions': max(0, polls - 1),
                     'modeled_after_wait_interactions': 0,
                     'actual_llm_tool_calls_before_after': 'NOT_MEASURED',
                     'actual_input_output_tokens_before_after': 'NOT_MEASURED',
                     'actual_llm_wait_turns_before_after': 'NOT_MEASURED',
                     'real_end_to_end_wall_before_after': 'NOT_MEASURED',
                     'unsafe_execution_permission': result['EVIDENCE']['CAN_EXECUTE']})
    return {'method': 'Offline fake GitHub and virtual sleeps. Modeled baseline: one LLM tool call per poll; not an observed session. Python timing excludes real network/wait latency.', 'tasks': rows}


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--benchmark']:
        print(json.dumps(benchmark(), indent=2))
    else:
        unittest.main(verbosity=2)
