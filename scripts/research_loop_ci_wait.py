#!/usr/bin/env python3
"""Bounded exact-revision CI observation; never dispatches, reruns, claims or executes."""
import re
import time
from urllib.parse import quote, urlencode

from research_loop_readonly import GitHub, Parser, SHA, Stop, emit, guarded, positive, report, require


def wait_ci(api, sha, workflow, event, branch='research-loop', attempt=None,
            timeout=600, interval=15, clock=time.monotonic, sleep=time.sleep):
    evidence = {}
    try:
        require(isinstance(sha, str) and bool(SHA.fullmatch(sha)), 'FULL_SHA_REQUIRED')
        require(bool(re.fullmatch(r'(?:[1-9][0-9]*|[A-Za-z0-9_.-]+\.ya?ml)', workflow)),
                'WORKFLOW_ID_OR_FILENAME_REQUIRED')
        require(event in ('push', 'pull_request', 'workflow_dispatch'), 'INVALID_EXPECTED_EVENT')
        require(isinstance(branch, str) and 0 < len(branch) <= 200 and
                not any(c.isspace() for c in branch), 'INVALID_BRANCH')
        require(attempt is None or type(attempt) is int and attempt > 0, 'INVALID_ATTEMPT')
        evidence.update(sha=sha, branch=branch, event=event)
        deadline = clock() + timeout
        require(api.head(branch) == sha, 'REMOTE_HEAD_MISMATCH')
        wf = api.get('/actions/workflows/' + quote(workflow, safe=''))
        wid = wf.get('id')
        require(type(wid) is int and isinstance(wf.get('name'), str) and
                isinstance(wf.get('path'), str), 'INVALID_WORKFLOW')
        require(str(wid) == workflow if workflow.isdigit()
                else wf['path'] == '.github/workflows/' + workflow, 'WORKFLOW_MISMATCH')
        evidence.update(workflow_id=wid, workflow=wf['name'], workflow_path=wf['path'])
        pinned_id = pinned_attempt = None
        polls = 0
        while True:
            if clock() >= deadline:
                return report('TIMED_OUT', evidence, 'WAIT_DEADLINE_EXCEEDED')
            require(api.head(branch) == sha, 'REMOTE_HEAD_CHANGED')
            q = urlencode({'head_sha': sha, 'branch': branch, 'event': event, 'per_page': 100})
            listing = api.get('/actions/workflows/' + str(wid) + '/runs?' + q)
            runs, count = listing.get('workflow_runs'), listing.get('total_count')
            require(isinstance(runs, list) and type(count) is int and count == len(runs),
                    'INCOMPLETE_RUN_LIST_OR_PAGINATION')
            polls += 1
            evidence['polls'] = polls
            if count == 0:
                return report('NOT_FOUND', evidence, 'NO_VISIBLE_MATCHING_RUN')
            require(count == 1, 'MULTIPLE_MATCHING_RUNS')
            summary = runs[0]
            rid = summary.get('id')
            require(type(rid) is int and rid > 0, 'INVALID_RUN_ID')
            run = api.get('/actions/runs/' + str(rid))
            for item in (summary, run):
                require(item.get('id') == rid and item.get('head_sha') == sha and
                        item.get('head_branch') == branch and item.get('event') == event and
                        item.get('workflow_id') == wid and item.get('name') == wf['name'] and
                        item.get('path') == wf['path'], 'RUN_IDENTITY_MISMATCH')
            actual_attempt = run.get('run_attempt')
            require(type(actual_attempt) is int and actual_attempt > 0 and
                    summary.get('run_attempt') == actual_attempt, 'ATTEMPT_RACE_OR_MISSING')
            require(actual_attempt == (attempt or 1), 'ATTEMPT_MISMATCH; EXPLICIT_RERUN_ATTEMPT_REQUIRED')
            if pinned_id is not None:
                require(rid == pinned_id and actual_attempt == pinned_attempt, 'RUN_OR_ATTEMPT_CHANGED')
            pinned_id, pinned_attempt = rid, actual_attempt
            evidence.update(run_id=rid, attempt=actual_attempt,
                            run_url=f'https://github.com/whatcanidowhat/SafeVLA/actions/runs/{rid}',
                            status=run.get('status'), conclusion=run.get('conclusion'))
            require(api.head(branch) == sha, 'REMOTE_HEAD_CHANGED')
            if clock() >= deadline:
                return report('TIMED_OUT', evidence, 'WAIT_DEADLINE_EXCEEDED')
            if run.get('status') == 'completed':
                outcomes = {'success': 'SUCCESS', 'failure': 'FAILURE', 'cancelled': 'CANCELLED',
                            'timed_out': 'TIMED_OUT', 'action_required': 'FAILURE',
                            'startup_failure': 'FAILURE'}
                verdict = outcomes.get(run.get('conclusion'), 'UNKNOWN')
                return report(verdict, evidence, None if verdict == 'SUCCESS' else 'CI_NOT_SUCCESS',
                              'Report CI evidence only; execution gates remain separate')
            require(run.get('status') in ('queued', 'in_progress', 'waiting', 'pending', 'requested')
                    and run.get('conclusion') is None, 'UNKNOWN_RUN_STATE')
            sleep(min(interval, max(0, deadline - clock())))
    except Stop as e:
        return report(e.verdict, evidence, e.reason)


def main(argv=None):
    def run():
        p = Parser(description=__doc__)
        p.add_argument('--sha', required=True)
        p.add_argument('--workflow', required=True)
        p.add_argument('--event', required=True)
        p.add_argument('--branch', default='research-loop')
        p.add_argument('--attempt', type=int)
        p.add_argument('--timeout', type=positive, default=600)
        p.add_argument('--interval', type=positive, default=15)
        a = p.parse_args(argv)
        return wait_ci(GitHub(a.timeout), a.sha, a.workflow, a.event, a.branch,
                       a.attempt, a.timeout, a.interval)
    return emit(guarded(run))


if __name__ == '__main__':
    raise SystemExit(main())
