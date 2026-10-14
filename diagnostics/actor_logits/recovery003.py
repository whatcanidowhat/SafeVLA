"""P2-003 control, immutable OFF reuse, and descriptive cross-cycle comparison.

No model construction or Episode execution. Network access is direct HTTPS only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from .storage import read_json, require, sha256, write_json

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = 'EXP-ACTOR-RESET-ON200-003'
CYCLE = 'actor-reset-on200-off002-reuse-003-20261014'
OFF = Path('/nvme2/user/qyy/SafeVLA_p2_002_results_20261014/OFF200')
OUTPUT = Path('/nvme2/user/qyy/SafeVLA_p2_003_on200_results_20261014')
EVIDENCE = Path('/nvme2/user/qyy/SafeVLA_p2_003_launch_20261014')
NETWORK_WITNESSES = Path('/nvme2/user/qyy/SafeVLA_p2_003_network_claim_20261014')
OLD_CODE = '06cde9adeb91d47c062a11e72e90c7edf56fb398'
OLD_CLAIM = '918170d26cca2fc385fb6f8fd87d0109b4a167b2'
OLD_RESULT = '88453def76051736edce2e8887f26b08d5b0c5e6'
PINS = {
    'RUN_MANIFEST.json': 'd94c3c905a023ac1fbfa31a7b866d9b84a9ca5b4c5d3147a179a220cfa5f1758',
    'episode_metrics.csv': '891973cbfc5d299ad76951efe6170dc12962729f9b991b81ba925577235a4296',
    'ARTIFACT_INDEX.json': '96e10b0fa5a5e7c8edd04289267c9ca2c7cd63679963e4dab7ee01647dd3be19',
}
MANIFEST_SHA = '8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0'
CHANGED_OLD_FILES = {'diagnostics/actor_logits/evaluate.py', 'diagnostics/actor_logits/analyze.py'}
NEW_SOURCE_FILES = {'diagnostics/actor_logits/recovery003.py',
                    'diagnostics/actor_logits/run_p2_on200_only.sh',
                    'diagnostics/actor_logits/tests/test_recovery003.py'}


def direct_environment():
    env = dict(os.environ)
    for key in list(env):
        if key.lower() in {'http_proxy', 'https_proxy', 'all_proxy'}:
            del env[key]
    env['NO_PROXY'] = '*'
    env['no_proxy'] = '*'
    env['GIT_TERMINAL_PROMPT'] = '0'
    return env


def git(root, *args):
    result = subprocess.run(['git', '-c', 'http.proxy=', '-C', str(root), *args],
                            env=direct_environment(), capture_output=True, text=True, timeout=60)
    require(result.returncode == 0, 'Git read failed (credential-bearing stderr suppressed)')
    return result.stdout.strip()


def ci_success(sha):
    require(len(sha) == 40 and all(c in '0123456789abcdef' for c in sha), 'Invalid CI SHA')
    request = urllib.request.Request(
        'https://api.github.com/repos/whatcanidowhat/SafeVLA/actions/runs?head_sha='+sha+'&per_page=100',
        headers={'Accept':'application/vnd.github+json','User-Agent':'SafeVLA-P2-003'})
    # Explicit empty ProxyHandler ignores any inherited interactive SSH proxy.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=30) as response:
        runs = json.load(response).get('workflow_runs', [])
    matches = [r for r in runs if r.get('head_sha') == sha
               and r.get('name') == 'research-loop-validate' and r.get('event') == 'push'
               and r.get('status') == 'completed' and r.get('conclusion') == 'success']
    require(matches, 'Exact push control CI is not SUCCESS')
    return {'head_sha':sha,'run_id':matches[0]['id'],'url':matches[0]['html_url'],'conclusion':'success'}


def network_probe(control):
    """One live read, with no retries or permission inference from reachability."""
    start = time.monotonic()
    report = {'observed_utc':datetime.now(timezone.utc).isoformat(),
              'transport':'DIRECT_HTTPS_NO_PROXY','status':'FAIL','authorizes_execution':False}
    # Hash the non-secret connection tuple, rather than copying environment data.
    connection = os.environ.get('SSH_CONNECTION', '')
    report['ssh_session_sha256'] = hashlib.sha256(connection.encode()).hexdigest() if connection else None
    try:
        local = git(control, 'rev-parse', 'HEAD')
        listing = git(control, 'ls-remote', '--heads', 'origin', 'refs/heads/research-loop').split()
        require(len(listing) == 2 and listing[0] == local, 'Remote control HEAD differs')
        report.update(status='PASS',head_sha=local,ci=ci_success(local))
    except Exception as exc:
        report['error_type'] = type(exc).__name__
        # Never record Git stderr, environment values, credentials or URLs with auth.
    report['elapsed_seconds'] = time.monotonic()-start
    return report


def require_disconnect_witnesses(claim):
    reports = [read_json(NETWORK_WITNESSES/claim/name) for name in ['session_a.json','session_b.json']]
    for report in reports:
        require(report['status']=='PASS' and report['transport']=='DIRECT_HTTPS_NO_PROXY'
                and report['head_sha']==claim and report['ci']['head_sha']==claim
                and report['ci']['conclusion']=='success', 'Invalid cross-session claim network evidence')
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(report['observed_utc'])).total_seconds()
        require(0 <= age <= 1800, 'Cross-session network evidence must be fresh (30 minutes)')
    require(reports[0]['ssh_session_sha256'] and reports[1]['ssh_session_sha256']
            and reports[0]['ssh_session_sha256'] != reports[1]['ssh_session_sha256'],
            'Two distinct disconnected/reconnected SSH probe sessions required')
    return reports


def verify_off():
    for name, digest in PINS.items():
        require(sha256(OFF/name) == digest, 'Immutable OFF identity mismatch: '+name)
    manifest = read_json(OFF/'RUN_MANIFEST.json')
    p = manifest['provenance']
    require(p['git_commit'] == OLD_CODE and p['actor_reset'] == 'off', 'Wrong historical OFF source')
    require(p['control_evidence']['claim_commit'] == OLD_CLAIM, 'Wrong historical OFF claim')
    require(p['manifest_sha256'] == MANIFEST_SHA, 'Wrong OFF TaskSpec manifest')
    index = read_json(OFF/'ARTIFACT_INDEX.json')['artifacts']
    require(len(index) == 1203, 'OFF raw index count mismatch')
    # The original strict validator also rehashes every file and checks all tensors,
    # official metrics, task order, action receipts, counts and boundary snapshots.
    from .validate import validate_run
    report = validate_run(OFF, 200)
    require(report['status'] == 'PASS', 'OFF stored evidence validation failed')
    require(report['decisions_checked'] == 19270, 'OFF decision count differs')
    es = report['outcomes']
    require(sum(e['success'] for e in es) == 169 and sum(e['cost'] for e in es) == 177,
            'OFF official outcomes differ')
    return {'status':'PASS','observed_utc':datetime.now(timezone.utc).isoformat(),
            'reference_result_commit':OLD_RESULT,'reference_claim_commit':OLD_CLAIM,
            'reference_code_commit':OLD_CODE,'server_root':str(OFF),'identity_sha256':PINS,
            'episodes_checked':200,'decisions_checked':19270,'indexed_artifacts_checked':1203,
            'OFF_written':False,'GPU_EQUIVALENCE':'NOT_RUN_PI_WAIVED'}


def science_parity(old_source, new_source):
    """Exact old scientific file hashes; only explicit orchestration files differ."""
    require(set(old_source) <= set(new_source), 'Historical source files removed')
    require(set(new_source)-set(old_source) <= NEW_SOURCE_FILES, 'Unreviewed new executable sources')
    for name, digest in old_source.items():
        if name not in CHANGED_OLD_FILES:
            require(new_source[name] == digest, 'Scientific source drift: '+name)


def strict_control(control):
    from .evaluate import codex_control_evidence, parse_args, code_identity
    state = read_json(control/'research/LOOP_STATE.json')
    claim = git(control,'rev-parse','HEAD')
    args = parse_args(['--actor-reset','on','--output',str(OUTPUT/'ON200'),
        '--checkpoint','/home/amax/public/datasets/qyy/checkpoints/safe_objnav.pt',
        '--shuffle','--test-augmentation','--execute-authorized','--pi-waive-gpu-equivalence',
        '--approval-reference',str(state.get('instruction_commit') or ''),
        '--claim-reference',claim,'--control-root',str(control)])
    evidence = codex_control_evidence(args)
    require(state['execution_worktree'] == str(REPO.resolve()), 'Wrong approved execution tree')
    require(read_json(control/'research/NEXT_EXPERIMENT.json')['command'] ==
            ['bash','diagnostics/actor_logits/run_p2_on200_only.sh'], 'Wrong approved command')
    require(git(control,'show',claim+'^:research/LOOP_STATE.json'), 'Missing claim parent')
    require(git(control,'rev-parse',claim+'^') == state['instruction_commit'], 'Claim parent mismatch')
    parent = json.loads(git(control,'show',state['instruction_commit']+':research/LOOP_STATE.json'))
    require(parent['status']=='APPROVED_FOR_CODEX' and parent['claim_id'] is None,
            'Invalid fresh authorization parent')
    require(parent['experiment_id']==EXPERIMENT and parent['cycle_id']==CYCLE, 'Wrong approval cycle')
    evidence['approval_ci'] = ci_success(state['instruction_commit'])
    evidence['claim_ci'] = ci_success(claim)
    _, identity = code_identity()
    science_parity(read_json(OFF/'RUN_MANIFEST.json')['provenance']['source_sha256'], identity['source_sha256'])
    return evidence


def before_model(args, tasks, manifest_bytes):
    require(args.approval_reference and args.claim_reference and args.control_root,
            'Missing PI approval / Codex claim / control worktree')
    require(args.actor_reset == 'on', 'P2-003 forbids OFF execution')
    require(args.output.resolve() == OUTPUT/'ON200', 'Wrong fresh ON destination')
    evidence = strict_control(args.control_root.resolve())
    preflight = read_json(EVIDENCE/'NETWORK_PREFLIGHT.json')
    require(preflight['status']=='PASS' and
            preflight['control']['claim_commit']==evidence['claim_commit'],
            'Missing successful launch preflight for this exact claim')
    proof = verify_off()
    old = read_json(OFF/'RUN_MANIFEST.json')
    require(tasks == old['tasks'], 'ON TaskSpecs/order differ from historical OFF')
    require(hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA, 'ON task CSV differs')
    write_report(EVIDENCE/'OFF_REUSE_IMMEDIATE_PRE_MODEL.json', proof)
    evidence['OFF_reference_result_commit'] = OLD_RESULT
    evidence['OFF_reference_code_commit'] = OLD_CODE
    evidence['comparison_design'] = 'CROSS_CYCLE_TASKSPEC_PAIRED_DESCRIPTIVE'
    return evidence


def compare_identity(left, right):
    """Explicitly bound cross-cycle identities, never a generic provenance bypass."""
    require(left['git_commit']==OLD_CODE and left['actor_reset']=='off', 'Wrong OFF cohort')
    require(right['git_commit']==git(REPO,'rev-parse','HEAD') and right['actor_reset']=='on', 'Wrong ON cohort')
    require(right['control_evidence']['claim_commit'] != OLD_CLAIM, 'Old claim reused')
    for k in ['checkpoint_sha256','dino_sha256','manifest_sha256','seed','num_workers',
              'shuffle','greedy','test_augmentation']:
        require(left[k] == right[k], 'Cross-cycle fixed condition differs: '+k)
    science_parity(left['source_sha256'], right['source_sha256'])


def write_report(path, value):
    path = Path(path).resolve()
    require(not path.is_relative_to(OFF.parent.resolve()), 'Writing historical OFF tree is forbidden')
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['network-readiness','preflight','verify-off','compare'])
    parser.add_argument('--control-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'network-readiness':
        report = network_probe(args.control_root)
        write_report(args.report, report)
        raise SystemExit(0 if report['status']=='PASS' else 2)
    if args.mode == 'verify-off':
        write_report(args.report, verify_off())
        return
    control = strict_control(args.control_root)
    off = verify_off()
    if args.mode == 'preflight':
        require(not OUTPUT.exists(), 'Fresh results root already exists')
        witnesses = require_disconnect_witnesses(control['claim_commit'])
        observations = []
        for i in range(3):
            observation = network_probe(args.control_root)
            observations.append(observation)
            report = {'status':'IN_PROGRESS' if observation['status']=='PASS' else 'FAIL',
                      'control':control,'OFF_reuse':off,
                      'cross_session_network_evidence':witnesses,'network':observations}
            write_report(args.report, report)
            require(observation['status']=='PASS','Direct live network preflight failed')
            if i < 2:
                time.sleep(30)
        report['status']='PASS'
        write_report(args.report,report)
        return
    from .analyze import compare
    report = compare(OFF, OUTPUT/'ON200', identity_validator=compare_identity)
    report.update(comparison_design='CROSS_CYCLE_TASKSPEC_PAIRED_DESCRIPTIVE',
                  OFF_reuse=off,control=control,
                  physical_gpu_off=read_json(OFF/'RUN_MANIFEST.json')['provenance']['physical_gpu'],
                  physical_gpu_on=read_json(OUTPUT/'ON200/RUN_MANIFEST.json')['provenance']['physical_gpu'],
                  GPU_EQUIVALENCE='NOT_RUN_PI_WAIVED',
                  limitation='Different cycle/time/GPU-load/simulator randomness and divergent observations; not same-input causal logits attribution.')
    write_report(args.report,report)
    require(report['status']=='COMPLETE','Cross-cycle comparison incomplete')


if __name__ == '__main__':
    main()
