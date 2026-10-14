"""Archive actual P2-002 terminal evidence; no model, GPU, retry or publication."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import subprocess

C = Path('/nvme2/user/qyy/SafeVLA_loop_control_p2_codex_20261013')
W = Path('/nvme2/user/qyy/SafeVLA_actor_logits_p2_002_20261014')
R = Path('/nvme2/user/qyy/SafeVLA_p2_002_results_20261014')
L = Path('/nvme2/user/qyy/SafeVLA_p2_002_launch_20261014')
CLAIM = '918170d26cca2fc385fb6f8fd87d0109b4a167b2'
CODE = '06cde9adeb91d47c062a11e72e90c7edf56fb398'
APPROVAL = '7fc797645d00fc38ab4ac8e08f4b8ed98053449f'
CYCLE = 'actor-reset-off200-on200-002-20261014'
H = C / 'research/handoffs' / CYCLE


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(name, value):
    (H / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def copy(src, name):
    assert src.is_file() and not src.is_symlink()
    (H / name).write_bytes(src.read_bytes())


assert git(C, 'rev-parse', 'HEAD') == CLAIM and not git(C, 'status', '--porcelain')
assert git(W, 'rev-parse', 'HEAD') == CODE and not git(W, 'status', '--porcelain')
state = read(C / 'research/LOOP_STATE.json')
assert state['status'] == 'CODEX_RUNNING' and state['instruction_commit'] == APPROVAL
assert state['claim_id'] == 'b89ac28dffc3490c9672afc9ef1afd15'
assert not (R / 'ON200').exists()
exit_record = read(L / 'EXIT.json')
assert exit_record['exit_code'] == 1
trace = (R / 'ON200.console.log').read_text()
assert 'TimeoutExpired' in trace and 'ls-remote' in trace and '60 seconds' in trace
process_text = subprocess.check_output(['ps', '-eo', 'pid,ppid,etime,args'], text=True)
active = [line for line in process_text.splitlines() if
          ('diagnostics.actor_logits.evaluate' in line or 'bash diagnostics/actor_logits/run_p2_full400.sh' in line)]
assert not active, active
v = read(R / 'OFF200.validation.json')
assert v['status'] == 'PASS' and v['episodes_checked'] == 200 and v['decisions_checked'] == 19270
m = read(R / 'OFF200/RUN_MANIFEST.json')
assert m['status'] == 'COMPLETE' and m['episodes_completed'] == 200
assert m['provenance']['git_commit'] == CODE
assert m['provenance']['control_evidence']['claim_commit'] == CLAIM
assert len(list((R / 'OFF200/episodes').glob('*.complete'))) == 200
assert len(list((R / 'OFF200/episodes').iterdir())) == 200
assert sha(L / 'TASK_MANIFEST.csv') == m['provenance']['manifest_sha256']

# Verify every already-validated raw artifact is still byte-identical; no forward.
raw_index = read(R / 'OFF200/ARTIFACT_INDEX.json')['artifacts']
for item in raw_index:
    p = R / 'OFF200' / item['path']
    assert p.is_file() and not p.is_symlink()
    assert p.stat().st_size == item['size'] and sha(p) == item['sha256'], str(p)
assert {item['path'] for item in raw_index} == {
    str(p.relative_to(R / 'OFF200')) for p in (R / 'OFF200').rglob('*')
    if p.is_file() and p.name != 'ARTIFACT_INDEX.json'}
for name, digest in m['provenance']['source_sha256'].items():
    assert sha(W / name) == digest, name

H.mkdir(parents=True, exist_ok=False)
for src, name in [
    (R/'OFF200.validation.json', 'OFF200_VALIDATION.json'),
    (R/'OFF200/RUN_MANIFEST.json', 'OFF200_RUN_MANIFEST.json'),
    (R/'OFF200/ARTIFACT_INDEX.json', 'OFF200_RAW_HASHES.json'),
    (R/'OFF200/episode_metrics.csv', 'OFF200_EPISODE_METRICS.csv'),
    (R/'OFF200.console.log', 'OFF200_CONSOLE.txt'),
    (R/'ON200.console.log', 'ON200_PRELAUNCH_FAILURE.txt'),
    (R/'P2_LAUNCH_CONTEXT.json', 'P2_LAUNCH_CONTEXT_AT_EXIT.json'),
    (L/'launcher.console.log', 'LAUNCHER_CONSOLE.txt'),
    (L/'PREFLIGHT.json', 'CPU_IDENTITY_PREFLIGHT.json'),
    (L/'CLAIM_CI.json', 'CLAIM_CI.json'),
    (L/'LAUNCH_ONCE.json', 'LAUNCH_ONCE.json'),
    (L/'LAUNCH_PID.json', 'LAUNCH_PID.json'),
    (L/'EXIT.json', 'EXIT.json'),
    (L/'TASK_MANIFEST.csv', 'TASK_MANIFEST.csv'),
    (C/'research/NEXT_EXPERIMENT.md', 'FROZEN_PROTOCOL.md'),
    (C/'research/NEXT_EXPERIMENT.json', 'FROZEN_DESIGN.json'),
    (L/'p2_002_handoff.py', 'archive_evidence.py'),
]:
    copy(src, name)

outcomes = v['outcomes']
cost = [e['cost'] for e in outcomes]
n = len(outcomes)
k = sum(e['success'] for e in outcomes)
z = 1.959963984540054
p = k / n
center = (p + z*z/(2*n))/(1+z*z/n)
half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)


def quantile(values, q):
    values = sorted(values)
    pos = (len(values)-1)*q
    low = math.floor(pos)
    high = math.ceil(pos)
    return values[low]+(values[high]-values[low])*(pos-low)


behavior = []
for directory in sorted((R/'OFF200/episodes').glob('*.complete')):
    rows = [json.loads(line) for line in (directory/'actions.jsonl').read_text().splitlines()]
    actions = [row['executed_action_name'] for row in rows]
    e = outcomes[len(behavior)]
    behavior.append({**e, 'last_action': actions[-1],
                     'failed_sub_horizon_end_proxy': not e['success'] and actions[-1]=='end' and len(rows)<600,
                     'adjacent_repeats': sum(a==b for a,b in zip(actions,actions[1:])),
                     'aba_oscillations': sum(actions[i]==actions[i-2]!=actions[i-1] for i in range(2,len(actions)))})
with (H/'OFF200_BEHAVIOR_PROXIES.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(behavior[0]));writer.writeheader();writer.writerows(behavior)
stats = {
    'status': 'OFF_ONLY_COMPLETE_PAIR_BLOCKED', 'n': n, 'successes': k, 'SR': p,
    'SR_Wilson_95_interval': [center-half, center+half],
    'uncertainty_scope': 'Descriptive binomial interval for this fixed task set; one seed, possible shared-house dependence, not seed uncertainty or causal inference.',
    'cost_sum': sum(cost), 'mean_cost': statistics.mean(cost), 'sample_sd_cost': statistics.stdev(cost),
    'cost_quantiles_linear': {str(q): quantile(cost,q) for q in [0, .25, .5, .75, .9, .95, .99, 1]},
    'components': {key: sum(e[key] for e in outcomes) for key in ['danger','corner','blind','fragile','critical']},
    'decisions': sum(e['eps_len'] for e in outcomes),
    'failed_sub_horizon_end_proxy_count': sum(e['failed_sub_horizon_end_proxy'] for e in behavior),
    'adjacent_repeat_count': sum(e['adjacent_repeats'] for e in behavior),
    'aba_oscillation_count': sum(e['aba_oscillations'] for e in behavior),
    'proxy_scope': 'Observed actions only; does not establish target invisibility, illegal END, or a causal oscillation mechanism.',
    'ON': 'NOT_STARTED_EPISODES_0_PRE_MODEL_CONTROL_CHECK_FAILED',
    'delta_SR': None, 'delta_cost': None, 'paired_logits': 'NOT_AVAILABLE',
    'gpu_logger_equivalence': 'NOT_RUN_PI_WAIVED',
}
write('OFF200_STATISTICS.json', stats)

# Snapshot all current raw file identities, preserving symlinks as link metadata.
inventory = []
for root in [R,L]:
    for folder, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs+files):
            path = Path(folder)/name
            if path.is_symlink():
                target=os.readlink(path)
                inventory.append({'server_path':str(path),'kind':'symlink_not_dereferenced','link_target':target,
                                  'sha256_of_link_text':hashlib.sha256(target.encode()).hexdigest()})
            elif path.is_file():
                inventory.append({'server_path':str(path),'kind':'file','size':path.stat().st_size,'sha256':sha(path)})
write('SERVER_RAW_INVENTORY.json', {'scope':'Server-only raw bytes are not uploaded; hashes and small review evidence are Git-readable.', 'entries':inventory})
raw_bytes=sum(i.get('size',0) for i in inventory if i['kind']=='file')
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
write('POST_RUN_PROVENANCE.json', {'observed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
    'source_head':git(W,'rev-parse','HEAD'),'source_status':git(W,'status','--porcelain'),
    'source_hashes_equal_run':True,'OFF_raw_index_verified_entries':len(raw_index),
    'active_experiment_processes':active,'gpu_compute_snapshot':compute.splitlines(),
    'gpu_snapshot':gpu.splitlines(),'raw_file_bytes':raw_bytes,
    'ON_result_root_exists':False,'no_cleanup_signals_sent':True})
launch_text=(L/'launcher.console.log').read_text()
write('GPU_QUEUE_REPORT.json', {'status':'ADMITTED_GPU0_AFTER_THREE_STABLE_POLLS','physical_gpu':0,
    'gpu_uuid':'GPU-ee966fa4-ba0e-54e5-aaa2-3fbbac38da37',
    'queue_start_utc':'2026-10-14T07:26:11Z',
    'stable_poll_utc':['2026-10-14T07:27:42Z','2026-10-14T07:28:12Z','2026-10-14T07:28:43Z'],
    'OFF_start_utc':'2026-10-14T07:28:43Z','OFF_validated_utc':'2026-10-14T11:36:48Z',
    'ON_command_attempt_utc':'2026-10-14T11:36:48Z','launcher_exit':exit_record,
    'source_evidence':'LAUNCHER_CONSOLE.txt; P2_LAUNCH_CONTEXT_AT_EXIT.json',
    'limitations':'Original runner logs admission and selected UUID, not each numerical nvidia-smi poll; do not fabricate missing poll values.',
    'other_user_processes_terminated':False,'ON_gpu_model_initialized':False})
elapsed=(dt.datetime.fromisoformat(exit_record['finished_at_utc'])-dt.datetime(2026,10,14,7,28,43,tzinfo=dt.timezone.utc)).total_seconds()
manifest={
    'schema_version':'1.0','cycle_id':CYCLE,'experiment_id':state['experiment_id'],
    'instruction_commit':APPROVAL,'claim_commit':CLAIM,'claim_id':state['claim_id'],'status':'BLOCKED',
    'handoff_actor':'CODEX','command':['bash','diagnostics/actor_logits/run_p2_full400.sh'],
    'implementation_commit':CODE,'execution_worktree':str(W),'result_root':str(R),'launch_evidence_root':str(L),
    'gpu_count':1,'physical_gpu':0,'episodes_started':200,'episodes_completed':200,
    'OFF_episodes_completed':200,'ON_episodes_started':0,'ON_episodes_completed':0,
    'online_decisions':19270,'offline_equivalence_forwards':0,
    'gpu_logger_equivalence':'NOT_RUN_PI_WAIVED','OFF_integrity':'PASS',
    'ON_command_attempted':True,'ON_model_load':'NOT_REACHED','ON_simulator':'NOT_REACHED',
    'comparison':'NOT_AVAILABLE','exit':exit_record,
    'failure_location':'ON200 pre-model codex_control_evidence -> git ls-remote',
    'failure_type':'subprocess.TimeoutExpired','failure_message':'GitHub remote claim check timed out after 60 seconds; Proxy CONNECT aborted due to timeout',
    'resource_wall_time_upper_bound_hours':elapsed/3600,
    'resource_time_scope':'OFF start through launcher exit, conservatively includes validation and failed ON CPU gate; not exact GPU kernel time.',
    'raw_file_bytes':raw_bytes,'automatic_retries':0,
    'approval_ci':'https://github.com/whatcanidowhat/SafeVLA/actions/runs/38525188076',
    'claim_ci':'https://github.com/whatcanidowhat/SafeVLA/actions/runs/38527821564',
    'runtime_manifest':'OFF200_RUN_MANIFEST.json','frozen_task_manifest_sha256':sha(H/'TASK_MANIFEST.csv'),
}
write('RUN_MANIFEST.json',manifest)
(H/'LOGGER_WAIVER.md').write_text('# GPU Logger equivalence\n\nNOT_RUN_PI_WAIVED, by this cycle\'s explicit PI authorization. No 600-pair replay, OFF/OFF Debug or GPU equivalence PASS was generated. OFF200 stored-evidence integrity PASS is a different and narrower check.\n')
(H/'RESULT_SUMMARY.md').write_text(f'''# P2-002 terminal BLOCKED: OFF200 complete, ON blocked before model load

OFF200 completed 200/200 tasks and passed the frozen integrity validator (19,270 decisions). Official success is {k}/200, SR {p:.1%}; Safety Cost total {sum(cost)}, mean {statistics.mean(cost):.3f}. Five component totals: danger 2, corner 87, blind 56, fragile 13, critical 19. These are OFF-only results, not a completed intervention comparison.

The same one-shot launcher attempted ON200 after OFF validation at 2026-10-14T11:36:48Z. ON stopped inside evaluate.py:79 at the remote Claim check: git ls-remote timed out after 60 seconds, with Proxy CONNECT timeout. execute():116 invokes this guard before model/CUDA/simulator initialization. ON result root was never created; ON Episodes=0. Launcher exited 1 at {exit_record['finished_at_utc']}. No retry, second claim, alternate GPU, parameter edit or continuation was executed.

Approval {APPROVAL}; unique claim {CLAIM}, claim ID {state['claim_id']}; exact execution code {CODE}. Approval and claim CI both SUCCESS before launch. One physical GPU 0; 200 total Episode starts; 19,270 decisions; 0 offline replay. Conservative start-to-exit wall-time upper bound {elapsed/3600:.4f} hours. Source tree remains clean at frozen SHA; source hashes and all {len(raw_index)} OFF indexed artifacts rechecked unchanged after exit.

The original monitoring SSH connection reported Connection reset; the server launcher continued and completed OFF. The later ON GitHub/HTTP proxy failure is directly evidenced. A lost session-dependent reverse tunnel is consistent with the traceback, but exact proxy/tunnel uptime was not recorded; this causal link is an inference, not a measured certainty.

Both GPU Logger equivalence and its unresolved noninterference limitation remain NOT_RUN_PI_WAIVED. Stored-log PASS does not establish GPU equivalence. ON/OFF SR, Cost, logits, early-END and oscillation differences are unavailable; H-RESET is not tested by this incomplete pair.

All required review files accompany this handoff. Full raw tensors/logs stay at {R}, hashed in SERVER_RAW_INVENTORY.json and OFF200_RAW_HASHES.json; compact actual metrics, validation, source/runtime manifest and complete console evidence are Git-readable. The original P2_LAUNCH_CONTEXT still says RUNNING because the launcher never updates it; preserved verbatim and superseded for outcome by EXIT.json and this terminal manifest.

Next actor PI. Review the valid OFF-only evidence and transport stop; any continuation/recovery design, code change, reuse of OFF as a control, or fresh execution requires an explicit separate PI decision and protocol authorization. No successor experiment is self-approved. STOP.
''')
(H/'OFF_ON_RESULTS.md').write_text(f'''# OFF-only result; paired comparison unavailable

| Measure | OFF200 | ON200 |
| --- | --- | --- |
| Completed Episodes | 200 | 0 (pre-model gate failed) |
| Successes / SR | {k}/200 / {p:.1%} | NOT_MEASURED |
| Safety Cost total / mean | {sum(cost)} / {statistics.mean(cost):.3f} | NOT_MEASURED |
| Cost sample SD / maximum | {statistics.stdev(cost):.6f} / {max(cost)} | NOT_MEASURED |
| Logged decisions | 19270 | 0 |
| Integrity | PASS | NOT_RUN |

Five OFF component totals: danger 2, corner 87, blind 56, fragile 13, critical 19. Sum=177. OFF SR Wilson 95% interval [{center-half:.6f}, {center+half:.6f}] is descriptive only: fixed tasks, one seed, possible within-house dependence; not uncertainty over seeds. Full cost quantiles and task-level metrics are in OFF200_STATISTICS.json and OFF200_EPISODE_METRICS.csv.

Failed sub-horizon final-END proxy count: {stats['failed_sub_horizon_end_proxy_count']}; adjacent repeats {stats['adjacent_repeat_count']}; ABA proxies {stats['aba_oscillation_count']}. Definitions and per-task rows are archived; these are descriptive action proxies, not proof of illegal END, target invisibility or mechanism. No ON samples exist, so all treatment differences and paired logits effects are NOT_AVAILABLE, never zero.
''')
(H/'VALIDATION_SUMMARY.md').write_text(f'''# Actual validation

- Frozen source {CODE}, clean before and after run; all runtime source hashes rechecked equal.
- Checkpoint/DINO/task CSV hashes matched before run. Real loader and loading events recorded in OFF200_RUN_MANIFEST.json, including 417 policy keys and no missing/unexpected entries.
- Existing official run validator: PASS, 200 Episodes and 19270 decisions. Checks include full 20-action vectors, tensor hashes/shapes/dtypes, action acknowledgements, forward/sample counts, task order, boundaries, unchanged OFF treatment/critic state, official cost aggregation and artifact coverage. See OFF200_VALIDATION.json.
- After termination, {len(raw_index)} raw index entries matched size and SHA256; no additional GPU/model call or Debug experiment was run.
- ON failed at remote claim lookup before model imports and result directory creation. No ON numerical, reset or comparison validation occurred.
- NOT_RUN_PI_WAIVED remains a waiver, not GPU equivalence PASS.
- Full console logs preserved as text copies; source/runtime manifest, actual per-task metrics and failure traceback are Git-readable. Large raw files are server-only, optional for this compact PI review and hashed; this does not claim upload or independent PI review.
''')
(H/'REVIEW_NOTES.md').write_text('''# PI review boundaries

Verified: OFF200 completed and passed stored-evidence validation; ON remote Claim check timed out before model/episode start. No retry. Terminal BLOCKED describes incomplete paired execution and does not discard the valid OFF-only data.

Observed transport loss: the live SSH monitoring session ended with Connection reset. The launcher continued OFF on the server. ON subsequently failed GitHub Proxy CONNECT. Tunnel loss causing the later timeout is plausible but not independently timestamp-proven.

Limitations: ON metrics and paired contrasts are missing; one-seed OFF cannot establish Actor Reset effects; GPU Logger noninterference was PI-waived; numerical GPU queue polls were not retained by the original launcher (only stable-poll/admission timestamps and selected UUID). The stale raw launch-context RUNNING marker is preserved, with explicit terminal EXIT evidence.

Recommendation only: PI reviews this packet and decides a separately authorized recovery design, including transport persistence and whether the existing OFF200 can serve as an approved control. Do not rerun/reuse the current terminal claim, bypass the remote guard, or independently launch ON. No code repair or new experiment was performed for this handoff.
''')

state.update(status='BLOCKED',next_actor='PI',updated_by='CODEX',state_version=state['state_version']+1,
             updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
(C/'research/LOOP_STATE.json').write_text(json.dumps(state,indent=2)+'\n')
note=f'''## 2026-10-14 P2-002 terminal BLOCKED, next actor PI

Claim {CLAIM} (CI 38527821564 SUCCESS), approval {APPROVAL}, source {CODE}. OFF200 completed 200/200 with integrity PASS, 19270 decisions, SR169/200=84.5%, cost177/mean0.885. ON command attempted only after OFF PASS; remote Claim git ls-remote/Proxy CONNECT timed out before ON model/simulator/Episode initialization (0 ON starts). Launcher exit1; no retry or code change. One GPU0; conservative run wall bound {elapsed/3600:.4f}h. NOT_RUN_PI_WAIVED retained, paired contrasts unavailable. Required actual evidence: [RESULT_SUMMARY.md](handoffs/{CYCLE}/RESULT_SUMMARY.md). Old outputs untouched. This entry supersedes earlier draft/running descriptions; PI review is the only next action. Executor STOP after publication.

'''
p=C/'research/CURRENT_STATE.md';p.write_text(note+p.read_text())
for name in ['EXPERIMENT_REGISTER.md','DECISION_LOG.md']:
    p=C/'research'/name;p.write_text(p.read_text().rstrip()+'\n\n'+note.rstrip()+'\n')
items=[]
for p in sorted(H.iterdir()):
    assert p.is_file() and not p.is_symlink() and p.stat().st_size <= 1024*1024, str(p)
    items.append({'name':p.name,'server_path':str(p),'git_path':str(p.relative_to(C)),
                  'sha256':sha(p),'size':p.stat().st_size,'required_for_PI_review':True})
for p in [R/'OFF200/ARTIFACT_INDEX.json',R/'OFF200.console.log',R/'ON200.console.log']:
    items.append({'name':'server_original_'+p.name,'server_path':str(p),'sha256':sha(p),
                  'size':p.stat().st_size,'required_for_PI_review':False})
write('ARTIFACT_INDEX.json',{'schema_version':'1.0','artifacts':items})
assert all((C/name).is_file() for name in state['required_outputs'])
print(json.dumps({'status':'HANDOFF_PREPARED_NOT_COMMITTED','statistics':stats,'files':len(list(H.iterdir())),'raw_bytes':raw_bytes},indent=2))
