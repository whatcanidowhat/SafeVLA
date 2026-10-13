"""CPU-only read-only raw evidence audit and terminal control handoff; no experiments."""
import csv, datetime, hashlib, json, os, pathlib, re, shutil, subprocess, sys

C=pathlib.Path('/nvme2/user/qyy/SafeVLA_loop_control_p2_codex_20261013')
R=pathlib.Path('/nvme2/user/qyy/SafeVLA_actor_logits_p2_full400_20261012')
O=pathlib.Path('/nvme2/user/qyy/SafeVLA_p2_full400_results_20261012')
A=pathlib.Path('/nvme2/user/qyy/SafeVLA_p2_full400_launch_20261013')
CLAIM='e27a5d25bba4e6a991960a733d64fd295480b7b9'
APPROVAL='2f153c3154caeb6266d463b4ba3e38d698bd9bad'
CODE='33ffa28000522e3715a0920bf3679c27e79b423e'
CYCLE='actor-reset-off200-on200-20261012'
H=C/'research/handoffs'/CYCLE
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def git(r,*args):return subprocess.check_output(['git','-C',str(r),*args],text=True).strip()
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert git(C,'rev-parse','HEAD')==CLAIM and not git(C,'status','--porcelain')
assert git(R,'rev-parse','HEAD')==CODE and not git(R,'status','--porcelain')
s=read(C/'research/LOOP_STATE.json')
assert s['status']=='CODEX_RUNNING' and s['instruction_commit']==APPROVAL
launch=read(A/'launch.json')
assert launch['status']=='EXITED' and launch['exit_code']==1
assert not (O/'ON200').exists()
H.mkdir(exist_ok=False)
sys.path.insert(0,str(R))
from diagnostics.actor_logits.storage import unpack_tensor
from diagnostics.actor_logits.validate import validate_run
from diagnostics.actor_logits.manifest import regenerate, MANIFEST_SHA
import numpy as np
P=O/'OFF200/episodes/0000.partial'
rows=lines(P/'actions.jsonl');acks=lines(P/'execution_results.jsonl')
boundaries=lines(P/'boundary_states.jsonl')
runtime=read(O/'OFF200/RUN_MANIFEST.json')
error=read(O/'OFF200/ERROR.json')
tasks,manifest=regenerate(R)
assert (O/'OFF200/TASK_MANIFEST.csv').read_bytes()==manifest
assert len(rows)==27 and len(acks)==27 and error['global_step']==27
assert len(list((O/'OFF200/episodes').iterdir()))==1
assert [b['event'] for b in boundaries]==['before_original_reset','after_original_reset','after_treatment','episode_end']
assert boundaries[1]['state']==boundaries[2]['state']
tensor_count=0
with np.load(P/'tensors.npz',allow_pickle=False) as z:
 expected={f'{i:06d}.{k}' for i,row in enumerate(rows) for k in row['tensors']}
 assert len(z.files)==len(expected) and set(z.files)==expected
 for i,row in enumerate(rows):
  assert row['step_index']==row['global_step']==i
  assert row['episode_order']==0 and row['sample_id']==tasks[0]['sample_id']
  assert row['raw_spec_sha256']==tasks[0]['raw_spec_sha256']
  assert row['task_source_index']==127
  assert all(v==1 for v in row['calls'].values())
  assert acks[i]=={'step_index':i,'returned':True}
  for key,spec in row['tensors'].items():
   arr=z[f'{i:06d}.{key}'];t=unpack_tensor(arr,spec)
   with np.load(P/f'step_{i:06d}.npz',allow_pickle=False) as shard:
    assert np.array_equal(arr,shard[key])
   if key in ['raw_actor_logits','normalized_logits','action_probs']:
    assert t.numel()==20 and np.isfinite(t.double().numpy()).all()
   if key=='sampled_action':assert int(t.item())==row['sampled_action_index']
   if key=='greedy_mode_action':assert int(t.item())==row['greedy_mode_action_index']
   tensor_count+=1
  assert runtime['action_names'][row['executed_action_index']]==row['executed_action_name']
index=read(O/'OFF200/ARTIFACT_INDEX.json')['artifacts']
for item in index:
 p=O/'OFF200'/item['path'];assert p.stat().st_size==item['size'] and sha(p)==item['sha256']
validation=validate_run(O/'OFF200',200)
assert validation['status']=='INCOMPLETE' and validation['outcomes'] is None
write(H/'OFF200_INTEGRITY_CHECK.json',validation)
keys=['danger','corner','blind','fragile','critical']
costs={k:sum(r['pre_step_cost'][k] for r in rows) for k in keys}
gpu=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
compute=subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
assert not compute
audit={'status':'PARTIAL_EVIDENCE_HASH_AND_ALIGNMENT_PASS_NOT_RUN_PASS','utc':now,
 'episode_initializations_observed':1,'episode_order':0,'sample_id':tasks[0]['sample_id'],
 'environment_loop_reached_end_episode':True,'archived_complete_episodes':0,
 'online_decisions':27,'returned_environment_steps':27,'tensor_fields_verified':tensor_count,
 'action_dimensions':20,'native_tensor_hash_shape_dtype_checks':'PASS',
 'per_step_shards_equal_packed_archive':'PASS','existing_artifact_index_checks':'PASS',
 'all_logged_forward_sample_counts_one_per_decision':True,
 'counts_are_logger_observations_not_independent_GPU_equivalence':True,
 'OFF_boundary_treatment_no_change':True,'first_row':{k:rows[0][k] for k in ['counter_before','counter_after','decoder_start_pos','executed_action_name']},
 'last_row':{k:rows[-1][k] for k in ['counter_before','counter_after','decoder_start_pos','executed_action_name']},
 'partial_pre_step_cost_components':costs,'partial_pre_step_cost_sum':sum(costs.values()),
 'original_manifest_decisions':runtime['decisions'],
 'manifest_counter_note':'Original manifest updates after successful Episode finalization; its zero decisions is stale, NOT actual usage. ERROR.global_step and27 rows/acks establish actual count.',
 'metric_fragment_note':'Incomplete metrics.json.tmp has success1.00000001,eps_len27,cost0 before failure at percentage_collision. It is preserved as a fragment, not promoted to valid official aggregate.',
 'OFF200_SR':None,'OFF200_cost':None,'ON200_status':'NOT_STARTED','comparison':None,
 'GPU_EQUIVALENCE':'NOT_RUN_PI_WAIVED','gpu_after_cleanup':gpu,'gpu_compute_after_cleanup':compute}
write(H/'PARTIAL_EVIDENCE_AUDIT.json',audit)
copies={
 'RUN_MANIFEST_AT_EXIT.json':O/'OFF200/RUN_MANIFEST.json',
 'ERROR.json':O/'OFF200/ERROR.json','ERROR_EVENTS.jsonl':O/'OFF200/ERROR_EVENTS.jsonl',
 'TASK_MANIFEST.csv':O/'OFF200/TASK_MANIFEST.csv',
 'EPISODE_STATUS_AT_EXIT.json':P/'EPISODE_STATUS.json',
 'METRICS_PARTIAL_AT_EXIT.txt':P/'metrics.json.tmp',
 'BOUNDARY_STATES.jsonl':P/'boundary_states.jsonl',
 'ACTIONS_METADATA.jsonl':P/'actions.jsonl','EXECUTION_ACKS.jsonl':P/'execution_results.jsonl',
 'LAUNCH_RECORD.json':A/'launch.json','LAUNCHER_LOG.txt':A/'launcher.log',
 'OWN_PROCESS_STOP.json':A/'own_process_stop.json',
 'P2_LAUNCH_CONTEXT_AT_EXIT.json':O/'P2_LAUNCH_CONTEXT.json',
 'OFF200_CONSOLE.txt':O/'OFF200.console.log',
 'GPU_OBSERVATIONS.jsonl':A/'gpu_observations.jsonl'}
for name,p in copies.items():shutil.copyfile(p,H/name)
shutil.copyfile(C/'research/NEXT_EXPERIMENT.md',H/'FROZEN_PROTOCOL.md')
shutil.copyfile(C/'research/NEXT_EXPERIMENT.json',H/'FROZEN_DESIGN.json')
with (H/'DECISIONS.csv').open('w',newline='') as f:
 fields=['episode_order','sample_id','step_index','global_step','sampled_action_index','greedy_mode_action_index','executed_action_index','executed_action_name','counter_before','counter_after','decoder_start_pos']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
all_raw=[p for b in [O,A] for p in sorted(b.rglob('*')) if p.is_file() and not p.is_symlink()]
inventory=[{'server_path':str(p),'sha256':sha(p),'size':p.stat().st_size} for p in all_raw]
write(H/'SERVER_ARTIFACT_HASHES.json',{'scope':'Post-exit immutable raw files; no tensor files copied to Git','files':inventory,'total_bytes':sum(x['size'] for x in inventory)})
base=pathlib.Path('/nvme2/user/qyy/SafeVLA_baseline_clean')
bdiff=subprocess.check_output(['git','-C',str(base),'diff','--binary'])
assert hashlib.sha256(bdiff).hexdigest()=='c4a8194d337c37a7a8765a01a11175778301b284211b3b32cd87de55088d3816'
write(H/'RUNTIME_PROVENANCE.json',{'source_commit':CODE,'source_worktree':str(R),'source_status':git(R,'status','--porcelain'),
 'B0_head':git(base,'rev-parse','HEAD'),'B0_diff_sha256':hashlib.sha256(bdiff).hexdigest(),
 'runtime_provenance':runtime['provenance'],
 'renderer_mapping':'Actual runtime CUDA_VISIBLE_DEVICES=2; original THOR Controller maps logical gpu_device0 via that list to physical2. Original controller commit printed in OFF console. No separate GPU renderer probe or per-process renderer UUID trace was captured.',
 'scientific_code_modified_during_run':False})
write(H/'GPU_QUEUE_REPORT.json',{'status':'SELECTED_THEN_RELEASED','hold_used':False,
 'physical_gpu':2,'logical_gpu':0,'uuid':'GPU-cb473e82-397b-a5a6-a17b-39dcd1852c1c',
 'queue_poll_utc':['2026-10-13T03:30:47Z','2026-10-13T03:31:18Z','2026-10-13T03:31:48Z'],
 'stability':'same candidate2 across3 polls, gaps31s and30s; launch rechecked UUID/capacity/no compute PID',
 'thresholds':{'max_used_MiB':4000,'min_free_MiB_strict':6823.390767,'max_util_percent':30,'foreign_compute_pids':0},
 'measurement_scope':'Frozen launcher logs candidates/stability but not every numeric sample. Separate read-only observations preserved; do not fabricate missing per-poll numeric telemetry.',
 'before_launch_readonly_observation':{'gpu2_used_MiB':5,'gpu2_free_MiB':32490,'gpu2_util_percent':0,'gpu2_compute_pids':[]},
 'after_cleanup':gpu,'after_cleanup_compute_pids':[],
 'foreign_jobs_terminated':False,'own_failed_process_cleanup':'OWN_PROCESS_STOP.json; TERM then after88.6s KILL of verified own PID3485003 only'})
started=datetime.datetime.fromisoformat('2026-10-13T03:31:49+00:00')
ended=datetime.datetime.fromisoformat(launch['ended_utc'])
manifest_out={'schema_version':'1.0','cycle_id':CYCLE,'experiment_id':s['experiment_id'],'instruction_commit':APPROVAL,
 'claim_id':s['claim_id'],'claim_commit':CLAIM,'approval_ci':'https://github.com/whatcanidowhat/SafeVLA/actions/runs/38339887619',
 'claim_ci':launch['claim_ci'],'status':'INVALID','next_actor':'PI','commands':[launch['command'],['CPU-only AST and frozen TaskSpec manifest check'],['CPU-only postmortem raw tensor/hash/alignment validation']],
 'execution_worktree':str(R),'implementation_commit':CODE,'raw_output_directory':str(O),'launch_evidence_directory':str(A),
 'started_utc':launch['started_utc'],'ended_utc':launch['ended_utc'],'exit_code':1,
 'gpu_count':1,'episodes_started':1,'episodes_completed':0,'episodes_environment_loop_finished':1,
 'online_decisions':27,'recorded_step_returns':27,'offline_replay_forwards':0,
 'logger_observed_root_forwards':27,'logger_observed_reward_critic_forwards':27,'logger_observed_cost_critic_forwards':27,
 'OFF200':'INVALID_INCOMPLETE_FIRST_EPISODE_FINALIZATION','ON200':'NOT_STARTED','SR':None,'Safety_Cost':None,'delta_SR':None,'delta_Cost':None,
 'GPU_EQUIVALENCE':'NOT_RUN_PI_WAIVED','raw_bytes':sum(x['size'] for x in inventory),
 'gpu_phase_wall_hours_upper_bound':(ended-started).total_seconds()/3600,
 'budget_note':'Wall-time upper bound includes model setup and failed-process cleanup; not measured kernel-active time. Episode initialization inferred from only order0 and fail-closed stack; no retry observed.',
 'failure':'TypeError: Unsupported JSON type: ndarray in metrics serialization at first Episode end',
 'counter_reconciliation':'Raw run manifest decisions0 is stale:27 actual decisions/acks and ERROR.global_step27. Preserve raw unchanged.',
 'missing_outputs':['200 completed OFF Episodes','any ON Episode','valid per-Episode official metrics JSON','full SR/Cost and OFF/ON comparison'],
 'stop_reason':'Frozen logger raised at first Episode finalization; no retry or code patch; result returned to PI.'}
write(H/'RUN_MANIFEST.json',manifest_out)
(H/'LOGGER_WAIVER.md').write_text('# PI Logger waiver\n\nGPU_EQUIVALENCE=NOT_RUN_PI_WAIVED.\n\nExplicit waiver from approval '+APPROVAL+'; no600-pair replay, OFF/OFF test or fabricated PASS. CPU static checks and partial artifact integrity do not establish real-GPU Logger noninterference. Both intended groups use the same frozen logger; ON was never started.\n')
(H/'OFF_ON_RESULTS.md').write_text('# OFF/ON results — INVALID / incomplete\n\nOFF:1 task initialized;27 decisions and27 returned steps; first task reached end_episode, then official metric serialization failed.0 complete Episode directories of200 required. ON:NOT_STARTED,0/200.\n\nFull OFF/ON SR, cost mean/sum/SD/quantiles, paired differences, JSD and uncertainty are NOT_MEASURED/null. No full-run Safety Cost is reported as0. Partial27-step pre-step components were all0, and the preserved incomplete metric fragment contains success1.00000001 and eps_len27. These are first-task observations only, not a valid200-task result.\n\nAll27 full20-action raw/normalized/probability vectors survive on the server with native tensor hashes and shapes. See PARTIAL_EVIDENCE_AUDIT.json and SERVER_ARTIFACT_HASHES.json. No ON data exists to align or compare; no reset hypothesis conclusion is supported.\n')
(H/'VALIDATION_SUMMARY.md').write_text('# Validation summary\n\nPrelaunch: frozen clean source SHA '+CODE+'; bash -n PASS;9 Python AST PASS;200 task manifest SHA '+MANIFEST_SHA+' PASS, first indices127,103,16,6,50. Historical22 CPU mock tests belong to bb71a65 development, not a newly run frozen-code suite. No new GPU equivalence tests.\n\nPostmortem CPU-only:27 rows/acks,189 native tensor fields, packed NPZ vs all27 shards, shapes/dtypes/SHA, complete20-action vectors, task/order/counter metadata and original artifact-index hashes checked. OFF after_original_reset vs after_treatment boundary states identical. These partial evidence checks PASS; run integrity correctly returns INCOMPLETE with null outcomes. They do not promote the partial directory.\n\nRoot cause: storage.py:23-26 json_value supports NumPy scalar, not ndarray. Original Worker.calculate_metrics:571 produces percentage_collision from observation values; metrics.json.tmp stops at that key. Logger.end_episode:217 -> EpisodeWriter.finish:133 -> write_json:33 raised. Exact ndarray shape/value was not serialized and is not inferred. Existing CPU mocks used scalar metrics and did not cover this carrier. No source fix or experiment retry performed.\n\nOriginal manifest decisions0 and launch context RUNNING were preserved unchanged; terminal handoff reconciles27 actual decisions and exit1. SIGTERM did not release failed own process within60s; verified exact own PID was killed after88.6s. GPU2 subsequently had no compute PID,5MiB used. Independent renderer UUID tracing and600-input GPU equivalence were not performed.\n')
(H/'RESULT_SUMMARY.md').write_text('# P2 executor handoff: INVALID\n\nExperiment '+s['experiment_id']+'; cycle '+CYCLE+'.\n\nUnique Claim '+CLAIM+' binds PI approval '+APPROVAL+'; independent Claim CI38341600763 SUCCESS preceded the only launcher invocation. Frozen source '+CODE+' stayed clean and unchanged.\n\nOFF200 failed at first Episode metrics serialization (ndarray unsupported).1 online task initialization,27 decisions/step returns,0 complete archived Episodes. ON200 NOT_STARTED. GPU2 was used and released; no foreign process terminated, no retry, no GPU replay, no code patch. Exact raw tensors/shards, actions, boundaries, traceback, truncated metric file, source provenance and launch/cleanup logs are preserved.\n\nThis is an engineering serialization failure, not evidence that Actor reset helps or hurts. Full SR/Safety Cost and OFF/ON comparisons remain unavailable. GPU_EQUIVALENCE=NOT_RUN_PI_WAIVED. The stale original decisions0 field is reconciled against27 rows/acks and error.global_step27, without overwriting raw evidence.\n\nNext actor PI. Recommend review of a narrowly scoped lossless ndarray metric serializer and a CPU regression using this real metric carrier; any new implementation/runtime requires a fresh independently approved cycle. No repair, replay or successor experiment is authorized by this handoff. STOP.\n')
(H/'REVIEW_NOTES.md').write_text('# PI review notes\n\nRead RESULT_SUMMARY, RUN_MANIFEST, ERROR, METRICS_PARTIAL_AT_EXIT, PARTIAL_EVIDENCE_AUDIT and raw source identity. All required review summaries are Git-readable; tensor NPZ/shards remain optional server evidence with SHA/size.\n\nVerified fact: first raw task source127,27 decisions/acks, writer failed at ndarray serialization before atomic Episode completion. Observation: truncated official fragment indicates first-task success and zero five-part cost. Conclusion: experiment INVALID/incomplete; no200-task metrics or ON comparison. Hypothesis H-RESET remains untested by a completed paired benchmark.\n\nReview limitations: PI consciously waived real GPU Logger equivalence; mock/static checks did not cover ndarray-valued auxiliary official metrics. Queue has candidate stability logs and separate occupancy observations but no full numeric telemetry for all three launcher polls. Renderer physical mapping follows original THOR source and runtime CUDA env; independent renderer UUID trace unavailable.\n\nRequested PI decision only: review failure and determine whether a separate code-only serialization repair/regression is warranted. Executor has not changed the frozen design, re-approved this cycle, rerun OFF, started ON, or edited B0/old evidence.\n')
entry='\n\n## 2026-10-13 — P2 OFF200/ON200 executor handoff INVALID\n\nClaim '+CLAIM+' (CI38341600763 SUCCESS), approval '+APPROVAL+', frozen source '+CODE+'. One OFF task reached Episode end after27 decisions; Logger metrics serialization failed on ndarray (percentage_collision), leaving0 complete archived Episodes. ON NOT_STARTED.1 GPU,1 task initialization,27 actual decisions,0 offline replay; GPU released after verified own failed-process cleanup. No retry/source edit. Full SR/Cost and paired comparison unavailable; GPU_EQUIVALENCE=NOT_RUN_PI_WAIVED. Required evidence: [RESULT_SUMMARY.md](handoffs/'+CYCLE+'/RESULT_SUMMARY.md). Next actor PI; STOP.\n'
for name in ['CURRENT_STATE.md','EXPERIMENT_REGISTER.md','DECISION_LOG.md']:
 with (C/'research'/name).open('a') as f:f.write(entry)
s.update(status='INVALID',next_actor='PI',updated_by='CODEX',updated_at_utc=now,state_version=s['state_version']+1)
write(C/'research/LOOP_STATE.json',s)
shutil.copyfile(pathlib.Path(__file__),H/'audit_partial_evidence.py')
artifacts=[]
for p in sorted(H.iterdir()):
 if p.name=='ARTIFACT_INDEX.json':continue
 artifacts.append({'name':p.name,'server_path':str(p),'git_path':str(p.relative_to(C)),'sha256':sha(p),'size':p.stat().st_size,'required_for_PI_review':True})
for item in inventory:
 artifacts.append({'name':'raw/'+str(pathlib.Path(item['server_path']).relative_to('/nvme2/user/qyy')),**item,'required_for_PI_review':False})
write(H/'ARTIFACT_INDEX.json',{'schema_version':'1.0','artifacts':artifacts})
print(json.dumps({'status':'INVALID_HANDOFF_PREPARED','files':len(list(H.iterdir())),'episodes_started':1,'decisions':27,'raw_bytes':manifest_out['raw_bytes'],'GPU_EQUIVALENCE':'NOT_RUN_PI_WAIVED'},indent=2))
