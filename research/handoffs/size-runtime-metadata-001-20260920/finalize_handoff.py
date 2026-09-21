"""Offline validation and BLOCKED handoff only; no simulator imports or launches."""
import collections,csv,datetime,hashlib,json,math,statistics,subprocess
from pathlib import Path
O=Path(__file__).resolve().parent
R=Path('/nvme2/user/qyy/SafeVLA')
C=Path('/nvme2/user/qyy/SafeVLA_loop_control')
P='research/handoffs/size-runtime-metadata-001-20260920/'
def rd(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def rows(n):
 with (O/n).open(newline='') as f:return list(csv.DictReader(f))
def git(*a):return subprocess.check_output(['git','-C',str(R),*a])
def eq(a,b):return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-15)
f=rd(O/'final_status.json');s=rd(O/'execution_start.json');state=s['state']
assert f['status']=='BLOCKED' and f['reason']=='BrokenPipeError: [Errno 32] Broken pipe'
assert f['scene_initializations']==44 and f['gpu_count']==1
assert all(f[k]==0 for k in ['safevla_episodes','model_checkpoint_loads','actor_critic_forwards'])
assert not f['outcome_columns_read'] and not f['size_success_association'] and not f['forbidden_modules_present']
assert not (O/'cleanup_error.json').exists()
priorpath=C/'research/handoffs/smalltarget-phenotype-001-20260918/metadata_preflight.json'
inputs=rd(O/'allowed_input_manifest.json')
assert sha(priorpath)==inputs[str(priorpath)]['sha256']
assert sha(O/'extract_runtime_geometry.py')==inputs[str(O/'extract_runtime_geometry.py')]['sha256']
expected={('ObjectNavType/val/'+r['task_key'],r['target_id']):r for r in rd(priorpath)['targets']}
assert len(expected)==368
tt=rows('target_geometry.csv');tasks=rows('task_geometry.csv');rep=rows('preflight_repeatability.csv')
assert (len(tt),len(tasks),len(rep))==(62,200,18)
samples={}
for p in O.glob('geometry_sample_*.json'):
 x=rd(p)
 for tid,g in x['targets'].items():
  k=(x['phase'],str(x['repeat']),x['task_key'],tid)
  assert k not in samples
  samples[k]=g
assert len(samples)==62
for r in tt:
 assert (r['task_key'],r['target_id']) in expected
 assert r['exact_id_matches']=='1' and r['valid_aabb']=='True'
 g=samples[(r['phase'],r['repeat'],r['task_key'],r['target_id'])]
 d=[float(r['aabb_'+a+'_m']) for a in 'xyz']
 assert all(math.isfinite(v) and v>0 for v in d) and d==g['dimensions']
 assert eq(math.prod(d),r['aabb_volume_m3']) and eq(max(d),r['max_side_m'])
 assert len(g['corners'])==8
 for i,v in enumerate(d):
  span=max(p[i] for p in g['corners'])-min(p[i] for p in g['corners'])
  assert abs(v-span)<=1e-6+1e-5*max(abs(v),abs(span))
for r in rep:
 a=samples[('preflight','1',r['task_key'],r['target_id'])]
 b=samples[('preflight','2',r['task_key'],r['target_id'])]
 assert a['vector']==b['vector'] and r['exact_equal']=='True' and r['within_tolerance']=='True'
 assert float(r['max_absolute_dimension_difference_m'])==float(r['max_relative_dimension_difference'])==0
full=[r for r in tt if r['phase']=='full']
keys={(r['task_key'],r['target_id']) for r in full}
assert len(keys)==len(full)==26
bt=collections.defaultdict(list)
for r in full:bt[r['task_key']].append(r)
assert len(bt)==20 and len({t['task_key'] for t in tasks})==200
assert sum(int(t['target_count']) for t in tasks)==368
for t in tasks:
 rr=bt[t['task_key']]
 if rr:
  assert t['complete']=='True' and len(rr)==int(t['target_count'])==int(t['valid_aabb_count'])
  assert eq(statistics.median(float(r['aabb_volume_m3']) for r in rr),t['median_aabb_volume_m3'])
  assert eq(statistics.median(float(r['max_side_m']) for r in rr),t['median_max_side_m'])
 else:
  assert t['complete']=='False' and t['status']=='NOT_EXTRACTED_STOP_CONDITION'
  assert not t['median_aabb_volume_m3'] and not t['median_max_side_m'] and not t['valid_aabb_count']
assert len(list(O.glob('geometry_sample_preflight_*.json')))==24
assert len(list(O.glob('geometry_sample_full_*.json')))==20
coverage=[]
for k,r in sorted(expected.items()):
 asset=r['asset_id'] or ''
 form='hex32_asset_id' if len(asset)==32 and all(c in '0123456789abcdef' for c in asset.lower()) else 'other_asset_id'
 coverage.append(dict(task_key=k[0],target_id=k[1],synset=r['synset'],asset_id=asset,source_id_form=form,static_candidate=bool(r['house_annotation_bbox'] or r['disk_metadata_bbox']),full_status='VALID_CREATION_STATE_AABB' if k in keys else 'NOT_ATTEMPTED_AFTER_STOP'))
with (O/'expected_target_coverage.csv').open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
groups=collections.Counter((r['synset'],r['source_id_form'],r['full_status']) for r in coverage)
static=sum(r['static_candidate'] for r in coverage);assert static==42
metrics=dict(preflight_tasks=12,preflight_initializations=24,preflight_target_observations=36,preflight_target_comparisons=18,preflight_exact_equal=18,max_absolute_dimension_difference_m=0,max_relative_dimension_difference=0,full_initializations=20,full_exact_mapping=26,full_valid_aabb=26,expected_full_targets=368,unattempted_targets=342,full_complete_tasks=20,expected_tasks=200,unattempted_tasks=180,static_candidate_total=42,static_candidate_full_valid=sum(r['static_candidate'] and r['full_status']=='VALID_CREATION_STATE_AABB' for r in coverage),full_obb_available=sum(r['obb_available']=='True' for r in full),observed_mapping_or_geometry_failures=0)
head=git('rev-parse','HEAD').decode().strip();diff=hashlib.sha256(git('diff','--binary')).hexdigest();status=git('status','--short').decode()
assert head==s['head'] and diff==s['tracked_diff_sha256'] and status==s['status']
save('development_preservation.json',dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),head=head,tracked_diff_sha256=diff,status=status,unchanged_from_execution_start=True,process_check='SSH ps -p 3505586 and ps -eo pid,ppid,args matching thor-CloudRendering found no extractor or simulator during handoff'))
save('validation_report.json',dict(status='PASS_PARTIAL_ARTIFACT_INTEGRITY',scientific_run_status='BLOCKED',metrics=metrics,checks=['snapshot versus CSV geometry','positive finite dimensions and corner extents','exact target membership','repeat vector equality','complete-target medians only','blank unattempted medians','resource counts','extractor hash unchanged','development status and diff unchanged'],limitations=['342 full-pass targets not attempted, not observed missing geometry','No full recovery or size-success claim','Offline validator starts no scenes']))
report='# Runtime geometry coverage: partial / BLOCKED\n\n| Measure | Observed |\n| --- | --- |\n'+''.join('| '+k+' | '+str(v)+' |\n' for k,v in metrics.items())
report+='\nAll 26 attempted full targets map exactly and have valid AABB. The remaining 342 targets were not attempted after the output-channel stop; they are not observed missing/ambiguous geometry. Preflight repeat observations are separate from full-pass coverage. Static candidates are checked for runtime availability only; no static/runtime numeric equivalence is asserted. OBB is secondary only.\n\nSource ID form describes identifier syntax, not validated asset-provider provenance.\n\n| Synset | Source ID form | Full status | Count |\n| --- | --- | --- | --- |\n'+''.join('| '+' | '.join(k)+' | '+str(v)+' |\n' for k,v in sorted(groups.items()))
(O/'coverage_report.md').write_text(report)
summary="""# EXP-SIZE-RUNTIME-METADATA-001: BLOCKED

The run stopped at 2026-09-20T13:25:25Z after 44 scene initializations (24 preflight + 20 full). Recorded exception: BrokenPipeError at the full-loop progress print after task 20 was saved. This is an output-channel interruption, not an observed mapping or AABB failure. The exact pipe closure trigger is not established. The final print also uses the same pipe; the process exit code was not recovered.

Preflight passed: 12 tasks loaded twice, 18 target comparisons, exact geometry equality and maximum absolute/relative dimension difference 0. Partial full-pass coverage is 26/368 valid exact-mapped targets and 20/200 complete tasks. The remaining 342 targets / 180 tasks were not attempted; task medians remain empty. Offline validation independently cross-checks snapshots, CSV geometry, repeatability and medians; see validation_report.json and coverage_report.md.

AI2-THOR build: 966bd7758586e05d18f6181f459c0e90ba318bec, CloudRendering. Archived requirements and committed initialization source bind the historical build; current executable/assembly hashes identify this run. Historical executable bytes were not separately hashed contemporaneously, so historical binary byte equality is not independently proven.

The descriptor is world-axis scene-instance AABB immediately after CreateHouse, with autoSimulation=False and no post-load physics advancement, task teleport, navigation or policy. It is creation-state extent, not canonical intrinsic volume or demonstrated post-settling evaluation geometry. Applicability to later phenotype work remains a PI decision.

Actual resources: one simulator graphics GPU (physical GPU 3, observed Vulkan device argument 4), 44/224 scene initializations, 0 SafeVLA/ObjectNav episodes, 0 model/checkpoint loads, 0 Actor/Critic forwards. Geometry extraction read no outcome files/columns and performed no size-success association. Frozen control documents read for handoff are not extraction inputs. No simulator process remained at handoff verification; original development HEAD, tracked diff and status are unchanged.

H-RUNTIME-AABB is not established for complete coverage. H-RUNTIME-GAP is not established by this output failure. H-SIZE remains untested. No imputation, category, visible-pixel or distance proxy was used.

Next actor: PI. One proposed next action (not approved): review this partial handoff and, if warranted, issue a fresh cycle/design for transport-resilient extraction, explicitly handling prior measurements and a new scene budget. No retry, resume or further experiment was started. Executor STOP after handoff publication.
"""
(O/'RESULT_SUMMARY.md').write_text(summary)
(O/'REVIEW_NOTES.md').write_text("""# PI review notes

Approval f5d0f56b975eab9f54e96343fb42d6b01df4d1a7 and claim c5ebd762bddc9db55c2acfd111ddf2daaf0f5a5e passed CI before execution.
Approval CI: https://github.com/whatcanidowhat/SafeVLA/actions/runs/35507016315
Claim CI: https://github.com/whatcanidowhat/SafeVLA/actions/runs/35507363526

Read RESULT_SUMMARY, failure_detail, final_status, frozen_measurement_plan, runtime_version_manifest, validation_report, the required CSVs and raw snapshots. ARTIFACT_INDEX binds every small shared file to its server path and bytes. expected_target_coverage.csv distinguishes unattempted targets from observed missing metadata.

The original extractor is preserved byte-for-byte, including its vulnerable stdout print. It was not repaired or rerun. Review partial coverage, creation-state versus settled-state applicability, world-axis orientation dependence and historical binary-hash limitations. Successful preflight is not complete benchmark recovery.

LOOP_STATE returns BLOCKED / next_actor PI; original approval, instruction commit, claim ID and frozen designs are preserved. No PI acknowledgement is fabricated. Any further experiment requires PI review and a fresh approved cycle. The interrupted claim must not auto-resume.
""")
m={k:state[k] for k in ['cycle_id','experiment_id','instruction_commit','claim_id']}
m.update(schema_version='1.0',status='BLOCKED',scientific_status='PARTIAL_GEOMETRY_OUTPUT_CHANNEL_INTERRUPTION',claim_commit=s['claim_commit'],claim_ci_url=s['claim_ci'],started_at_utc=f['started_at_utc'],completed_at_utc=f['completed_at_utc'],command=['/home/amax/.conda/envs/safevla/bin/python','-u',P+'extract_runtime_geometry.py'],execution_worktree=str(R),gpu_count=1,physical_gpu_index=3,episodes_started=0,model_checkpoint_loads=0,actor_critic_forwards=0,scene_initializations=44,max_scene_initializations=224,outcome_columns_read=False,size_success_association=False,exit_code=None,exit_code_interpretation='Not recovered after output pipe interruption; run BLOCKED.',reason=f['reason'],metrics=metrics,development_head=head,development_tracked_diff_sha256=diff,extractor_sha256=sha(O/'extract_runtime_geometry.py'),frozen_design_sha256={n:sha(C/'research'/n) for n in ['NEXT_EXPERIMENT.md','NEXT_EXPERIMENT.json']},input_manifest='allowed_input_manifest.json',runtime_identity='runtime_version_manifest.json',missing_not_started=dict(full_tasks=180,full_targets=342,policy_evaluation='NOT_STARTED_BY_DESIGN',outcome_analysis='NOT_STARTED_BY_DESIGN'),handoff_validation_command=['/home/amax/.conda/envs/safevla/bin/python',P+'finalize_handoff.py'])
save('RUN_MANIFEST.json',m)
art=[]
for p in sorted(O.iterdir()):
 if p.name=='ARTIFACT_INDEX.json':continue
 assert p.is_file() and p.suffix in {'.json','.csv','.py','.md'} and p.stat().st_size<1048576
 art.append(dict(name=p.name,server_path=str(p),git_path=P+p.name,sha256=sha(p),size=p.stat().st_size,required_for_PI_review=True))
save('ARTIFACT_INDEX.json',dict(schema_version='1.0',status='BLOCKED',cycle_id=state['cycle_id'],artifacts=art))
assert all((O/Path(p).name).is_file() for p in state['required_outputs'])
print(json.dumps(dict(status='HANDOFF_READY_BLOCKED',metrics=metrics,files=len(art)+1)))
