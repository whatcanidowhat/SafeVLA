"""Independent CSV/source validation and handoff assembly; no research rerun."""
import csv,collections,json,hashlib,datetime,subprocess,statistics,math,re,sys
from pathlib import Path
ROOT=Path('/nvme2/user/qyy/SafeVLA');CONTROL=Path('/nvme2/user/qyy/SafeVLA_loop_control');OUT=Path(__file__).resolve().parent
ARCHIVE=CONTROL/'research/history/evidence-alignment-20260919'
PREFIX='research/handoffs/premature-end-dynamics-001-20260923/'
def rd(p):return json.loads(p.read_text())
def csvrd(p):return list(csv.DictReader(p.open(newline='')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(OUT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def equal(a,b):return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12)
def sid(r):return int(re.search(r'sub_house_id=(\d+)',r['video_path']).group(1))
def classify(first,last,n):
 if first>=.5:return 'EARLY_HIGH_PRIOR'
 if first<.1 and n>5 and last>=.5:return 'LATE_RISE'
 if last<.1:return 'LOW_PROB_END'
 return 'OTHER_OR_UNCLEAR'
result=rd(OUT/'analysis_result.json');cases=csvrd(OUT/'premature_end_cases.csv');controls=csvrd(OUT/'matched_success_controls.csv')
assert result['status']=='AWAITING_PI_REVIEW' and len(cases)==len(controls)==16
pop=csvrd(ARCHIVE/'full200/episode_results.csv');bysid={sid(r):r for r in pop}
expected={sid(r) for r in pop if r['success']=='False' and int(r['eps_len'])<600}
assert expected=={int(c['sub_house_id']) for c in cases}
summaryrows=csvrd(OUT/'pdone_trace_summary.csv');assert len(summaryrows)==27
traces={int(p.stem.split('sub')[1]):csvrd(p) for p in OUT.glob('trace_sub*.csv')}
assert len(traces)==28 # 16 failures + 11 distinct successes + the sub120 decoder reference
for n,tr in traces.items():
 assert len(tr)==int(bysid[n]['eps_len'])
 assert [int(r['frame_0based']) for r in tr]==list(range(len(tr)))
 assert all(r['action_thresholds_agree']=='True' for r in tr)
 assert all(equal(r['p_end_approx'],min(1,int(r['end_legacy_width_px'])/55)) for r in tr)
 assert all(abs(int(r['end_legacy_width_px'])-int(r['end_chroma_endpoint_px']))<=3 for r in tr)
for c in cases:
 tr=traces[int(c['sub_house_id'])];ps=[float(r['p_end_approx']) for r in tr];last=len(ps)-1
 assert tr[-1]['executed_action']==c['final_executed_action']=='end'
 assert all(r['executed_action']!='end' for r in tr[:-1])
 assert int(c['end_step_0based'])==last and equal(c['p_end_step0'],ps[0]) and equal(c['p_end_at_end'],ps[-1])
 assert equal(c['max_p_end_first5'],max(ps[:5])) and equal(c['max_p_end_before_end'],max(ps[:-1]))
 assert equal(c['delta_p_end'],ps[-1]-ps[0])
 for threshold,col in [(.5,'first_step_ge_0_5'),(.9,'first_step_ge_0_9')]:
  first=next((i for i,p in enumerate(ps) if p>=threshold),None)
  assert c[col]==('' if first is None else str(first))
 assert classify(max(ps[:5]),ps[-1],len(ps))==c['morphology']
 assert c['threshold_robust']=='True' and c['chroma_morphology']==c['morphology']
for c in controls:
 fail=bysid[int(c['failure_sub_house_id'])];chosen=bysid[int(c['control_sub_house_id'])]
 pool=[r for r in pop if r['success']=='True' and r['object_types']==fail['object_types']]
 assert chosen==min(pool,key=lambda r:(abs(int(r['gt_episode_len'])-int(fail['gt_episode_len'])),sid(r),r['task_path']))
 tr=traces[sid(chosen)];w=min(int(fail['eps_len']),len(tr))-1
 assert int(c['window_last_frame_0based'])==w
 assert equal(c['control_p_end_window_last'],tr[w]['p_end_approx'])
 assert equal(c['control_max_p_end_first5'],max(float(r['p_end_approx']) for r in tr[:min(5,w+1)]))
 assert tr[-1]['executed_action']=='end'
assert all(result[k]==0 for k in ['gpu_count','episodes','simulator_launches','model_checkpoint_loads','actor_critic_forwards'])
assert not result['forbidden_modules_present']
for path,item in rd(OUT/'input_manifest.json').items():
 p=Path(path);assert sha(p)==item['sha256'] and p.stat().st_size==item['size']
start=rd(OUT/'execution_start.json')
def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a])
assert git('rev-parse','HEAD').decode().strip()==start['head']
assert hashlib.sha256(git('diff','--binary')).hexdigest()==start['tracked_diff_sha256']
assert git('status','--short').decode()==start['status']
counts={k:sum(c['morphology']==k for c in cases) for k in ['EARLY_HIGH_PRIOR','LATE_RISE','LOW_PROB_END','OTHER_OR_UNCLEAR']}
premax=[float(c['max_p_end_before_end']) for c in cases]
terminal_only_05=sum(c['first_step_ge_0_5']==c['end_step_0based'] for c in cases)
control_terminal_first=sum(next((i for i,r in enumerate(traces[int(c['control_sub_house_id'])]) if float(r['p_end_approx'])>=.5),None)==len(traces[int(c['control_sub_house_id'])])-1 for c in controls)
unique_control_terminal_first=sum(next((i for i,r in enumerate(traces[n]) if float(r['p_end_approx'])>=.5),None)==len(traces[n])-1 for n in {int(c['control_sub_house_id']) for c in controls})
metrics=dict(morphology_counts=counts,terminal_only_first_crossing_0_5=terminal_only_05,all_step0_subresolution=all(float(c['p_end_step0'])==0 for c in cases),all_preterminal_max_below_0_5=all(p<.5 for p in premax),preterminal_max_across_cases=max(premax),control_terminal_first_crossing_pairs=control_terminal_first,control_terminal_first_crossing_unique=unique_control_terminal_first,expert_gap_median=statistics.median(int(c['absolute_expert_gap']) for c in controls),expert_gap_range=[min(int(c['absolute_expert_gap']) for c in controls),max(int(c['absolute_expert_gap']) for c in controls)],paired_first5_difference_median=statistics.median(float(c['failure_minus_control_first5']) for c in controls),paired_window_endpoint_difference_median=statistics.median(float(c['failure_minus_control_at_window_end']) for c in controls),frames_decoded=sum(len(t) for t in traces.values()),source_videos=len(traces))
save('independent_validation.json',dict(status='PASS',validated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=['all16 population membership','all frame indices and episode lengths','pixel widths versus probability approximations','terminal end and absence of earlier executed end','per-case summary metrics and frozen morphology thresholds','same-category globally nearest expert match with deterministic tie-break','control window endpoints','28 video/input SHA256 identities','original development HEAD/diff/status unchanged','all resource counters zero'],metrics=metrics))
# Explicitly bind the archived run metadata without copying secrets or full environment.
run_dir=ROOT/Path(bysid[0]['video_path']).parent
meta=run_dir/'wandb/wandb/run-20260803_011640-z9jilvit/files/wandb-metadata.json'
m=rd(meta)
save('historical_run_identity.json',{'server_path':str(meta),'sha256':sha(meta),'size':meta.stat().st_size,'git_commit':m.get('git',{}).get('commit'),'program':m.get('program'),'python':m.get('python'),'note':'Selected provenance fields only; no full environment or credentials copied.'})
# Scientific plot from derived, verified probabilities.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(4,4,figsize=(13,9),sharey=True)
for ax,c,co in zip(axs.flat,cases,controls):
 n=int(c['sub_house_id']);cn=int(co['control_sub_house_id']);ftr=traces[n];ctr=traces[cn];end=int(co['window_last_frame_0based'])
 ax.step(range(len(ftr)),[float(r['p_end_approx']) for r in ftr],where='post',color='#b91c1c',label='failure')
 ax.step(range(end+1),[float(r['p_end_approx']) for r in ctr[:end+1]],where='post',color='#2563eb',alpha=.75,label='matched success window')
 ax.axhline(.5,color='gray',ls=':',lw=.7);ax.set_ylim(-.03,1.05)
 ax.set_title('sub'+str(n)+' / control '+str(cn),fontsize=9)
 ax.set_xlabel('decision index (0-based)',fontsize=7);ax.tick_params(labelsize=7)
handles,labels=axs.flat[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=2)
fig.suptitle('Historical rendered p(end): quantized approximation, not exact logits',y=.995,fontsize=11)
fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'p_end_traces.svg');plt.close(fig)
table='| Morphology | n/16 | Fraction |\n| --- | --- | --- |\n'+''.join('| '+k+' | '+str(v)+'/16 | '+format(v/16,'.1%')+' |\n' for k,v in counts.items())
case_table='| Sample ID | Decisions | Pre-end max | At end approx | Morphology | Scope |\n| --- | --- | --- | --- | --- | --- |\n'+''.join('| '+c['sub_house_id']+' | '+c['episode_length']+' | '+format(float(c['max_p_end_before_end']),'.3f')+' | '+format(float(c['p_end_at_end']),'.3f')+' | '+c['morphology']+' | '+c['target_scope']+' |\n' for c in cases)
report=f"""# Premature-end dynamics: historical observational audit

Status: AWAITING_PI_REVIEW. All 16/16 candidate failure traces were recovered, with exact archived video hashes and frame counts equal to episode length. Every final executed action is end; none has an earlier executed end. No stop condition was triggered. The matched-control audit recovered 16/16 pairs using 11 unique same-category success videos, with replacement and no cross-category fallback.

## Frozen morphology results

{table}
The fractions describe this fixed historical 16-case census; they are not estimates of a new rollout's performance or independent causal effects. No new SR or Safety Cost was measured. Reference only: the aligned historical run had 173/200 success and total Safety Cost 145.

## What the traces actually show

All 16 first-decision end bars are sub-resolution. All preterminal p(end) maxima are below 0.5; the largest recovered value is {max(premax):.3f}. For all {terminal_only_05} cases that ever cross 0.5, the first crossing occurs at the executed end decision itself. Thus LATE_RISE is the preregistered label for a low early window and a high terminal value; these data do not show a gradual buildup after prolonged unsuccessful search.

The one EARLY_HIGH_PRIOR case is sub55, a two-decision sofa failure: step0 is sub-resolution and step1 is the terminal high-probability end. Its first-five window necessarily includes termination. This is an early-window morphology, not evidence of a high task-start prior before exploration. LOW_PROB_END has 0/16 under the specified <0.1 threshold. Sub24 (~0.418 at end) and sub117 (~0.200) are OTHER_OR_UNCLEAR; stochastic execution remains possible without meeting that low-probability category.

{case_table}

## Matched-success contrast and its limits

All 16 pair-specific success early windows have max p(end)<0.5 (indeed sub-resolution in the first five available frames). However, {control_terminal_first}/16 matched pairs ({unique_control_terminal_first}/11 unique controls) also first cross 0.5 only at their own successful terminal decision. Abrupt terminal probability increase is therefore not specific to failed termination.

Thirteen of 16 prescribed windows include the control's successful terminal frame because those controls finish earlier. The median paired window-end probability difference is {metrics['paired_window_endpoint_difference_median']:.3f}; this contrast often compares failed end to successful end, not equal-duration search states. The median expert-length gap is {metrics['expert_gap_median']} decisions (range {metrics['expert_gap_range'][0]}–{metrics['expert_gap_range'][1]}). Matching is an incomplete task-difficulty proxy; controls are reused and pairs are not 16 independent control episodes. The two-decision sub55 comparison is descriptive only, with a 42-decision expert-length gap.

## Measurement validation

Eight row/action identities were frozen from the archived rendering source and the existing sub120 method. One black-text row must be uniquely detected at three thresholds (60/70/80). End probabilities use the preserved strict-blue endpoint estimator divided by 55. A secondary chromatic endpoint method gives the same 16 morphology labels; its end endpoints differ by at most one pixel in analyzed case/control frames. All 16 labels also survive the declared +/-2/55 nonzero-value sensitivity perturbation (zero represented as [0,1/55]). These are measurement sensitivity checks, not statistical confidence intervals.

The inclusive PIL rectangle endpoint and lossy video color thresholds prevent exact probability recovery. A zero bar means below approximately 1/55, not exact zero. A displayed estimate capped at 1 means near-saturated, not proven true probability 1. Threshold-crossing indices are based on quantized approximate values. The decoder reproduced all 600 preserved sub120 executed actions and end approximations; sub120 is only a validation reference, excluded from the 16-case result.

Frame0 maps to the first policy decision; the on-video counter is 1-based. Archived worker source passes the same decision's probabilities and actual executed action into the frame and saves the terminal frame before breaking. Source excerpts and exact hashes are in source_semantics.json. Current worker source differs from that commit and is not substituted for historical evidence. This audit never imports or runs that worker.

## Target-evidence boundary

12 cases have STRICT narrow==broad target IDs; 4 (sub0, sub24, sub110, sub175) are AMBIGUOUS. The CSV preserves both ID sets. Historical narrow visibility and room visitation are descriptive aggregate fields, not per-decision stop-legality proof. For ambiguous cases, zero narrow visibility cannot establish that every success-eligible target was unseen. Stable sub_house_id is used only as an ID/tie-break, not as house identity or difficulty.

## Interpretation and one proposed next branch

The preregistered leading morphology is LATE_RISE (13/16); there is no evidence for a population-wide high initial prior or <0.1 stochastic-end morphology. The more precise observational description is an abrupt terminal Actor-end probability jump, which also occurs in matched successful episodes. The present audit does not identify why an end decision is invalid, whether a target was correctly grounded, or which training component caused it.

Proposed next causal branch, not approved: following the frozen LATE_RISE decision rule, PI may design a matched hard-task policy comparison between SafeVLA and the closest justified non-safety/base policy (ideally FLaRe if available), with protocol/checkpoint comparability explicitly reviewed. This is a discrimination test, not attribution to safety alignment. No comparison, checkpoint search/load, hidden-state Probe, replay or rollout was started. Gate B remains unresolved for later hidden-state interpretation. Current Executor stops after handoff publication.

## Evidence

- premature_end_cases.csv: complete case-level metrics and target scope.
- matched_success_controls.csv: exact matches, gaps, windows and contrasts.
- pdone_trace_summary.csv and trace_sub*.csv: all approximate probability/action traces.
- frozen_extraction_plan.json, reference_validation.json, independent_validation.json: frozen rules and checks.
- input_manifest.json and ARTIFACT_INDEX.json: original video/table/script identities; large original media remain server-side.

![Historical p(end) traces](p_end_traces.svg)
"""
(OUT/'premature_end_analysis.md').write_text(report)
(OUT/'RESULT_SUMMARY.md').write_text(f"""# EXP-PREMATURE-END-DYNAMICS-001 — AWAITING_PI_REVIEW

All 16 historical sub-horizon failure videos are reliable and confirm final executed end. The frozen morphology counts are LATE_RISE 13/16 (81.25%), EARLY_HIGH_PRIOR 1/16 (6.25%), OTHER_OR_UNCLEAR 2/16 (12.5%), LOW_PROB_END 0/16.

The key refinement is temporal: all first-decision end bars are sub-resolution; all preterminal maxima are <0.5. Every first >=0.5 crossing is at the terminal decision. The single early-high case is a two-step episode whose terminal decision falls inside the first-five window, not evidence of a high initial prior.

All 16 same-category success matches were recovered (11 unique controls, reused); {unique_control_terminal_first}/11 unique controls also first cross 0.5 at their own terminal decision. Thirteen prescribed control windows include successful termination. Thus the common abrupt terminal jump does not distinguish failure causation. Expert-length gaps are imperfect matching, median {metrics['expert_gap_median']} (range {metrics['expert_gap_range'][0]}–{metrics['expert_gap_range'][1]}).

These are video-rendered quantized probability approximations (~1/55 resolution), not exact logits. Zero does not mean true zero; capped1 does not mean true1. All labels agree under a secondary color method and the declared quantization sensitivity check. Target scope is STRICT12 / AMBIGUOUS4; narrow visibility is not broad-target stop legality.

Independent validation passed for source/video hashes, frame alignment, all case metrics, control selection/windows and unchanged development identity. Actual resources: 0 GPU, 0 episode, 0 simulator, 0 model/checkpoint load, 0 Actor/Critic forward. No behavior intervention or causal attribution was made.

The proposed next branch follows the frozen LATE_RISE rule: PI review of a matched-task SafeVLA versus comparable non-safety/base-policy causal comparison. It is not approved or executed. No new experiment is self-approved. See premature_end_analysis.md for full evidence and limitations. Next actor PI; STOP after handoff.
""")
(OUT/'REVIEW_NOTES.md').write_text("""# PI review notes

Approval instruction HEAD: 681cd28355cfa79a6eaec7f663d9b3776da7f37e.
Existing claim: a5045c5255b67f3e0f1d4f559d2d6078f70f9544.
Claim ID: 37f4a6ae9b5d41ada6753d75f2b30caa.
Claim CI: https://github.com/whatcanidowhat/SafeVLA/actions/runs/35845852871 (success before execution).
The same preserved claim was recovered and published after explicit user authorization; no new claim was generated. The user separately authorized experiment execution after claim recovery.

Review the raw derived traces and frozen plan alongside case/control CSVs, source_semantics, reference_validation and independent_validation. Original source media hashes are provided; raw videos and inspection PNGs are server-only optional evidence, not uploaded into the shared control plane. All required review outputs and derived numerical evidence have Git-readable copies.

Interpretation pitfalls: LATE_RISE includes a jump occurring only at the terminal decision; first-five EARLY_HIGH_PRIOR includes the terminal decision in the two-step sub55 case; successful controls show similar terminal jumps. Expert matching has replacement and large gaps in several cases, and most windows include the successful end. Quantized p=0/p=1 are not exact probabilities. Four target scopes are ambiguous. Do not infer SafeRL cause, cost-critic gating, deliberate scene training, leakage or hidden-state mechanisms.

No stop condition was triggered. No PI acknowledgement is fabricated. Approved NEXT_EXPERIMENT files and claim/authorization identities remain frozen. Next action is PI review; any causal comparison requires a separate approved cycle. Executor STOP after publication.
""")
manifest={k:start['state'][k] for k in ['cycle_id','experiment_id','instruction_commit','claim_id']}
manifest.update(schema_version='1.0',status='AWAITING_PI_REVIEW',claim_commit=start['claim_commit'],claim_ci_url=start['claim_ci'],command=['/home/amax/.conda/envs/safevla/bin/python',PREFIX+'analyze_premature_end.py'],execution_worktree=str(ROOT),started_at_utc=result['started_at_utc'],completed_at_utc=result['completed_at_utc'],gpu_count=0,episodes_started=0,simulator_launches=0,model_checkpoint_loads=0,actor_critic_forwards=0,exit_code=0,scope='historical CPU video decode and descriptive analysis only',development_head=start['head'],development_status=start['status'],development_tracked_diff_sha256=start['tracked_diff_sha256'],development_preserved=True,actual_python=sys.executable,python_version=sys.version,inputs='input_manifest.json',results='analysis_result.json',independent_validation='independent_validation.json',frozen_design_sha256={n:sha(CONTROL/'research'/n) for n in ['NEXT_EXPERIMENT.md','NEXT_EXPERIMENT.json']},source_sha256=sha(OUT/'analyze_premature_end.py'),preflight_command=['/home/amax/.conda/envs/safevla/bin/python','/tmp/prepare-premature-evidence.py'],handoff_validation_command=['/home/amax/.conda/envs/safevla/bin/python',PREFIX+'finalize_premature_handoff.py'],stdout_path='/tmp/premature-end-analysis-20260923.stdout',stderr_path='/tmp/premature-end-analysis-20260923.stderr',seed_workers_checkpoint='Not applicable: no policy or environment execution',resources={'opencv':'CPU FFMPEG backend, HW acceleration NONE, OpenCL disabled, threads1'})
save('RUN_MANIFEST.json',manifest)
art=[]
for p in sorted(OUT.iterdir()):
 if p.name=='ARTIFACT_INDEX.json':continue
 if p.suffix in {'.json','.csv','.py','.md','.svg'}:
  assert p.stat().st_size<1048576
  art.append(dict(name=p.name,server_path=str(p),git_path=PREFIX+p.name,sha256=sha(p),size=p.stat().st_size,required_for_PI_review=True))
 elif p.suffix=='.png':
  art.append(dict(name=p.name,server_path=str(p),sha256=sha(p),size=p.stat().st_size,required_for_PI_review=False,access_boundary='server-only optional inspection frame'))
 else:raise AssertionError('Unexpected output '+p.name)
for path,item in rd(OUT/'input_manifest.json').items():
 if path.endswith('.mp4'):art.append(dict(name=Path(path).name,server_path=path,sha256=item['sha256'],size=item['size'],required_for_PI_review=False,access_boundary='server-only original video; Git-readable extracted traces provide review evidence'))
save('ARTIFACT_INDEX.json',dict(schema_version='1.0',cycle_id=manifest['cycle_id'],status='AWAITING_PI_REVIEW',artifacts=art))
assert all((OUT/Path(p).name).is_file() for p in start['state']['required_outputs'])
print(json.dumps({'status':'HANDOFF_READY','metrics':metrics,'shared_files':sum('git_path' in a for a in art)+1}))
