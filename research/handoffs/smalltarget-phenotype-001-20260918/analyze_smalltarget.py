#!/usr/bin/env python3
"""Finalize the metadata-first BLOCKED handoff. Standard library only, no runtime imports."""
import csv,datetime,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path('/nvme2/user/qyy/SafeVLA')
CONTROL=Path('/nvme2/user/qyy/SafeVLA_loop_control')
OUT=Path(__file__).resolve().parent
PACKET=CONTROL/'research/history/evidence-alignment-20260919'
CLAIM='72634cf7c3a954dd99456b2d75c3af539da9e68e'
APPROVAL='290b426ac65d35ef1f8cb4819e9662c4fae1bb31'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,v):(OUT/name).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def text(name,v):(OUT/name).write_text(v,encoding='utf-8')
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])
def writecsv(name,rows,fields):
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 assert Path.cwd()==ROOT
 start=json.loads((OUT/'execution_start.json').read_text())
 state=json.loads((CONTROL/'research/LOOP_STATE.json').read_text())
 assert state==start['state'] and state['status']=='CODEX_RUNNING'
 assert state['instruction_commit']==APPROVAL
 audit=json.loads((OUT/'metadata_preflight.json').read_text())
 s=audit['summary'];targets=audit['targets']
 assert (s['tasks'],s['target_occurrences'],s['either_bbox_occurrences'],s['tasks_all_targets_have_candidate_bbox'])==(200,368,42,31)
 # These are candidate asset-space bounds, not a certified instance-space measurement.
 # No outcome, category, distance, visibility, or policy feature participates in the gate.
 gate={'status':'BLOCKED','reason_code':'STATIC_PHYSICAL_SIZE_NOT_VALIDATED',
 'tasks':200,'mapped_broad_target_instances':368,'unmatched_target_instances':0,
 'candidate_asset_bbox_instances':42,'instances_without_candidate_bbox':326,
 'tasks_with_candidate_bbox_for_every_target':31,'tasks_missing_one_or_more_candidate_bbox':169,
 'validated_task_physical_sizes':0,'size_analyzable_n':0,
 'units_scale_instance_transform_contract':'NOT_ESTABLISHED',
 'decision':'Available static evidence cannot establish the approved per-task physical size. Stop before category SR, tertiles, association or regression.',
 'scope_limit':'This is a failure to construct a valid variable from inspected local sources; not proof no additional static dataset exists.',
 'forbidden_proxies_used':[],'model_loads':0,'simulator_launches':0,'gpu_count':0,'episodes_started':0}
 save('metadata_gate.json',gate)
 # After the gate stops analysis, copy existing raw fields only for a reviewable handoff.
 summary=json.loads((PACKET/'full200/wandb-summary.json').read_text())
 tablepath=PACKET/'full200'/summary['VideoTable/ObjectNavType']['path']
 assert digest(tablepath)==summary['VideoTable/ObjectNavType']['sha256']
 table=json.loads(tablepath.read_text());rows=[dict(zip(table['columns'],r)) for r in table['data']]
 tasks=[json.loads(l) for l in (PACKET/'task_specs/objectnavtype_val.jsonl').read_text().splitlines() if l]
 def key(p):return p[p.index('ObjectNavType/val/'):]
 taskmap={key(t['task_path']):t for t in tasks}
 assert len(rows)==len(taskmap)==200 and len({key(r['task_path']) for r in rows})==200
 assert sum(r['success'] for r in rows)==173 and sum(r['sum_cost'] for r in rows)==145
 outrows=[]
 for r in rows:
  t=taskmap[key(r['task_path'])]
  assert r['gt_episode_len']==t['expert_length']
  oid=sorted(set(t['broad_synset_to_object_ids'][t['synsets'][0]]))
  cand=[a for a in targets if a['house_index']==t['house_index'] and a['target_id'] in oid]
  assert len(cand)==len(oid)
  outrows.append({'task_key':key(t['task_path']),'house_index':t['house_index'],'target_synset':t['synsets'][0],
   'success':r['success'],'episode_length':r['eps_len'],'expert_length':t['expert_length'],
   'broad_target_ids_json':json.dumps(oid),'broad_target_count':len(oid),
   'candidate_asset_bbox_count':sum(bool(a['house_annotation_bbox'] or a['disk_metadata_bbox']) for a in cand),
   'physical_size_volume':'','physical_size_max_side':'','physical_size_status':'UNVALIDATED_METADATA_GATE_BLOCKED',
   'failure_termination_type':'NOT_DERIVED_GATE_BLOCKED','initial_target_distance':'',
   'max_nav_visible_pixels':'','has_agent_been_in_room':r['has_agent_been_in_room'],
   'raw_source_status':'EXISTING_HISTORICAL_EVIDENCE_NO_NEW_ROLLOUT'})
 writecsv('episode_table.csv',outrows,list(outrows[0]))
 categories=sorted({t['synsets'][0] for t in tasks})
 writecsv('category_sr.csv',[{'target_synset':c,'n':'','successes':'','sr':'','wilson95_low':'','wilson95_high':'','status':'NOT_EXECUTED_METADATA_GATE_BLOCKED'} for c in categories],
  ['target_synset','n','successes','sr','wilson95_low','wilson95_high','status'])
 save('raw_identity_check.json',{'status':'PASS_EXISTING_EVIDENCE_IDENTITY_ONLY','rows':200,'unique_task_keys':200,'successes':173,'sum_cost':145,'length_matches':200,'source_table':str(tablepath),'source_table_sha256':digest(tablepath),'category_statistics_computed':False,'size_associations_computed':False})
 refs=[]
 for rel,ranges,purpose in [
  ('tasks/object_nav_task.py',[[119,129]],'Success eligible IDs use broad_synset_to_object_ids, not narrow IDs.'),
  ('online_evaluation/online_evaluator.py',[[461,479]],'minival selects static val houses dataset.'),
  ('utils/data_utils.py',[[25,70],[158,204]],'JSONL dataset indexing and loader preserve row identity.'),
  ('tasks/abstract_task_sampler.py',[[59,65]],'House index maps to selected house list.'),
  ('utils/constants/objaverse_data_dirs.py',[],'Dataset path constants; source identity only.')]:
  p=ROOT/rel
  if p.exists():refs.append({'path':str(p),'sha256':digest(p),'size':p.stat().st_size,'line_ranges':ranges,'purpose':purpose})
 save('source_references.json',refs)
 after={'head':git('rev-parse','HEAD').decode().strip(),'status':git('status','--porcelain').decode(),'tracked_diff_sha256':hashlib.sha256(git('diff','--binary')).hexdigest()}
 assert all(after[k]==start[k] for k in ['head','status','tracked_diff_sha256'])
 save('development_preservation.json',{'before':{k:start[k] for k in after},'after':after,'identical':True,'note':'Only new handoff files under the already-untracked research directory were added; existing tracked changes and status preserved.'})
 ended=datetime.datetime.now(datetime.timezone.utc).isoformat()
 note="""# EXP-SMALLTARGET-PHENOTYPE-001 — BLOCKED

The metadata-first stop condition is met. Existing local static sources do not establish the approved policy-independent task-level physical size. No category SR, Wilson intervals, size tertiles, effect estimates, or regression were executed.

All 200 task specifications map to 200 scene records. Every one of the 368 broad-synset success-eligible target IDs maps uniquely to a scene object. Candidate asset bounding boxes exist for 42 targets and agree between house annotations and per-asset thor_metadata.json. The other 326 targets have no candidate bounding box in the inspected static sources. Only 31/200 tasks have candidate boxes for every valid target; 169/200 lack at least one. Candidate coverage is not validated size coverage: units, scale, scene transformation and actual instance-bound equivalence were not established. Validated size count and analyzable n are 0.

The general annotations.size field contains annotated dimensions (including GPT-4 labels); it was not substituted for physical size. No distance, visible-pixel measure or category-name estimate was used. No incomplete-target median and no 31-task selected-subset association was computed.

The local search covered the archived scene and asset annotation files, per-target thor_metadata.json, relevant project and installed package metadata candidates, and bounded standard cache locations. Three THOR build metadata files contain only server_types, not object dimensions. This bounded audit does not prove another static source could never resolve the gap.

Historical raw identity remains 200 unique tasks, 173 successes, cost sum 145 and 200 expert-length matches. These are integrity checks of existing evidence, not a new evaluation.

episode_table.csv contains existing raw outcomes and task IDs with all physical-size cells empty. category_sr.csv is an explicit NOT_EXECUTED_METADATA_GATE_BLOCKED status artifact: blank numeric cells are not zeros or completed category results.

Next actor: PI. A proposed prerequisite is a version-bound static geometry/instance-transform source for all broad targets, including built-in THOR assets, with documented units and scaling. This is a suggestion only, not approval or a new run. Preserve the current design and stop; any restart must follow PI's recovery protocol.

Resources used by this task: 0 GPU / 0 episodes / 0 model loads / 0 simulator launches. No Probe, replay, video analysis, reset treatment, Safe-vs-IL or new full-200 evaluation.
"""
 text('RESULT_SUMMARY.md',note)
 text('size_analysis.md',"""# Size analysis — NOT EXECUTED / BLOCKED

Primary variable: per-task median 3D bounding-box volume across all valid broad-synset target instances; median maximum side length as robustness descriptor.

The static metadata gate failed before estimating this variable. See metadata_gate.json and metadata_preflight.json. Asset-space candidates are not declared actual scene-instance sizes. Missing geometry for 326/368 targets affects 169/200 tasks; even the 31 candidate-complete tasks lack a validated units/scale/instance-bound contract. Therefore physical-size values are missing for all 200 tasks and analyzable n=0.

No size effects, confidence intervals, tertiles, logistic models, or category-statistical analyses were run. No conclusion for or against H-SIZE follows from unavailable measurements. H-CATEGORY, H-DIFFICULTY and H-SAMPLE remain untested alternatives in this execution.
""")
 text('REVIEW_NOTES.md',"""# PI review notes

- Confirm claim 72634cf7c3a954dd99456b2d75c3af539da9e68e binds approval 290b426ac65d35ef1f8cb4819e9662c4fae1bb31. Claim CI 35499476480 succeeded before any static metadata execution.
- Review the distinction between 42 candidate asset boxes, 31 candidate-complete tasks, and zero validated task sizes.
- Broad targets were used; narrow IDs and the chosen exemplar were not substituted.
- category_sr.csv deliberately contains blank numeric fields with NOT_EXECUTED status because the user required metadata validation first and STOP on failure.
- Annotated size, target distance, trajectory visibility, category names and incomplete target subsets were not used as physical size.
- The bounded search supports BLOCKED for current evidence, not a claim of global nonexistence.
- Preserve original experiment design and baseline. No self-approval, renewal or next experiment was performed. PI must decide the recovery path before any restart.
""")
 save('RUN_MANIFEST.json',{'schema_version':'1.0','cycle_id':state['cycle_id'],'experiment_id':state['experiment_id'],'instruction_commit':APPROVAL,'claim_id':state['claim_id'],'claim_commit':CLAIM,'claim_ci_url':'https://github.com/whatcanidowhat/SafeVLA/actions/runs/35499476480','status':'BLOCKED','scientific_status':'STATIC_PHYSICAL_SIZE_NOT_VALIDATED','started_at_utc':start['started_at_utc'],'completed_at_utc':ended,'command':['python3','research/handoffs/smalltarget-phenotype-001-20260918/analyze_smalltarget.py'],'execution_worktree':str(ROOT),'actual_python':sys.executable,'python_version':sys.version,'gpu_count':0,'episodes_started':0,'model_loads':0,'simulator_launches':0,'analysis_imports':'Python standard library only','exit_code':0,'exit_code_interpretation':'Successfully produced a BLOCKED handoff; not a successful scientific size analysis.','execution_stages':['claim via repository script and wait for green CI','read-only static schema/source inspection using Python over SSH','python3 research/handoffs/smalltarget-phenotype-001-20260918/metadata_preflight.py','bounded local static database search','python3 research/handoffs/smalltarget-phenotype-001-20260918/analyze_smalltarget.py'],'frozen_design_sha256':{x:digest(CONTROL/'research'/x) for x in ['NEXT_EXPERIMENT.md','NEXT_EXPERIMENT.json']},'source_files':audit['sources'],'development_head':after['head'],'development_tracked_diff_sha256':after['tracked_diff_sha256'],'source_references':'source_references.json','limitations':['Unmatched geometry/size evidence, not model failure','No global filesystem exhaustiveness claim','No scientific category or size model executed']})
 files=[p for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='ARTIFACT_INDEX.json']
 prefix='research/handoffs/'+state['cycle_id']+'/'
 save('ARTIFACT_INDEX.json',{'schema_version':'1.0','artifacts':[{'name':p.name,'server_path':str(p),'sha256':digest(p),'size':p.stat().st_size,'required_for_PI_review':True,'git_path':prefix+p.name} for p in files]})
 print(json.dumps({'status':'BLOCKED','outputs':len(files)+1,'validated_sizes':0,'raw_rows':200,'development_preserved':True}))
if __name__=='__main__':main()
