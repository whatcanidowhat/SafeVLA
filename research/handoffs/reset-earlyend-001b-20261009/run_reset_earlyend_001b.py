"""Frozen 001B P0 audit. No policy/simulator execution. Fails closed on new evidence.

Reproduces the unreconstructable-counter handoff from original historical files.
It deliberately does not implement P1 after the mandatory P0 stopping condition.
"""
import collections, csv, datetime, hashlib, io, json, os, re, subprocess, sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2

DEV=Path('/nvme2/user/qyy/SafeVLA')
CONTROL=Path('/nvme2/user/qyy/SafeVLA_loop_control')
CYCLE='reset-earlyend-001b-20261009'
OUT=DEV/'research/handoffs'/CYCLE
PRIOR=CONTROL/'research/handoffs/premature-end-dynamics-001-20260923'
HIST=CONTROL/'research/history/evidence-alignment-20260919'
BASE=DEV/'eval/objectnav-full-minival-200-20260803-gpu0-w4'
CLAIM='fd3a11bbc5bb6ef365f3f1edc68e1e6a9888264c'
INSTRUCTION='919f62294d1ebcbdff1035112e7cf8c1c4530737'
HIST_COMMIT='60bc54fbdedaf5745d0476c25321e808708273aa'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def git(root,*args): return subprocess.check_output(['git','-C',str(root),*args])
def rows(p): return list(csv.DictReader(io.StringIO(Path(p).read_text())))
def writecsv(name,fields,data):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)
def identity(p): return {'server_path':str(p),'sha256':sha(p),'size':p.stat().st_size}

def main():
    assert Path.cwd()==DEV, 'Execute CPU audit from development checkout only'
    assert git(CONTROL,'rev-parse','HEAD').decode().strip()==CLAIM
    state=json.loads((CONTROL/'research/LOOP_STATE.json').read_text())
    assert state['status']=='CODEX_RUNNING' and state['claim_id']=='279393e2f8a44f58b71c01ced81f0bbe'
    assert state['instruction_commit']==INSTRUCTION
    assert not (OUT/'RUN_MANIFEST.json').exists(), 'Never overwrite an existing audit'
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    before=git(DEV,'diff','HEAD','--binary')
    provenance={'development_HEAD':git(DEV,'rev-parse','HEAD').decode().strip(),
                'tracked_diff_sha256_before':hashlib.sha256(before).hexdigest(),
                'tracked_status_before':git(DEV,'status','--porcelain','--untracked-files=no').decode(),
                'python':sys.executable,'python_version':sys.version,'pyarrow_version':pa.__version__,
                'cwd':str(Path.cwd()),'runner_sha256':sha(Path(__file__))}
    full=rows(HIST/'full200/episode_results.csv')
    assert len(full)==200
    byid={int(re.search(r'sub_house_id=(\d+)',r['video_path']).group(1)):r for r in full}
    candidates={i:r for i,r in byid.items() if r['success']=='False' and int(r['eps_len'])<600}
    cases=rows(PRIOR/'premature_end_cases.csv')
    assert set(map(lambda r:int(r['sub_house_id']),cases))==set(candidates) and len(cases)==16
    oldinputs=json.loads((PRIOR/'input_manifest.json').read_text())
    inputs=[identity(HIST/'full200/episode_results.csv'),identity(PRIOR/'premature_end_cases.csv'),
            identity(PRIOR/'input_manifest.json'),identity(PRIOR/'independent_validation.json'),
            identity(HIST/'FILE_INVENTORY.json')]
    assert sha(HIST/'full200/episode_results.csv')==oldinputs[str(HIST/'full200/episode_results.csv')]['sha256']
    assert json.loads((PRIOR/'independent_validation.json').read_text())['status']=='PASS'
    manifest=[]; exposure=[]
    reason='Original task-to-worker assignment and worker-local order/counter not present in inspected persisted evidence'
    for c in sorted(cases,key=lambda r:int(r['sub_house_id'])):
        sid=int(c['sub_house_id']); r=candidates[sid]; trace=PRIOR/c['trace_file']; t=rows(trace)
        assert c['confirmed_invalid_end']=='True' and c['reliable_trace']=='True'
        assert c['final_executed_action']=='end' and t[-1]['executed_action']=='end'
        assert len(t)==int(r['eps_len'])==int(c['episode_length'])
        assert int(t[-1]['frame_0based'])==int(c['end_step_0based'])==len(t)-1
        assert all(x['executed_action']!='end' for x in t[:-1])
        assert r['task_path']==c['task_key']
        video=DEV/r['video_path']; assert sha(video)==oldinputs[str(video)]['sha256']
        inputs.extend([identity(trace),identity(video)])
        manifest.append({'sub_house_id':sid,'task_key':c['task_key'],'house_index':c['house_index'],
                         'success':False,'eps_len':len(t),'historical_end_step_0based':len(t)-1,
                         'final_executed_action':'end','confirmed_failed_end':True,
                         'confirmation_source':str(trace.relative_to(CONTROL)),
                         'trace_sha256':sha(trace),'video_sha256':sha(video),'eligible':True})
        exposure.append({'sub_house_id':sid,'task_key':c['task_key'],'historical_end_step_0based':len(t)-1,
                         'worker_id':'','worker_local_order':'','counter_start':'','rollover_local_step':'',
                         'exposure':'UNKNOWN','included_primary':False,'paired_execution':'NOT_STARTED','reason':reason})
    # Inspect all original W&B table schemas, not only the last aggregate.
    journal=next(BASE.rglob('run-z9jilvit.wandb')); wb=journal.parent
    tables=[]; suspicious=[]
    fieldpat=re.compile(r'worker_id|sample_id|counter_start|time_step_counter|worker_local|episode_order|^iter$')
    for p in sorted((wb/'files/media/table').rglob('*.table.json')):
        obj=json.loads(p.read_text()); cols=obj['columns']
        tables.append({**identity(p),'columns':cols,'rows':len(obj['data'])})
        # sample_id alone identifies the task but not a worker-local history.
        if any(re.search(r'worker_id|counter|worker_local|episode_order|^iter$',str(k)) for k in cols): suspicious.append(str(p))
    ds=DataStore(); ds.open_for_scan(str(journal))
    counts=collections.Counter(); hkeys=set(); skeys=set(); console=collections.defaultdict(list); hits=[]
    while True:
        b=ds.scan_data()
        if b is None: break
        rec=wandb_internal_pb2.Record(); rec.ParseFromString(b); kind=rec.WhichOneof('record_type'); counts[kind]+=1
        if kind in ('history','summary'):
            items=rec.history.item if kind=='history' else rec.summary.update
            for item in items:
                key=item.key or '.'.join(item.nested_key)
                (hkeys if kind=='history' else skeys).add(key)
                if fieldpat.search(key): hits.append({'record_type':kind,'key':key,'value':item.value_json[:300]})
        if kind in ('output','output_raw'):
            part=getattr(rec,kind); console[str(part.output_type)].append(part.line)
    consoletext='\n'.join(''.join(v) for v in console.values())
    mappingpat=re.compile(r'worker_id|counter_start|time_step_counter|worker_local|episode_order|[\x27\"]iter[\x27\"]\s*:')
    consolehits=[x[:500] for x in consoletext.splitlines() if mappingpat.search(x)]
    logs=[]
    for p in [BASE/'launcher.log',wb/'files/output.log',wb/'logs/debug.log',wb/'logs/debug-internal.log']:
        txt=p.read_text(errors='replace'); found=[x[:500] for x in txt.splitlines() if mappingpat.search(x)]
        logs.append({**identity(p),'worker_mapping_hits':found,
                     'worker_lifecycle_lines':[x[:300] for x in txt.splitlines() if re.search(r'Starting worker|Worker \d+ processed|Joining worker|Joined worker',x)]})
    inventory=json.loads((HIST/'FILE_INVENTORY.json').read_text())['records']
    invpaths=[r['server_path'] for r in inventory if str(BASE) in r.get('server_path','')]
    nonmedia=[p for p in invpaths if not p.endswith(('.mp4','.png','.table.json'))]
    # Persist a bounded search scope; do not claim absence from unavailable external logs.
    audit={'scope':str(BASE),'table_count':len(tables),'table_schemas':tables,'suspicious_table_paths':suspicious,
           'journal':identity(journal),'journal_record_counts':dict(counts),
           'history_keys':sorted(hkeys),'summary_keys':sorted(skeys),'history_summary_mapping_hits':hits,
           'journal_reassembled_console_mapping_hits':consolehits,'log_audit':logs,
           'historical_inventory_matching_records':len(invpaths),'historical_inventory_nonmedia_paths':nonmedia,
           'limitation':'Bounded to retained original run, aligned historical evidence and referenced source; no assertion about unavailable external worker logs.',
           'inference':'Global completion order, progress timestamps, filenames and aggregate num_workers cannot identify each dynamic worker queue history.'}
    dump('source_audit.json',audit)
    assert not suspicious and not hits and not consolehits and not any(x['worker_mapping_hits'] for x in logs), 'New potential mapping evidence: inspect before continuing'
    source=[]
    for name in ['online_evaluation/online_evaluator_worker.py','online_evaluation/online_evaluator.py']:
        raw=git(DEV,'show',HIST_COMMIT+':'+name); lines=raw.decode().splitlines(); selected=set()
        for n,line in enumerate(lines):
            if any(k in line for k in ['verbose =','iter=num_tasks','results_queue.put','results_queue.get','contributing_workers','tab_data','worker_id','VideoTable','metrics_table']):
                selected.update(range(max(0,n-3),min(len(lines),n+12)))
        source.append({'commit':HIST_COMMIT,'path':name,'sha256':hashlib.sha256(raw).hexdigest(),
                       'excerpts':[{'line':n+1,'text':lines[n]} for n in sorted(selected)]})
    dump('historical_source_excerpts.json',source)
    writecsv('earlyend_case_manifest.csv',list(manifest[0]),manifest)
    writecsv('counter_exposure_audit.csv',list(exposure[0]),exposure)
    writecsv('paired_episode_results.csv',['sub_house_id','condition','seed','failed_end','end_step','success','eps_len','official_safety_cost','first_action_divergence_step','status'],[])
    schema=pa.schema([('sub_house_id',pa.int64()),('condition',pa.string()),('step_0based',pa.int64()),
                      ('executed_action',pa.string()),('p_end',pa.float64()),('end_rank',pa.int64()),
                      ('end_margin',pa.float64()),('rollover_local_step',pa.int64())],
                     metadata={b'status':b'NOT_STARTED',b'reason':b'P0_BLOCKED_ALL_COUNTERS_UNKNOWN'})
    pq.write_table(pa.Table.from_pylist([],schema=schema),OUT/'paired_step_trace.parquet')
    assert pq.read_table(OUT/'paired_step_trace.parquet').num_rows==0
    assert before==git(DEV,'diff','HEAD','--binary')
    provenance['tracked_diff_sha256_after']=hashlib.sha256(git(DEV,'diff','HEAD','--binary')).hexdigest()
    dump('runtime_provenance.json',provenance); dump('input_identities.json',inputs)
    summary='''# EXP-RESET-EARLYEND-001B — B4 / BLOCKED

P0 confirms 16 historical unsuccessful sub-horizon cases with executed terminal `end`.
All 16 have UNKNOWN historical worker-local counter_start. No EXPOSED, BOUNDARY or
UNEXPOSED assignment is justified. The frozen P0 stopping condition is met.

The retained original four-worker run persists global result-arrival tables and
aggregate worker counts, not per-task worker_id/worker-local iteration. Its W&B
binary journal (history, summary and reassembled raw console), launcher/output/debug
logs and table schemas contain no recoverable mapping. Historical source creates
worker_id and iter in an in-memory queue payload; Linux disables its per-task print,
and the persisted tables omit those fields. This is an evidence limitation, not a
proof that every possible external worker log is absent. See source_audit.json and
historical_source_excerpts.json for the exact bounded search and source references.

Eligibility is verified against the aligned full200 table, accepted terminal-action
traces and their independent validation; all 16 original video hashes match the
accepted input manifest. Terminal end is not inferred from episode length alone.
Historical end steps are zero based. Missing counters and rollover steps stay blank.
No counter, worker assignment, cache or order is invented from timestamps or global
completion order. The prescribed 500-counter_start rule therefore cannot be applied.

P1 and subsequent online comparisons: NOT_STARTED. GPU use 0; live episodes 0;
policy checkpoint loads 0; simulator starts 0. paired_episode_results.csv is header
only; paired_step_trace.parquet is a typed zero-row table marked NOT_STARTED.
These empty outputs are missing experimental observations, not zero treatment effects.

No causal conclusion B1/B2/B3 is supported, and the accepted 001A R1 result remains
unchanged. This audit neither supports nor refutes reset effects on early termination,
success or official Safety Cost. It cannot identify valid negative-control cases.

Next actor PI. Specific next action: determine whether an authentic task-to-worker
ordered log can be supplied; if unavailable, PI must decide a new design and approval.
No successor experiment, rerun or replacement claim is authorized by this handoff.
Executor STOP after publication.
'''
    (OUT/'RESULT_SUMMARY.md').write_text(summary)
    (OUT/'analysis.md').write_text(summary+'\n## Claim and provenance\n\nInstruction `'+INSTRUCTION+'`; claim `'+CLAIM+'`. Claim control CI 37872063939 succeeded before P0. The audit is CPU-only and does not import the policy or simulator. Source hashes and unchanged tracked development diff are recorded.\n')
    (OUT/'REVIEW_NOTES.md').write_text('''# PI review notes

- Review all 16 eligibility confirmations and exact video/input hashes.
- Review source_audit.json: table fields, history/summary keys, raw console scan,
  original log scans and bounded historical inventory search.
- In historical_source_excerpts.json, distinguish transient worker_id/iter queue
  fields from persisted global tables. num_workers is an aggregate, not assignment.
- All counters and exposure labels remain unknown. Do not relabel unknown as unexposed.
- Empty paired outputs explicitly represent NOT_STARTED. B4 is unreconstructable,
  not a failed policy experiment or evidence for B3.
- A legitimate future source must establish worker assignment and within-worker
  predecessor order/counters. Global completion order alone is insufficient.
- Review or replace the design as PI; no automatic execution or PI acknowledgement.
''')
    m={k:state[k] for k in ['cycle_id','experiment_id','instruction_commit','claim_id']}
    m.update(status='BLOCKED',result_class='B4',claim_commit=CLAIM,claim_ci_run_id=37872063939,
             claim_ci_conclusion='success',started_at_utc=start,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             command=[sys.executable,str(OUT/'run_reset_earlyend_001b.py')],gpu_count=0,episodes_started=0,
             episodes_completed=0,policy_checkpoint_loads=0,simulator_starts=0,policy_forwards=0,
             eligible_cases=16,unknown_exposure_cases=16,reconstructable_cases=0,paired_cases_started=0,
             stages={'P0':'BLOCKED','P1':'NOT_STARTED','online_comparison':'NOT_STARTED'},
             stopping_reason='No early-stop case has reconstructable worker-local counter state',
             budget=state['authorization'],runtime_provenance='runtime_provenance.json',
             design_sha256={n:sha(CONTROL/'research'/n) for n in ['NEXT_EXPERIMENT.md','NEXT_EXPERIMENT.json']})
    dump('RUN_MANIFEST.json',m)
    dump('validation.json',{'status':'PASS','scientific_status':'BLOCKED','checks':['200 unique historical rows','all 16 failed end cases matched','16 terminal action traces checked','16 original video hashes matched','no persisted worker mapping found in specified scope','all counter fields missing rather than invented','paired CSV and Parquet have zero rows','tracked development diff unchanged'],'resource_use':{'gpu':0,'episodes':0}})
    print(json.dumps({'status':'BLOCKED','result_class':'B4','eligible':16,'unknown':16,'episodes_started':0,'gpu_count':0}))

if __name__=='__main__': main()
