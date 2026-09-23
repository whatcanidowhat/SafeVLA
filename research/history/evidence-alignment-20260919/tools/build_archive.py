"""Archive existing SafeVLA evidence. No model, simulator, or experiment execution."""
from pathlib import Path
import csv, datetime, gzip, hashlib, io, json, os, re, subprocess, collections

DEV = Path('/nvme2/user/qyy/SafeVLA')
CONTROL = Path('/nvme2/user/qyy/SafeVLA_evidence_alignment_20260919')
REL = Path('research/history/evidence-alignment-20260919')
OUT = CONTROL / REL
RUN = DEV / 'eval/objectnav-full-minival-200-20260803-gpu0-w4/ObjectNavType/safevla-objectnav-full-minival-200-20260803-gpu0-w4/08_03_2026_01_16_39_731179'
WB = RUN / 'wandb/wandb/run-20260803_011640-z9jilvit/files'
ORIGINALS = [Path('/nvme2/user/qyy/SafeVLA_Original'), Path('/home/amax/public/users/qyy/SafeVLA_Original')]
NOW = datetime.datetime.now(datetime.timezone.utc).isoformat()
LIMIT = 1024 * 1024
SECRET = [rb'gh[pousr]_[A-Za-z0-9_]{20,}', rb'github_pat_[A-Za-z0-9_]{20,}',
          rb'https?://[^\s/"<>]+@', rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
          rb'(?i)(?:authorization\s*:\s*(?:bearer|token)|(?:api_key|access_token|password)\s*[=:])\s*["\x27]?[A-Za-z0-9_+/=-]{20,}']
records, artifacts, warnings = [], [], []
seen = {}
blob_refs = collections.defaultdict(list)

def git(*args, root=DEV):
    return subprocess.check_output(['git', '-C', str(root), *args])

def jbytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()

def digest(path):
    size = path.stat().st_size
    h, blob = hashlib.sha256(), hashlib.sha1(b'blob ' + str(size).encode() + b'\0')
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block); blob.update(block)
    return h.hexdigest(), blob.hexdigest(), size

base = git('rev-parse', 'HEAD', root=CONTROL).decode().strip()
refs = {}
for ref in ['origin/research-loop', 'origin/baseline-experiment', 'origin/feature/probing-experiment', 'origin/main']:
    refs[ref] = git('rev-parse', ref).decode().strip()
    for line in git('ls-tree', '-r', '-z', ref).split(b'\0'):
        if not line: continue
        meta, path = line.split(b'\t', 1)
        mode, typ, sha = meta.decode().split()
        if typ == 'blob':
            blob_refs[sha].append({'ref': ref, 'commit': refs[ref], 'path': path.decode()})
before = {'head': git('rev-parse','HEAD').decode().strip(),
          'status': git('status','--porcelain=v1').decode(),
          'tracked_diff_sha256': hashlib.sha256(git('diff','--binary','HEAD')).hexdigest()}
OUT.mkdir(parents=True, exist_ok=True)

def emit(relative, data, source=None, transformation=None):
    if isinstance(data, str): data = data.encode()
    if len(data) > LIMIT: raise RuntimeError('Oversized shared file: ' + str(relative))
    if any(re.search(p, data) for p in SECRET): raise RuntimeError('Secret pattern in proposed shared file: ' + str(relative))
    dest = OUT / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    item = {'name': str(relative), 'server_path': str(dest), 'git_path': str(REL / relative),
            'sha256': sha, 'size': len(data), 'required_for_PI_review': True}
    if source:
        item['source_server_path'] = source['server_path']
        item['source_sha256'] = source['sha256']
    if transformation: item['transformation'] = transformation
    artifacts.append(item)
    return str(REL / relative)

def register(path, cls, reason, copy=None, group=None):
    path = Path(path)
    key = str(path)
    if key in seen:
        item = seen[key]
        if copy and not item.get('git_path'):
            item['git_path'] = emit(Path(copy), path.read_bytes(), item, 'byte-identical copy')
            item['classification'] = 'MUST_GIT'
        return item
    item = {'server_path': key, 'realpath': str(path.resolve()), 'classification': cls,
            'reason': reason, 'group': group, 'readable': False}
    records.append(item); seen[key] = item
    if not path.is_file():
        item.update(exists=False, sha256=None, size=None)
        return item
    sha, blob, size = digest(path)
    item.update(exists=True, readable=True, sha256=sha, size=size,
                mtime_ns=path.stat().st_mtime_ns, git_blob_sha1=blob,
                exact_github_copies=blob_refs.get(blob, []))
    try:
        relative = path.relative_to(DEV)
        tracked = subprocess.run(['git','-C',str(DEV),'ls-files','--error-unmatch',str(relative)],
                                 stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0
        ignored = subprocess.run(['git','-C',str(DEV),'check-ignore','-q',str(relative)]).returncode == 0
        item['development_git_status'] = 'tracked' if tracked else ('ignored' if ignored else 'untracked')
    except ValueError:
        item['development_git_status'] = 'outside_development_worktree'
    if copy:
        item['git_path'] = emit(Path(copy), path.read_bytes(), item, 'byte-identical copy')
        item['classification'] = 'MUST_GIT'
    elif item['exact_github_copies']:
        item['disposition_note'] = 'Already present byte-for-byte in the listed GitHub commit; no duplicate upload.'
    return item

# Complete diagnostic and development-research census, preserving old material as historical snapshots.
for folder in ['diagnostics/end_causal_audit', 'research']:
    for p in sorted((DEV / folder).rglob('*')):
        if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts: continue
        rel = p.relative_to(DEV)
        cls, why, copy = 'PATH_HASH_ONLY', 'Raw runtime artifact, patch, binary, or log; retain original server bytes.', None
        if p.suffix in ['.md','.json','.jsonl'] and p.stat().st_size <= LIMIT:
            sha, blob, size = digest(p)
            if not blob_refs.get(blob):
                cls, why, copy = 'MUST_GIT', 'Small historical report or structured evidence absent from GitHub.', Path('legacy_development') / rel
            else:
                why = 'Same bytes already shared on a GitHub branch; register immutable reference.'
        if p.suffix in ['.py','.sh'] or p.name in ['environment.txt'] or p.suffix == '.pid':
            cls, why = 'DO_NOT_SHARE_CONTROL', 'Runtime code/launcher, environment dump, or ephemeral process state; excluded from shared control plane.'
        register(p, cls, why, copy, folder)

# Exact final W&B tables, not the incomplete progress snapshots.
summary = json.loads((WB / 'wandb-summary.json').read_text())
register(WB / 'wandb-summary.json','MUST_GIT','Original final table pointers and source hashes.',Path('full200/wandb-summary.json'),'full200')
for name, value in summary.items():
    if isinstance(value,dict) and 'path' in value:
        p = WB / value['path']
        register(p,'MUST_GIT','Final W&B table explicitly referenced by the historical run summary.',Path('full200') / value['path'],'full200')
for name in ['分析output.md','RECOVERY_AUDIT.md','narrow_broad_reanalysis.csv','narrow_broad_reanalysis.py']:
    p = WB / name
    if p.is_file():
        register(p,'MUST_GIT','Historical analysis artifact; archived claims/recommendations are not current authority.',
                 Path('full200/legacy_analysis') / name,'full200')
for name in ['output.log','wandb-metadata.json','requirements.txt','config.yaml']:
    register(WB / name,'PATH_HASH_ONLY','Complete runtime log/config metadata stays on server; selected non-sensitive identity is shared separately.',group='full200')
for p in sorted(RUN.iterdir()):
    if p.is_file():
        if p.name in ['sub120_actions.csv','sub120_actions.json','sub120_action_probs_approx.csv','extract_actions.py','extract_action_probs.py']:
            register(p,'MUST_GIT','Small single-case evidence or its extraction script; not exact logits or population evidence.',
                     Path('sub120') / p.name,'sub120')
        elif p.suffix == '.py':
            register(p,'DO_NOT_SHARE_CONTROL','Runtime replay/source is not part of this evidence-only control commit.',group='sub120')
        else:
            register(p,'PATH_HASH_ONLY','Raw video/image retained on server; no media uploaded.',group='full200_media')
print('ARCHIVE_PROGRESS full200 and diagnostic files indexed', flush=True)

# Task-spec payload decompressed byte-for-byte, with non-scientific integrity/key checks.
task_source = register(DEV / 'benchmark/objectnavtype_val.jsonl.gz','PATH_HASH_ONLY',
                       'Compressed benchmark source; its exact decompressed task-spec payload is shared.',group='task_spec')
task_raw = gzip.decompress((DEV / 'benchmark/objectnavtype_val.jsonl.gz').read_bytes())
emit(Path('task_specs/objectnavtype_val.jsonl'),task_raw,task_source,'gzip decompression only; payload bytes unchanged')
tasks = [json.loads(line) for line in task_raw.splitlines() if line.strip()]
table_source = WB / summary['VideoTable/ObjectNavType']['path']
table = json.loads(table_source.read_text())
rows = [dict(zip(table['columns'],row)) for row in table['data']]
def stable(path):
    marker='ObjectNavType/val/'
    if marker not in path: raise ValueError('Unrecognized task identity format')
    return path[path.index(marker):]
taskmap = {stable(t['task_path']): t for t in tasks}
keyrows = []
for row in rows:
    key = stable(row['task_path'])
    task = taskmap.get(key)
    keyrows.append({'stable_task_key':key,'task_spec_sha256':hashlib.sha256(json.dumps(task,sort_keys=True,separators=(',',':')).encode()).hexdigest() if task else None,
                    'house_index':task.get('house_index') if task else None,
                    'expert_length':task.get('expert_length') if task else None,
                    'gt_episode_len':row['gt_episode_len'],
                    'expert_lengths_equal':task.get('expert_length') == row['gt_episode_len'] if task else False})
emit(Path('task_specs/task_identity_check.json'),jbytes({'key_rule':'Keep the ObjectNavType/val/... suffix; do not use transient episode IDs or result row order.','rows':keyrows}))
checks = {'scope':'Archive integrity only; no category/size association analysis, experiment claim, model or simulator run.',
          'rows':len(rows),'successes':sum(r['success'] for r in rows),'cost_total':sum(r['sum_cost'] for r in rows),
          'task_spec_rows':len(tasks),'unique_result_keys':len({stable(r['task_path']) for r in rows}),
          'unique_task_spec_keys':len(taskmap),'matched_result_keys':sum(stable(r['task_path']) in taskmap for r in rows),
          'expert_length_matches':sum(r['expert_lengths_equal'] for r in keyrows),
          'room_visitation_non_null':sum(r['has_agent_been_in_room'] is not None for r in rows),
          'columns':table['columns'],
          'final_table_sha256_matches_wandb_summary':digest(table_source)[0] == summary['VideoTable/ObjectNavType']['sha256'],
          'historical_report_reconciled':len(rows)==200 and sum(r['success'] for r in rows)==173 and sum(r['sum_cost'] for r in rows)==145}
emit(Path('full200/INTEGRITY_CHECK.json'),jbytes(checks))
csvbuf=io.StringIO(newline='');writer=csv.DictWriter(csvbuf,fieldnames=table['columns']);writer.writeheader();writer.writerows(rows)
emit(Path('full200/episode_results.csv'),csvbuf.getvalue(),seen[str(table_source)],'Lossless row/column projection of final table; JSON remains authoritative for types.')
meta_path=WB/'wandb-metadata.json'
if meta_path.is_file():
    meta=json.loads(meta_path.read_text())
    allowed={k:meta[k] for k in ['startedAt','program','codePath','python','gpu','gpu_count'] if k in meta}
    allowed['git_commit']=(meta.get('git') or {}).get('commit')
    emit(Path('full200/run_identity.json'),jbytes(allowed),seen[str(meta_path)],'Allowlisted fields; remote URL, user/environment details omitted.')

# Probe artifacts: never deserialize tensors or execute collectors.
probe_log_suffix=Path('eval/ObjectNavType/OnlineEval/06_15_2026_09_52_57_364270/wandb/wandb/run-20260615_095258-0biawkdd/files/output.log')
for number, root in enumerate(ORIGINALS,1):
    for name in ['probe_data.pt','probe_data_worker0.pt','probe_data_worker1.pt','probe_small_object.py',
                 'online_evaluation/probe_small_object.py','l_probing.py','steering_vector.py']:
        p=root/name
        register(p,'PATH_HASH_ONLY' if p.suffix=='.pt' else 'DO_NOT_SHARE_CONTROL',
                 'Tensor identity only; never loaded.' if p.suffix=='.pt' else 'Experimental/runtime source: use existing GitHub branch when byte-identical; do not copy into control plane.',group='probe')
    p=root/probe_log_suffix
    entry=register(p,'PATH_HASH_ONLY','Full 11MB historical Probe log; only a line-numbered excerpt is shared.',group='probe')
    if p.is_file():
        selected=[]
        with p.open(errors='replace') as f:
            for lineno,line in enumerate(f,1):
                if re.search(r'(?i)(decoder.layers|dim=512|10000.*3.*512|1000.*3.*512|positive|is_close_and_visible|saved.*probe|probe.*enabled)',line):
                    selected.append({'line':lineno,'text':line.rstrip()[:1200]})
        sample=selected[:12]+selected[-12:] if len(selected)>24 else selected
        emit(Path('probe')/('replica%d_log_excerpt.json'%number),
             jbytes({'source':str(p),'source_sha256':entry.get('sha256'),'matching_lines':len(selected),'selected_lines':sample}),
             entry,'Selected original log lines with line numbers; not a complete log or new analysis.')
print('ARCHIVE_PROGRESS probe identities indexed', flush=True)

# Static dataset identities and schemas. No simulator, unsafe pickle loader, or size-effect analysis.
metadata_schema=[]
for p in [Path('/home/amax/public/datasets/qyy/objaverse_houses/houses_2023_07_28/val.jsonl.gz'),
          Path('/home/amax/public/datasets/qyy/objaverse_assets/2023_07_28/annotations.json.gz')]:
    entry=register(p,'PATH_HASH_ONLY','Large source dataset remains server-only; static field availability inspected.',group='scene_object_metadata')
    schema={'server_path':str(p),'exists':p.is_file(),'sha256':entry.get('sha256')}
    if p.is_file():
        with gzip.open(p,'rt') as f:
            data=json.load(f) if p.name=='annotations.json.gz' else json.loads(next(line for line in f if line.strip()))
        schema.update(container=type(data).__name__,top_level_count=len(data) if hasattr(data,'__len__') else None)
        if p.name=='annotations.json.gz':
            example=next(iter(data.values())) if isinstance(data,dict) else data[0]
            schema.update(example_fields=list(example) if isinstance(example,dict) else [],
                          size_field_present=isinstance(example,dict) and 'size' in example,
                          size_field_shape=type(example.get('size')).__name__ if isinstance(example,dict) else None,
                          interpretation='Asset-level metadata availability only; target-instance transformation/scaling and complete per-task size coverage have not been validated.')
        else:
            schema.update(house_fields=list(data),room_fields=list(data.get('rooms',[{}])[0]),object_fields=list(data.get('objects',[{}])[0]),
                          interpretation='Static scene structure exists; full task-to-house-to-asset reconciliation is a later authorized audit.')
        del data
    metadata_schema.append(schema)
emit(Path('metadata/STATIC_METADATA_AVAILABILITY.json'),jbytes(metadata_schema))
after={'head':git('rev-parse','HEAD').decode().strip(),'status':git('status','--porcelain=v1').decode(),
       'tracked_diff_sha256':hashlib.sha256(git('diff','--binary','HEAD')).hexdigest()}
if before != after: raise RuntimeError('Development state changed during archive collection; inspect before publication.')
emit(Path('DEVELOPMENT_PRESERVATION.json'),jbytes({'before':before,'after':after,'identical':True}))
counts=collections.Counter(r['classification'] for r in records)
manifest={'created_at_utc':NOW,'scope':'Historical-file inventory and evidence publication only; not EXP-SMALLTARGET-PHENOTYPE-001 execution.',
          'source_refs':refs,'control_parent':base,'development':before,'classification_counts':dict(counts),
          'coverage':['Complete files under development diagnostics/end_causal_audit and research.',
                      'Historical full-200 run top-level files plus final W&B tables, selected reports and configuration identities.',
                      'Two known Original replicas: named Probe tensors/collectors and the registered 2026-06-15 collection log.',
                      'Registered task-spec and static scene/asset sources. Not a census of all videos/datasets/other users on the server.'],
          'records':records}
emit(Path('FILE_INVENTORY.json'),jbytes(manifest))
emit(Path('ARCHIVE_MANIFEST.json'),jbytes({'created_at_utc':NOW,'repository':'https://github.com/whatcanidowhat/SafeVLA',
     'control_parent_commit':base,'source_refs':refs,'classification_counts':dict(counts),'gpu_used':0,'episodes_started':0,
     'models_loaded':0,'claim_created':False,'loop_state_modified':False,'approved_design_modified':False,
     'preserved_development_state':before,'limitations':['Main 02 conversation was not independently retrieved; its full available branch and migrated research summaries were read.',
     'No tensor deserialization. Historical AUC cannot be attached to any recovered PT merely from filename.',
     'Static metadata source presence is verified, not full task-level physical-size coverage.']}))
index={'schema_version':'1.0','status':'HISTORICAL_EVIDENCE_ARCHIVE','artifacts':artifacts}
(OUT/'ARTIFACT_INDEX.json').write_bytes(jbytes(index))
print(json.dumps({'archive':str(OUT),'files_shared':len(artifacts),'inventory_records':len(records),'classes':dict(counts),
                  'integrity':checks,'metadata':metadata_schema},ensure_ascii=False),flush=True)
