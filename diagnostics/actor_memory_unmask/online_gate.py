"""Fail-closed, stdlib-only authorization for the new Slurm unmask pair."""
import importlib.metadata
import json
import os
import sys
from pathlib import Path
import subprocess
import urllib.request
from datetime import datetime, timezone

EXPERIMENT='EXP-ACTOR-MEMORY-UNMASK-002'
CYCLE='actor-memory-unmask-002-20261029'
COMMAND=['sbatch','diagnostics/actor_memory_unmask/unmask_pair.slurm']


def require(ok,message):
    if not ok:raise ValueError(message)


def validate_binding(state, design, source_sha, worktree, approval, now=None):
    now=now or datetime.now(timezone.utc)
    require(state.get('status')=='CODEX_RUNNING' and state.get('next_actor')=='CODEX' and state.get('updated_by')=='CODEX','No current CODEX_RUNNING claim')
    require(state.get('experiment_id')==EXPERIMENT and state.get('cycle_id')==CYCLE,'Wrong experiment/cycle; old Claim forbidden')
    require(state.get('instruction_commit')==approval and state.get('claim_id'),'Missing approval binding/unique Claim')
    require(state.get('execution_worktree')==str(worktree),'Wrong approved worktree')
    auth=state.get('authorization',{})
    require(auth.get('status')=='APPROVED' and auth.get('approved_by')=='PI' and auth.get('max_gpu')==1 and auth.get('max_episodes')==400,'PI resource authorization must cover ON200 + OFF200 = 400 starts')
    require(datetime.fromisoformat(auth['expires_at_utc'].replace('Z','+00:00'))>now,'Authorization expired')
    require(design.get('status')=='APPROVED' and design.get('experiment_id')==EXPERIMENT and design.get('cycle_id')==CYCLE,'Wrong approved design')
    require(design.get('code_sha')==source_sha and design.get('command')==COMMAND,'Source pin or command mismatch')
    require(design.get('gpu_logger_equivalence') in ['PASS_FOR_THIS_SOURCE','PI_WAIVED_FOR_THIS_EXPERIMENT'],'New experiment logger evidence/explicit PI decision required; old waiver forbidden')
    require(design.get('resources')=={'max_gpu':1,'max_episodes':400},'Design resource bounds mismatch')
    require(design.get('max_decisions')==240000 and design.get('max_runtime_seconds')==43200,'Design decision/runtime bounds mismatch')
    fixed=design.get('fixed_conditions',{})
    require(isinstance(fixed,dict) and fixed.get('max_steps')==500 and fixed.get('kv_cache_capacity')==500 and fixed.get('episode_horizon')==600 and fixed.get('actor_reset')=='off','Effective Baseline must be500/cache500/horizon600/resetOFF in both arms')
    require(design.get('asset_inventory_sha256') and design.get('asset_inventory_path'),'Missing reviewed target asset inventory')
    require(design.get('rendering_review')=='PI_ACCEPTED_FOR_BOTH_ARMS','Target rendering adaptation not reviewed')
    require(design.get('environment_review')=='PI_ACCEPTED_TARGET_ENVIRONMENT_FOR_BOTH_ARMS','Full target dependency differences need explicit review')
    return auth


def check_ci(sha):
    request=urllib.request.Request('https://api.github.com/repos/whatcanidowhat/SafeVLA/actions/runs?head_sha='+sha+'&per_page=100',headers={'User-Agent':'SafeVLA-unmask-gate','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=25) as response:runs=json.load(response).get('workflow_runs',[])
    require(any(r.get('head_sha')==sha and r.get('name')=='research-loop-validate' and r.get('event')=='push' and r.get('status')=='completed' and r.get('conclusion')=='success' for r in runs),'Exact approval/Claim control CI not SUCCESS: '+sha)


def authorize(args,repo):
    from diagnostics.actor_logits.storage import sha256
    require(args.control_root and args.approval_reference and args.claim_reference,'Missing independent PI approval, new Claim and control checkout')
    control=args.control_root.resolve()
    # Compute-side commands below are read-only and support the target Git1.8.
    # The independent claim writer must use the protocol's modern Git client;
    # do not invent an atomic-push requirement for this read-only runtime gate.
    def git(root,*argv):return subprocess.check_output(['git',*argv],cwd=root,text=True,timeout=30).strip()
    require(git(control,'symbolic-ref','--short','HEAD')=='research-loop','Wrong control branch')
    require(not git(control,'status','--porcelain'),'Dirty control checkout')
    require(git(control,'rev-parse','HEAD')==args.claim_reference,'Claim checkout mismatch')
    remote=git(control,'ls-remote','--heads','origin','refs/heads/research-loop').split()
    require(remote and remote[0]==args.claim_reference,'True remote Claim HEAD mismatch')
    state=json.loads((control/'research/LOOP_STATE.json').read_text())
    design=json.loads((control/'research/NEXT_EXPERIMENT.json').read_text())
    source=git(repo,'rev-parse','HEAD')
    auth=validate_binding(state,design,source,repo.resolve(),args.approval_reference)
    require(design.get('python_executable')==sys.executable,'Python path differs from reviewed target environment')
    render_keys=['VK_ICD_FILENAMES','LD_LIBRARY_PATH','DISPLAY','XDG_RUNTIME_DIR']
    require(design.get('rendering_environment')=={k:os.environ.get(k) for k in render_keys},'Actual renderer environment differs from reviewed design')
    require(not git(repo,'status','--porcelain'),'Dirty source tree')
    check_ci(args.approval_reference); check_ci(args.claim_reference)
    require(args.approval_reference!=args.claim_reference,'Approval and Claim must be distinct')
    reference=json.loads((repo/'diagnostics/actor_logits/reference_identity.json').read_text())
    for name,expected in reference['dependencies'].items():
        require(importlib.metadata.version(name)==expected,'Dependency drift: '+name+'; target migration is not equivalence approval')
    target_versions=design.get('target_dependency_versions',{})
    require(target_versions and 'pandas' in target_versions and 'Pillow' in target_versions,'Reviewed target dependency versions missing')
    for name,expected in target_versions.items():require(importlib.metadata.version(name)==expected,'Target dependency differs from reviewed version: '+name)
    inventory=Path(design['asset_inventory_path'])
    require(sha256(inventory)==design['asset_inventory_sha256'],'Reviewed asset inventory identity mismatch')
    inventory_record=json.loads(inventory.read_text())
    require(inventory_record.get('status')=='COMPLETE_EXTERNAL_CLOSURE_REQUIRES_BUILTIN_RENDERER_PIN' and inventory_record.get('task_count')==200 and not inventory_record.get('missing_external_assets'),'Selected200 asset closure inventory incomplete')
    entries=inventory_record['files']
    require(entries,'Empty target asset inventory')
    for item in entries:require(sha256(Path(item['path']))==item['sha256'],'Asset content drift: '+item['path'])
    renderer_files=design.get('renderer_file_sha256',{})
    require(renderer_files,'Simulator build and renderer binary/resource pins missing')
    for path,digest in renderer_files.items():require(sha256(Path(path))==digest,'Simulator/renderer file drift: '+path)
    # Preserve scheduler CUDA mapping. Never replace it with logical GPU 0.
    require(os.environ.get('SLURM_JOB_ID'),'Requires separately authorized Slurm allocation')
    allocated=os.environ.get('SLURM_JOB_GPUS','').split(',')
    require(len(allocated)==1 and allocated[0].isdigit(),'Expected one physical Slurm GPU')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    require(len(visible)==1 and visible[0],'Expected exactly one scheduler-visible GPU')
    gpu=int(allocated[0])
    row=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20).strip().split(',')
    require(len(row)==4,'Malformed GPU resource query')
    uuid,used,free,util=[x.strip() for x in row]
    require(int(used)<=4000 and int(free)>=6824 and int(util)<=30,'Allocated GPU capacity guard failed')
    pids=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=20).strip()
    require(not pids,'Allocated GPU has preexisting compute processes')
    return {'approval_commit':args.approval_reference,'claim_commit':args.claim_reference,'claim_id':state['claim_id'],
            'approval_expires_at_utc':auth['expires_at_utc'],'gpu_logger_equivalence':design['gpu_logger_equivalence'],
            'physical_gpu':gpu,'gpu_uuid':uuid,'slurm_job_id':os.environ['SLURM_JOB_ID'],'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
            'rendering_environment':{k:os.environ.get(k) for k in render_keys}}
