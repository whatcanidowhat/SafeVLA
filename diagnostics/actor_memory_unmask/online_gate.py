"""Fail-closed, stdlib-only authorization for the new Slurm unmask pair."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import urllib.request
from datetime import datetime, timezone

EXPERIMENT='EXP-ACTOR-MEMORY-UNMASK-001'
CYCLE='actor-memory-unmask-001-20261028'
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
    require(design.get('asset_inventory_sha256') and design.get('asset_inventory_path'),'Missing reviewed target asset inventory')
    require(design.get('rendering_review')=='PI_ACCEPTED_FOR_BOTH_ARMS','Target rendering adaptation not reviewed')
    return auth


def check_ci(sha):
    request=urllib.request.Request('https://api.github.com/repos/whatcanidowhat/SafeVLA/actions/runs?head_sha='+sha+'&per_page=100',headers={'User-Agent':'SafeVLA-unmask-gate','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=25) as response:runs=json.load(response).get('workflow_runs',[])
    require(any(r.get('head_sha')==sha and r.get('name')=='research-loop-validate' and r.get('event')=='push' and r.get('status')=='completed' and r.get('conclusion')=='success' for r in runs),'Exact approval/Claim control CI not SUCCESS: '+sha)


def authorize(args,repo):
    from diagnostics.actor_logits.storage import sha256
    require(args.control_root and args.approval_reference and args.claim_reference,'Missing independent PI approval, new Claim and control checkout')
    control=args.control_root.resolve()
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
    render_keys=['VK_ICD_FILENAMES','LD_LIBRARY_PATH','DISPLAY','XDG_RUNTIME_DIR']
    require(design.get('rendering_environment')=={k:os.environ.get(k) for k in render_keys},'Actual renderer environment differs from reviewed design')
    require(not git(repo,'status','--porcelain'),'Dirty source tree')
    check_ci(args.approval_reference); check_ci(args.claim_reference)
    require(args.approval_reference!=args.claim_reference,'Approval and Claim must be distinct')
    reference=json.loads((repo/'diagnostics/actor_logits/reference_identity.json').read_text())
    for name,expected in reference['dependencies'].items():
        require(importlib.metadata.version(name)==expected,'Dependency drift: '+name+'; target migration is not equivalence approval')
    inventory=Path(design['asset_inventory_path'])
    require(sha256(inventory)==design['asset_inventory_sha256'],'Reviewed asset inventory identity mismatch')
    entries=json.loads(inventory.read_text())['files']
    require(entries,'Empty target asset inventory')
    for item in entries:require(sha256(Path(item['path']))==item['sha256'],'Asset content drift: '+item['path'])
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
