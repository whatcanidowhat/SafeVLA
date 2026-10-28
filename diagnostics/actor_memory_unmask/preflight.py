"""Read-only CPU/resource inventory; never allocate GPU or run simulator."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from diagnostics.actor_logits.manifest import regenerate, MANIFEST_SHA
from diagnostics.actor_memory_unmask.cpu_model import ROOT


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    out=ROOT/'research/proposals/EXP-ACTOR-MEMORY-UNMASK-001'
    reference=json.loads((ROOT/'diagnostics/actor_logits/reference_identity.json').read_text())
    tasks,content=regenerate(ROOT); (out/'TASK_MANIFEST.csv').write_bytes(content)
    resources={}
    for name,path,expected in [
        ('policy','/home/amax/public/datasets/qyy/checkpoints/safe_objnav.pt','05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301'),
        ('dino','/home/amax/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth','b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9')]:
        p=Path(path); actual=sha(p) if p.is_file() else None
        resources[name]=dict(path=path,sha256=actual,expected=expected,match=actual==expected,size=p.stat().st_size if p.is_file() else None)
    dino=Path('/home/amax/.cache/torch/hub/facebookresearch_dinov2_main')
    dino_checks={name:(dino/name).is_file() and sha(dino/name)==digest for name,digest in reference['dino_source'].items()}
    dependencies={}
    for name,expected in reference['dependencies'].items():
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual=None
        dependencies[name]=dict(actual=actual,expected=expected,match=actual==expected)
    protected=['training/online/third_party_models/llama/model.py',
               'architecture/models/allenact_transformer_models/separate_actor_critic.py',
               'architecture/models/allenact_transformer_models/inference_agent.py',
               'online_evaluation/online_evaluator_worker.py']
    protected_checks={}
    for name in protected:
        original=subprocess.check_output(['git','show','06cde9adeb91d47c062a11e72e90c7edf56fb398:'+name],cwd=ROOT)
        protected_checks[name]=sha(ROOT/name)==hashlib.sha256(original).hexdigest()
    def query(*args):
        r=subprocess.run(args,capture_output=True,text=True,timeout=20)
        return dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
    gpu=query('nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits')
    compute=query('nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits')
    disk=shutil.disk_usage(ROOT)
    paths=['/home/amax/public/datasets/qyy/objaverse_houses/houses_2023_07_28',
           '/home/amax/public/datasets/qyy/objaverse_assets',
           '/home/amax/public/datasets/qyy/nltk_data']
    result=dict(status='PREPARATION_ONLY_NO_GPU_AUTHORIZATION',utc=datetime.now(timezone.utc).isoformat(),
                host=socket.gethostname(),python=sys.executable,platform=platform.platform(),cwd=str(ROOT),
                command=sys.argv,resources=resources,dino_source_checks=dino_checks,dependencies=dependencies,
                tasks=200,manifest_sha256=MANIFEST_SHA,protected_source_unchanged=protected_checks,
                disk_total=disk.total,disk_free=disk.free,online_disk_required_bytes=100*1024**3,
                online_disk_gate=disk.free>=100*1024**3,gpu_snapshot=gpu,foreign_compute_snapshot=compute,
                memory_info=query('free','-b'),asset_paths_exist={p:Path(p).exists() for p in paths},
                asset_content_identity='NOT_HASHED_REQUIRES_FUTURE_DATASET_VERSION_PIN',
                gpu_admission='NOT_RUN_BUSY_SNAPSHOT_ONLY',network_claim_ci_gate='NOT_RUN_NO_NEW_CLAIM',
                episodes_started=0,gpu_allocations=0,simulator_launches=0,
                environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE']})
    (out/'RESOURCE_PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')
    assert all(x['match'] for x in resources.values()) and all(dino_checks.values()) and all(protected_checks.values())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
