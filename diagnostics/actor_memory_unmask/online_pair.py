"""Future one-shot Slurm ON200 then OFF200, never self-authorizes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from .online_gate import authorize, EXPERIMENT, CYCLE

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--control-root',type=Path,required=True)
    p.add_argument('--approval-reference',required=True)
    p.add_argument('--claim-reference',required=True)
    p.add_argument('--execute-authorized',action='store_true')
    a=p.parse_args()
    if not a.execute_authorized:
        print(json.dumps({'status':'PLAN_ONLY','experiment':EXPERIMENT,'cycle':CYCLE,'groups':['MASK_ON200','MASK_OFF200'],'starts':400,'GPU':'NOT_ALLOCATED_BY_THIS_COMMAND'}));return
    evidence=authorize(a,ROOT)
    if a.output.exists():raise ValueError('Output exists; no overwrite/resume/retry')
    if shutil.disk_usage(a.output.parent).free<100*1024**3:raise ValueError('Requires100GiB free disk')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'PAIR_LAUNCH.json').write_text(json.dumps({'experiment':EXPERIMENT,'cycle':CYCLE,'evidence':evidence,'command':sys.argv,'status':'STARTED'},indent=2)+'\n')
    expiry=datetime.fromisoformat(evidence['approval_expires_at_utc'].replace('Z','+00:00')).timestamp()
    end=time.monotonic()+min(43200,expiry-time.time())
    def used_bytes():return sum(x.stat().st_size for x in a.output.rglob('*') if x.is_file())
    def check():
        if time.monotonic()>=end or time.time()>=expiry:raise ValueError('Runtime/authorization deadline reached')
        if used_bytes()>=90*1024**3 or shutil.disk_usage(a.output).free<10*1024**3:raise ValueError('Disk/output guard reached')
    try:
        for mode,label in [('on','MASK_ON200'),('off','MASK_OFF200')]:
            check()
            command=[sys.executable,'-B','-m','diagnostics.actor_memory_unmask.evaluate_online',
                     '--actor-mask',mode,'--output',str(a.output/label),'--checkpoint',str(a.checkpoint),
                     '--num-workers','1','--seed','123','--shuffle','--greedy','false','--test-augmentation','--episodes','200',
                     '--execute-authorized','--control-root',str(a.control_root),
                     '--approval-reference',a.approval_reference,'--claim-reference',a.claim_reference]
            with (a.output/(label+'.console.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    while child.poll() is None:
                        check();time.sleep(15)
                    if child.returncode:raise ValueError(label+' failed; later arm NOT_STARTED')
                except BaseException:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
            from diagnostics.actor_logits.validate import validate_run
            report=validate_run(a.output/label,200)
            (a.output/(label+'.validation.json')).write_text(json.dumps(report,indent=2)+'\n')
            if report['status']!='PASS' or report['decisions_checked']>120000:raise ValueError(label+' integrity/decision cap failure')
        from .online_results import analyze
        result=analyze(a.output/'MASK_ON200',a.output/'MASK_OFF200')
        (a.output/'COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
        (a.output/'PAIR_STATUS.json').write_text(json.dumps({'status':'COMPLETE','episodes':400,'next_actor':'PI'})+'\n')
    except BaseException as exc:
        (a.output/'PAIR_STATUS.json').write_text(json.dumps({'status':'STOPPED','error':type(exc).__name__+': '+str(exc),'next_actor':'PI','no_retry':True})+'\n')
        raise


if __name__=='__main__':main()
