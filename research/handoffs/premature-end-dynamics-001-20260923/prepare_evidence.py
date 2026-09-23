import csv,json,hashlib,subprocess,os,sys
from pathlib import Path
import cv2,numpy as np
R=Path('/nvme2/user/qyy/SafeVLA');C=Path('/nvme2/user/qyy/SafeVLA_loop_control');A=C/'research/history/evidence-alignment-20260919';O=R/'research/handoffs/premature-end-dynamics-001-20260923'
O.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,indent=2)+'\n')
def git(*a):return subprocess.check_output(['git','-C',str(R),*a])
state=json.loads((C/'research/LOOP_STATE.json').read_text())
save('execution_start.json',dict(state=state,claim_commit='a5045c5255b67f3e0f1d4f559d2d6078f70f9544',claim_ci='https://github.com/whatcanidowhat/SafeVLA/actions/runs/35845852871',head=git('rev-parse','HEAD').decode().strip(),status=git('status','--short').decode(),tracked_diff_sha256=hashlib.sha256(git('diff','--binary')).hexdigest(),command_python=sys.executable,cv2_version=cv2.__version__,gpu_count=0,episodes=0,simulator_launches=0,model_loads=0))
idx=json.loads((A/'ARTIFACT_INDEX.json').read_text())
for name in ['full200/episode_results.csv','task_specs/objectnavtype_val.jsonl','sub120/extract_action_probs.py','sub120/extract_actions.py']:
 p=A/name;item=next(x for x in idx['artifacts'] if x.get('git_path')=='research/history/evidence-alignment-20260919/'+name);assert sha(p)==item['sha256']
rows=list(csv.DictReader((A/'full200/episode_results.csv').open()))
fails=[r for r in rows if r['success']=='False' and int(r['eps_len'])<600];assert len(fails)==16
inv={x['server_path']:x for x in json.loads((A/'FILE_INVENTORY.json').read_text())['records']}
out=[]
for i,r in enumerate(fails):
 p=R/r['video_path'];assert p.is_file()
 item=inv[str(p)];assert sha(p)==item['sha256']
 cap=cv2.VideoCapture(str(p));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
 cap.set(cv2.CAP_PROP_POS_FRAMES,n-1);ok,im=cap.read();assert ok
 cv2.imwrite(str(O/('inspect_last_'+str(i)+'.png')),im)
 cap.set(cv2.CAP_PROP_POS_FRAMES,0);ok,first=cap.read();assert ok
 if i==0:cv2.imwrite(str(O/'inspect_first_0.png'),first)
 cap.release()
 rgb=im[:,:,::-1];counts=[]
 for cy in [36,45,54,63,72,81,90,99]:
  counts.append(int(np.all(rgb[cy-4:cy+4,800:910,:]<70,axis=2).sum()))
 out.append(dict(case=i,task_key=r['task_path'],video=str(p),sha256=item['sha256'],size=p.stat().st_size,eps_len=int(r['eps_len']),frames=n,width=w,height=h,last_action_black_counts=counts))
save('video_preflight.json',out)
sources={}
for name in ['utils/visualization_utils.py','utils/constants/stretch_initialization_utils.py','online_evaluation/online_evaluator_worker.py']:
 old=git('show','60bc54fbdedaf5745d0476c25321e808708273aa:'+name)
 current=(R/name).read_bytes()
 sources[name]=dict(historical_commit='60bc54fbdedaf5745d0476c25321e808708273aa',historical_sha256=hashlib.sha256(old).hexdigest(),current_sha256=hashlib.sha256(current).hexdigest(),current_equal_historical=old==current)
 # Only provenance excerpts are shared, no runtime imports.
 lines=old.decode().splitlines()
 if 'visualization' in name: spans=[(160,215),(400,500)]
 elif 'initialization' in name:spans=[(135,200)]
 else:
  hits=[i for i,l in enumerate(lines) if 'get_action(' in l or 'get_video_frame(' in l or 'all_video_frames.append' in l]
  spans=[(max(0,i-8),min(len(lines),i+20)) for i in hits]
 sources[name]['historical_excerpts']=[dict(start_1based=a+1,lines=lines[a:b]) for a,b in spans]
save('source_semantics.json',sources)
print(json.dumps(out))
