#!/usr/bin/env python3
"""Historical-video-only premature-end audit. No simulator or policy imports."""
import sys,os,importlib.abc
os.environ['CUDA_VISIBLE_DEVICES']=''
class NoModels(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','transformers','safetensors','ai2thor','allenact','architecture','environment','tasks','training','online_evaluation'}:
   raise RuntimeError('FORBIDDEN_MODEL_OR_SIMULATOR_IMPORT '+fullname)
sys.meta_path.insert(0,NoModels())
import csv,json,hashlib,datetime,collections,statistics,math,re,subprocess,traceback
from pathlib import Path
import cv2,numpy as np
cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
ROOT=Path('/nvme2/user/qyy/SafeVLA')
CONTROL=Path('/nvme2/user/qyy/SafeVLA_loop_control')
ARCHIVE=CONTROL/'research/history/evidence-alignment-20260919'
OUT=Path(__file__).resolve().parent
CLAIM='a5045c5255b67f3e0f1d4f559d2d6078f70f9544'
ACTIONS=['m','r','l','b','end','sub_done','ls','rs']
LONG=['move_ahead','rotate_right','rotate_left','move_back','done','sub_done','rotate_left_small','rotate_right_small']
INPUTS={}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def save(n,v):
 p=OUT/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def csvsave(n,rr,fields=None):
 with (OUT/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rr[0]));w.writeheader();w.writerows(rr)
def readcsv(p):return list(csv.DictReader(p.open(newline='')))
def require(x,msg):
 if not x:raise RuntimeError(msg)
def record(p,expected=None):
 d=sha(p);require(expected is None or d==expected,'SOURCE_HASH_MISMATCH '+str(p))
 INPUTS[str(p)]={'sha256':d,'size':p.stat().st_size};return p
def key(p):return 'ObjectNavType/val/'+p.split('ObjectNavType/val/')[1]
def sid(r):return int(re.search(r'sub_house_id=(\d+)',r['video_path']).group(1))
def summary(trace):
 vals=[r['p_end_approx'] for r in trace]
 return {'p_end_step0':vals[0],'max_p_end_first5':max(vals[:5]),'max_p_end_before_end':max(vals[:-1]) if len(vals)>1 else '',
 'p_end_at_end':vals[-1],'delta_p_end':vals[-1]-vals[0],
 'first_step_ge_0_5':next((i for i,p in enumerate(vals) if p>=.5),''),
 'first_step_ge_0_9':next((i for i,p in enumerate(vals) if p>=.9),'')}
def label(first,last,n,action):
 if action!='end':return 'ABNORMAL_TERMINATION'
 if first>=.5:return 'EARLY_HIGH_PRIOR'
 if first<.1 and n>5 and last>=.5:return 'LATE_RISE'
 if last<.1:return 'LOW_PROB_END'
 return 'OTHER_OR_UNCLEAR'
def decode(r,inventory):
 p=ROOT/r['video_path'];record(p,inventory[str(p)]['sha256'])
 cap=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_HW_ACCELERATION,cv2.VIDEO_ACCELERATION_NONE])
 require(cap.isOpened(),'VIDEO_UNAVAILABLE')
 require(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==int(r['eps_len']),'FRAME_COUNT_MISMATCH')
 require((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))==(1068,304),'LAYOUT_MISMATCH')
 trace=[]
 while True:
  ok,bgr=cap.read()
  if not ok:break
  rgb=bgr[:,:,::-1];hits=[]
  for threshold in [60,70,80]:
   counts=[int(np.all(rgb[cy-4:cy+4,800:910,:]<threshold,axis=2).sum()) for cy in [36,45,54,63,72,81,90,99]]
   choices=[i for i,n in enumerate(counts) if n>=4]
   require(len(choices)==1,'AMBIGUOUS_ACTION '+str(sid(r))+' frame '+str(len(trace)))
   hits.append(choices[0])
  require(len(set(hits))==1,'ACTION_THRESHOLD_DISAGREEMENT')
  lens=[];other=[]
  for cy in [35,44,53,62,71,80,89,98]:
   a=rgb[cy-1:cy+2,913:971,:].astype(np.int16)
   strict=(a[:,:,2]>200)&(a[:,:,0]<60)&(a[:,:,1]<60)
   cc=np.where(strict.any(axis=0))[0]
   legacy=int(cc.max()+1) if len(cc) else 0
   chrom=(a[:,:,2]-a[:,:,0]>60)&(a[:,:,2]-a[:,:,1]>60)
   dd=np.where(chrom.any(axis=0))[0]
   chromend=int(dd.max()) if len(dd) else 0
   require(legacy<=56 and chromend<=56,'BAR_LAYOUT_OUT_OF_RANGE')
   # Independent color criterion; singleton at x0 is the inclusive rectangle's baseline.
   require(abs(legacy-chromend)<=3,'PROBABILITY_EXTRACTION_AMBIGUOUS')
   lens.append(legacy);other.append(chromend)
  require(.80<=sum(lens)/55<=1.20,'BAR_SUM_SCHEMA_MISMATCH')
  row={'frame_0based':len(trace),'displayed_decision_1based':len(trace)+1,'executed_action':ACTIONS[hits[0]],'p_end_approx':min(1,lens[4]/55),'p_end_chroma_endpoint_approx':min(1,other[4]/55),'end_legacy_width_px':lens[4],'end_chroma_endpoint_px':other[4],'action_thresholds_agree':True}
  row.update({a+'_approx':min(1,n/55) for a,n in zip(LONG,lens)})
  trace.append(row)
 cap.release()
 require(len(trace)==int(r['eps_len']),'DECODED_FRAME_COUNT_MISMATCH')
 csvsave('trace_sub'+str(sid(r))+'.csv',trace)
 save('progress.json',{'stage':'VIDEO_DECODED','sub_house_id':sid(r),'frames':len(trace),'gpu_count':0,'episodes':0,'updated_at_utc':now()})
 return trace
def run():
 require(Path.cwd()==ROOT,'WRONG_EXECUTION_WORKTREE')
 require(not (OUT/'analysis_result.json').exists(),'ALREADY_FINALIZED_NO_RERUN')
 started=now()
 state=json.loads((CONTROL/'research/LOOP_STATE.json').read_text())
 require(state['status']=='CODEX_RUNNING' and state['claim_id']=='37f4a6ae9b5d41ada6753d75f2b30caa','CLAIM_MISMATCH')
 require(subprocess.check_output(['git','-C',str(CONTROL),'rev-parse','HEAD']).decode().strip()==CLAIM,'CLAIM_HEAD_CHANGED')
 idx=json.loads((ARCHIVE/'ARTIFACT_INDEX.json').read_text())
 def shared(rel):
  p=ARCHIVE/rel;ent=next(x for x in idx['artifacts'] if x.get('git_path')=='research/history/evidence-alignment-20260919/'+rel)
  return record(p,ent['sha256'])
 population=readcsv(shared('full200/episode_results.csv'))
 specs={key(x['task_path']):x for x in map(json.loads,shared('task_specs/objectnavtype_val.jsonl').read_text().splitlines())}
 inventory={x['server_path']:x for x in json.loads(record(ARCHIVE/'FILE_INVENTORY.json').read_text())['records']}
 require(len(population)==len(specs)==200 and sum(r['success']=='True' for r in population)==173,'POPULATION_MISMATCH')
 for r in population:
  require(int(r['gt_episode_len'])==specs[key(r['task_path'])]['expert_length'],'EXPERT_IDENTITY_MISMATCH')
 fails=sorted([r for r in population if r['success']=='False' and int(r['eps_len'])<600],key=sid)
 require(len(fails)==16,'FAILURE_POPULATION_MISMATCH')
 successes=[r for r in population if r['success']=='True'];pairs=[]
 for f in fails:
  same=[s for s in successes if s['object_types']==f['object_types']]
  pool=same or successes
  c=min(pool,key=lambda s:(abs(float(s['gt_episode_len'])-float(f['gt_episode_len'])),sid(s),key(s['task_path'])))
  pairs.append({'failure':f,'control':c,'cross_category_fallback':not bool(same)})
 plan={'frozen_at_utc':now(),'population':'All 16 historical failures with eps_len<600; no category/size/house preselection','action_order':ACTIONS,'rendered_names':LONG,'frame_alignment':'frame index 0 is first decision; on-video counter is 1-based; frame count must equal eps_len; terminal frame saved before loop break','bar_semantics':'Legacy strict RGB B>200,R<60,G<60; farthest detected column+1 divided by55, capped1. This is video-rendered quantized probability approximation, not exact logits. Zero means sub-resolution (~1/55).','inclusive_endpoint_caveat':'PIL rectangle draws x0 through x0+int(55*p), inclusive; H264 color thresholds can move endpoints. Secondary chroma endpoint excludes singleton stem and checks extraction within3 pixels; sensitivity uses +/-2/55 on nonzero estimates, zero [0,1/55]. These are sensitivity bands, not confidence bounds.','action_validation':'Unique black-text row >=4 pixels and agreement across RGB thresholds60/70/80','morphology_precedence':['EARLY_HIGH_PRIOR','LATE_RISE','LOW_PROB_END','OTHER_OR_UNCLEAR'],'thresholds':'Exact frozen NEXT_EXPERIMENT thresholds .5, .1, >5; no fitted cutoffs','matching':'same exact target category where available, min absolute expert_length difference; tie by numeric stable sub_house_id then stable task path; with replacement; index only tie-break, not difficulty','control_window':'frames0 through min(failure_end_step0,control_length-1), inclusive','pairs':[{'failure_sub_id':sid(p['failure']),'control_sub_id':sid(p['control']),'failure_task':p['failure']['task_path'],'control_task':p['control']['task_path'],'fallback':p['cross_category_fallback']} for p in pairs]}
 save('frozen_extraction_plan.json',plan)
 cache={};cases=[];controls=[];summaries=[]
 # Validate the generalized decoder against the preserved 600-frame sub120 sequence.
 sub120=next(r for r in population if sid(r)==120)
 old=readcsv(shared('sub120/sub120_action_probs_approx.csv'))
 ref=decode(sub120,inventory)
 require(len(ref)==len(old)==600,'REFERENCE_LENGTH_MISMATCH')
 require(all(a['executed_action']==b['executed_action'] and round(a['p_end_approx'],3)==float(b['done']) for a,b in zip(ref,old)),'SUB120_REFERENCE_MISMATCH')
 save('reference_validation.json',{'sub_house_id':120,'frames':600,'actions_match_legacy':True,'p_end_match_legacy':True,'role':'decoder validation only; not added to the 16-case population'})
 for pair in pairs:
  f=pair['failure'];ident=sid(f);tr=decode(f,inventory);cache[ident]=tr
  spec=specs[key(f['task_path'])]
  narrow=set(v for vv in spec['synset_to_object_ids'].values() for v in vv)
  broad=set(v for vv in spec['broad_synset_to_object_ids'].values() for v in vv)
  met=summary(tr);n=len(tr);act=tr[-1]['executed_action']
  primary=label(met['max_p_end_first5'],met['p_end_at_end'],n,act)
  alt=label(max(t['p_end_chroma_endpoint_approx'] for t in tr[:5]),tr[-1]['p_end_chroma_endpoint_approx'],n,act)
  lo=lambda p:max(0,p-2/55)
  hi=lambda p:min(1,p+2/55) if p else 1/55
  possibilities=sorted({label(a,b,n,act) for a in [lo(met['max_p_end_first5']),hi(met['max_p_end_first5'])] for b in [lo(met['p_end_at_end']),hi(met['p_end_at_end'])]})
  row={'sub_house_id':ident,'task_key':f['task_path'],'house_index':spec['house_index'],'category':f['object_types'],'goal':f['goal'],'expert_length':spec['expert_length'],'episode_length':n,'end_step_0based':n-1,'end_decision_1based':n,'final_executed_action':act,'confirmed_invalid_end':act=='end','reliable_trace':True,**met,'morphology':primary,'chroma_morphology':alt,'sensitivity_labels':'|'.join(possibilities),'threshold_robust':possibilities==[primary] and alt==primary,'target_scope':'STRICT' if narrow==broad else 'AMBIGUOUS','scope_exact':narrow==broad,'narrow_ids':json.dumps(sorted(narrow)),'broad_ids':json.dumps(sorted(broad)),'historical_narrow_vis_pix_navigation':f['vis_pix_navigation'],'historical_narrow_room_visited':f['has_agent_been_in_room'],'trace_file':'trace_sub'+str(ident)+'.csv'}
  cases.append(row);summaries.append(dict(role='failure',sub_house_id=ident,frames=n,**met))
  c=pair['control'];ci=sid(c)
  if ci not in cache:cache[ci]=decode(c,inventory)
  ct=cache[ci];require(ct[-1]['executed_action']=='end','SUCCESS_FINAL_ACTION_AMBIGUOUS')
  wi=min(n,len(ct))-1;window=ct[:wi+1];cm=summary(window)
  controls.append({'failure_sub_house_id':ident,'control_sub_house_id':ci,'control_task_key':c['task_path'],'category':c['object_types'],'cross_category_fallback':pair['cross_category_fallback'],'failure_expert_length':int(f['gt_episode_len']),'control_expert_length':int(c['gt_episode_len']),'absolute_expert_gap':abs(int(c['gt_episode_len'])-int(f['gt_episode_len'])),'control_episode_length':len(ct),'window_last_frame_0based':wi,'window_includes_control_terminal_frame':wi==len(ct)-1,'control_p_end_step0':cm['p_end_step0'],'control_max_p_end_first5':cm['max_p_end_first5'],'control_max_p_end_window':max(t['p_end_approx'] for t in window),'control_p_end_window_last':cm['p_end_at_end'],'failure_minus_control_first5':met['max_p_end_first5']-cm['max_p_end_first5'],'failure_minus_control_at_window_end':met['p_end_at_end']-cm['p_end_at_end'],'control_final_action':ct[-1]['executed_action'],'trace_file':'trace_sub'+str(ci)+'.csv','control_status':'RELIABLE'})
  if not any(x['role']=='control' and x['sub_house_id']==ci for x in summaries):summaries.append(dict(role='control',sub_house_id=ci,frames=len(ct),**summary(ct)))
 csvsave('premature_end_cases.csv',cases);csvsave('matched_success_controls.csv',controls);csvsave('pdone_trace_summary.csv',summaries)
 counts=dict(collections.Counter(x['morphology'] for x in cases))
 result={'status':'AWAITING_PI_REVIEW','started_at_utc':started,'completed_at_utc':now(),'reliable_failures':len(cases),'confirmed_invalid_end':sum(x['confirmed_invalid_end'] for x in cases),'morphology_counts':counts,'unique_success_controls':len(set(x['control_sub_house_id'] for x in controls)),'matched_pairs':len(controls),'cross_category_fallbacks':sum(x['cross_category_fallback'] for x in controls),'scope_counts':dict(collections.Counter(x['target_scope'] for x in cases)),'threshold_robust_cases':sum(x['threshold_robust'] for x in cases),'chroma_label_agreement':sum(x['morphology']==x['chroma_morphology'] for x in cases),'max_end_pixel_method_difference':max(abs(t['end_legacy_width_px']-t['end_chroma_endpoint_px']) for tr in cache.values() for t in tr),'control_early_high_pairs':sum(x['control_max_p_end_first5']>=.5 for x in controls),'windows_including_control_end':sum(x['window_includes_control_terminal_frame'] for x in controls),'gpu_count':0,'episodes':0,'simulator_launches':0,'model_checkpoint_loads':0,'actor_critic_forwards':0,'forbidden_modules_present':[m for m in sys.modules if m.split('.')[0] in {'torch','ai2thor','architecture','allenact','transformers'}]}
 record(Path(__file__));save('input_manifest.json',INPUTS);save('analysis_result.json',result)
 # File is canonical; stdout is deliberately unnecessary for experiment completion.
if __name__=='__main__':
 try:run()
 except Exception as e:
  save('failure_detail.json',{'status':'BLOCKED','reason':type(e).__name__+': '+str(e),'traceback':traceback.format_exc(),'utc':now()})
  save('input_manifest.json',INPUTS)
  sys.exit(1)
