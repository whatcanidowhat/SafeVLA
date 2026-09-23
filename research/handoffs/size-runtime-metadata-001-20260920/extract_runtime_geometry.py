#!/usr/bin/env python3
"""Outcome-blind runtime AABB recovery. No SafeVLA/task/model imports."""
import collections,copy,csv,datetime,gzip,hashlib,importlib.abc,importlib.metadata,json,math,os,statistics,subprocess,sys,traceback
from pathlib import Path
ROOT=Path('/nvme2/user/qyy/SafeVLA')
CONTROL=Path('/nvme2/user/qyy/SafeVLA_loop_control')
OUT=Path(__file__).resolve().parent
CYCLE='size-runtime-metadata-001-20260920'
APPROVAL='f5d0f56b975eab9f54e96343fb42d6b01df4d1a7'
CLAIM='c5ebd762bddc9db55c2acfd111ddf2daaf0f5a5e'
BUILD='966bd7758586e05d18f6181f459c0e90ba318bec'
BUILD_DIR=Path('/home/amax/.ai2thor/releases')/('thor-CloudRendering-'+BUILD)
ASSETS=Path('/home/amax/public/datasets/qyy/objaverse_assets/2023_07_28/assets')
HOUSES=Path('/home/amax/public/datasets/qyy/objaverse_houses/houses_2023_07_28/val.jsonl.gz')
PRIOR=CONTROL/'research/handoffs/smalltarget-phenotype-001-20260918/metadata_preflight.json'
MAX_SCENES=224
ABS_TOL=1e-6
REL_TOL=1e-5
GPU=3
os.environ['CUDA_VISIBLE_DEVICES']=str(GPU)
SCENES=0
GPU_USED=0
ACTION_COUNTS=collections.Counter()
TARGET_ROWS=[]
PREFLIGHT_ROWS=[]
TASK_ROWS=[]
READ_INPUTS={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def save(name,v):(OUT/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def record(p):
 p=Path(p);READ_INPUTS[str(p)]={'sha256':sha(p),'size':p.stat().st_size}
 return p
class Blocked(RuntimeError):pass
def require(ok,msg):
 if not ok:raise Blocked(msg)
class NoModelImports(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','transformers','safetensors','allenact','architecture','tasks','training','online_evaluation','environment'}:
   raise Blocked('FORBIDDEN_MODEL_OR_POLICY_IMPORT: '+fullname)
sys.meta_path.insert(0,NoModelImports())
def file_guard(event,args):
 if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
 p=os.fsdecode(args[0]).replace('\\','/')
 # Block outcomes/checkpoints even if a dependency accidentally requests them.
 forbidden=('wandb-summary.json','episode_results.csv','episode_table.csv','category_sr.csv','output.log','analysis_validation_report','narrow_broad_reanalysis')
 if any(n in p for n in forbidden) or '/checkpoints/' in p or p.endswith(('.pt','.pth','.ckpt','.safetensors')):
  raise Blocked('FORBIDDEN_OUTCOME_OR_CHECKPOINT_FILE')
sys.addaudithook(file_guard)
def csvwrite(name,rows,fields):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def checkpoint_status(status,reason=''):
 save('progress.json',{'status':status,'reason':reason,'updated_at_utc':stamp(),'scene_initializations':SCENES,'max_scene_initializations':MAX_SCENES,'gpu_count':GPU_USED,'action_counts':dict(ACTION_COUNTS),'outcome_columns_read':False})
def get_inputs():
 idx=json.loads((PRIOR.parent/'ARTIFACT_INDEX.json').read_text())
 ident=next(a for a in idx['artifacts'] if a['name']=='metadata_preflight.json')
 require(sha(PRIOR)==ident['sha256'],'PREVIOUS_GEOMETRY_ONLY_INPUT_HASH_MISMATCH')
 prior=json.loads(record(PRIOR).read_text())
 require(sha(HOUSES)==prior['sources'][str(HOUSES)]['sha256'],'SCENE_DATASET_HASH_MISMATCH')
 record(HOUSES)
 # Benchmark bytes are hashed only. No task outcome or episode-length field is parsed.
 benchmark=ROOT/'benchmark/objectnavtype_val.jsonl.gz'
 require(sha(benchmark)==prior['sources'][str(benchmark)]['sha256'],'TASK_SPEC_HASH_MISMATCH')
 record(benchmark)
 grouped=collections.defaultdict(list)
 for r in prior['targets']:grouped[r['task_key']].append(r)
 require(len(grouped)==200 and sum(map(len,grouped.values()))==368,'EXPECTED_TASK_TARGET_COUNT_MISMATCH')
 tasks=[]
 for key,rr in sorted(grouped.items()):
  require(len({r['house_index'] for r in rr})==1,'AMBIGUOUS_HOUSE_KEY')
  require(len({r['target_id'] for r in rr})==len(rr),'DUPLICATE_TARGET_ID')
  tasks.append({'key':'ObjectNavType/val/'+key,'house_index':rr[0]['house_index'],'synset':rr[0]['synset'],'targets':[{'id':r['target_id'],'asset_id':r['asset_id'],'has_static_candidate':bool(r['house_annotation_bbox'] or r['disk_metadata_bbox'])} for r in sorted(rr,key=lambda r:r['target_id'])]})
 selected=[t for t in tasks if any(r['has_static_candidate'] for r in t['targets'])][:6]+[t for t in tasks if not any(r['has_static_candidate'] for r in t['targets'])][:6]
 require(len(selected)==12,'PREFLIGHT_SELECTION_INCOMPLETE')
 save('frozen_measurement_plan.json',{'created_at_utc':stamp(),'preflight_tasks':selected,'all_task_count':200,'all_target_count':368,'selection':'lexicographic stable task keys: first six with >=1 static candidate, then first six with zero candidates; no outcomes',
 'measurement_point':'Immediately after Controller.reset(scene=archived_house) returns CreateHouse metadata, with autoSimulation=False; no post-load AdvancePhysicsStep, camera calibration, task teleport, navigation or policy. This is creation-state scene-instance geometry, not after-policy or settled evaluation geometry.',
 'scene_handling':'Deep copy of archived house; no object/agent transforms modified. Explicit scene at Controller construction prevents an extra default-scene reset.',
 'repeatability':'Fresh independent Controller/Unity process for each of 24 preflight loads. Same input house/configuration; no automatic retries.',
 'absolute_tolerance_m':ABS_TOL,'relative_tolerance':REL_TOL,'tolerance_rule':'abs(a-b) <= 1e-6 + 1e-5 * max(abs(a),abs(b)); max difference also checked for AABB centers/corners if present',
 'max_scenes':MAX_SCENES,'gpu_index':GPU,'no_models':True,'no_episodes':True,'no_outcome_association':True})
 needed={t['house_index'] for t in tasks};houses={}
 with gzip.open(HOUSES,'rt') as f:
  for i,l in enumerate(f):
   if i in needed:houses[i]=json.loads(l)
   if i>=max(needed):break
 require(len(houses)==200,'MISSING_HOUSES')
 return tasks,selected,houses
def version_gate():
 historical=json.loads((OUT/'historical_runtime_metadata.json').read_text())
 req=json.loads((OUT/'requirements.json').read_text())
 source=json.loads((OUT/'initialization_source_identity.json').read_text())
 require('ai2thor==0+'+BUILD in req['simulator_dependencies'],'HISTORICAL_PACKAGE_IDENTITY_UNRESOLVED')
 require(importlib.metadata.version('ai2thor')=='0+'+BUILD,'INSTALLED_PACKAGE_MISMATCH')
 require(source['byte_identical_to_historical_commit'],'HISTORICAL_INITIALIZATION_SOURCE_CHANGED')
 require(historical['runtime_identity']['git']['commit']=='60bc54fbdedaf5745d0476c25321e808708273aa','HISTORICAL_RUN_COMMIT_MISMATCH')
 init=ROOT/'utils/constants/stretch_initialization_utils.py'
 require(sha(init)==source['sha256'],'CURRENT_INITIALIZATION_SOURCE_CHANGED')
 require('STRETCH_COMMIT_ID = "'+BUILD+'"' in init.read_text(),'FIXED_BUILD_IDENTITY_UNRESOLVED')
 required=[BUILD_DIR/('thor-CloudRendering-'+BUILD),BUILD_DIR/'UnityPlayer.so',BUILD_DIR/'metadata.json',
  BUILD_DIR/('thor-CloudRendering-'+BUILD+'_Data/Managed/Assembly-CSharp.dll')]
 for p in required:require(p.is_file(),'EXACT_BUILD_FILE_MISSING: '+p.name);record(p)
 site=Path('/home/amax/.conda/envs/safevla/lib/python3.10/site-packages')
 for rel in ['ai2thor/controller.py','ai2thor/server.py','ai2thor/fifo_server.py','ai2thor/build.py','ai2thor/platform.py','ai2thor/hooks/procedural_asset_hook.py']:
  record(site/rel)
 for rel in ['utils/constants/stretch_initialization_utils.py','environment/stretch_controller.py','online_evaluation/online_evaluator_worker.py']:
  record(ROOT/rel)
 v={'status':'PASS_BUILD_IDENTITY_BEFORE_SCENE','historical_package':'0+'+BUILD,'installed_package':importlib.metadata.version('ai2thor'),'build_commit':BUILD,'platform':'CloudRendering',
 'historical_identity_chain':['W&B metadata fixes historical run code commit and Python environment','Historical requirements pin ai2thor package to this build commit','Byte-identical committed initialization source fixes same Unity commit_id; StretchController asserts runtime build commit','Linux evaluation source selects CloudRendering; matching cached executable and Unity assembly hashed before launch'],
 'limitation':'Historical executable bytes were not separately hashed contemporaneously; current cached binary hashes identify the exact files used here. No other installed build is substituted.',
 'current_executable':str(required[0]),'hashes_before_scene':dict(READ_INPUTS),
 'units_and_extent_semantics':{'source':'https://ai2thor.allenai.org/ithor/documentation/environment-state/','field':'axisAlignedBoundingBox.size and cornerPoints','length_unit':'Unity scene units interpreted as meters, consistent with the task source maxDistance=2 and meter-valued gridSize','meaning':'World-axis scene-instance extents at creation state; not intrinsic/canonical volume','runtime_validation':'finite positive dimensions and consistency with eight corner points; OBB separately recorded if available'},
 'outcome_files_read':False,'model_modules_imported':False}
 save('runtime_version_manifest.json',v)
 return v
def geometry(obj):
 b=obj.get('axisAlignedBoundingBox')
 require(isinstance(b,dict),'MISSING_PRIMARY_AABB')
 size=b.get('size',{});corners=b.get('cornerPoints')
 if all(a in size for a in 'xyz'):dims=[float(size[a]) for a in 'xyz']
 elif isinstance(corners,list) and len(corners)==8:dims=[max(float(p[i]) for p in corners)-min(float(p[i]) for p in corners) for i in range(3)]
 else:raise Blocked('AABB_EXTENTS_UNAVAILABLE')
 require(all(math.isfinite(x) and x>0 for x in dims),'INVALID_PRIMARY_AABB_EXTENTS')
 vec=list(dims)
 if corners is not None:
  require(len(corners)==8 and all(len(p)==3 and all(math.isfinite(float(v)) for v in p) for p in corners),'INVALID_AABB_CORNERS')
  spans=[max(float(p[i]) for p in corners)-min(float(p[i]) for p in corners) for i in range(3)]
  require(all(abs(a-b)<=ABS_TOL+REL_TOL*max(abs(a),abs(b)) for a,b in zip(dims,spans)),'AABB_SIZE_CORNER_INCONSISTENCY')
  vec.extend(float(x) for p in sorted(corners) for x in p)
 center=b.get('center')
 if center:
  require(all(math.isfinite(float(center[a])) for a in 'xyz'),'INVALID_AABB_CENTER')
  vec.extend(float(center[a]) for a in 'xyz')
 obb=obj.get('objectOrientedBoundingBox');points=obb.get('cornerPoints') if isinstance(obb,dict) else None
 obb_ok=bool(isinstance(points,list) and len(points)==8 and all(len(p)==3 and all(math.isfinite(float(v)) for v in p) for p in points))
 return {'dimensions':dims,'volume':math.prod(dims),'max_side':max(dims),'corners':corners,'center':center,'vector':vec,'obb_available':obb_ok,'obb_points':points if obb_ok else None}
def make_classes():
 from ai2thor.controller import Controller
 from ai2thor.hooks.procedural_asset_hook import ProceduralAssetHookRunner,get_all_asset_ids_recursively,create_assets_if_not_exist
 class Hook(ProceduralAssetHookRunner):
  def CreateHouse(self,action,controller):
   ids=get_all_asset_ids_recursively(action['house']['objects'],[])
   controller.step(action='DeleteLRUFromProceduralCache',assetLimit=0)
   return create_assets_if_not_exist(controller=controller,asset_ids=ids,asset_directory=self.asset_directory,asset_symlink=True,stop_if_fail=True,copy_to_dir=os.path.join(controller._build.base_dir,self.target_dir),load_file_in_unity=False)
 class GeometryController(Controller):
  def prune_releases(self):pass # Preserve all existing user simulator builds.
  def reset(self,scene=None,**kw):
   global SCENES
   require(isinstance(scene,dict),'UNEXPECTED_DEFAULT_SCENE')
   require(SCENES<MAX_SCENES,'SCENE_BUDGET_EXCEEDED')
   SCENES+=1;checkpoint_status('SCENE_INITIALIZING')
   return super().reset(scene=scene,**kw)
  def step(self,action=None,**kw):
   name=action.get('action') if isinstance(action,dict) else action
   allowed={'ChangeResolution','ChangeQuality','Initialize','CreateHouse','DeleteLRUFromProceduralCache','AssetsInDatabase','CreateRuntimeAsset','CreateRuntimeAssets','CreateAsset','CreateAssets'}
   require(name in allowed,'UNAUTHORIZED_SIMULATOR_ACTION: '+str(name))
   ACTION_COUNTS[name]+=1
   return super().step(action,**kw)
 return GeometryController,Hook
def load_once(task,house,phase,rep):
 global GPU_USED
 from ai2thor.platform import CloudRendering
 from ai2thor.fifo_server import FifoServer
 Controller,Hook=make_classes()
 controller=None
 try:
  GPU_USED=1
  # No policy modules or task sampler. Same simulator physics flags as archived source.
  controller=Controller.__new__(Controller)
  Controller.__init__(controller,scene=copy.deepcopy(house),commit_id=BUILD,platform=CloudRendering,server_class=FifoServer,gpu_device=0,
   width=396,height=224,gridSize=0.15,visibilityDistance=0.8673349051766235,visibilityScheme='Distance',fieldOfView=59,
   useMassThreshold=False,massThreshold=1,autoSimulation=False,autoSyncTransforms=True,renderInstanceSegmentation=True,
   agentMode='stretch',renderDepthImage=False,cameraNearPlane=0.01,snapToGrid=False,fastActionEmit=True,
   server_timeout=1200,server_start_timeout=180,action_hook_runner=Hook(asset_directory=str(ASSETS),asset_symlink=True,verbose=False,asset_limit=200,stop_if_fail=True))
  require(controller._build.commit_id==BUILD,'RUNTIME_BUILD_MISMATCH')
  # lastActionSuccess is simulator API completion, never an ObjectNav outcome.
  meta=controller.last_event.metadata
  require(meta.get('lastActionSuccess') is True,'SCENE_CREATION_FAILED')
  mapping=collections.defaultdict(list)
  for obj in meta.get('objects',[]):mapping[obj['objectId']].append(obj)
  result={}
  for target in task['targets']:
   matches=mapping.get(target['id'],[])
   row={'phase':phase,'repeat':rep,'task_key':task['key'],'house_index':task['house_index'],'synset':task['synset'],'target_id':target['id'],'asset_id':target['asset_id'],'static_candidate':target['has_static_candidate'],'exact_id_matches':len(matches),'valid_aabb':False,'status':'NOT_MEASURED','aabb_x_m':'','aabb_y_m':'','aabb_z_m':'','aabb_volume_m3':'','max_side_m':'','obb_available':False}
   TARGET_ROWS.append(row)
   require(len(matches)==1,'INCOMPLETE_OR_AMBIGUOUS_PREFLIGHT_TARGET_MAPPING')
   try:g=geometry(matches[0])
   except Blocked as e:row['status']=str(e);raise
   row.update(valid_aabb=True,status='VALID_CREATION_STATE_AABB',aabb_x_m=g['dimensions'][0],aabb_y_m=g['dimensions'][1],aabb_z_m=g['dimensions'][2],aabb_volume_m3=g['volume'],max_side_m=g['max_side'],obb_available=g['obb_available'])
   result[target['id']]=g
  save('geometry_sample_'+phase+'_'+str(rep)+'_'+str(task['house_index'])+'.json',{'task_key':task['key'],'phase':phase,'repeat':rep,'scene_initializations':SCENES,'runtime_build':BUILD,'targets':result})
  return result
 finally:
  if controller is not None and getattr(controller,'server',None) is not None:
   try:controller.stop()
   except Exception as cleanup_error:
    save('cleanup_error.json',{'type':type(cleanup_error).__name__,'message':str(cleanup_error)[:300]})
def main():
 global SCENES
 require(Path.cwd()==ROOT,'WRONG_EXECUTION_WORKTREE')
 require(not (OUT/'final_status.json').exists(),'RUN_ALREADY_FINALIZED_NO_RETRY')
 started=stamp();reason='';status='BLOCKED';tasks=[];selected=[];version={}
 checkpoint_status('PREFLIGHT_VERSION')
 try:
  state=json.loads((CONTROL/'research/LOOP_STATE.json').read_text())
  require(state['status']=='CODEX_RUNNING' and state['instruction_commit']==APPROVAL,'CLAIM_STATE_MISMATCH')
  require(subprocess.check_output(['git','-C',str(CONTROL),'rev-parse','HEAD']).decode().strip()==CLAIM,'CLAIM_HEAD_CHANGED')
  version=version_gate()
  tasks,selected,houses=get_inputs()
  save('allowed_input_manifest.json',READ_INPUTS)
  for n,t in enumerate(selected,1):
   samples=[]
   for rep in [1,2]:
    samples.append(load_once(t,houses[t['house_index']],'preflight',rep))
    print(json.dumps({'phase':'preflight','task':n,'repeat':rep,'scene_initializations':SCENES}),flush=True)
   for target in t['targets']:
    a,b=[x[target['id']] for x in samples];av,bv=a['vector'],b['vector']
    require(len(av)==len(bv),'REPEAT_GEOMETRY_SCHEMA_CHANGED')
    diff=[abs(x-y) for x,y in zip(av,bv)]
    rel=[d/max(abs(x),abs(y),1e-12) for d,x,y in zip(diff,av,bv)]
    stable=all(d<=ABS_TOL+REL_TOL*max(abs(x),abs(y)) for d,x,y in zip(diff,av,bv))
    dimdiff=[abs(x-y) for x,y in zip(a['dimensions'],b['dimensions'])]
    row={'task_key':t['key'],'target_id':target['id'],'exact_equal':av==bv,'max_absolute_dimension_difference_m':max(dimdiff),'max_relative_dimension_difference':max(d/max(abs(x),abs(y),1e-12) for d,x,y in zip(dimdiff,a['dimensions'],b['dimensions'])),'max_absolute_geometry_difference_m':max(diff),'within_tolerance':stable,'status':'PASS' if stable else 'BLOCKED_UNSTABLE_GEOMETRY'}
    PREFLIGHT_ROWS.append(row)
    require(stable,'UNEXPLAINED_PREFLIGHT_GEOMETRY_INSTABILITY')
   checkpoint_status('PREFLIGHT_PASS_TASK_'+str(n))
  require(SCENES==24,'PREFLIGHT_INITIALIZATION_COUNT_MISMATCH')
  for n,t in enumerate(tasks,1):
   result=load_once(t,houses[t['house_index']],'full',1)
   require(len(result)==len(t['targets']),'INCOMPLETE_FULL_TASK')
   TASK_ROWS.append({'task_key':t['key'],'house_index':t['house_index'],'target_count':len(t['targets']),'valid_aabb_count':len(result),'complete':True,'median_aabb_volume_m3':statistics.median(x['volume'] for x in result.values()),'median_max_side_m':statistics.median(x['max_side'] for x in result.values()),'status':'COMPLETE_CREATION_STATE_GEOMETRY'})
   print(json.dumps({'phase':'full','task':n,'scene_initializations':SCENES}),flush=True)
  status='AWAITING_PI_REVIEW';reason='COMPLETE_RUNTIME_GEOMETRY_RECOVERY'
 except Exception as e:
  reason=str(e) if isinstance(e,Blocked) else type(e).__name__+': '+str(e)[:500]
  save('failure_detail.json',{'exception_type':type(e).__name__,'reason':reason,'traceback':traceback.format_exc(),'scene_initializations':SCENES})
 finally:
  targetfields=['phase','repeat','task_key','house_index','synset','target_id','asset_id','static_candidate','exact_id_matches','valid_aabb','status','aabb_x_m','aabb_y_m','aabb_z_m','aabb_volume_m3','max_side_m','obb_available']
  csvwrite('target_geometry.csv',TARGET_ROWS,targetfields)
  prefields=['task_key','target_id','exact_equal','max_absolute_dimension_difference_m','max_relative_dimension_difference','max_absolute_geometry_difference_m','within_tolerance','status']
  csvwrite('preflight_repeatability.csv',PREFLIGHT_ROWS,prefields)
  taskfields=['task_key','house_index','target_count','valid_aabb_count','complete','median_aabb_volume_m3','median_max_side_m','status']
  completed={r['task_key'] for r in TASK_ROWS}
  for t in tasks:
   if t['key'] not in completed:TASK_ROWS.append({'task_key':t['key'],'house_index':t['house_index'],'target_count':len(t['targets']),'valid_aabb_count':'','complete':False,'median_aabb_volume_m3':'','median_max_side_m':'','status':'NOT_EXTRACTED_STOP_CONDITION'})
  csvwrite('task_geometry.csv',TASK_ROWS,taskfields)
  record(Path(__file__))
  save('allowed_input_manifest.json',READ_INPUTS)
  final={'status':status,'reason':reason,'started_at_utc':started,'completed_at_utc':stamp(),'scene_initializations':SCENES,'max_scene_initializations':MAX_SCENES,'gpu_count':GPU_USED,'gpu_index':GPU if GPU_USED else None,'safevla_episodes':0,'model_checkpoint_loads':0,'actor_critic_forwards':0,'outcome_columns_read':False,'size_success_association':False,'preflight_comparisons':len(PREFLIGHT_ROWS),'full_target_rows':sum(r['phase']=='full' for r in TARGET_ROWS),'complete_tasks':sum(r['complete'] for r in TASK_ROWS),'action_counts':dict(ACTION_COUNTS),'forbidden_modules_present':[x for x in sys.modules if x.split('.')[0] in {'torch','transformers','architecture','tasks','training','online_evaluation'}]}
  save('final_status.json',final);checkpoint_status(status,reason);print(json.dumps(final),flush=True)
if __name__=='__main__':main()
