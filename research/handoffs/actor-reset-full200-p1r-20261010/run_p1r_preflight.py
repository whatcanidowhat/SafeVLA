"""001D P1 one-shot supervisor and isolated child phases. No live ON, no retry."""
import csv,datetime,gzip,hashlib,importlib.metadata,json,os,random,signal,subprocess,sys,time,traceback,types
from pathlib import Path
R=Path('/nvme2/user/qyy/SafeVLA_p1r_001d')
O=R/'research/handoffs/actor-reset-full200-p1r-20261010'
W=R/'research/runs/actor-reset-full200-p1r-20261010'
C=Path('/nvme2/user/qyy/SafeVLA_loop_control_p1_renewal_20261010')
D=Path('/nvme2/user/qyy/SafeVLA')
CLAIM='d35f645cabf45bfb2ad4c3dffa548cdd7da7c8e5'
CLAIM_ID='fa92b7b37bb6457383ea4f3f19fa1b8b'
ATOL=1e-5
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda a:a.tolist() if hasattr(a,'tolist') else str(a))+'\n')
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,default=lambda a:a.tolist()).encode()
def note(x):print(now()+' '+x,flush=True)
def jread(p):return json.loads(Path(p).read_text())
def budget():
 assert time.time()-jread(W/'launch.json')['epoch']<8*3600,'8 GPU-hour ceiling'
 size=sum(p.stat().st_size for p in W.rglob('*') if p.is_file())
 assert size<50*1024**3,'50 GiB ceiling'
def verify_identity():
 f=jread(W/'frozen_identity.json')
 for n,h in f['source'].items():assert sha(R/n)==h,'B0 source mismatch '+n
 for label,v in f['resources'].items():assert sha(v['path'])==v['sha256'],label+' weight mismatch'
 for n,h in jread(W/'dino_source_sha256.json').items():assert sha(Path(f['environment']['DINOV2_REPO'])/n)==h,'DINO source mismatch '+n
 for n,v in f['dependencies'].items():assert importlib.metadata.version(n)==v,'Dependency mismatch '+n
 assert sys.executable=='/home/amax/.conda/envs/safevla/bin/python'
 assert hashlib.sha256(gzip.decompress((R/'benchmark/objectnavtype_val.jsonl.gz').read_bytes())).hexdigest()=='f10ce169dc71e0ee2b4f9bb803a464babf8c72e28ad7d7d2a28b0d80a174863e'
 return f

def configure():
 f=jread(W/'frozen_identity.json')
 for k,v in f['environment'].items():
  if v is not None:os.environ[k]=str(v)
 os.environ.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(R),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='offline',RESET_STATE_FIX='0',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='False')
 assert not os.environ.get('ACTION_DICT'),'Unexpected action override'
 os.chdir(R);sys.path.insert(0,str(R))

def restore_temporal(b,s):
 b.time_step_counter=s[0]
 for layer,(k,v) in zip(b.decoder.layers,s[1]):
  layer.attention.cache_k=k.clone()
  layer.attention.cache_v=v.clone()

def cpu_gate():
 configure();os.environ['CUDA_VISIBLE_DEVICES']='';start=time.monotonic()
 with (W/'cpu.started').open('x') as marker:marker.write(now())
 report={'phase':'cpu','status':'RUNNING','gpu_count':0,'initialization_attempts':0,'episodes_started':0,'episodes_completed':0,'online_decisions':0,'offline_combined_forwards':0,'offline_root_forwards':0,'started_at':now(),'cases':[]}
 dump(W/'cpu.json',report)
 try:
  verify_identity()
  import numpy as np,torch
  from training.online.third_party_models.llama.model import Attention,ModelArgs
  assert not torch.cuda.is_initialized()
  random.seed(123);np.random.seed(123);torch.manual_seed(123)
  torch.set_num_threads(1)
  def th(t):return hashlib.sha256(str((tuple(t.shape),str(t.dtype),str(t.device))).encode()+t.detach().contiguous().numpy().tobytes()).hexdigest()
  def snapshot(b):return (b.time_step_counter,[(a.attention.cache_k.clone(),a.attention.cache_v.clone()) for a in b.decoder.layers])
  def info(s):return {'counter':s[0],'layers':[{'k_shape':list(k.shape),'v_shape':list(v.shape),'dtype':str(k.dtype),'device':str(k.device),'k_hash':th(k),'v_hash':th(v)} for k,v in s[1]]}
  def exact(b,s):return info(snapshot(b))==info(s)
  def rng():return (random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
  def rngset(s):random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2])
  def rh():
   s=rng();return hashlib.sha256(repr(s[0]).encode()+repr(s[1]).encode()+s[2].numpy().tobytes()).hexdigest()
  def weights(b):return [[th(t) for t in x.attention.state_dict().values()] for x in b.decoder.layers]
  for dtype in [torch.float32,torch.float64]:
   args=ModelArgs(dim=512,n_layers=3,n_heads=8,n_kv_heads=8,max_batch_size=0,max_seq_len=500)
   b=types.SimpleNamespace(time_step_counter=0,decoder=types.SimpleNamespace(layers=[types.SimpleNamespace(attention=Attention(args).to(dtype=dtype).eval()) for _ in range(3)]))
   x=torch.arange(512,dtype=dtype).reshape(1,1,512)/512
   before_weights=weights(b)
   def step():
    out=x
    for layer in b.decoder.layers:out=layer.attention(out,b.time_step_counter,None)
    b.time_step_counter+=1
    return out.clone()
   with torch.no_grad():
    for case in ['empty','allocated']:
     saved=snapshot(b);saved_info=info(saved);original_rng=rng();initial_rng=rh()
     if case=='empty':assert all(list(k.shape)==[0,500,8,64] for k,v in saved[1]) and saved[0]==0
     else:assert all(list(k.shape)==[1,500,8,64] for k,v in saved[1]) and saved[0]==1
     y=step();after=snapshot(b);after_rng=rh()
     stale=[(l.attention.cache_k,l.attention.cache_v) for l in b.decoder.layers];stale_hashes=[(th(k),th(v)) for k,v in stale]
     restore_temporal(b,saved)
     assert exact(b,saved) and info(saved)==saved_info and rh()==after_rng
     for layer,(k,v),(oldk,oldv) in zip(b.decoder.layers,saved[1],stale):
      assert layer.attention.cache_k is not k and layer.attention.cache_v is not v
      assert layer.attention.cache_k is not oldk and layer.attention.cache_v is not oldv
      if k.numel():assert layer.attention.cache_k.data_ptr()!=k.data_ptr() and layer.attention.cache_v.data_ptr()!=v.data_ptr()
     restored=info(snapshot(b));rngset(original_rng);assert rh()==initial_rng
     yy=step()
     assert torch.equal(y,yy) and exact(b,after) and rh()==after_rng
     assert info(saved)==saved_info and [(th(k),th(v)) for k,v in stale]==stale_hashes
     assert weights(b)==before_weights
     report['cases'].append({'dtype':str(dtype),'case':case,'before':saved_info,'after_first':info(after),'restored':restored,'after_repeated':info(snapshot(b)),'output_max_abs':(y-yy).abs().max().item(),'output_hash':th(y),'rng_exact':True,'snapshot_immutable':True,'no_alias':True,'stale_references_unchanged':True,'weights_unchanged':True,'attention_calls':6})
     dump(W/'cpu.json',report)
  assert len(report['cases'])==4 and not torch.cuda.is_initialized()
  report.update(status='PASS',real_attention_source=str(R/'training/online/third_party_models/llama/model.py'),real_attention_source_sha256=sha(R/'training/online/third_party_models/llama/model.py'),cuda_initialized=False,attention_calls=24,scope='Three genuine Attention modules per dtype, same 512/8/64/500 dimensions; deterministic CPU fixture weights, no policy checkpoint or simulator. Two forwards per lifecycle case, four cases, three layers =24 calls.')
 except BaseException as e:
  report.update(status='INVALID',error=type(e).__name__+': '+str(e));(W/'cpu.error.txt').write_text(traceback.format_exc());traceback.print_exc()
 finally:
  report.update(finished_at=now(),elapsed_seconds=time.monotonic()-start);dump(W/'cpu.json',report);note('Gate A '+report['status'])
 return 0 if report['status']=='PASS' else 2

def child(phase):
 assert jread(W/'cpu.json')['status']=='PASS','Gate A required'
 configure();started=time.monotonic();f=verify_identity()
 with (W/(phase+'.started')).open('x') as marker:marker.write(now())
 import numpy as np, torch
 from training.online import online_eval as entry
 from online_evaluation.online_evaluator import OnlineEvaluatorManager as Manager
 from architecture.models.allenact_transformer_models.inference_agent import InferenceAgentVIDA as Agent
 from architecture.models.allenact_transformer_models.allenact_dino_transformer import DinoLLAMATxNavActorCritic as Base
 from allenact.utils import spaces_utils as su
 assert torch.cuda.device_count()==1,'Exactly one visible GPU required'
 random.seed(123);np.random.seed(123);torch.manual_seed(123);torch.cuda.manual_seed_all(123)
 def cpu(x):return x.detach().cpu().clone()
 def rng():return (random.getstate(),np.random.get_state(),torch.get_rng_state().clone(),[x.clone() for x in torch.cuda.get_rng_state_all()])
 def rngset(x):random.setstate(x[0]);np.random.set_state(x[1]);torch.set_rng_state(x[2]);torch.cuda.set_rng_state_all(x[3])
 def rnghash():
  x=rng();h=hashlib.sha256(repr(x[0]).encode()+repr(x[1]).encode());h.update(x[2].numpy().tobytes())
  for y in x[3]:h.update(y.numpy().tobytes())
  return h.hexdigest()
 def tensorhash(x):return hashlib.sha256(str((str(x.dtype),tuple(x.shape))).encode()+x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
 def inputhash(kwargs):
  h=hashlib.sha256()
  def walk(x,path):
   if isinstance(x,torch.Tensor):h.update(path.encode()+tensorhash(x).encode())
   elif isinstance(x,dict):
    for k in sorted(x):walk(x[k],path+'/'+k)
   elif x is not None:h.update((path+repr(x)).encode())
  walk(kwargs,'');return h.hexdigest()
 def caches(b):return [(l.attention.cache_k.clone(),l.attention.cache_v.clone()) for l in b.decoder.layers]
 def save_temporal(b):return (b.time_step_counter,caches(b))
 def same_temporal(b,s):return b.time_step_counter==s[0] and all(torch.equal(l.attention.cache_k,k) and torch.equal(l.attention.cache_v,v) for l,(k,v) in zip(b.decoder.layers,s[1]))
 def root_reset(root):
  root.time_step_counter=0
  for layer in root.decoder.layers:layer.attention.cache_k.zero_();layer.attention.cache_v.zero_()
 def state_digest(root):
  h=hashlib.sha256()
  for k,v in root.state_dict().items():h.update(k.encode()+tensorhash(v).encode())
  return h.hexdigest()
 def non_target_digest(root):
  h=hashlib.sha256()
  target={id(l.attention) for l in root.decoder.layers}
  def walk(x):
   if isinstance(x,torch.Tensor):return tensorhash(x)
   if isinstance(x,dict):return [(str(k),walk(v)) for k,v in x.items()]
   if isinstance(x,(list,tuple)):return [walk(v) for v in x]
   return repr(x)
  for name,module in root.named_modules():
   for key,value in vars(module).items():
    if key in ['_modules','_parameters','_buffers']:continue
    if module is root and key=='time_step_counter':continue
    if id(module) in target and key in ['cache_k','cache_v']:continue
    h.update(canonical([name,key,walk(value)]))
  return h.hexdigest()
 specs=jread(W/'TASK_SPECS.json');loads=[];holder={};counts={'root':0,'reward':0,'cost':0,'sample':0,'mode':0};resets=[]
 report={'phase':phase,'status':'RUNNING','started_at':now(),'episodes_started':0,'episodes_completed':0,'initialization_attempts':0,'online_decisions':0,'offline_root_forwards':0,'offline_combined_forwards':0,'entry_rng_sha256':rnghash(),'pre_model_seed':123,'gpu':torch.cuda.get_device_name(0),'python':sys.executable}
 dump(W/(phase+'.json'),report)
 def assert_loads():
  policy=[x for x in loads if x['class'].endswith('SafeDinoLLAMATxNavActorCriticSeparate')]
  assert policy and policy[-1]['nkeys']==417 and not policy[-1]['missing'] and not policy[-1]['unexpected'],'Incomplete policy load'
 def checkroot(agent):
  root=agent.actor_critic
  assert root.max_steps==500 and len(root.decoder.layers)==3
  assert len(agent.get_action_list())==20 and agent.get_action_list().index('end')==4
  assert root.decoder is not root.critic_tsfm.decoder and root.decoder is not root.c_critic_tsfm.decoder
  assert not agent.greedy_sampling and agent.test_augmentation
  assert_loads();holder.update(agent=agent,root=root)
  report['model_ready_rng_sha256']=rnghash();report['action_names']=agent.get_action_list()
  report['module_paths']={k:v.__file__ for k,v in sys.modules.items() if getattr(v,'__file__',None) and k.split('.')[0] in ['architecture','training','online_evaluation','tasks','environment','dinov2']}
  for k,p in report['module_paths'].items():
   expected=Path(f['environment']['DINOV2_REPO']) if k.startswith('dinov2') else R
   assert Path(p).resolve().is_relative_to(expected.resolve()),'Unexpected runtime import '+p
  dump(W/(phase+'.json'),report)
 def load_profile(frame,event,arg):
  if frame.f_code.co_name=='load_state_dict' and event=='return' and frame.f_code.co_filename.endswith('/torch/nn/modules/module.py') and hasattr(arg,'missing_keys'):
   obj=frame.f_locals['self'];loads.append({'class':type(obj).__module__+'.'+type(obj).__name__,'strict':frame.f_locals.get('strict'),'nkeys':len(frame.f_locals.get('state_dict',{})),'missing':list(arg.missing_keys),'unexpected':list(arg.unexpected_keys)})
 def augmentation(agent):
  return {str(k):{'num_steps':p.num_steps,'interval':p.num_steps_to_change,'enabled':p.use_augmentation,'transform':str(p.augmentations)} for k,p in agent.sensor_preprocessor_graph.preprocessors.items() if hasattr(p,'num_steps_to_change')}
 # Logger used in both the offline equivalence gate and live OFF sessions.
 class Logger:
  def __init__(self,root):self.root=root;self.current={};self.handles=[]
  def guarded(self,fn):
   def cb(*a,**kw):
    rh=rnghash();r=fn(*a,**kw);assert rnghash()==rh,'Logger consumed RNG';return r
   return cb
  def attach(self):
   def linear(m,a,y):self.current['raw']=cpu(y)
   def dec(m,a):self.current.update(start_pos=int(a[1]),mask_sha256=tensorhash(a[2]) if a[2] is not None else None)
   self.handles=[self.root.actor.linear.register_forward_hook(self.guarded(linear)),self.root.decoder.register_forward_pre_hook(self.guarded(dec))]
  def detach(self):
   for h in self.handles:h.remove()
   self.handles=[]
  def result(self,output,sample,mode,history):
   before=rnghash();r={'raw':self.current['raw'],'normalized':cpu(output.distributions.logits),'probs':cpu(output.distributions.probs),'sample':cpu(sample),'mode':cpu(mode),'history':cpu(history),'counter':self.root.time_step_counter,'start_pos':self.current['start_pos']}
   assert rnghash()==before;return r
 try:
  if phase=='offline':
   sys.setprofile(load_profile)
   params=entry.model_config_params['InferenceDINOv2ViTSLLAMATxTxBaseDist']();params.num_train_processes=0;params.use_bbox=False
   agent=Agent.build_agent(exp_config_type=entry.model_config_type['InferenceDINOv2ViTSLLAMATxTxBaseDist'],params=params,device=0,img_encoder_rgb_mean=entry.img_encoder_type['DinoV2']['mean'],img_encoder_rgb_std=entry.img_encoder_type['DinoV2']['std'],greedy_sampling=False,test_augmentation=True,ckpt_path=f['resources']['checkpoint']['path'])
   sys.setprofile(None);checkroot(agent);root=agent.actor_critic;root.eval()
   inp=D/'research/runs/EXP-RESET-001A/execution_20261008/captured_actor_inputs.pt'
   assert sha(inp)=='417e684e96563b043138d328776d693f05dc74bb7c074099dea39c3b5881500e'
   all_inputs=torch.load(inp,map_location='cpu',weights_only=False);seq=[x for x in all_inputs if x['episode_order']==2];assert len(seq)==600
   class SavedEncoder(torch.nn.Module):
    def forward(self,obs):return obs['saved_obs_embeds'],None
   root.visual_encoder=SavedEncoder();logger=Logger(root);oracle={};steps=[];rollovers=[]
   # Common read-only reference observer remains identical on both sides.
   def oracle_head(m,a,y):oracle['raw']=cpu(y)
   def oracle_decoder(m,a):oracle['input']=cpu(a[0]);oracle['start_pos']=int(a[1])
   handles=[root.actor.linear.register_forward_hook(oracle_head),root.decoder.register_forward_pre_hook(oracle_decoder)]
   def step(record,logged):
    report['offline_root_forwards']+=1;assert report['offline_root_forwards']<=2400
    obs={k:v.to('cuda:0') for k,v in record['observations'].items()};obs['saved_obs_embeds']=record['obs_embeds'].to('cuda:0')
    output,_=Base.forward(root,observations=obs,memory=None,prev_actions=record['prev_actions'].to('cuda:0'),masks=record['masks'].to('cuda:0'))
    assert torch.equal(oracle['input'],record['decoder_input']),'Saved Actor input reconstruction changed'
    s=output.distributions.sample();mode=output.distributions.mode();hist=su.flatten(root.action_space,s)
    counts['root']+=1;counts['sample']+=1;counts['mode']+=1
    return logger.result(output,s,mode,hist) if logged else {'raw':oracle['raw'],'normalized':cpu(output.distributions.logits),'probs':cpu(output.distributions.probs),'sample':cpu(s),'mode':cpu(mode),'history':cpu(hist),'counter':root.time_step_counter,'start_pos':oracle['start_pos']}
   with torch.no_grad():
    root_reset(root)
    for i,record in enumerate(seq):
     temporal=save_temporal(root);rs=rng();ca=dict(counts)
     a=step(record,False);after=save_temporal(root);rha=rnghash();counta={k:counts[k]-ca[k] for k in ca}
     restore_temporal(root,temporal);rngset(rs);cb=dict(counts)
     logger.attach()
     try:b=step(record,True)
     finally:logger.detach()
     countb={k:counts[k]-cb[k] for k in cb}
     delta={k:(a[k]-b[k]).abs().max().item() for k in ['raw','normalized','probs']}
     cachemax=max((l.attention.cache_k-k).abs().max().item() for l,(k,v) in zip(root.decoder.layers,after[1]))
     cachemax=max(cachemax,max((l.attention.cache_v-v).abs().max().item() for l,(k,v) in zip(root.decoder.layers,after[1])))
     exact=all(torch.equal(a[k],b[k]) for k in ['sample','mode','history']) and a['counter']==b['counter'] and rha==rnghash() and counta==countb
     steps.append({'step':i,**delta,'cache_max_abs':cachemax,'exact_discrete_rng_counts':exact,'start_pos':a['start_pos']})
     dump(W/'logger_pairs_partial.json',steps)
     assert max(delta.values())<=ATOL and cachemax<=ATOL and exact,'Logger equivalence failure'
     if temporal[0]>=500:rollovers.append(i)
     if i%100==0:budget();dump(W/'offline_progress.json',{'completed_input_pairs':i+1,'root_forwards':report['offline_root_forwards']});note('Offline input pairs '+str(i+1))
    assert rollovers==[500]
    for h in handles:h.remove()
    critics=[root.critic_tsfm,root.c_critic_tsfm]
    # Engineering sentinels are never forwarded or represented as valid rollout carry.
    # Nonzero critic fixtures make an accidental recursive zero detectable.
    for j,branch in enumerate(critics):
     branch.time_step_counter=37+j
     for layer in branch.decoder.layers:layer.attention.cache_k.fill_(0.125*(j+1));layer.attention.cache_v.fill_(-0.25*(j+1))
    crit=[save_temporal(x) for x in critics];weights=state_digest(root);other=non_target_digest(root);rh=rnghash();aug=augmentation(agent);actor_before=save_temporal(root)
    # OFF is deliberately a no-op treatment, original bookkeeping remains outside this package.
    assert same_temporal(root,actor_before)
    root_reset(root)
    assert root.time_step_counter==0 and all(torch.count_nonzero(l.attention.cache_k)==0 and torch.count_nonzero(l.attention.cache_v)==0 for l in root.decoder.layers)
    assert all(same_temporal(b,s) for b,s in zip(critics,crit)) and state_digest(root)==weights and non_target_digest(root)==other and rnghash()==rh and augmentation(agent)==aug
   dump(W/'logger_equivalence.json',{'status':'PASS','atol':ATOL,'pairs':600,'root_forwards':1200,'combined_forwards':0,'rollover_steps':rollovers,'steps':steps,'scope':'Actual saved root Actor encoder-output boundary; exact original Base.forward. Common read-only oracle observer in both arms; passive Logger toggled only. No visual frontend or simulator equivalence claim.'})
   dump(W/'reset_invariants.json',{'status':'PASS','root_before_counter':actor_before[0],'root_after_counter':0,'root_all_KV_zero':True,'critic_counter_KV_exact':True,'weights_buffers_exact':True,'rng_exact':True,'augmentation_exact':True,'critic_cache_shapes':[[list(k.shape),list(v.shape)] for branch in crit for k,v in branch[1]],'critic_fixture':'Isolated counter/fill sentinels; cache shapes recorded explicitly; empty tensors are not nonzero carry histories','root_fixture':'600 sequential saved real Actor inputs using original root forward; no fabricated root counter'})
   report.update(status='PASS',logger_gate='PASS',reset_gate='PASS')
  else:
   assert phase in ['off_a','off_b'];assert jread(W/'offline.json')['status']=='PASS'
   records=[];episodes=[];logger=None;active={};hooks=[];sampler_depth=0
   expected=[x['sample_id'] for x in specs[:5]]
   from online_evaluation.online_evaluator_worker import OnlineEvaluatorWorker
   original_loader=Manager.load_minival_eval_samples_per_task
   def loader(manager,task_type,use_local_path=None):
    samples=original_loader(manager,task_type,use_local_path)
    assert len(samples)==200
    for sample,spec in zip(samples,specs):
     assert sample['sample_id']==spec['sample_id']
     assert hashlib.sha256(canonical(json.loads(sample['observations']['templated_task_type']))).hexdigest()==spec['spec_sha256']
    report['full_manifest_crosscheck']='PASS';report['manifest_sha256']=sha(O/'TASK_MANIFEST.csv');dump(W/(phase+'.json'),report)
    return samples
   Manager.load_minival_eval_samples_per_task=loader
   class PrefixCompleted(Exception):pass
   def attach(agent):
    nonlocal logger
    checkroot(agent);root=agent.actor_critic;logger=Logger(root);logger.attach()
    def pre(m,args,kwargs):
     counts['root']+=1;active['input_sha256']=inputhash(kwargs);active['counter_before']=root.time_step_counter;active['rng_before_forward']=rnghash()
    hooks.append(root.register_forward_pre_hook(pre,with_kwargs=True))
    for key,branch in [('reward',root.critic_tsfm),('cost',root.c_critic_tsfm)]:
     def counter(m,a,key=key):counts[key]+=1
     hooks.append(branch.register_forward_pre_hook(counter))
   trace=(W/(phase+'.steps.jsonl')).open('x')
   def profile(frame,event,arg):
    nonlocal sampler_depth
    load_profile(frame,event,arg);name=frame.f_code.co_name;file=frame.f_code.co_filename
    if file==str(R/'architecture/models/allenact_transformer_models/inference_agent.py'):
     if name=='build_agent' and event=='return' and arg is not None:attach(arg)
     if name=='reset' and event=='call':resets.append({'completed_episodes':len([e for e in episodes if e['status']=='COMPLETED']),'root_counter':frame.f_locals['self'].actor_critic.time_step_counter})
     if name=='act' and event=='return' and arg is not None:
      agent=frame.f_locals['self'];x=logger.result(frame.f_locals['actor_critic_output'],frame.f_locals['action'],frame.f_locals['action_greedy'],agent.last_action_flat)
      r={k:v.tolist() if isinstance(v,torch.Tensor) else v for k,v in x.items()}
      r.update(episode=len(episodes)-1,local_step=agent.steps_taken_in_task-1,executed_action=arg[0],**active,rng_after_action=rnghash(),augmentation=augmentation(agent))
      assert r['executed_action']==agent.get_action_list()[int(frame.f_locals['action'].item())]
      records.append(r);trace.write(json.dumps(r,ensure_ascii=False)+'\n');trace.flush()
      report['online_decisions']+=1;assert report['online_decisions']<=3000
      if report['online_decisions']%100==0:budget();dump(W/(phase+'.json'),report)
    if event=='call' and isinstance(frame.f_locals.get('self'),torch.distributions.Categorical):
     if name=='sample' and file.endswith('/torch/distributions/categorical.py'):counts['sample']+=1
     if name=='mode' and file.endswith('/allenact/base_abstractions/distributions.py'):counts['mode']+=1
    if file==str(R/'tasks/multi_task_eval_sampler.py') and name=='next_task':
     if event=='call':
      report['initialization_attempts']+=1;assert report['initialization_attempts']<=5,'Initialization attempt budget'
      assert sampler_depth==0,'Unexpected recursive task initialization';sampler_depth+=1;dump(W/(phase+'.json'),report)
     elif event=='return':sampler_depth-=1
    if file==str(R/'online_evaluation/online_evaluator_worker.py') and name=='evaluate_on_task':
     if event=='call':
      task=frame.f_locals['task'];sid=task.task_info['eval_info']['sample_id'];assert sid==expected[len(episodes)] and task.max_steps==600
      assert report.get('full_manifest_crosscheck')=='PASS';assert_loads()
      ep={'sample_id':sid,'status':'STARTED','start_record':len(records),'initial_task_spec_sha256':hashlib.sha256(canonical({k:task.task_info[k] for k in ['house_index','natural_language_spec','agent_starting_position','agent_y_rotation'] if k in task.task_info})).hexdigest(),'initial_rng':rnghash()}
      episodes.append(ep);report['episodes_started']+=1;dump(W/(phase+'.episodes.json'),episodes);note(phase+' episode '+str(len(episodes))+' '+sid)
     elif event=='return' and isinstance(arg,dict):
      task=frame.f_locals['task'];ep=episodes[-1];metrics=arg['metrics'];components={k:frame.f_locals['sum_'+k] for k in ['danger','corner','blind','fragile','critical']}
      assert metrics['cost']==sum(components.values()) and all(metrics[k]==v for k,v in components.items())
      assert len(records)-ep['start_record']==metrics['eps_len']
      assert (metrics['success']>0.1)==bool(frame.f_locals['success'])
      ep.update(status='COMPLETED',decisions=len(records)-ep['start_record'],metrics=metrics,components=components,success=bool(frame.f_locals['success']),actions=[r['executed_action'] for r in records[ep['start_record']:]],trajectory=task.task_info.get('followed_path'),final_rng=rnghash())
      report['episodes_completed']+=1;dump(W/(phase+'.episodes.json'),episodes);dump(W/(phase+'.json'),report);budget();note(phase+' completed '+str(len(episodes))+' decisions='+str(ep['decisions']))
      if len(episodes)==5:raise PrefixCompleted()
   args=['--task_type','ObjectNavType','--eval_subset','minival','--eval_set_size','200','--shuffle','--num_workers','1','--seed','123','--test_augmentation','--max_eps_len','-1','--gpu_devices','0','--house_set','objaverse','--input_sensors','raw_navigation_camera','raw_manipulation_camera','last_actions','an_object_is_in_hand','--ckpt_path',f['resources']['checkpoint']['path'],'--output_basedir',str(W/(phase+'_eval'))]
   report['official_args']=args;sys.argv=['training/online/online_eval.py']+args
   try:
    sys.setprofile(profile);entry.main(entry.parse_args())
   except PrefixCompleted:note(phase+' frozen five-task prefix complete')
   finally:
    sys.setprofile(None);trace.close()
    if logger:logger.detach()
    for h in hooks:h.remove()
    Manager.load_minival_eval_samples_per_task=original_loader
   assert report['episodes_started']==report['episodes_completed']==report['initialization_attempts']==5
   assert all(v==len(records) for v in counts.values()),'Original forward/sample count mismatch'
   assert any(x['class'].startswith('dinov2.') and x['strict'] is True and not x['missing'] and not x['unexpected'] for x in loads),'DINO strict runtime load missing'
   report.update(status='PASS',counts=counts,reset_events=resets,official_metrics_reconciled=True)
 except BaseException as e:
  report.update(status='INVALID',error=type(e).__name__+': '+str(e));(W/(phase+'.error.txt')).write_text(traceback.format_exc());traceback.print_exc()
 finally:
  sys.setprofile(None);report.update(finished_at=now(),elapsed_seconds=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),load_events=loads)
  dump(W/(phase+'.json'),report);note(phase+' '+report['status'])
 return 0 if report['status']=='PASS' else 2

def supervisor():
 configure();state=jread(C/'research/LOOP_STATE.json')
 assert state['claim_id']==CLAIM_ID and state['status']=='CODEX_RUNNING'
 assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(state['authorization']['expires_at_utc'].replace('Z','+00:00'))
 with (W/'launch.json').open('x') as f:json.dump({'epoch':time.time(),'started_at':now(),'one_shot':True},f)
 m={k:state[k] for k in ['cycle_id','experiment_id','instruction_commit','claim_id']}
 m.update(status='RUNNING',claim_commit=CLAIM,claim_ci=38045418732,command=[sys.executable,str(Path(__file__).resolve().relative_to(R))],gpu_count=0,episodes_started=0,episodes_completed=0,started_at=now(),phases=[],no_live_ON=True,execution_worktree=str(R),runner_sha256=sha(__file__),manifest_sha256=sha(O/'TASK_MANIFEST.csv'))
 dump(O/'RUN_MANIFEST.json',m)
 try:
  verify_identity()
  gates={'A':'NOT_STARTED','B':'NOT_STARTED','C':'NOT_STARTED'};dump(O/'GATE_STATUS.json',gates)
  for phase in ['cpu','offline','off_a','off_b']:
   gate='A' if phase=='cpu' else ('B' if phase=='offline' else 'C')
   gates[gate]='RUNNING';dump(O/'GATE_STATUS.json',gates)
   budget();note('Starting bounded phase '+phase)
   if phase=='offline':
    q=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits']).decode().splitlines()
    free={int(x.split(',')[0]):int(x.split(',')[1]) for x in q};m['initial_free_gpu_mib']=free.get(0,0)
    minimum=3*Path(jread(W/'frozen_identity.json')['resources']['checkpoint']['path']).stat().st_size/1024**2+1024
    assert free.get(0,0)>minimum,'GPU0 insufficient free capacity: BLOCKED'
   if phase!='cpu':m['gpu_count']=1
   dump(O/'RUN_MANIFEST.json',m)
   remaining=8*3600-(time.time()-jread(W/'launch.json')['epoch'])
   with (W/(phase+'.stdout.log')).open('x') as log:
    proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--phase',phase],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=20)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
     raise RuntimeError('8 GPU-hour ceiling reached')
   result=jread(W/(phase+'.json')) if (W/(phase+'.json')).exists() else {'phase':phase,'status':'INVALID','error':'Child terminated before report initialization; inspect stdout','initialization_attempts':0,'episodes_completed':0}
   result['exit_code']=code;m['phases'].append(result);m['episodes_started']=sum(x['initialization_attempts'] for x in m['phases']);m['episodes_completed']=sum(x['episodes_completed'] for x in m['phases']);dump(O/'RUN_MANIFEST.json',m)
   gates[gate]=result['status'] if phase!='off_a' or result['status']!='PASS' else 'RUNNING';dump(O/'GATE_STATUS.json',gates)
   assert code==0 and result['status']=='PASS','Phase failed: '+phase+' '+result.get('error','')
  a=jread(W/'off_a.episodes.json');b=jread(W/'off_b.episodes.json');contrasts=[]
  for x,y in zip(a,b):
   assert x['sample_id']==y['sample_id'] and x['initial_task_spec_sha256']==y['initial_task_spec_sha256']
   contrasts.append({'sample_id':x['sample_id'],'actions_exact':x['actions']==y['actions'],'trajectory_exact':x['trajectory']==y['trajectory'],'metrics_exact':x['metrics']==y['metrics'],'initial_rng_exact':x['initial_rng']==y['initial_rng']})
  ra=[json.loads(x) for x in (W/'off_a.steps.jsonl').read_text().splitlines()];rb=[json.loads(x) for x in (W/'off_b.steps.jsonl').read_text().splitlines()]
  first=None
  for i,(x,y) in enumerate(zip(ra,rb)):
   if any(x[k]!=y[k] for k in ['input_sha256','raw','sample','executed_action']):first={'record':i,'episode':x['episode'],'step':x['local_step'],'input_identical':x['input_sha256']==y['input_sha256'],'action_identical':x['executed_action']==y['executed_action']};break
  aa={'status':'PASS' if all(all(v for k,v in c.items() if k!='sample_id') for c in contrasts) and first is None else 'REVIEW_REQUIRED','cases':contrasts,'first_divergence':first,'scope':'Two OFF five-task sessions; engineering validity only, no reset benefit estimate'}
  dump(W/'aa_result.json',aa);m['status']='AWAITING_PI_REVIEW' if aa['status']=='PASS' else 'BLOCKED'
  if aa['status']!='PASS':m['error']='Independent online A/A divergence requires PI assessment before P2; no effect interpretation'
 except BaseException as e:
  traceback.print_exc();m['status']='INVALID' if m['phases'] else 'BLOCKED';m['error']=type(e).__name__+': '+str(e);(W/'supervisor.error.txt').write_text(traceback.format_exc())
 finally:
  m['finished_at']=now();m['source_unchanged']=all(sha(R/n)==h for n,h in jread(W/'frozen_identity.json')['source'].items());dump(O/'RUN_MANIFEST.json',m)
  note('P1 terminal '+m['status'])

if __name__=='__main__':
 if len(sys.argv)==3 and sys.argv[1]=='--phase':sys.exit(cpu_gate() if sys.argv[2]=='cpu' else child(sys.argv[2]))
 assert len(sys.argv)==1;supervisor()
