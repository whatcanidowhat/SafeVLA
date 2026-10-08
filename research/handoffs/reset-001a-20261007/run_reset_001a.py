"""Frozen EXP-RESET-001A executor. One live capture, bounded offline replay, no automatic retry."""
import ast, csv, datetime, difflib, gzip, hashlib, importlib.metadata, json, os
import random, shutil, socket, subprocess, sys, tarfile, traceback, types
from pathlib import Path

DEV=Path('/nvme2/user/qyy/SafeVLA')
CLEAN=Path('/nvme2/user/qyy/SafeVLA_baseline_clean')
CONTROL=Path('/nvme2/user/qyy/SafeVLA_loop_control')
OUT=DEV/'research/handoffs/reset-001a-20261007'
RAW=DEV/'research/runs/EXP-RESET-001A/execution_20261008'
RUNTIME=RAW/'official_runtime'
CLAIM='c4ab4b395817f439a0239514f3469480aa0d8500'
INSTRUCTION='0493548a70aa192bb04457bc15e9343d934bb259'
CLAIM_ID='24f414396c8f4ef39b34b179dee8c6fa'
ATOL=1e-5
RTOL=1e-5

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''): h.update(b)
    return h.hexdigest()
def dump(path,x):
    Path(path).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(type(x)))+'\n')
def git(root,*args): return subprocess.check_output(['git','-C',str(root),*args]).decode().strip()
def note(x): print(now()+' '+x,flush=True)

def prepare():
    assert git(CONTROL,'rev-parse','HEAD')==CLAIM
    state=json.loads((CONTROL/'research/LOOP_STATE.json').read_text())
    assert state['claim_id']==CLAIM_ID and state['status']=='CODEX_RUNNING'
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(state['authorization']['expires_at_utc'])
    assert not (RAW/'model_phase_started.json').exists(), 'Model phase already entered; do not retry'
    RAW.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    assert git(CLEAN,'rev-parse','HEAD')=='2aa82559d272b5f888e53433e258914057f15bed'
    assert git(CLEAN,'diff','--name-only')=='architecture/allenact_preprocessors/dino_preprocessors.py'
    assert not git(CLEAN,'diff','--cached','--name-only')
    paths=git(CLEAN,'ls-files').splitlines()
    prior=json.loads((DEV/'research/runs/EXP-B0-REPRO-001A/execution_20260912/reference/source_sha256.json').read_text())
    for p in paths:
        assert sha(CLEAN/p)==prior[p], 'Accepted B0 source drift: '+p
        target=RUNTIME/p; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(CLEAN/p,target)
    shutil.copytree(CLEAN/'benchmark',RUNTIME/'benchmark',dirs_exist_ok=True)
    frozen={p:sha(RUNTIME/p) for p in paths}
    dump(RAW/'source_sha256.json',frozen)
    with tarfile.open(RAW/'source_snapshot.tar.gz','w:gz') as t:
        for p in paths: t.add(RUNTIME/p,arcname=p,recursive=False)
    (RAW/'development_HEAD.txt').write_text(git(DEV,'rev-parse','HEAD')+'\n')
    (RAW/'development_status.txt').write_text(git(DEV,'status','--short')+'\n')
    diff=subprocess.check_output(['git','-C',str(DEV),'diff','--binary'])
    (RAW/'development.patch').write_bytes(diff)
    resources={}
    for key,p,expected in [('checkpoint','/home/amax/public/datasets/qyy/checkpoints/safe_objnav.pt','05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301'),('DINO',os.environ['DINOV2_CKPT'],'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9')]:
        resources[key]={'path':p,'sha256':sha(p)}
        assert resources[key]['sha256']==expected,key+' resource drift'
    dino=Path(os.environ['DINOV2_REPO'])
    dino_hash={str(p.relative_to(dino)):sha(p) for p in sorted(dino.rglob('*.py'))}
    dump(RAW/'dino_source_sha256.json',dino_hash)
    with tarfile.open(RAW/'dino_source_snapshot.tar.gz','w:gz') as t:
        for p in dino_hash: t.add(dino/p,arcname=p,recursive=False)
    rows=[json.loads(x) for x in gzip.open(RUNTIME/'benchmark/objectnavtype_val.jsonl.gz','rt')]
    selected=sorted(range(len(rows)),key=lambda i:(-rows[i]['expert_length'],i))[:4]
    selection=[{'index':i,'expert_length':rows[i]['expert_length'],'task_path':rows[i]['task_path'],'house_index':rows[i]['house_index']} for i in selected]
    dump(OUT/'capture_plan.json',{'selection_rule':'Four largest recorded expert_length values, index tie-break; no outcome selection. Stop after first complete episode with >=220 decisions. No extra episodes or retry.','selected':selection,'seed':123,'max_episodes':4,'worker_count':1,'greedy':False,'test_augmentation':True,'horizon':600,'atol':ATOL,'rtol':RTOL,'warmup':'Replay first 300 decisions; if the sequence is shorter, concatenate another replay prefix after the original official boundary semantics. All warmup inputs are captured real inputs.','sampling':'Greedy primary; paired Torch categorical draws use restored common RNG state.','checkpoint':resources,'capture_boundary':'Root Actor forward_encoder output plus masks, previous actions, episode timestep, object-in-hand; also capture actual decoder input/output and raw linear logits. No added live forward or simulator query.'})
    manifest={'experiment_id':'EXP-RESET-001A','cycle_id':'reset-001a-20261007','instruction_commit':INSTRUCTION,'claim_id':CLAIM_ID,'claim_commit':CLAIM,'claim_ci':'https://github.com/whatcanidowhat/SafeVLA/actions/runs/37752169988','status':'RUNNING','started_utc':now(),'host':socket.gethostname(),'execution_worktree':str(DEV),'runtime_source_root':str(RUNTIME),'python':sys.executable,'command':['python3 research/handoffs/reset-001a-20261007/run_reset_001a.py'],'commands':['python3 research/handoffs/reset-001a-20261007/run_reset_001a.py'],'gpu_count':0,'episodes_started':0,'episodes_completed':0,'resources':resources,'source_manifest_sha256':sha(RAW/'source_sha256.json'),'dino_source_manifest_sha256':sha(RAW/'dino_source_sha256.json'),'development_diff_sha256':hashlib.sha256(diff).hexdigest(),'environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','DINOV2_REPO','DINOV2_CKPT','OBJAVERSE_HOUSES_DIR','OBJAVERSE_DATA_DIR','NLTK_DATA','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','WANDB_MODE']},'source_frozen':frozen,'selected_tasks':selection,'seed':123,'worker_count':1,'raw_directory':str(RAW)}
    dump(OUT/'RUN_MANIFEST.json',manifest)
    make_audit()
    return manifest,selected

def make_audit():
    entries=[]
    for carrier,owner,lifetime,roll,effect,reset in [
        ('time_step_counter','root DinoLLAMATxNavActorCritic','model instance', '>=max_steps (500) before decoder, or sequence length >1','sets cache write/read length and episode mask start','set root counter=0 at episode boundary'),
        ('cache_k','root.decoder.layers[*].attention','model instance; ordinary Tensor, not state_dict buffer','prefix overwritten after counter rollover','per-layer attention keys','zero root cache_k tensors, preserving shape/device/dtype'),
        ('cache_v','root.decoder.layers[*].attention','model instance; ordinary Tensor, not state_dict buffer','prefix overwritten after counter rollover','per-layer attention values','zero root cache_v tensors, preserving shape/device/dtype'),
        ('episode-local timestep / masks / previous action','InferenceAgentVIDA and rollout storage','episode','independent of model counter','time positional encoding, previous-action token, episode attention mask','retain original reset; no extra treatment change'),
        ('reward/cost branch state','critic_tsfm / c_critic_tsfm','independent model instances','same formula, independently owned','not used for Actor action distribution','do not change')]:
        entries.append(dict(state_carrier=carrier,owner_module=owner,lifetime=lifetime,episode_reset_now='original agent reset leaves model-side counter and K/V unchanged',rollover_condition=roll,actor_effect_path=effect,correct_reset_operation=reset))
    dump(OUT/'decoder_state_map.json',{'state_carriers':entries,'cache_shape':'[batch,500,8,64] per layer, 3 layers; runtime verified separately','actual_decoder':'training/online/third_party_models/llama/model.py','actor_only_reset_isolatable':True})
    (OUT/'decoder_reset_audit.md').write_text('''# P0 decoder reset audit

The actual imported decoder is `training/online/third_party_models/llama/model.py`, not the similarly named unused architecture decoder.
The root Actor and `critic_tsfm`/`c_critic_tsfm` are independently constructed model objects. Actor distributions come from the root Actor head; critic values do not select actions.

`allenact_dino_transformer.py` forward resets its own counter at >=max_steps before constructing the mask, calls its decoder with start_pos=counter, then increments for a single decision. The default runtime max_steps is 500. Episode-local time remains independent and drives the sinusoidal time encoder.
The current-episode attention mask is `max(counter-local_timestep,0) <= arange(counter+1)`. Previous episodes are masked before rollover. On rollover, retained current-episode history before the new write origin is no longer addressed.

Each of 3 decoder layers owns K and V ordinary Tensor attributes. One-step attention writes [start_pos:start_pos+1], reads only [:start_pos+1], and uses scaled_dot_product_attention with dropout=0. Cache entries outside that prefix are not read. No RoPE is applied on this path. `sampler_select` resizes/selects sampler caches; it is not an episode-reset API.

Original `InferenceAgentVIDA.reset()` resets rollout bookkeeping, steps, trajectory index and memory, but not the root or critic counters/caches. Setting counter=0 alone is computationally consistent on the sequential overwrite-before-read path, but canonical clean state is defined here as counter=0 plus zero all root K/V tensors. The explicit treatment performs this complete package; it leaves both critics untouched.

CARRY-300 is generated by 300 actual sequential calls on recorded inputs with valid caches, never by assigning 300 to a counter with empty caches. A current-episode boundary retains that package using original reset semantics. The two replay conditions share episode masks/timesteps/recorded inputs; their derived attention-mask dimensions necessarily depend on their counters and are part of the treatment pathway.

Source bytes are frozen against the accepted official-source snapshot in `source_sha256.json`. Runtime identity and independence are checked before capture. This audit establishes implementation semantics, not an SR effect.
''')
    p='architecture/models/allenact_transformer_models/inference_agent.py'
    before=(RUNTIME/p).read_text()
    needle='    def reset(self):\n'
    assert before.count(needle)==1
    after=before.replace(needle,needle+'''        if os.getenv("RESET_STATE_FIX", "0") == "1":
            actor = self.actor_critic
            actor.time_step_counter = 0
            for layer in actor.decoder.layers:
                layer.attention.cache_k.zero_()
                layer.attention.cache_v.zero_()
''')
    (RAW/'patched_inference_agent.py').write_text(after)
    (OUT/'reset_fix.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+p,tofile='b/'+p)))
    (OUT/'reset_fix_design.md').write_text('''# RESET_STATE_FIX design frozen before capture

Only the episode-boundary reset method is changed in the review patch. OFF is the exact original reset path; ON first resets only the root Actor counter and root decoder K/V, then executes original rollout bookkeeping. No branch-critic reset, action selection change, simulator query, forward addition or weight change occurs.
The patch is not applied to any B0 source checkout. The exact patched reset method is AST-loaded into the offline harness. Live capture uses untouched official reset. OFF A/A uses original versus patched reset and the same captured Actor encoder outputs, observation fields, previous-action tensors, masks, root forward implementation, weights and common RNG. Both hidden outputs and 20-D raw logits, probabilities, greedy/sample/history selections are checked. Reconstructed decoder input must match captured input exactly.

Tolerance predeclared: abs=1e-5, rel=1e-5 for float outputs; exact equality for actions/history and reconstructed input. Within-condition replay must pass twice. Metrics include raw maxima even below tolerance. Sampling is optional CRN categorical sampling, with the original selection/history code statically unchanged. No stochastic trajectory equivalence claim.

CLEAN calls patched reset ON; CARRY replays 300 decisions and calls original reset. Warmup wraps a recorded sequence with an official boundary only if fewer than 300 inputs were captured. No divergent prediction feeds back: recorded previous actions remain fixed. Warmup is a synthetic offline valid history, not a claim of 300 live decisions from an independent episode.
''')

class CapturedEnough(Exception): pass

def execute(manifest,selected):
    with (RAW/'model_phase_started.json').open('x') as f:
        json.dump({'started_utc':now(),'single_entry':True},f)
    os.chdir(RUNTIME)
    sys.path.insert(0,str(RUNTIME))
    os.environ['PYTHONPATH']=str(RUNTIME)
    os.environ['RESET_STATE_FIX']='0'
    import numpy as np, torch
    from training.online import online_eval as entry
    from online_evaluation.online_evaluator import OnlineEvaluatorManager
    from architecture.models.allenact_transformer_models.inference_agent import InferenceAgentVIDA
    from architecture.models.allenact_transformer_models.allenact_dino_transformer import DinoLLAMATxNavActorCritic
    from allenact.utils import spaces_utils as su
    manifest['gpu_count']=1
    manifest['dependencies']={k:importlib.metadata.version(k) for k in ['torch','numpy','transformers','ai2thor','jsonschema']}
    manifest['gpu']=torch.cuda.get_device_name(0)
    dump(OUT/'RUN_MANIFEST.json',manifest)
    def cpu(x): return x.detach().cpu().clone() if isinstance(x,torch.Tensor) else x
    def fingerprint():
        h=hashlib.sha256(repr(random.getstate()).encode()+repr(np.random.get_state()).encode())
        h.update(torch.get_rng_state().numpy().tobytes())
        for x in torch.cuda.get_rng_state_all(): h.update(x.cpu().numpy().tobytes())
        return h.hexdigest()
    def guarded(fn):
        def wrapped(*a,**k):
            before=fingerprint(); result=fn(*a,**k)
            assert fingerprint()==before,'Observer changed RNG'
            return result
        return wrapped
    records=[]; hooks=[]; holder={}; episodes=[]; active={}; loads=[]
    counts={'actor':0,'reward':0,'cost':0,'top':0}
    original_loader=OnlineEvaluatorManager.load_minival_eval_samples_per_task
    def select_tasks(manager,task_type,use_local_path=None):
        saved=(manager.eval_set_size,manager.shuffle)
        manager.eval_set_size=None; manager.shuffle=False
        try: all_rows=original_loader(manager,task_type,use_local_path)
        finally: manager.eval_set_size,manager.shuffle=saved
        chosen=[all_rows[i] for i in selected]
        dump(RAW/'selected_normalized_tasks.json',chosen)
        return chosen
    OnlineEvaluatorManager.load_minival_eval_samples_per_task=select_tasks
    def attach(agent):
        root=agent.actor_critic
        assert root.max_steps==500 and root.traj_idx_uuid is not None
        assert root.decoder is not root.critic_tsfm.decoder and root.decoder is not root.c_critic_tsfm.decoder
        assert set(map(id,root.decoder.parameters())).isdisjoint(map(id,root.critic_tsfm.decoder.parameters()))
        holder.update(agent=agent,root=root)
        dump(RAW/'runtime_state_layout.json',{'max_steps':root.max_steps,'action_names':agent.get_action_list(),'independent_decoder_owners':True,'cache_shapes_before_first_forward':[list(l.attention.cache_k.shape) for l in root.decoder.layers]})
        @guarded
        def top(m,a,kw):
            counts['top']+=1
            obs=kw['observations']
            fields={k:cpu(obs[k]) for k in [root.time_step_uuid,root.traj_idx_uuid,root.an_object_is_in_hand_uuid] if k is not None}
            records.append({'episode_order':len(episodes)-1,'observations':fields,'prev_actions':cpu(kw['prev_actions']),'masks':cpu(kw['masks']),'counter_before':root.time_step_counter})
        hooks.append(root.register_forward_pre_hook(top,with_kwargs=True))
        @guarded
        def enc(m,a,y): records[-1]['obs_embeds']=cpu(y[0])
        hooks.append(root.visual_encoder.register_forward_hook(enc))
        @guarded
        def decpre(m,a):
            records[-1].update(decoder_input=cpu(a[0]),start_pos=int(a[1]),attention_mask=cpu(a[2]))
        hooks.append(root.decoder.register_forward_pre_hook(decpre))
        @guarded
        def decpost(m,a,y):
            counts['actor']+=1; records[-1]['hidden']=cpu(y)
        hooks.append(root.decoder.register_forward_hook(decpost))
        @guarded
        def rawlogits(m,a,y): records[-1]['logits']=cpu(y)
        hooks.append(root.actor.linear.register_forward_hook(rawlogits))
        for key,branch in [('reward',root.critic_tsfm),('cost',root.c_critic_tsfm)]:
            def count(m,a,y,key=key): counts[key]+=1
            hooks.append(branch.decoder.register_forward_hook(guarded(count)))
    def profile(frame,event,arg):
        name=frame.f_code.co_name; file=frame.f_code.co_filename
        if name=='load_state_dict' and event=='return' and file.endswith('/torch/nn/modules/module.py') and hasattr(arg,'missing_keys'):
            obj=frame.f_locals['self']; cls=type(obj)
            loads.append({'class':cls.__module__+'.'+cls.__name__,'strict':frame.f_locals.get('strict'),'missing':list(arg.missing_keys),'unexpected':list(arg.unexpected_keys),'nkeys':len(frame.f_locals.get('state_dict',{}))})
        if file==str(RUNTIME/'architecture/models/allenact_transformer_models/inference_agent.py'):
            if name=='build_agent' and event=='return' and arg is not None: attach(arg)
            if name=='act' and event=='return' and records:
                r=records[-1]; agent=frame.f_locals['self']
                r['history_action']=cpu(agent.last_action_flat); r['returned_action']=arg[0]
                r['sampled_action']=cpu(frame.f_locals['action']); r['greedy_action']=cpu(frame.f_locals['action_greedy'])
        if file==str(RUNTIME/'online_evaluation/online_evaluator_worker.py') and name=='evaluate_on_task':
            if event=='call':
                assert len(episodes)<4,'Episode cap exceeded'
                task=frame.f_locals['task']; assert task.max_steps==600
                ep={'order':len(episodes),'task_info':task.task_info,'start_record':len(records),'status':'STARTED'}
                episodes.append(ep); manifest['episodes_started']=len(episodes)
                dump(RAW/'episodes.json',episodes); dump(OUT/'RUN_MANIFEST.json',manifest)
                note('Capture episode '+str(len(episodes))+' started')
            elif event=='return' and isinstance(arg,dict):
                ep=episodes[-1]; ep.update(status='COMPLETED',decisions=len(records)-ep['start_record'],metrics=arg['metrics'])
                manifest['episodes_completed']+=1
                torch.save(records,RAW/'captured_actor_inputs.pt')
                dump(RAW/'episodes.json',episodes); dump(OUT/'RUN_MANIFEST.json',manifest)
                note('Capture episode completed: '+str(ep['decisions'])+' decisions')
                if ep['decisions']>=220: raise CapturedEnough()
    args=['--task_type','ObjectNavType','--eval_subset','minival','--eval_set_size','4','--num_workers','1','--seed','123','--test_augmentation','--max_eps_len','-1','--gpu_devices','0','--house_set','objaverse','--input_sensors','raw_navigation_camera','raw_manipulation_camera','last_actions','an_object_is_in_hand','--ckpt_path',manifest['resources']['checkpoint']['path'],'--output_basedir',str(RAW/'capture_output')]
    manifest['official_entrypoint_args']=args
    sys.argv=['training/online/online_eval.py']+args
    note('Starting one bounded official B0 capture')
    try:
        sys.setprofile(profile)
        entry.main(entry.parse_args())
    except CapturedEnough: note('Sufficient fixed sequence captured; remaining episodes not started')
    finally:
        sys.setprofile(None)
        for h in hooks: h.remove()
        OnlineEvaluatorManager.load_minival_eval_samples_per_task=original_loader
        torch.save(records,RAW/'captured_actor_inputs.pt')
        dump(RAW/'load_events.json',loads)
        dump(RAW/'forward_counts.json',counts)
        dump(RAW/'runtime_module_paths.json',{k:getattr(v,'__file__',None) for k,v in sys.modules.items() if k.split('.')[0] in ['architecture','training','online_evaluation','tasks','environment','allenact','ai2thor','torch','dinov2'] and getattr(v,'__file__',None)})
    assert counts['top']==counts['actor']==counts['reward']==counts['cost']==len(records),'Forward count mismatch'
    assert any('dinov2.' in l['class'] and l['strict'] is True and not l['missing'] and not l['unexpected'] for l in loads),'DINO strict load unverified'
    assert any('SafeDinoLLAMATxNavActorCriticSeparate' in l['class'] and not l['missing'] and not l['unexpected'] for l in loads),'Policy checkpoint load incomplete'
    candidates=[e for e in episodes if e.get('decisions',0)>=220 and e['status']=='COMPLETED']
    if not candidates: raise RuntimeError('BLOCKED: No valid >=220-decision sequence within four live episodes')
    ep=candidates[0]; seq=records[ep['start_record']:ep['start_record']+ep['decisions']]
    assert all(int(r['observations']['time_step'].item())==i for i,r in enumerate(seq))
    manifest['fixed_sequence']={'episode_order':ep['order'],'decisions':len(seq),'raw_input_path':str(RAW/'captured_actor_inputs.pt'),'raw_input_sha256':sha(RAW/'captured_actor_inputs.pt')}
    note('Beginning offline A/A and rollover replay on '+str(len(seq))+' captured decisions')
    replay(holder['root'],seq,manifest,torch,InferenceAgentVIDA,DinoLLAMATxNavActorCritic,su)

def replay(root,seq,manifest,torch,Agent,Base,su):
    class SavedEncoder(torch.nn.Module):
        def forward(self,obs): return obs['saved_obs_embeds'],None
    root.visual_encoder=SavedEncoder()
    root.eval()
    tree=ast.parse((RAW/'patched_inference_agent.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='InferenceAgentVIDA')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='reset')
    ns={'os':os}; exec(compile(ast.Module(body=[method],type_ignores=[]),str(RAW/'patched_inference_agent.py'),'exec'),ns)
    patched_reset=ns['reset']
    stub=types.SimpleNamespace(actor_critic=root,has_initialized=False,steps_taken_in_task=0,num_evaluated_traj=0,memory=None)
    def reset(on,patched=True):
        os.environ['RESET_STATE_FIX']=str(int(on))
        (patched_reset if patched else Agent.reset)(stub)
    captured={}
    def pre(m,a): captured.update(x=a[0].detach().cpu().clone(),pos=int(a[1]))
    def post(m,a,y): captured['h']=y.detach().cpu().clone()
    def head(m,a,y): captured['z']=y.detach().cpu().clone()
    hooks=[root.decoder.register_forward_pre_hook(pre),root.decoder.register_forward_hook(post),root.actor.linear.register_forward_hook(head)]
    def step(r):
        obs={k:v.to('cuda:0') for k,v in r['observations'].items()}
        obs['saved_obs_embeds']=r['obs_embeds'].to('cuda:0')
        before=root.time_step_counter
        result,_=Base.forward(root,observations=obs,memory=None,prev_actions=r['prev_actions'].to('cuda:0'),masks=r['masks'].to('cuda:0'))
        assert torch.equal(captured['x'],r['decoder_input']),'Captured decoder input reconstruction changed'
        sampled=result.distributions.sample(); greedy=result.distributions.mode()
        history=su.flatten(root.action_space,sampled)
        return {'hidden':captured['h'],'logits':captured['z'],'probs':result.distributions.probs.detach().cpu().clone(),'sampled':sampled.detach().cpu().clone(),'greedy':greedy.detach().cpu().clone(),'history':history.detach().cpu().clone(),'counter_before':before,'start_pos':captured['pos'],'counter_after':root.time_step_counter}
    def setrng():
        torch.manual_seed(20261008); torch.cuda.manual_seed_all(20261008)
    def warm():
        reset(True)
        for i in range(300):
            if i and i%len(seq)==0: reset(False,False)
            step(seq[i%len(seq)])
        assert root.time_step_counter==300
    def run(kind,patched=True):
        if kind=='carry': warm(); reset(False,patched)
        else: reset(True); reset(False,patched)
        setrng()
        return [step(r) for r in seq]
    def comparison(a,b):
        report={}; passed=True
        for field in ['hidden','logits','probs']:
            x=torch.stack([r[field] for r in a]); y=torch.stack([r[field] for r in b]); delta=(x-y).abs()
            okay=torch.allclose(x,y,atol=ATOL,rtol=RTOL)
            report[field]={'max_abs':delta.max().item(),'max_rel':(delta/x.abs().clamp_min(1e-12)).max().item(),'within_tolerance':okay}; passed &= okay
        for field in ['sampled','greedy','history']:
            okay=all(torch.equal(x[field],y[field]) for x,y in zip(a,b)); report[field+'_exact']=okay; passed &= okay
        report['pass']=bool(passed); return report
    with torch.no_grad():
        critics=[root.critic_tsfm,root.c_critic_tsfm]
        critic_identity=[(x.time_step_counter,[(l.attention.cache_k.clone(),l.attention.cache_v.clone()) for l in x.decoder.layers]) for x in critics]
        aa_ref=run('carry',False); aa_off=run('carry',True)
        aa=comparison(aa_ref,aa_off)
        aa.update(status='PASS' if aa['pass'] else 'FAIL',atol=ATOL,rtol=RTOL,decisions=len(seq),reference='exact original reset + original root forward',patched='AST-extracted reset_fix.patch OFF + same original root forward',environment_path='patch changes only reset; OFF branch performs no operation before original reset',input_reconstruction_exact=True)
        dump(OUT/'aa_off_equivalence.json',aa)
        if not aa['pass']: raise RuntimeError('R4: A/A OFF failed')
        clean1=run('clean'); clean2=run('clean')
        carry1=run('carry'); carry2=run('carry')
        stable={'clean':comparison(clean1,clean2),'carry':comparison(carry1,carry2)}
        assert all(v['pass'] for v in stable.values()),'R4: Within-condition replay unstable'
        for branch,(counter,caches) in zip(critics,critic_identity):
            assert branch.time_step_counter==counter
            for layer,(k,v) in zip(branch.decoder.layers,caches):
                assert torch.equal(layer.attention.cache_k,k) and torch.equal(layer.attention.cache_v,v),'Critic state changed'
        names=__import__('utils.constants.stretch_initialization_utils',fromlist=['ALL_STRETCH_ACTIONS']).ALL_STRETCH_ACTIONS
        end_index=names.index('end')
        rows=[]; firstlog=firstarg=firstsample=None
        for t,(a,b) in enumerate(zip(clean1,carry1)):
            z=a['logits'].flatten(); w=b['logits'].flatten(); h=a['hidden'].flatten(); j=b['hidden'].flatten()
            if firstlog is None and not torch.allclose(z,w,atol=ATOL,rtol=RTOL): firstlog=t
            if firstarg is None and not torch.equal(a['greedy'],b['greedy']): firstarg=t
            if firstsample is None and not torch.equal(a['sampled'],b['sampled']): firstsample=t
            row={'local_timestep':t,'end_index':end_index,'hidden_max_abs':(h-j).abs().max().item(),'hidden_max_rel':((h-j).abs()/h.abs().clamp_min(1e-12)).max().item(),'logit_max_abs':(z-w).abs().max().item(),'logit_max_rel':((z-w).abs()/z.abs().clamp_min(1e-12)).max().item()}
            for tag,r in [('clean',a),('carry',b)]:
                probs=r['probs'].flatten(); logits=r['logits'].flatten()
                row.update({tag+'_counter_before':r['counter_before'],tag+'_counter_after':r['counter_after'],tag+'_start_pos':r['start_pos'],tag+'_logits':json.dumps(logits.tolist()),tag+'_probabilities':json.dumps(probs.tolist()),tag+'_policy_end_prob':probs[end_index].item(),tag+'_end_rank':int((probs>probs[end_index]).sum())+1,tag+'_end_vs_best_other_margin':(logits[end_index]-torch.cat([logits[:end_index],logits[end_index+1:]]).max()).item(),tag+'_argmax_action':int(r['greedy'].item()),tag+'_common_rng_sample':int(r['sampled'].item())})
            rows.append(row)
        with (OUT/'rollover_trace.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        rollover=next((t for t,r in enumerate(carry1) if r['counter_before']>=500 and r['start_pos']==0),None)
        assert rollover==200 and len(seq)>=220,'Rollover boundary not observed'
        label='R1' if firstlog is not None and (firstarg is not None or firstsample is not None) else ('R2' if firstlog is not None else 'R3')
        regions={}
        for name,subset in [('pre_rollover',rows[:200]),('boundary',rows[200:201]),('post_rollover',rows[201:])]:
            regions[name]={'n':len(subset),'max_abs_logit':max(r['logit_max_abs'] for r in subset),'max_abs_hidden':max(r['hidden_max_abs'] for r in subset)}
        result={'status':'COMPLETED','classification':label,'atol':ATOL,'rtol':RTOL,'decisions':len(seq),'first_logit_divergence_step':firstlog,'first_argmax_action_divergence_step':firstarg,'first_common_rng_sample_divergence_step':firstsample,'first_carry_rollover_step':rollover,'divergence_minus_rollover':None if firstlog is None else firstlog-rollover,'regions':regions,'within_condition_replay':stable,'critic_state_unchanged':True,'teacher_forced':True,'end_index':end_index,'action_names':names,'sampling_note':'same manual seed is restored per complete replay; no extra RNG calls in differing paths','warmup_decisions_per_carry':300,'warmup_wraps_if_needed':len(seq)<300,'scope':'Actor state -> distribution/action only. No SR or Safety Cost effect is established.'}
        dump(OUT/'rollover_stress_result.json',result)
        torch.save({'clean1':clean1,'clean2':clean2,'carry1':carry1,'carry2':carry2},RAW/'offline_replays.pt')
        manifest['result_class']=label
    for h in hooks: h.remove()

def main():
    manifest=None
    try:
        manifest,selected=prepare()
        execute(manifest,selected)
        manifest['status']='AWAITING_PI_REVIEW'
    except BaseException as e:
        traceback.print_exc()
        if manifest is None: raise
        manifest['status']='BLOCKED' if 'BLOCKED:' in str(e) else 'INVALID'
        manifest['error']=type(e).__name__+': '+str(e)
        manifest['result_class']='R4'
        (RAW/'traceback.txt').write_text(traceback.format_exc())
        for name in ['aa_off_equivalence.json','rollover_stress_result.json']:
            if not (OUT/name).exists(): dump(OUT/name,{'status':'NOT_COMPLETED','reason':manifest['error'],'classification':'R4','missing_not_started':True})
        if not (OUT/'rollover_trace.csv').exists(): (OUT/'rollover_trace.csv').write_text('local_timestep,clean_counter_before,carry_counter_before,logit_max_abs\n')
    finally:
        if manifest is not None:
            manifest['ended_utc']=now()
            manifest['official_snapshot_unchanged']=all(sha(RUNTIME/p)==v for p,v in manifest['source_frozen'].items())
            manifest['development_diff_unchanged']=hashlib.sha256(subprocess.check_output(['git','-C',str(DEV),'diff','--binary'])).hexdigest()==manifest['development_diff_sha256']
            dump(OUT/'RUN_MANIFEST.json',manifest)
            note('Executor finished: '+manifest['status'])

if __name__=='__main__': main()
