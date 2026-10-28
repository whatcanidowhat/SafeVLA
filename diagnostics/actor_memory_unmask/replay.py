"""Independent CPU checkpoint Actor-component replay, no online policy."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
import torch
from diagnostics.actor_memory_unmask.cpu_model import make_model, fixed_input, cache, ROOT


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def js(a,b):
    p,q=a.double().softmax(-1),b.double().softmax(-1)
    m=(p+q)/2
    return float(((p*(p.log()-m.log())).sum()+(q*(q.log()-m.log())).sum())/2)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(123)
    digest=sha(a.checkpoint)
    assert digest=='05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301'
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    state=checkpoint.get('model_state_dict',checkpoint.get('state_dict',checkpoint))
    state={k.removeprefix('model.'):v for k,v in state.items()}
    model=make_model(dim=512,layers=3,heads=8,max_steps=1000)
    loads=[]
    for prefix,module in [('decoder.',model.decoder),('actor.',model.actor),('last_actions_embed.',model.last_actions_embed),('time_encoder.',model.time_encoder)]:
        subset={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        if prefix=='actor.' and 'weight' in subset:
            subset={'linear.'+k:v for k,v in subset.items()}
        result=module.load_state_dict(subset,strict=True)
        loads.append(dict(prefix=prefix,nkeys=len(subset),strict=True,missing=result.missing_keys,unexpected=result.unexpected_keys))
    del checkpoint,state
    embeddings=torch.randn(5,1,512)
    torch.save({'embeddings':embeddings,'seed':123,'timesteps':[0,1,2,3,0],
                'episodes':[0,0,0,0,1], 'previous_actions':[3]*5,
                'masks':[0,1,1,1,0], 'origin':'synthetic post-perception; generated after fixture initialization'},a.output/'fixed_inputs.pt')
    calls=[]
    with torch.no_grad():
        for i in range(4):
            model(**fixed_input(embeddings[i:i+1],i,0))
    torch.save({'cache':cache(model),'counter':model.time_step_counter,'origin':'4 genuine warmup forwards from checkpoint temporal weights'},a.output/'warm_cache.pt')
    # Fresh independent copy for every intervention. Only old tokens are mutated.
    variants={}
    for enabled in [True,False]:
        for mutation in ['none','keys','values']:
            m=copy.deepcopy(model)
            m.actor_cross_episode_attention_mask=enabled
            attn=m.decoder.layers[-1].attention
            with torch.no_grad():
                if mutation=='keys': attn.cache_k[:,:4] += 7
                if mutation=='values': attn.cache_v[:,:4] += 7
            before=cache(m); ids=[(id(l.attention.cache_k),id(l.attention.cache_v)) for l in m.decoder.layers]
            seen=[]; handles=[]
            for l in m.decoder.layers:
                # TransformerBlock calls attention.forward directly, bypassing
                # Attention __call__ hooks. Block hooks see the effective mask.
                handles.append(l.register_forward_pre_hook(lambda _,args:seen.append(dict(start_pos=args[1],mask=None if args[2] is None else args[2].tolist()))))
            rng=torch.get_rng_state().clone()
            with torch.no_grad(): out=m(**fixed_input(embeddings[4:5],0,1))[0]
            for h in handles:h.remove()
            logits=m.actor.raw.detach().reshape(-1)
            for (ka,va),(kb,vb) in zip(before,cache(m)):
                assert torch.equal(ka[:,:4],kb[:,:4]) and torch.equal(va[:,:4],vb[:,:4])
            assert ids==[(id(l.attention.cache_k),id(l.attention.cache_v)) for l in m.decoder.layers]
            assert m.time_step_counter==5 and torch.equal(rng,torch.get_rng_state())
            assert len(seen)==3 and all(x['start_pos']==4 for x in seen)
            label=('mask_on' if enabled else 'mask_off')+'_'+mutation
            variants[label]={'logits':logits.tolist(),'p_end':float(logits.double().softmax(-1)[4]),
                             'counter_before':4,'counter_after':5,'attention_calls':len(seen),'attention_inputs':seen,
                             'history_unchanged_during_forward':True,'cache_objects_preserved':True,'rng_unchanged':True}
    contrasts=[]
    pairs=[('mask_on_none','mask_off_none')]+[(f'mask_{mode}_none',f'mask_{mode}_{mutation}') for mode in ['on','off'] for mutation in ['keys','values']]
    for x,y in pairs:
        lx,ly=torch.tensor(variants[x]['logits']),torch.tensor(variants[y]['logits'])
        contrasts.append(dict(a=x,b=y,max_abs_logit=float((lx-ly).abs().max()),js_nats=js(lx,ly),p_end_delta=variants[y]['p_end']-variants[x]['p_end']))
    assert all(c['max_abs_logit']==0 for c in contrasts if c['a']=='mask_on_none' and c['b'].startswith('mask_on_'))
    assert all(c['max_abs_logit']>1e-6 and c['js_nats']>0 for c in contrasts if c['a']=='mask_off_none')
    result=dict(status='PASS',scope='checkpoint Actor component, source-extracted forward, fixed synthetic post-perception input; NOT full policy/online',
                checkpoint_sha256=digest,strict_submodule_loads=loads,variants=variants,contrasts=contrasts,
                live_episodes=0,gpu_count=0,warmup_forwards=4,independent_boundary_forwards=6,
                device='cpu',torch_version=torch.__version__,python=sys.executable,host=socket.gethostname(),
                utc=datetime.now(timezone.utc).isoformat(),command=sys.argv,
                module_paths={'fixture':str(Path(__file__).with_name('cpu_model.py')),
                              'decoder':str(ROOT/'training/online/third_party_models/llama/model.py')},
                inputs_sha256=sha(a.output/'fixed_inputs.pt'),warm_cache_sha256=sha(a.output/'warm_cache.pt'))
    (a.output/'REPLAY_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'strict_submodule_loads':loads,'contrasts':contrasts},indent=2))


if __name__=='__main__':main()
