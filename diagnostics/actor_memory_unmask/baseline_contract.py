"""Read-only effective evaluation configuration and model/cache assertions."""
import ast
import hashlib
from pathlib import Path


def source_baseline(repo):
    repo=Path(repo)
    paths={
        'params':'training/online/base.py',
        'factory':'training/online/dinov2_vits_tsfm_base.py',
        'horizon':'online_evaluation/max_episode_configs.py',
        'model':'architecture/models/allenact_transformer_models/allenact_dino_transformer.py'}
    trees={k:ast.parse((repo/p).read_text()) for k,p in paths.items()}
    params=next(n for n in trees['params'].body if isinstance(n,ast.ClassDef) and n.name=='BaseConfigParams')
    max_steps=next(ast.literal_eval(n.value) for n in params.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='max_steps')
    horizon=next(ast.literal_eval(n.value)['ObjectNavType'] for n in trees['horizon'].body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MAX_EPISODE_LEN_PER_TASK' for t in n.targets))
    factory=next(n for n in ast.walk(trees['factory']) if isinstance(n,ast.FunctionDef) and n.name=='create_model')
    call=next(n for n in ast.walk(factory) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='model_init_func')
    kwargs={k.arg:k.value for k in call.keywords}
    assert ast.unparse(kwargs['max_steps'])=='self.params.max_steps','Model max_steps binding changed'
    assert isinstance(kwargs['initial_tgt_cache_shape'],ast.Tuple)
    assert ast.unparse(kwargs['initial_tgt_cache_shape'].elts[0])=='self.params.max_steps','Cache capacity binding changed'
    assert max_steps==500 and horizon==600,'Effective Baseline differs; never repair it to satisfy experiment'
    model=next(n for n in trees['model'].body if isinstance(n,ast.ClassDef) and n.name=='DinoLLAMATxNavActorCritic')
    init=next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    defaults=dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):],init.args.defaults))
    return {'max_steps':max_steps,'kv_cache_capacity':max_steps,'episode_horizon':horizon,
            'model_class_default_not_effective_config':ast.literal_eval(defaults['max_steps']),
            'model_binding':ast.unparse(kwargs['max_steps']),
            'cache_capacity_binding':ast.unparse(kwargs['initial_tgt_cache_shape'].elts[0]),
            'sources':{p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths.values()}}


def assert_branch(branch,name,expected=500):
    assert branch.max_steps==expected,f'{name}.max_steps={branch.max_steps}, Baseline={expected}'
    assert branch.decoder.params.max_seq_len==expected,f'{name} decoder capacity differs'
    assert branch.time_encoder.max_len==expected,f'{name} positional encoder configuration differs'
    layers=[]
    for i,layer in enumerate(branch.decoder.layers):
        attention=layer.attention
        kshape=tuple(attention.cache_k.shape);vshape=tuple(attention.cache_v.shape)
        assert len(kshape)==len(vshape)==4,f'{name} layer{i} invalid cache rank'
        assert kshape==vshape and kshape[1]==expected,f'{name} layer{i} K/V capacity mismatch: {kshape}, {vshape}'
        layers.append({'layer':i,'cache_k_shape':list(kshape),'cache_v_shape':list(vshape)})
    assert layers,f'{name} decoder has no layers'
    return {'max_steps':branch.max_steps,'decoder_max_seq_len':branch.decoder.params.max_seq_len,
            'time_encoder_max_len':branch.time_encoder.max_len,'counter':branch.time_step_counter,'layers':layers}


def assert_runtime_baseline(root,repo):
    baseline=source_baseline(repo)
    branches={name:assert_branch(branch,name,baseline['max_steps']) for name,branch in [
        ('actor',root),('reward_critic',root.critic_tsfm),('cost_critic',root.c_critic_tsfm)]}
    # Inspect metadata only. No forward, reset, tensor write, resize or RNG call.
    return {'source_baseline':baseline,'loaded_branches':branches,'read_only':True}
