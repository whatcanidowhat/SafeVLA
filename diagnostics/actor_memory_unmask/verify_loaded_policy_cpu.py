"""Load original evaluator's full CPU policy, inspect500-slot state, NO forward."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
from .baseline_contract import assert_runtime_baseline


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not a.output.exists()
    assert digest(a.checkpoint)=='05b3f7f4db356a24999cd2177b59634b4c9d8f0a4f581af613dcadc5fec6a301'
    torch.set_num_threads(1)
    from training.online import online_eval as entry
    key='InferenceDINOv2ViTSLLAMATxTxBaseDist'
    params=entry.model_config_params[key]()
    # Exact original online_eval.main adjustments for the registered sensors.
    params.num_train_processes=0
    params.use_bbox=False
    # Deliberately DO NOT assign params.max_steps or alter model/cache config.
    assert params.max_steps==500
    loads=[];calls=[]
    def profile(frame,event,result):
        if event=='return' and frame.f_code.co_name=='load_state_dict' and frame.f_code.co_filename.endswith('/torch/nn/modules/module.py') and hasattr(result,'missing_keys'):
            loads.append({'class':type(frame.f_locals['self']).__name__,'strict':frame.f_locals['strict'],
                          'nkeys':len(frame.f_locals['state_dict']),'missing':list(result.missing_keys),'unexpected':list(result.unexpected_keys)})
    def forbid_policy_forward(module,args):
        if type(module).__name__ in ['DinoLLAMATxNavActorCritic','DinoLLAMATxNavActorCriticSeparate','SafeDinoLLAMATxNavActorCriticSeparate']:
            calls.append(type(module).__name__)
            raise AssertionError('CPU load verification must not forward the policy')
    hook=torch.nn.modules.module.register_module_forward_pre_hook(forbid_policy_forward)
    assert sys.getprofile() is None
    try:
        sys.setprofile(profile)
        agent=entry.InferenceAgentVIDA.build_agent(
            exp_config_type=entry.model_config_type[key],params=params,device=torch.device('cpu'),
            img_encoder_rgb_mean=entry.DINO_RGB_MEANS,img_encoder_rgb_std=entry.DINO_RGB_STDS,
            greedy_sampling=False,test_augmentation=True,ckpt_path=str(a.checkpoint))
    finally:
        sys.setprofile(None);hook.remove()
    assert not calls and not torch.cuda.is_initialized()
    assert any(x['class']=='SafeDinoLLAMATxNavActorCriticSeparate' and x['nkeys']==417 and not x['missing'] and not x['unexpected'] for x in loads),'Full policy load not417/417'
    assert all(not x['missing'] and not x['unexpected'] for x in loads)
    baseline=assert_runtime_baseline(agent.actor_critic,Path(__file__).resolve().parents[2])
    result={'status':'PASS','scope':'Actual full original evaluator policy loaded on CPU, no policy forward/simulator/Episode/GPU allocation',
            'actual_model_class':type(agent.actor_critic).__name__,'runtime_baseline':baseline,
            'load_events':loads,'policy_forwards':len(calls),'cuda_initialized':torch.cuda.is_initialized(),
            'python':sys.executable,'torch_version':torch.__version__,'parameter_max_steps_not_overridden':True}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
