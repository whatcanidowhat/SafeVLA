"""CPU-only import/dependency provenance; never constructs policy or simulator."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU import check requires hidden CUDA devices'
    expected={'torch':'2.4.1+cu121','torchvision':'0.19.1+cu121','torchaudio':'2.4.1+cu121','numpy':'2.1.2','transformers':'4.46.1','ai2thor':'0+966bd7758586e05d18f6181f459c0e90ba318bec','jsonschema':'4.25.1'}
    versions={}
    for name,target in expected.items():
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual=None
        versions[name]={'expected':target,'actual':actual,'match':actual==target}
    imports={}
    for name in ['torch','torchvision','torchaudio','numpy','pandas','cv2','architecture.models.allenact_transformer_models.separate_actor_critic','training.online.online_eval']:
        try:
            module=importlib.import_module(name)
            imports[name]={'status':'PASS','path':getattr(module,'__file__',None),'version':getattr(module,'__version__',None)}
        except Exception as exc:imports[name]={'status':'FAIL','error':type(exc).__name__+': '+str(exc)}
    import torch
    initialized=torch.cuda.is_initialized()
    result={'status':'PASS' if all(v['match'] for v in versions.values()) and all(v['status']=='PASS' for v in imports.values()) and not initialized else 'BLOCKED',
            'python':sys.executable,'python_version':sys.version,'versions':versions,'imports':imports,
            'cuda_initialized':initialized,'model_constructed':False,'simulator_started':False,
            'pandas_legacy_version':'2.3.3','pandas_target_version':importlib.metadata.version('pandas'),
            'full_environment_equivalence':'NOT_ESTABLISHED; retain dependency resolver conflicts and Pandas difference for PI review'}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 2)


if __name__=='__main__':main()
