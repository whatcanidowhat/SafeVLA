import json
from pathlib import Path
from types import SimpleNamespace
from manual_gate import authorize, ROOT, BASE, CONTROL
from diagnostics.actor_memory_unmask import online_gate, online_pair
from pair_entry import load_patched_pair

assert online_gate.authorize is authorize and online_pair.authorize is authorize, 'NVIDIA adapter not loaded'
assert load_patched_pair().authorize is authorize, 'Patched orchestrator lost the NVIDIA adapter'

evidence = authorize(SimpleNamespace(control_root=CONTROL), ROOT)
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expected one CUDA-visible GPU'
report = dict(status='PASS', model_constructions=0, policy_forwards=0,
              simulator_launches=0, episodes_started=0,
              cuda_device_count=torch.cuda.device_count(), cuda_device_name=torch.cuda.get_device_name(0),
              sequence=['MASK_ON200', 'MASK_OFF200'], evidence=evidence)
print(json.dumps(report, indent=2))
Path(__file__).with_name('last_preflight.json').write_text(json.dumps(report, indent=2) + '\n')
