"""Check frozen source/dependencies/runtime paths; declare the NVIDIA renderer.

No model imports, policy forwards, simulator launches, or formal Claim/CI writes.
"""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone

BASE = Path('/public/home/zhangyh/QYY')
HERE = BASE / 'unmask_nvidia_20261103'
ROOT = BASE / 'SafeVLA_actor_memory_unmask_002_20261029'
CONTROL = BASE / 'SafeVLA_research_control_20261028'
PYTHON = BASE / 'envs/safevla_unmask500_20261028/bin/python'
SOURCE_SHA = '2689081ec324c67a35f2a4692cfa009826fcf9d4'


def check(value, message):
    if not value:
        raise RuntimeError('UNMASK_NVIDIA_PREFLIGHT: ' + message)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def command(*args, cwd=None, timeout=45):
    return subprocess.check_output(args, cwd=cwd, text=True, timeout=timeout).strip()


def allocation():
    """Find the real Sothis allocation by the current container ID, not fake Slurm env."""
    host = socket.gethostname()
    for path in Path('/public/home/zhangyh/SothisAI/instance/ssh').glob('*/_dockerlist_*'):
        for line in path.read_text().splitlines():
            fields = dict(x.split('=', 1) for x in line.split(',') if '=' in x)
            container = fields.get('ID', '')
            if container and container.startswith(host) and fields.get('STATUS') == 'running':
                check(fields.get('NODE'), 'allocation node missing')
                ids = fields.get('GPUIDS', '').split(':')
                check(all(x.isdigit() for x in ids), 'malformed allocated host GPU list')
                return path.name.removeprefix('_dockerlist_'), fields, ids
    raise RuntimeError('UNMASK_NVIDIA_PREFLIGHT: no running Sothis allocation for this container')


def authorize(args, repo):
    check(Path(repo).resolve() == ROOT, 'wrong experiment source')
    check(Path(sys.executable) == PYTHON, 'use frozen unmask500 Python')
    check(command('git', 'rev-parse', 'HEAD', cwd=ROOT) == SOURCE_SHA, 'source SHA changed')
    check(not command('git', 'status', '--porcelain', cwd=ROOT), 'source tree dirty')
    check(Path(args.control_root).resolve() == CONTROL, 'unexpected control reference')
    design = json.loads((CONTROL / 'research/NEXT_EXPERIMENT.json').read_text())
    check(design['code_sha'] == SOURCE_SHA, 'design/source pin mismatch')
    expected = dict(max_steps=500, kv_cache_capacity=500, episode_horizon=600,
                    actor_reset='off', seed=123, num_workers=1, greedy=False,
                    test_augmentation=True, shuffle=True, TaskSpecs=200)
    check(all(design['fixed_conditions'].get(k) == v for k, v in expected.items()),
          'scientific conditions changed')
    for name, version in design['target_dependency_versions'].items():
        check(metadata.version(name) == version, 'dependency mismatch: ' + name)
    manifest = json.loads((HERE / 'renderer_manifest.json').read_text())
    for path, sha in manifest['files'].items():
        check(digest(path) == sha, 'renderer/integration file drift: ' + path)
    for name, value in manifest['environment'].items():
        check(os.environ.get(name) == value, 'renderer environment mismatch: ' + name)
    for name in ('DISPLAY', 'VK_DRIVER_FILES', 'SDL_VIDEODRIVER', 'QT_QPA_PLATFORM'):
        check(not os.environ.get(name), 'unexpected graphics override: ' + name)
    # The user disabled full asset-content auditing. Validate only the fixed
    # resource roots and two input files, without opening the asset inventory or
    # visiting/hashing its 15,237 individual resources in any of the three gates.
    resource_roots = {
        'OBJAVERSE_HOUSES_DIR': BASE / 'data/objaverse_houses/houses_2023_07_28',
        'OBJAVERSE_DATA_DIR': BASE / 'data/objaverse_assets',
        'NLTK_DATA': BASE / 'data/nltk_data',
    }
    for name, expected_path in resource_roots.items():
        check(os.environ.get(name) == str(expected_path), 'resource path mismatch: ' + name)
        check(expected_path.is_dir(), 'resource directory missing: ' + str(expected_path))
    check((resource_roots['OBJAVERSE_HOUSES_DIR'] / 'val.jsonl.gz').is_file(), 'val houses missing')
    check((ROOT / 'benchmark/objectnavtype_val.jsonl.gz').is_file(), 'navigation tasks missing')
    print('Resource paths OK; per-file asset hash scan disabled.', flush=True)
    job_id, record, physical_ids = allocation()
    index = os.environ['UNMASK_GPU_INDEX']
    check(index.isdigit() and int(index) < len(physical_ids), 'GPU index outside allocation')
    query = command('nvidia-smi', '-i', index,
                    '--query-gpu=uuid,driver_version,memory.used,memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits').split(',')
    uuid, driver, used, free, util = (x.strip() for x in query)
    check(driver == '535.104.05', 'NVIDIA bundle requires driver 535.104.05')
    check(uuid == os.environ['UNMASK_GPU_UUID'] == os.environ['CUDA_VISIBLE_DEVICES'],
          'CUDA must select exactly the allocated GPU UUID')
    check(int(used) <= 4000 and int(free) >= 6824 and int(util) <= 30, 'selected GPU busy/low memory')
    check(not command('nvidia-smi', '-i', index, '--query-compute-apps=pid',
                      '--format=csv,noheader,nounits'), 'selected GPU has compute processes')
    vulkan = command('vulkaninfo', timeout=60)
    devices = {}
    current = None
    for line in vulkan.splitlines():
        m = re.match(r'GPU(\d+):', line)
        if m:
            current = int(m.group(1))
        elif 'deviceUUID' in line and current is not None:
            devices[line.split('=', 1)[1].strip()] = current
    check(uuid.removeprefix('GPU-') in devices, 'selected CUDA GPU absent from Vulkan')
    vulkan_index = devices[uuid.removeprefix('GPU-')]
    # AI2-THOR receives CUDA logical index 0; pin it to its verified Vulkan index.
    cache = Path(os.environ['HOME']) / '.ai2thor/cuda-vulkan-mapping.json'
    cache.write_text(json.dumps({'0': vulkan_index}) + '\n')
    deadline = os.environ['UNMASK_NVIDIA_DEADLINE_UTC']
    check(datetime.fromisoformat(deadline.replace('Z', '+00:00')) > datetime.now(timezone.utc),
          'manual wall-time deadline expired')
    return dict(mode='MANUAL_OFFLINE_NVIDIA', formal_control_authorized=False,
                github_ci_verified=False, claim_id=None, approval_commit=None, claim_commit=None,
                approval_expires_at_utc=deadline, deadline_field_role='MANUAL_WALLTIME_ONLY',
                physical_gpu=int(physical_ids[int(index)]), container_nvml_index=int(index),
                gpu_uuid=uuid, cuda_logical_index=0, vulkan_index=vulkan_index,
                slurm_job_id=job_id, slurm_node=record['NODE'], container_id=record['ID'],
                cuda_visible_devices=uuid, renderer='NVIDIA 535.104.05',
                renderer_changed_from_legacy_mesa=True,
                asset_validation={
                    'mode': 'PATHS_ONLY', 'full_asset_hash_scan': False,
                    'reason': 'disabled_by_user',
                    'resource_roots': {k: str(v) for k, v in resource_roots.items()},
                },
                orchestration_patch=manifest.get('orchestration_patch'),
                renderer_manifest_sha256=digest(HERE / 'renderer_manifest.json'),
                rendering_environment={k: os.environ.get(k) for k in
                    ['HOME', 'VK_ICD_FILENAMES', 'LD_LIBRARY_PATH', 'XDG_DATA_DIRS',
                     'XDG_RUNTIME_DIR', '__EGL_VENDOR_LIBRARY_FILENAMES']})
