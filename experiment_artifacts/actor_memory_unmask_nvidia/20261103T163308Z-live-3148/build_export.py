"""Archive an already completed run. Does not import policy/simulator packages."""
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/public/home/zhangyh/QYY')
RUN_ID = '20261103T163308Z-live-3148'
RUN = BASE / 'runs/EXP-ACTOR-MEMORY-UNMASK-NVIDIA' / RUN_ID
SOURCE = BASE / 'SafeVLA_actor_memory_unmask_002_20261029'
OUT = BASE / 'migration_exports/github-unmask-20261106'
REL = Path('experiment_artifacts/actor_memory_unmask_nvidia') / RUN_ID
CAP = 40 * 1024**2


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def copy_exact(source, target):
    assert source.is_file() and not source.is_symlink(), str(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def archive_group(group, files, compressed):
    chunks = []
    chunk, size = [], 0
    for path in files:
        assert not path.is_symlink(), str(path)
        n = path.stat().st_size
        assert n < CAP, str(path)
        if chunk and size + n > CAP:
            chunks.append(chunk)
            chunk, size = [], 0
        chunk.append(path)
        size += n
    if chunk:
        chunks.append(chunk)
    for number, chunk in enumerate(chunks, 1):
        relative = REL / 'archives' / f'{group}.part{number:02d}.zip'
        destination = OUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
        with zipfile.ZipFile(destination, 'w', compression=compression, compresslevel=6) as z:
            for path in chunk:
                member = path.relative_to(RUN).as_posix()
                h = hashlib.sha256()
                info = zipfile.ZipInfo.from_file(path, arcname=member)
                info.compress_type = compression
                with path.open('rb') as src, z.open(info, 'w') as dst:
                    for block in iter(lambda: src.read(1024**2), b''):
                        h.update(block)
                        dst.write(block)
                artifact_index.append({'path': member, 'bytes': path.stat().st_size,
                                       'sha256': h.hexdigest(), 'archive': str(relative.relative_to(REL)),
                                       'archive_member': member})
        assert destination.stat().st_size < 45 * 1024**2
        volumes.append({'path': str(relative.relative_to(REL)), 'bytes': destination.stat().st_size,
                        'sha256': digest(destination), 'members': len(chunk), 'group': group})
        print(json.dumps({'archive_ready': destination.name, 'bytes': destination.stat().st_size}), flush=True)


status = json.loads((RUN / 'PAIR_STATUS.json').read_text())
assert status.get('status') == 'COMPLETE' and status.get('episodes') == 400
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
assert head == '2689081ec324c67a35f2a4692cfa009826fcf9d4'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE, text=True).strip()
assert shutil.disk_usage(BASE).free > 2 * 1024**3
OUT.mkdir(parents=True, exist_ok=False)
package = OUT / REL
package.mkdir(parents=True)
artifact_index, volumes, group_summaries = [], [], {}

for path in sorted(RUN.iterdir()):
    if path.is_file():
        copy_exact(path, package / path.name)
        artifact_index.append({'path': path.name, 'bytes': path.stat().st_size,
                               'sha256': digest(path), 'published_file': path.name})

for group in ['MASK_ON200', 'MASK_OFF200']:
    manifest = json.loads((RUN / group / 'RUN_MANIFEST.json').read_text())
    assert manifest['status'] == 'COMPLETE' and manifest['episodes_completed'] == 200
    assert manifest['provenance']['git_commit'] == head
    for name, expected in manifest['provenance']['source_sha256'].items():
        assert digest(SOURCE / name) == expected, name
    rows = []
    for episode in sorted((RUN / group / 'episodes').glob('*.complete')):
        s = json.loads((episode / 'EPISODE_STATUS.json').read_text())
        m = json.loads((episode / 'metrics.json').read_text())['official']
        assert s['status'] == 'COMPLETE'
        costs = {key: int(m[key]) for key in ['danger', 'corner', 'blind', 'fragile', 'critical']}
        assert sum(costs.values()) == m['cost']
        rows.append({'episode_order': s['episode_order'], 'sample_id': s['sample_id'],
                     'success': m['success'] > .1, 'steps': m['eps_len'], 'cost': m['cost'], **costs})
    assert len(rows) == 200
    csv_path = package / group / 'episode_metrics.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    group_summaries[group] = {
        'episodes': len(rows), 'successes': sum(r['success'] for r in rows),
        'success_rate': sum(r['success'] for r in rows) / len(rows),
        'cost_sum': sum(r['cost'] for r in rows), 'mean_cost': sum(r['cost'] for r in rows) / len(rows),
        'decisions': sum(r['steps'] for r in rows),
        'components': {key: sum(r[key] for r in rows) for key in ['danger', 'corner', 'blind', 'fragile', 'critical']},
    }
    for path in sorted((RUN / group).iterdir()):
        if path.is_file():
            copy_exact(path, package / group / path.name)
    raw_files = sorted(p for p in (RUN / group).rglob('*') if p.is_file())
    archive_group(group + '.raw', raw_files, compressed=True)
    media_files = sorted(p for p in (RUN / (group + '_official')).rglob('*') if p.is_file())
    assert len([p for p in media_files if p.suffix == '.mp4']) == 200
    assert len([p for p in media_files if p.suffix == '.png']) == 200
    archive_group(group + '.media', media_files, compressed=False)

assert group_summaries['MASK_ON200']['cost_sum'] == 107
assert group_summaries['MASK_OFF200']['cost_sum'] == 1076
assert group_summaries['MASK_ON200']['decisions'] == 17754
assert group_summaries['MASK_OFF200']['decisions'] == 29124
dump(package / 'SUMMARY.json', {'run_id': RUN_ID, 'status': 'COMPLETE', 'groups': group_summaries,
                              'scope': 'Stored outcomes; descriptive comparison, not controlled causal attribution.'})
dump(package / 'ORIGINAL_ARTIFACT_INDEX.json', {'source_root': str(RUN), 'files': artifact_index})
dump(package / 'ARCHIVE_VOLUMES.json', {'format': 'independent ZIP volumes; extract all into the same empty directory', 'volumes': volumes})
dump(package / 'SOURCE_IDENTITY.json', {
    'runtime_source_root': str(SOURCE), 'git_commit': head, 'git_status_at_export': 'clean',
    'repository_source_ref': 'codex/unmask-002-batch-20261029',
    'root_source_is_retained_at_parent_commit': True,
    'policy_checkpoint': '/public/home/zhangyh/QYY/data/checkpoints/safe_objnav.pt',
    'policy_sha256': manifest['provenance']['checkpoint_sha256'],
    'dino_sha256': manifest['provenance']['dino_sha256'],
    'actor_boundary_reset': 'off in both arms',
    'root_actor_attention_mask': {'MASK_ON200': True, 'MASK_OFF200': False},
    'effective_cache_capacity': 500, 'episode_horizon': 600,
    'source_hashes': manifest['provenance']['source_sha256'],
})
packages = sorted([{'name': d.metadata['Name'], 'version': d.version} for d in metadata.distributions()], key=lambda d: d['name'].lower())
dump(package / 'ENVIRONMENT_OBSERVED_AT_EXPORT.json', {
    'observed_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Current evaluation Python installation; not a fabricated historical environment dump.',
    'python': sys.executable, 'python_version': sys.version, 'platform': platform.platform(), 'packages': packages,
})

adapter = BASE / 'unmask_nvidia_20261103'
for name in ['environment.sh', 'manual_gate.py', 'output_scan.py', 'pair_entry.py', 'preflight.py',
             'run_pair.sh', 'run_pair_live.sh', 'sitecustomize.py', 'test_output_scan.py', 'README.md',
             'renderer_manifest.json', 'last_preflight.json', 'check_after_removing_asset_hashes.sh',
             'validate_output_scan_fix.sh', 'output_scan_fix_validation.log']:
    copy_exact(adapter / name, OUT / 'deployment/unmask_nvidia_20261103' / name)
vulkan = BASE / 'vulkan_repair_20261031'
for name in ['activate.sh', 'run_objectnav.sh', 'README.md', 'NVIDIA-Linux-x86_64-535.104.05.run.sha256sum',
             'share/vulkan/icd.d/nvidia_icd.json', 'share/glvnd/egl_vendor.d/10_nvidia.json']:
    if (vulkan / name).is_file():
        copy_exact(vulkan / name, OUT / 'deployment/vulkan_repair_20261031' / name)

dump(OUT / 'SERVER_EXPORT_READY.json', {
    'status': 'READY', 'run_id': RUN_ID, 'source_commit': head, 'archives': len(volumes),
    'original_file_count': len(artifact_index), 'original_bytes': sum(x['bytes'] for x in artifact_index),
    'archive_bytes': sum(x['bytes'] for x in volumes), 'source_and_deployment_files': True,
    'not_packaged': ['policy/DINO weights', 'Conda environments', 'Objaverse datasets/assets', 'NVIDIA driver binaries'],
    'source_root': str(RUN), 'export_root': str(OUT),
})
print(json.dumps({'export_ready': str(OUT), 'archives': len(volumes), 'original_files': len(artifact_index),
                  'archive_bytes': sum(x['bytes'] for x in volumes)}), flush=True)
