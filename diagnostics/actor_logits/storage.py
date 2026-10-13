"""Lossless per-episode storage, atomic completion, and explicit partial evidence."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def json_value(value):
    """Encode NumPy values in JSON metadata without touching lossless tensor shards.

    NumPy arrays may appear in original evaluator metrics. JSON has no ndarray
    type, so retain their full nested values/shape as JSON lists. Scalar NumPy
    values retain the existing conversion; unsupported objects and NaN/Inf
    remain fatal via TypeError / allow_nan=False (no silent dropping).
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported JSON type: {type(value).__name__}')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, default=json_value, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_json(path, value):
    with Path(path).open('a', encoding='utf8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, default=json_value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines()]


def pack_tensor(tensor):
    """Preserve logical dtype/shape; BF16 has an explicit lossless uint16 encoding."""
    import torch
    t = tensor.detach().cpu().contiguous().clone()
    if t.dtype == torch.bfloat16:
        array, encoding = t.view(torch.uint16).numpy().copy(), 'bfloat16_bits'
    else:
        array, encoding = t.numpy().copy(), 'native_numpy'
    require(not array.dtype.hasobject, 'Object arrays are forbidden')
    spec = {'dtype': str(t.dtype), 'shape': list(t.shape), 'storage_dtype': array.dtype.str,
            'encoding': encoding, 'order': 'C', 'sha256': hashlib.sha256(array.tobytes()).hexdigest()}
    return array, spec


def unpack_tensor(array, spec):
    import torch
    require(list(array.shape) == spec['shape'], 'Tensor shape mismatch')
    require(array.dtype.str == spec['storage_dtype'], 'Tensor storage dtype mismatch')
    require(hashlib.sha256(array.tobytes()).hexdigest() == spec['sha256'], 'Tensor digest mismatch')
    t = torch.from_numpy(array.copy())
    if spec['encoding'] == 'bfloat16_bits':
        t = t.view(torch.bfloat16)
    else:
        require(spec['encoding'] == 'native_numpy', 'Unknown encoding')
    require(str(t.dtype) == spec['dtype'], 'Logical dtype mismatch')
    return t


def write_npz(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class EpisodeWriter:
    def __init__(self, run_dir, episode):
        self.base = Path(run_dir) / 'episodes'
        self.base.mkdir(exist_ok=True)
        self.order = episode['episode_order']
        self.path = self.base / f'{self.order:04d}.partial'
        require(not (self.base / f'{self.order:04d}.complete').exists(), 'Duplicate episode')
        self.path.mkdir()  # exclusive: no resume/overwrite
        self.episode = episode
        self.count = 0
        self.acks = 0
        for name in ['actions.jsonl', 'execution_results.jsonl', 'boundary_states.jsonl']:
            (self.path / name).touch(exist_ok=False)
        write_json(self.path / 'EPISODE_STATUS.json', {'status': 'INCOMPLETE', **episode})

    def append(self, record, tensors):
        require(record['step_index'] == self.count, 'Missing/duplicate/out-of-order decision')
        require(self.acks == self.count, 'Previous execution did not return')
        arrays, specs = {}, {}
        for name, tensor in tensors.items():
            arrays[name], specs[name] = pack_tensor(tensor)
        write_npz(self.path / f'step_{self.count:06d}.npz', arrays)
        append_json(self.path / 'actions.jsonl', {**record, 'tensors': specs})
        self.count += 1

    def acknowledge(self):
        require(self.count == self.acks + 1, 'Duplicate/missing execution acknowledgement')
        append_json(self.path / 'execution_results.jsonl', {'step_index': self.acks, 'returned': True})
        self.acks += 1

    def boundary(self, event):
        append_json(self.path / 'boundary_states.jsonl', event)

    def finish(self, metrics):
        require(self.count > 0 and self.acks == self.count, 'Incomplete execution')
        arrays = {}
        for i in range(self.count):
            with np.load(self.path / f'step_{i:06d}.npz', allow_pickle=False) as shard:
                arrays.update({f'{i:06d}.{k}': shard[k] for k in shard.files})
        write_npz(self.path / 'tensors.npz', arrays)
        write_json(self.path / 'metrics.json', metrics)
        # A partial directory is never considered complete, even if interrupted here.
        for i in range(self.count):
            (self.path / f'step_{i:06d}.npz').unlink()
        write_json(self.path / 'EPISODE_STATUS.json', {'status': 'COMPLETE', **self.episode, 'decisions': self.count})
        final = self.base / f'{self.order:04d}.complete'
        require(not final.exists(), 'Refusing to overwrite completed episode')
        os.replace(self.path, final)
        self.path = final
        return final


def read_episode(path):
    """No pickle. Partial shards remain inspectable but never count as complete."""
    path = Path(path)
    require(path.name.endswith('.complete'), 'INCOMPLETE episode directory')
    status = read_json(path / 'EPISODE_STATUS.json')
    require(status['status'] == 'COMPLETE', 'INCOMPLETE episode status')
    rows = read_jsonl(path / 'actions.jsonl')
    with np.load(path / 'tensors.npz', allow_pickle=False) as data:
        expected = {f"{i:06d}.{k}" for i, row in enumerate(rows) for k in row['tensors']}
        require(len(data.files) == len(expected) and set(data.files) == expected, 'Missing/extra/duplicate tensor keys')
        values = [{k: unpack_tensor(data[f'{i:06d}.{k}'], spec) for k, spec in row['tensors'].items()}
                  for i, row in enumerate(rows)]
    return status, rows, values, read_json(path / 'metrics.json')
