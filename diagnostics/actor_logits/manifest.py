"""CPU-only reconstruction of original loader selection and frozen CSV bytes."""
import ast
import csv
import gzip
import io
import json
import random
from pathlib import Path
from types import SimpleNamespace

from .storage import require, sha256

MANIFEST_SHA = '8ca476af1250460a3cd8a5101bb20e24b22c635358367b15bf1da7264d567bd0'
DATA_SHA = 'fa0dffc2dece071716dcf648e19c6944caf1621ee24dadc6996eadcd46eee04a'


def spec_hash(task):
    import hashlib
    return hashlib.sha256(json.dumps(task, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def regenerate(repo):
    """Execute the real loader's selection body with identity-only normalization.

    This does not import evaluator/model/simulator dependencies. The actual full
    normalization and TaskSpecs are checked separately by the runtime loader.
    Preserve caller's Python RNG despite the original selection's seed/shuffle.
    """
    repo = Path(repo)
    dataset = repo / 'benchmark/objectnavtype_val.jsonl.gz'
    require(sha256(dataset) == DATA_SHA, 'Dataset identity mismatch')
    raw = [json.loads(line) for line in gzip.decompress(dataset.read_bytes()).splitlines() if line]
    source = repo / 'online_evaluation/online_evaluator.py'
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'load_minival_eval_samples_per_task')
    namespace = {'random': random, 'inverse_map_task_type': lambda task_type: task_type,
                 'eval_sample_to_normalized_eval_sample': lambda task_type, sample, index: index}
    exec(compile('from __future__ import annotations\n' + ast.unparse(method), str(source), 'exec'), namespace)
    state = random.getstate()
    try:
        adapter = SimpleNamespace(shuffle=True, seed=123, eval_set_size=200,
                                  _load_dataset_from_local_path=lambda **kw: {'val': raw})
        indices = namespace[method.name](adapter, 'ObjectNavType', use_local_path=str(dataset))
    finally:
        random.setstate(state)
    require(len(raw) == len(indices) == len(set(indices)) == 200, 'Incomplete original selection')
    rows, tasks = [], []
    for order, i in enumerate(indices):
        task = raw[i]
        sid = f"task=ObjectNavType,house={task['house_index']},sub_house_id={i}"
        digest = spec_hash(task)
        rows.append(dict(ordinal_0based=order, original_row_index=i, sample_id=sid,
                         house_index=task['house_index'], task_path=task['task_path'],
                         raw_spec_sha256=digest, p1_selected=order < 5, worker_id=0, seed=123))
        tasks.append(dict(sample_id=sid, task_source_index=i, raw_spec_sha256=digest))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    content = stream.getvalue().encode()
    import hashlib
    require(hashlib.sha256(content).hexdigest() == MANIFEST_SHA, 'Frozen manifest mismatch')
    return tasks, content


def check_actual_loader(samples, tasks):
    require(len(samples) == len(tasks) == 200, 'Actual loader count mismatch')
    for sample, task in zip(samples, tasks):
        require(sample['sample_id'] == task['sample_id'] and sample['sub_house_id'] == task['task_source_index'], 'Actual loader order/identity mismatch')
        require(spec_hash(json.loads(sample['observations']['templated_task_type'])) == task['raw_spec_sha256'], 'Actual loader TaskSpec mismatch')
