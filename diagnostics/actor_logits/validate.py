"""Integrity checks only. Passing this does not prove GPU logger equivalence."""
import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np

from .logger import COST_KEYS
from .storage import read_episode, read_json, read_jsonl, require, sha256, write_json


def zero_digest(info):
    remaining = math.prod(info['shape']) * np.dtype(info['storage_dtype']).itemsize
    h = hashlib.sha256()
    while remaining:
        count = min(remaining, 65536)
        h.update(bytes(count))
        remaining -= count
    return h.hexdigest()


def validate_run(directory, expected=200):
    root = Path(directory)
    problems, episodes, global_step = [], [], 0
    try:
        manifest = read_json(root / 'RUN_MANIFEST.json')
        require(manifest['status'] == 'COMPLETE', 'Run marked INCOMPLETE')
        require(manifest['provenance']['actor_reset'] in ('off', 'on'), 'Unknown treatment')
        require(manifest['episodes_expected'] == manifest['episodes_completed'] == len(manifest['tasks']) == expected, 'Expected/completed task count mismatch')
        require(len({t['sample_id'] for t in manifest['tasks']}) == expected, 'Duplicate task IDs')
        paths = sorted((root / 'episodes').iterdir())
        require(len(paths) == expected and all(p.name.endswith('.complete') for p in paths), 'Missing/partial/extra episodes')
        seen = set()
        for order, path in enumerate(paths):
            status, rows, values, metrics = read_episode(path)
            task = manifest['tasks'][order]
            require(status['episode_order'] == order and status['sample_id'] == task['sample_id'], 'Episode order/ID mismatch')
            require(status['decisions'] == len(rows) == metrics['official']['eps_len'] > 0, 'Missing decisions')
            acks = read_jsonl(path / 'execution_results.jsonl')
            require(acks == [{'step_index': i, 'returned': True} for i in range(len(rows))], 'Missing/duplicate execution acknowledgements')
            for step, (row, tensors) in enumerate(zip(rows, values)):
                require(row['sample_id'] == task['sample_id'] and row['task_source_index'] == task['task_source_index'] and row['raw_spec_sha256'] == task['raw_spec_sha256'], 'TaskSpec alignment mismatch')
                require(row['step_index'] == step and row['episode_order'] == order and row['global_step'] == global_step, 'Missing/duplicate/reordered row')
                require(row['decision_id'] == f"{order:04d}:{task['sample_id']}:{step:06d}" and row['decision_id'] not in seen, 'Duplicate/malformed decision ID')
                seen.add(row['decision_id'])
                global_step += 1
                require(set(row['calls']) == {'root_forward','reward_forward','cost_forward','raw_head','decoder','sample','mode'} and all(v == 1 for v in row['calls'].values()), 'Forward/sample count mismatch')
                require(set(tensors) == {'raw_actor_logits','normalized_logits','action_probs','sampled_action','greedy_mode_action','action_history','prev_action'}, 'Missing tensor fields')
                for key in ['raw_actor_logits', 'normalized_logits', 'action_probs']:
                    require(tensors[key].numel() == len(manifest['action_names']), 'Truncated action vector')
                    require(np.isfinite(tensors[key].double().numpy()).all(), 'Nonfinite policy vector')
                require(bool((tensors['action_probs'] >= 0).all()), 'Negative probability')
                require(int(tensors['sampled_action'].item()) == row['sampled_action_index'], 'Sample metadata mismatch')
                require(int(tensors['greedy_mode_action'].item()) == row['greedy_mode_action_index'], 'Mode metadata mismatch')
                require(all(0 <= row[k] < len(manifest['action_names']) for k in ['sampled_action_index','greedy_mode_action_index','executed_action_index']), 'Invalid action index')
                require(manifest['action_names'][row['executed_action_index']] == row['executed_action_name'], 'Executed action map mismatch')
                require(int(tensors['action_history'].item()) == row['sampled_action_index'], 'Sampled history mismatch')
                require(row['counter_after'] == row['decoder_start_pos'] + 1, 'Counter/decoder boundary mismatch')
            components = {k: sum(r['pre_step_cost'][k] for r in rows) for k in COST_KEYS}
            require(all(metrics['official'][k] == components[k] for k in COST_KEYS), 'Cost component mismatch')
            require(metrics['official']['cost'] == sum(components.values()), 'Official cost mismatch')
            require(metrics['summary']['success'] == (metrics['official']['success'] > 0.1), 'Success mismatch')
            require(metrics['summary']['cost'] == metrics['official']['cost'] and all(metrics['summary'][k] == components[k] for k in COST_KEYS), 'Summary mismatch')
            boundaries = read_jsonl(path / 'boundary_states.jsonl')
            require([b['event'] for b in boundaries] == ['before_original_reset', 'after_original_reset', 'after_treatment', 'episode_end'], 'Missing/duplicate boundaries')
            require(all(b['episode_order'] == order and b['sample_id'] == task['sample_id'] for b in boundaries), 'Boundary task alignment mismatch')
            before, after = boundaries[1]['state'], boundaries[2]['state']
            require(before['reward_critic'] == after['reward_critic'] and before['cost_critic'] == after['cost_critic'], 'Treatment changed critic temporal state')
            if manifest['provenance']['actor_reset'] == 'off':
                require(before == after, 'OFF treatment changed state')
            else:
                require(after['actor']['counter'] == 0, 'ON root counter not reset')
                require(len(before['actor']['layers']) == len(after['actor']['layers']), 'ON changed layer count')
                for old, new in zip(before['actor']['layers'], after['actor']['layers']):
                    for key in ['k', 'v']:
                        require(all(old[key][field] == new[key][field] for field in ['shape','dtype','device']), 'ON changed cache layout')
                        require(new[key]['sha256'] == zero_digest(new[key]), 'ON root K/V not zero')
            episodes.append(metrics['summary'])
        require(global_step == manifest['decisions'], 'Global decision count mismatch')
        with (root / 'episode_metrics.csv').open() as stream:
            table = list(csv.DictReader(stream))
        require(len(table) == expected and all(row == {k:str(v) for k,v in e.items()} for row,e in zip(table,episodes)), 'Metrics CSV alignment mismatch')
        index = read_json(root / 'ARTIFACT_INDEX.json')['artifacts']
        indexed = set()
        for item in index:
            relative = Path(item['path'])
            require(not relative.is_absolute() and '..' not in relative.parts and item['path'] not in indexed, 'Unsafe/duplicate artifact path')
            indexed.add(item['path'])
            p = root / relative
            require(not p.is_symlink() and p.stat().st_size == item['size'] and sha256(p) == item['sha256'], 'Artifact digest/size mismatch')
        actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'ARTIFACT_INDEX.json'}
        require(indexed == actual, 'Unindexed/missing artifact')
    except Exception as exc:
        problems.append(type(exc).__name__ + ': ' + str(exc))
    return {'status': 'INCOMPLETE' if problems else 'PASS', 'problems': problems,
            'episodes_checked': len(episodes), 'decisions_checked': global_step,
            'outcomes': None if problems else episodes,
            'scope': 'Stored evidence integrity; NOT GPU equivalence validation'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--expected', type=int, default=200)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    result = validate_run(args.directory, args.expected)
    write_json(args.report, result)
    return 0 if result['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
