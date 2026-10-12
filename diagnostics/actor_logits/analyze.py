"""Task-aligned descriptive OFF/ON comparison, never a causal replay claim."""
import argparse
from pathlib import Path
import numpy as np

from .logger import COST_KEYS
from .storage import read_episode, read_json, require, write_json
from .validate import validate_run


def entropy(p):
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    return float(-np.sum(p[p > 0] * np.log(p[p > 0])))


def jsd(p, q):
    p, q = np.asarray(p, dtype=np.float64).reshape(-1), np.asarray(q, dtype=np.float64).reshape(-1)
    m = (p + q) / 2
    def kl(x):
        take = x > 0
        return np.sum(x[take] * np.log(x[take] / m[take]))
    return float((kl(p) + kl(q)) / 2)


def repetition(rows):
    actions = [r['executed_action_name'] for r in rows]
    repeat = sum(a == b for a, b in zip(actions, actions[1:]))
    oscillations = sum(actions[i] == actions[i-2] != actions[i-1] for i in range(2, len(actions)))
    corner_repeat = sum(rows[i]['pre_step_cost']['corner'] > 0 and actions[i] == actions[i-1] for i in range(1, len(actions)))
    return {'adjacent_repeats': repeat, 'aba_oscillations': oscillations, 'pre_step_corner_repeats': corner_repeat,
            'repeat_fraction': repeat / max(1, len(actions)-1), 'scope': 'Descriptive proxies, not diagnosed dead corners'}


def compare(off, on, expected=200):
    left, right = validate_run(off, expected), validate_run(on, expected)
    result = {'status': 'INCOMPLETE', 'delta_SR': None, 'delta_mean_cost': None,
              'scope': 'Online same task/step comparison is descriptive. After action divergence observations may differ; only separately authorized identical-input/state offline controls support direct logits attribution.'}
    if left['status'] != 'PASS' or right['status'] != 'PASS':
        return {**result, 'validation': {'off': left, 'on': right}}
    off, on = Path(off), Path(on)
    lm, rm = read_json(off/'RUN_MANIFEST.json'), read_json(on/'RUN_MANIFEST.json')
    if lm['tasks'] != rm['tasks'] or lm['action_names'] != rm['action_names']:
        return {**result, 'error': 'TaskSpecs/order/action map mismatch'}
    identity_keys = ['git_commit','source_sha256','checkpoint_sha256','dino_sha256','manifest_sha256',
                     'seed','num_workers','shuffle','greedy','test_augmentation','physical_gpu']
    if any(k not in lm['provenance'] or k not in rm['provenance'] or
           lm['provenance'][k] != rm['provenance'][k] for k in identity_keys):
        return {**result, 'error': 'Runtime provenance/configuration mismatch or missing identity'}
    if lm['provenance']['actor_reset'] != 'off' or rm['provenance']['actor_reset'] != 'on':
        return {**result, 'error': 'Expected OFF then ON runs'}
    task_results = []
    end_index = lm['action_names'].index('end')
    for lp, rp in zip(sorted((off/'episodes').glob('*.complete')), sorted((on/'episodes').glob('*.complete'))):
        _, lr, lv, lmet = read_episode(lp)
        _, rr, rv, rmet = read_episode(rp)
        first, first_action, steps = None, None, []
        for i, (a, b, x, y) in enumerate(zip(lr, rr, lv, rv)):
            p = x['action_probs'].double().numpy().reshape(-1)
            q = y['action_probs'].double().numpy().reshape(-1)
            delta = float((x['raw_actor_logits'].double()-y['raw_actor_logits'].double()).abs().max())
            row = {'step_index': i, 'raw_logit_max_abs': delta, 'jsd': jsd(p, q),
                   'entropy_off': entropy(p), 'entropy_on': entropy(q),
                   'p_end_off': float(p[end_index]), 'p_end_on': float(q[end_index]),
                   'executed_off': a['executed_action_name'], 'executed_on': b['executed_action_name'],
                   'input_hash_equal': a['input_sha256'] == b['input_sha256']}
            if delta != 0 and first is None:
                first = row
            if a['executed_action_name'] != b['executed_action_name'] and first_action is None:
                first_action = i
            steps.append(row)
        task_results.append({'sample_id': lr[0]['sample_id'], 'delta_success': int(rmet['summary']['success'])-int(lmet['summary']['success']),
                             'delta_cost': rmet['summary']['cost']-lmet['summary']['cost'],
                             'delta_components': {k:rmet['summary'][k]-lmet['summary'][k] for k in COST_KEYS},
                             'first_logit_divergence': first, 'first_action_divergence_step': first_action,
                             'unpaired_steps_off': max(0,len(lr)-len(rr)), 'unpaired_steps_on':max(0,len(rr)-len(lr)),
                             'repetition_off': repetition(lr), 'repetition_on': repetition(rr), 'aligned_steps': steps})
    lo, ro = left['outcomes'], right['outcomes']
    summarize = lambda es: {'n':len(es), 'successes':sum(e['success'] for e in es),
                            'SR':sum(e['success'] for e in es)/len(es),
                            'cost_sum':sum(e['cost'] for e in es), 'mean_cost':sum(e['cost'] for e in es)/len(es),
                            'cost_max':max(e['cost'] for e in es), 'components':{k:sum(e[k] for e in es) for k in COST_KEYS}}
    ls, rs = summarize(lo), summarize(ro)
    return {**result, 'status':'COMPLETE', 'off':ls, 'on':rs, 'delta_SR':rs['SR']-ls['SR'],
            'delta_mean_cost':rs['mean_cost']-ls['mean_cost'], 'delta_cost_sum':rs['cost_sum']-ls['cost_sum'], 'tasks':task_results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('off'); parser.add_argument('on'); parser.add_argument('--report', required=True)
    args = parser.parse_args()
    report = compare(args.off, args.on)
    write_json(args.report, report)
    raise SystemExit(0 if report['status']=='COMPLETE' else 2)
