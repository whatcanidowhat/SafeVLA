"""Passive root-head hooks plus explicit Agent/Worker observation points."""
import csv
import functools
import hashlib
import json
import traceback
from pathlib import Path

from .storage import EpisodeWriter, pack_tensor, require, sha256, write_json, append_json

COST_KEYS = ('danger', 'corner', 'blind', 'fragile', 'critical')


def guarded(method):
    @functools.wraps(method)
    def call(self, *args, **kwargs):
        require('error_location' not in self.manifest, 'Logger has failed; continuation is forbidden')
        try:
            return method(self, *args, **kwargs)
        except BaseException as exc:
            self.abort(method.__name__, exc)
            raise
    return call


def tensor_info(tensor):
    _, spec = pack_tensor(tensor)
    return {**spec, 'device': str(tensor.device)}


def branch_state(branch):
    return {'counter': int(branch.time_step_counter), 'layers': [
        {'k': tensor_info(layer.attention.cache_k), 'v': tensor_info(layer.attention.cache_v)}
        for layer in branch.decoder.layers]}


def boundary_state(root):
    return {'actor': branch_state(root), 'reward_critic': branch_state(root.critic_tsfm),
            'cost_critic': branch_state(root.c_critic_tsfm)}


def input_digest(value):
    """Hash existing input values; never invokes preprocessing or a controller."""
    import torch
    def visit(x):
        if isinstance(x, torch.Tensor):
            return tensor_info(x)
        if isinstance(x, dict):
            return {str(k): visit(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple)):
            return [visit(v) for v in x]
        if x is None or isinstance(x, (str, bool, int, float)):
            return x
        # AllenAct Memory is a dict subclass. Reject unknown input carriers.
        raise TypeError(f'Unsupported input carrier {type(x).__name__}')
    return hashlib.sha256(json.dumps(visit(value), sort_keys=True).encode()).hexdigest()


class ActorLogitsLogger:
    def __init__(self, root, output, tasks, action_names, provenance):
        self.root = root
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.tasks = tasks
        self.action_names = list(action_names)
        require(len(set(self.action_names)) == len(self.action_names), 'Duplicate action names')
        self.manifest = {'schema_version': 1, 'status': 'INCOMPLETE', 'episodes_expected': len(tasks),
                         'episodes_completed': 0, 'decisions': 0, 'action_names': self.action_names,
                         'tasks': tasks, 'provenance': provenance}
        write_json(self.output / 'RUN_MANIFEST.json', self.manifest)
        self.handles = []
        self.writer = None
        self.active = None
        self.ready = None
        self.dispatched = False
        self.global_step = 0
        self.metrics = []
        self.handles.append(root.register_forward_pre_hook(self._root_before))
        self.handles.append(root.actor.linear.register_forward_hook(self._raw))
        self.handles.append(root.decoder.register_forward_pre_hook(self._decoder))
        for name, branch in [('reward_forward', root.critic_tsfm), ('cost_forward', root.c_critic_tsfm)]:
            self.handles.append(branch.register_forward_pre_hook(functools.partial(self._critic, name)))

    def abort(self, location, error):
        self.manifest['status'] = 'INCOMPLETE'
        self.manifest.setdefault('error_location', location)
        self.manifest.setdefault('error', f'{type(error).__name__}: {error}')
        try:
            write_json(self.output / 'RUN_MANIFEST.json', self.manifest)
            event = {'location': location, 'error': str(error), 'traceback': traceback.format_exc(),
                     'episode_order': len(self.metrics), 'global_step': self.global_step,
                     'active_calls': None if self.active is None else self.active['calls']}
            if not (self.output / 'ERROR.json').exists():
                write_json(self.output / 'ERROR.json', event)
            append_json(self.output / 'ERROR_EVENTS.jsonl', event)
        except OSError:
            pass  # Original exception still propagates; existing INCOMPLETE marker remains.

    @guarded
    def begin_episode(self, task_info, worker_id):
        require(self.writer is None and self.active is None, 'Overlapping episodes')
        order = len(self.metrics)
        require(order < len(self.tasks) and worker_id == 0, 'Unexpected task/worker')
        expected = self.tasks[order]
        require(task_info['eval_info']['sample_id'] == expected['sample_id'], 'Task order/ID mismatch')
        self.episode = {'episode_order': order, **expected, 'goal': task_info['natural_language_spec'],
                        'target_ids': task_info.get('synset_to_object_ids', {}), 'worker_id': worker_id}
        self.writer = EpisodeWriter(self.output, self.episode)
        self.boundary('before_original_reset')

    @guarded
    def boundary(self, label):
        require(self.writer is not None and self.active is None, 'Boundary inside decision')
        self.writer.boundary({'event': label, 'episode_order': self.episode['episode_order'],
                              'sample_id': self.episode['sample_id'], 'state': boundary_state(self.root)})

    @guarded
    def begin_decision(self, agent_input):
        require(self.writer is not None and self.active is None and not self.dispatched, 'Missing/duplicate decision boundary')
        self.active = {'calls': dict(root_forward=0, reward_forward=0, cost_forward=0,
                                    raw_head=0, decoder=0, sample=0, mode=0),
                       'prev_action': agent_input['prev_actions'].detach().cpu().clone(),
                       'input_sha256': input_digest(agent_input)}

    @guarded
    def _root_before(self, module, args):
        require(self.active is not None, 'Unregistered policy forward')
        self.active['calls']['root_forward'] += 1
        require(self.active['calls']['root_forward'] == 1, 'Extra root forward')
        self.active['counter_before'] = int(module.time_step_counter)

    @guarded
    def _critic(self, name, module, args):
        require(self.active is not None, 'Unregistered critic forward')
        self.active['calls'][name] += 1
        require(self.active['calls'][name] == 1, 'Extra critic forward')

    @guarded
    def _raw(self, module, args, output):
        require(self.active is not None, 'Head outside decision')
        self.active['calls']['raw_head'] += 1
        require(self.active['calls']['raw_head'] == 1, 'Duplicate root head')
        self.active['raw_actor_logits'] = output.detach().cpu().clone()

    @guarded
    def _decoder(self, module, args):
        require(self.active is not None, 'Decoder outside decision')
        self.active['calls']['decoder'] += 1
        self.active['decoder_start_pos'] = int(args[1])
        mask = args[2]
        self.active['mask'] = None if mask is None else tensor_info(mask)

    @guarded
    def note_sampling_call(self, name):
        """Called at the existing Agent invocation sites; never samples itself."""
        require(self.active is not None and name in ('sample', 'mode'), 'Sampling outside decision')
        self.active['calls'][name] += 1
        require(self.active['calls'][name] == 1, 'Duplicate sample/mode call')

    @guarded
    def policy_result(self, output, sampled, mode, history, proposed_action):
        require(self.ready is None and self.active is not None, 'Duplicate/missing policy result')
        require(all(v == 1 for v in self.active['calls'].values()), 'Forward/sampling count mismatch')
        tensors = {'raw_actor_logits': self.active['raw_actor_logits'],
                   'normalized_logits': output.distributions.logits,
                   'action_probs': output.distributions.probs, 'sampled_action': sampled,
                   'greedy_mode_action': mode, 'action_history': history,
                   'prev_action': self.active['prev_action']}
        tensors = {k: v.detach().cpu().clone() for k, v in tensors.items()}
        for k in ['raw_actor_logits', 'normalized_logits', 'action_probs']:
            require(tensors[k].numel() == len(self.action_names), 'Not one full action distribution')
        sampled_index, mode_index = int(sampled.item()), int(mode.item())
        require(0 <= sampled_index < len(self.action_names) and 0 <= mode_index < len(self.action_names), 'Invalid action index')
        require(proposed_action in self.action_names, 'Unknown proposed action')
        self.ready = (tensors, {'sampled_action_index': sampled_index, 'greedy_mode_action_index': mode_index,
                               'agent_action_name': proposed_action,
                               'counter_after': int(self.root.time_step_counter)})

    @guarded
    def execution_requested(self, action, pre_step_cost):
        require(self.ready is not None and not self.dispatched, 'Missing/duplicate execution')
        require(action in self.action_names and set(pre_step_cost) == set(COST_KEYS), 'Action/cost schema mismatch')
        tensors, result = self.ready
        step = self.writer.count
        row = {**self.episode, 'step_index': step, 'global_step': self.global_step,
               'decision_id': f"{self.episode['episode_order']:04d}:{self.episode['sample_id']}:{step:06d}",
               **result, 'executed_action_index': self.action_names.index(action),
               'executed_action_name': action, 'execution_status': 'DISPATCHED',
               'counter_before': self.active['counter_before'],
               'decoder_start_pos': self.active['decoder_start_pos'], 'mask': self.active['mask'],
               'calls': self.active['calls'], 'input_sha256': self.active['input_sha256'],
               'pre_step_cost': pre_step_cost}
        self.writer.append(row, tensors)
        self.ready = None
        self.active = None
        self.dispatched = True
        self.global_step += 1

    @guarded
    def execution_returned(self):
        require(self.dispatched, 'Execution acknowledgement without dispatch')
        self.writer.acknowledge()
        self.dispatched = False

    @guarded
    def end_episode(self, metrics, success):
        from .storage import read_jsonl
        require(self.active is None and self.ready is None and not self.dispatched, 'Unfinished decision')
        rows = read_jsonl(self.writer.path / 'actions.jsonl')
        require(metrics['eps_len'] == len(rows), 'Official decision count mismatch')
        components = {k: sum(r['pre_step_cost'][k] for r in rows) for k in COST_KEYS}
        require(all(metrics[k] == v for k, v in components.items()) and metrics['cost'] == sum(components.values()), 'Official pre-step cost mismatch')
        require(bool(success) == (metrics['success'] > 0.1), 'Official success mismatch')
        self.boundary('episode_end')
        summary = {'episode_order': self.episode['episode_order'], 'sample_id': self.episode['sample_id'],
                   'success': bool(success), 'eps_len': metrics['eps_len'], 'cost': metrics['cost'], **components}
        self.writer.finish({'official': metrics, 'summary': summary})
        self.metrics.append(summary)
        self.writer = None
        self.manifest.update(episodes_completed=len(self.metrics), decisions=self.global_step)
        write_json(self.output / 'RUN_MANIFEST.json', self.manifest)

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @guarded
    def finish_run(self):
        require(self.writer is None and len(self.metrics) == len(self.tasks), 'INCOMPLETE run')
        with (self.output / 'episode_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.metrics[0]))
            writer.writeheader()
            writer.writerows(self.metrics)
        self.manifest.update(status='COMPLETE', decisions=self.global_step)
        write_json(self.output / 'RUN_MANIFEST.json', self.manifest)
        self.write_index()

    def write_index(self):
        artifacts = [{'path': str(p.relative_to(self.output)), 'sha256': sha256(p), 'size': p.stat().st_size}
                     for p in sorted(self.output.rglob('*')) if p.is_file() and p.name != 'ARTIFACT_INDEX.json']
        write_json(self.output / 'ARTIFACT_INDEX.json', {'artifacts': artifacts})
