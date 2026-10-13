"""CPU mocks and real integration function bodies; never real policy weights/THOR."""
import ast
import copy
import json
import os
from pathlib import Path
import random
import tempfile
import typing
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from diagnostics.actor_logits.logger import ActorLogitsLogger, boundary_state, COST_KEYS
from diagnostics.actor_logits.reset import reset_root_actor
from diagnostics.actor_logits.storage import pack_tensor, unpack_tensor, read_episode, read_json, read_jsonl
from diagnostics.actor_logits.validate import validate_run
from diagnostics.actor_logits.analyze import compare, jsd, entropy
from diagnostics.actor_logits.manifest import regenerate, check_actual_loader

ROOT = Path(__file__).resolve().parents[3]
ACTIONS = ['m', 'end', 'sub_done']


class Cache(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache_k = torch.zeros(1, 8, 1, 2)
        self.cache_v = torch.zeros_like(self.cache_k)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Module() for _ in range(3)])
        for layer in self.layers:
            layer.attention = Cache()

    def forward(self, x, start_pos, mask):
        for layer in self.layers:
            layer.attention.cache_k[:, start_pos] = x.reshape(1, 1, 2)
            layer.attention.cache_v[:, start_pos] = -x.reshape(1, 1, 2)
        return x


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.time_step_counter = 0
        self.actor = nn.Module()
        self.actor.linear = nn.Linear(2, 3)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        self.decoder(x, self.time_step_counter, None)
        self.time_step_counter += 1
        return self.actor.linear(x)


class Distribution(torch.distributions.Categorical):
    def __init__(self, logits, counts):
        super().__init__(logits=logits)
        self.counts = counts

    def sample(self, sample_shape=torch.Size()):
        self.counts['sample'] += 1
        return super().sample(sample_shape)

    def mode(self):
        self.counts['mode'] += 1
        return self.logits.argmax(dim=-1)


class Root(Branch):
    def __init__(self):
        super().__init__()
        self.critic_tsfm = Branch()
        self.c_critic_tsfm = Branch()
        self.recurrent_memory_specification = {}
        self.action_space = None
        self.counts = {'sample': 0, 'mode': 0}

    def forward(self, observations, memory=None, prev_actions=None, masks=None):
        x = observations['x']
        logits = super().forward(x)
        self.critic_tsfm(x + 17)
        self.c_critic_tsfm(x - 31)
        return NS(distributions=Distribution(logits, self.counts)), None


class Rollout:
    def initialize(self, observations, **kwargs):
        self.observations = observations
    def after_updates(self):
        pass
    def add(self, observations, **kwargs):
        self.observations = observations
    def agent_input_for_next_step(self):
        return dict(observations=self.observations, memory=None,
                    prev_actions=torch.zeros((1,1), dtype=torch.long), masks=torch.ones(1,1,1))


def load_method(file, method, namespace):
    node = next(n for n in ast.walk(ast.parse((ROOT/file).read_text())) if isinstance(n,ast.FunctionDef) and n.name==method)
    exec(compile('from __future__ import annotations\n'+ast.unparse(node), str(ROOT/file), 'exec'), namespace)
    return namespace[method]


agent_act = load_method('architecture/models/allenact_transformer_models/inference_agent.py', 'act',
                       {'torch':torch, 'cast':lambda typ,x:x, 'Tuple':typing.Tuple, 'ActorCriticOutput':dict,
                        'DistributionType':int, 'Memory':int, 'Optional':typing.Optional,
                        'batch_observations':lambda xs,device:xs[0],
                        'su':NS(flatten=lambda space,x:x, action_list=lambda space,x:[int(x.item())])})


class Agent:
    act = agent_act
    def __init__(self, root):
        self.actor_critic = root
        self.rollout_storage = Rollout()
        self.device = torch.device('cpu')
        self.sensor_preprocessor_graph = None
        self.has_initialized = False
        self.steps_before_rollout_refresh = 10000
        self.greedy_sampling = False
        self.memory = None
        self.reset()
    def reset(self):
        self.steps_taken_in_task = 0
    def get_action_list(self):
        return ACTIONS
    def get_action(self, frame, goal):
        return self.act(frame, goal)


class Task:
    def __init__(self, sid):
        self.task_info = {'natural_language_spec':'mock goal', 'eval_info':{'sample_id':sid,'task_path':'a/b/c/d'},'followed_path':[],'task_type':'ObjectNavType'}
        self.controller = NS(navigation_camera=np.zeros((2,2,3),dtype=np.uint8),manipulation_camera=np.zeros((2,2,3),dtype=np.uint8))
        self.max_steps = 2
        self.steps = 0
        self.queries = 0
        self.last_action_success = True
        self.debug_info = {}
        for key in COST_KEYS:
            setattr(self, 'last_action_'+key, 0)
    def get_observations(self):
        self.queries += 1
        return {'x':torch.tensor([[[1.0,2.0]]])}
    def step_with_action_str(self, action):
        self.steps += 1
        self.last_action_corner = 1
    def is_done(self):
        return self.steps == 2
    def is_successful(self):
        return True


worker_method = load_method('online_evaluation/online_evaluator_worker.py', 'evaluate_on_task',
                           {'torch':torch, 'np':np, 'get_top_down_frame':lambda *a:(np.zeros((2,2,3)),[]),
                            'chain':__import__('itertools').chain})


class Worker:
    evaluate_on_task = worker_method
    input_sensors = ['x']
    skip_done = True
    logging_sensor = NS(get_video_frame=lambda **kw: np.zeros((2,2,3)))
    def calculate_metrics(self, task, actions, success, additional, number, *components):
        return dict(eps_len=len(actions), success=float(success)+1e-8, cost=sum(components), **dict(zip(COST_KEYS, components)))


def run_mock(directory=None, reset='off', episodes=2):
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    root = Root().eval()
    # Force sampled 'end' while still making one genuine stochastic sample call.
    with torch.no_grad():
        root.actor.linear.weight.zero_()
        root.actor.linear.bias.copy_(torch.tensor([-1000.0,1000.0,-1000.0]))
    tasks = [dict(sample_id=f'task{i}',task_source_index=i,raw_spec_sha256=str(i)*64) for i in range(episodes)]
    agent = Agent(root)
    logger = None
    if directory is not None:
        provenance = dict(actor_reset=reset, git_commit='CPU_MOCK', source_sha256={'fixture':'CPU_MOCK'},
                          checkpoint_sha256='CPU_MOCK_NO_WEIGHTS',dino_sha256='CPU_MOCK_NO_DINO',
                          manifest_sha256='CPU_MOCK_TASKS',seed=123,num_workers=1,shuffle=True,
                          greedy=False,test_augmentation=True,physical_gpu='NONE_CPU_MOCK')
        logger = ActorLogitsLogger(root,directory,tasks,ACTIONS,provenance)
        agent.actor_logits_logger = logger
    agent.actor_boundary_reset = reset_root_actor if reset=='on' else None
    worker = Worker()
    queries = []
    for spec in tasks:
        task = Task(spec['sample_id'])
        worker.evaluate_on_task(task,agent,0)
        queries.append(task.queries)
    if logger:
        logger.finish_run(); logger.detach()
    return root, torch.get_rng_state().clone(), random.getstate(), np.random.get_state(), queries


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'run'

    def test_native_roundtrip_all_dtypes(self):
        for dtype in [torch.float16,torch.float32,torch.float64,torch.bfloat16,torch.int64,torch.bool]:
            x=torch.arange(12).reshape(3,4).to(dtype).t().contiguous()
            array,spec=pack_tensor(x); y=unpack_tensor(array,spec)
            self.assertEqual(x.dtype,y.dtype);self.assertEqual(x.shape,y.shape)
            self.assertTrue(torch.equal(x.view(torch.uint8),y.view(torch.uint8)))

    def test_full_agent_worker_integration_lossless_and_override(self):
        root,*_=run_mock(self.path)
        report=validate_run(self.path,2)
        self.assertEqual(report['status'],'PASS',report)
        status,rows,values,metrics=read_episode(self.path/'episodes/0000.complete')
        self.assertEqual(rows[0]['sampled_action_index'],1)
        self.assertEqual(rows[0]['executed_action_name'],'sub_done')
        self.assertEqual(rows[0]['executed_action_index'],2)
        self.assertEqual(values[0]['raw_actor_logits'].flatten().tolist(),[-1000,1000,-1000])
        self.assertFalse(torch.equal(values[0]['raw_actor_logits'],values[0]['normalized_logits']))
        self.assertEqual(metrics['official']['corner'],1)  # terminal action cost is not appended

    def test_no_added_rng_forward_sample_mode_or_observation_calls(self):
        plain=run_mock();logged=run_mock(self.path)
        self.assertTrue(torch.equal(plain[1],logged[1]));self.assertEqual(plain[2],logged[2])
        self.assertEqual(repr(plain[3]),repr(logged[3]));self.assertEqual(plain[4],logged[4])
        for root in [plain[0],logged[0]]:
            self.assertEqual(root.calls,4);self.assertEqual(root.critic_tsfm.calls,4)
            self.assertEqual(root.c_critic_tsfm.calls,4);self.assertEqual(root.counts,{'sample':4,'mode':4})
        self.assertEqual(boundary_state(plain[0]),boundary_state(logged[0]))

    def test_shared_logger_on_reset_keeps_critics(self):
        root,*_=run_mock(self.path,reset='on')
        self.assertEqual(root.time_step_counter,2);self.assertEqual(root.critic_tsfm.time_step_counter,4)
        self.assertEqual(validate_run(self.path,2)['status'],'PASS')
        boundaries=read_jsonl(self.path/'episodes/0001.complete/boundary_states.jsonl')
        self.assertEqual(boundaries[1]['state']['reward_critic'],boundaries[2]['state']['reward_critic'])

    def test_reset_changes_no_weights_or_rng(self):
        root,*_=run_mock()
        weights={k:v.clone() for k,v in root.state_dict().items()};before=boundary_state(root);rng=torch.get_rng_state()
        reset_root_actor(root);after=boundary_state(root)
        self.assertEqual(before['reward_critic'],after['reward_critic']);self.assertEqual(before['cost_critic'],after['cost_critic'])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,root.state_dict()[k]) for k,v in weights.items()))

    def test_partial_crash_and_preexisting_output_rejected(self):
        root=Root();tasks=[dict(sample_id='x',task_source_index=0,raw_spec_sha256='x')]
        logger=ActorLogitsLogger(root,self.path,tasks,ACTIONS,{'actor_reset':'off'})
        logger.begin_episode(Task('x').task_info,0)
        logger.abort('mock_crash',RuntimeError('simulated crash'));logger.detach()
        self.assertTrue((self.path/'episodes/0000.partial').exists())
        self.assertEqual(validate_run(self.path,1)['status'],'INCOMPLETE')
        with self.assertRaises(FileExistsError):ActorLogitsLogger(root,self.path,tasks,ACTIONS,{})

    def test_duplicate_sample_fail_closed(self):
        logger=ActorLogitsLogger(Root(),self.path,[dict(sample_id='x')],ACTIONS,{'actor_reset':'off'})
        self.addCleanup(logger.detach)
        logger.begin_episode(Task('x').task_info,0)
        logger.begin_decision({'prev_actions':torch.zeros(1,dtype=torch.long)})
        logger.note_sampling_call('sample')
        with self.assertRaisesRegex(ValueError,'Duplicate'):logger.note_sampling_call('sample')
        self.assertEqual(read_json(self.path/'RUN_MANIFEST.json')['status'],'INCOMPLETE')

    def test_missing_or_duplicate_rows_detected(self):
        for duplicate in [False,True]:
            p=Path(self.tmp.name)/str(duplicate);run_mock(p)
            log=p/'episodes/0000.complete/actions.jsonl';lines=log.read_text().splitlines()
            log.write_text('\n'.join([lines[0],lines[0]] if duplicate else lines[:1])+'\n')
            self.assertEqual(validate_run(p,2)['status'],'INCOMPLETE')

    def test_corrupt_tensor_detected(self):
        run_mock(self.path)
        p=self.path/'episodes/0000.complete/tensors.npz'
        with np.load(p) as z:arrays={k:z[k] for k in z.files}
        arrays['000000.raw_actor_logits'][...]=123
        np.savez(p,**arrays)
        self.assertEqual(validate_run(self.path,2)['status'],'INCOMPLETE')

    def test_missing_ack_detected(self):
        run_mock(self.path)
        (self.path/'episodes/0000.complete/execution_results.jsonl').write_text('')
        self.assertEqual(validate_run(self.path,2)['status'],'INCOMPLETE')

    def test_no_zero_outcome_for_incomplete_200(self):
        run_mock(self.path)
        report=validate_run(self.path)
        self.assertEqual(report['status'],'INCOMPLETE');self.assertIsNone(report['outcomes'])
        report=compare(self.path,self.path)
        self.assertEqual(report['status'],'INCOMPLETE');self.assertIsNone(report['delta_SR'])

    def test_analysis_alignment_entropy_and_jsd(self):
        off=Path(self.tmp.name)/'OFF';on=Path(self.tmp.name)/'ON'
        run_mock(off);run_mock(on,reset='on')
        result=compare(off,on,expected=2)
        self.assertEqual(result['status'],'COMPLETE');self.assertEqual(result['delta_SR'],0)
        self.assertEqual(result['delta_mean_cost'],0)
        self.assertAlmostEqual(entropy([0.5,0.5]),np.log(2));self.assertAlmostEqual(jsd([1,0],[0,1]),np.log(2))

    def test_ndarray_official_metrics_episode_completion_preserves_logits(self):
        original = Worker.calculate_metrics
        def metrics_with_arrays(worker, *args, **kwargs):
            metrics = original(worker, *args, **kwargs)
            metrics['percentage_collision'] = np.array([0.25], dtype=np.float32)
            metrics['nested_fixture'] = {'scalar': np.asarray(2.5),
                                         'matrix': np.array([[0.125, 0.5]], dtype=np.float32)}
            return metrics
        with patch.object(Worker, 'calculate_metrics', metrics_with_arrays):
            run_mock(self.path)
        report = validate_run(self.path, 2)
        self.assertEqual(report['status'], 'PASS', report)
        for episode in sorted((self.path / 'episodes').iterdir()):
            status, rows, tensors, metrics = read_episode(episode)
            self.assertEqual(status['status'], 'COMPLETE')
            self.assertEqual(metrics['official']['percentage_collision'], [0.25])
            self.assertEqual(metrics['official']['nested_fixture'],
                             {'scalar': 2.5, 'matrix': [[0.125, 0.5]]})
            for values in tensors:
                raw = values['raw_actor_logits']
                expected = torch.tensor([-1000.0, 1000.0, -1000.0], dtype=torch.float32).reshape_as(raw)
                self.assertEqual(raw.dtype, expected.dtype)
                self.assertTrue(torch.equal(raw.view(torch.uint8), expected.view(torch.uint8)))
            self.assertFalse((episode / 'metrics.json.tmp').exists())

    def test_actual_loader_selection_and_manifest(self):
        state=random.getstate();tasks,content=regenerate(ROOT)
        self.assertEqual(state,random.getstate());self.assertEqual(len(tasks),200)
        self.assertEqual([t['task_source_index'] for t in tasks[:5]],[127,103,16,6,50])
        with self.assertRaises(ValueError):check_actual_loader([],tasks)

    def test_analysis_rejects_different_provenance(self):
        from diagnostics.actor_logits.storage import write_json
        off=Path(self.tmp.name)/'OFF';on=Path(self.tmp.name)/'ON'
        run_mock(off);run_mock(on,reset='on')
        manifest=read_json(on/'RUN_MANIFEST.json')
        manifest['provenance']['seed']=999
        write_json(on/'RUN_MANIFEST.json',manifest)
        # Reindex intentionally, so this exercises comparison rather than corruption detection.
        indexer=NS(output=on)
        ActorLogitsLogger.write_index(indexer)
        self.assertEqual(validate_run(on,2)['status'],'PASS')
        result=compare(off,on,expected=2)
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertIn('provenance',result['error'])

    def test_off_on_command_plan_has_same_evaluation_parameters(self):
        from diagnostics.actor_logits.evaluate import parse_args,original_arguments
        common=['--checkpoint','same.pt','--shuffle','--test-augmentation']
        off=original_arguments(parse_args(common+['--actor-reset','off','--output','/tmp/OFF200']))
        on=original_arguments(parse_args(common+['--actor-reset','on','--output','/tmp/ON200']))
        self.assertEqual(off[:-1],on[:-1])
        self.assertNotEqual(off[-1],on[-1])

    def test_io_error_propagates_and_records_location(self):
        logger=ActorLogitsLogger(Root(),self.path,[dict(sample_id='x')],ACTIONS,{'actor_reset':'off'})
        self.addCleanup(logger.detach)
        with patch('diagnostics.actor_logits.logger.EpisodeWriter',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):logger.begin_episode(Task('x').task_info,0)
        self.assertEqual(read_json(self.path/'ERROR.json')['location'],'begin_episode')

    def test_dispatched_step_survives_crash_without_ack(self):
        root=Root().eval();agent=Agent(root)
        logger=ActorLogitsLogger(root,self.path,[dict(sample_id='x',task_source_index=0,raw_spec_sha256='x')],ACTIONS,{'actor_reset':'off'})
        self.addCleanup(logger.detach);agent.actor_logits_logger=logger
        logger.begin_episode(Task('x').task_info,0)
        with torch.no_grad():
            action,_=agent.act({'x':torch.ones(1,1,2)},'goal')
        logger.execution_requested(action,dict.fromkeys(COST_KEYS,0))
        logger.abort('environment_step',RuntimeError('mock crash before return'))
        partial=self.path/'episodes/0000.partial'
        self.assertEqual(len(read_jsonl(partial/'actions.jsonl')),1)
        self.assertTrue((partial/'step_000000.npz').exists())
        self.assertEqual(read_jsonl(partial/'execution_results.jsonl'),[])
        with self.assertRaises(ValueError):logger.execution_returned()
        self.assertEqual(read_json(self.path/'ERROR.json')['location'],'environment_step')

    def test_completed_episode_survives_later_failure(self):
        root=Root().eval();agent=Agent(root)
        tasks=[dict(sample_id=f'x{i}',task_source_index=i,raw_spec_sha256=str(i)) for i in range(2)]
        logger=ActorLogitsLogger(root,self.path,tasks,ACTIONS,{'actor_reset':'off'})
        self.addCleanup(logger.detach);agent.actor_logits_logger=logger
        Worker().evaluate_on_task(Task('x0'),agent,0)
        logger.begin_episode(Task('x1').task_info,0)
        logger.abort('mock_second_episode',RuntimeError('mock crash'))
        self.assertTrue((self.path/'episodes/0000.complete/tensors.npz').exists())
        self.assertTrue((self.path/'episodes/0001.partial').exists())
        self.assertEqual(read_json(self.path/'RUN_MANIFEST.json')['episodes_completed'],1)
        self.assertEqual(validate_run(self.path,2)['status'],'INCOMPLETE')

    def test_missing_boundary_and_count_mismatch(self):
        for field in ['boundary','calls']:
            p=Path(self.tmp.name)/field;run_mock(p)
            if field=='boundary':
                (p/'episodes/0000.complete/boundary_states.jsonl').write_text('')
            else:
                q=p/'episodes/0000.complete/actions.jsonl';rows=read_jsonl(q)
                rows[0]['calls']['sample']=2
                q.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertEqual(validate_run(p,2)['status'],'INCOMPLETE')

    def test_cli_plan_never_imports_runtime(self):
        from diagnostics.actor_logits.evaluate import main
        import contextlib,io
        with patch('diagnostics.actor_logits.evaluate.execute',side_effect=AssertionError('execution forbidden')):
            with contextlib.redirect_stdout(io.StringIO()) as capture:
                main(['--actor-reset','off','--output',str(self.path),'--checkpoint','missing.pth','--shuffle','--test-augmentation'])
        self.assertEqual(json.loads(capture.getvalue())['status'],'PLAN_ONLY_NOT_EXECUTED')
        self.assertFalse(self.path.exists())

    def test_gpu_gate_rejects_before_runtime_import(self):
        from diagnostics.actor_logits.evaluate import execute,parse_args
        args=parse_args(['--actor-reset','off','--output',str(self.path),'--checkpoint','missing.pth','--shuffle','--test-augmentation','--execute-authorized'])
        with self.assertRaisesRegex(ValueError,'Missing PI approval / Codex claim / control worktree'):
            execute(args,[],b'')
        self.assertFalse(self.path.exists())

    def test_no_cuda_initialized(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    unittest.main()
