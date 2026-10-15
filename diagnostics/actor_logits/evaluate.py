"""Future authorized OFF200/ON200 entry. Default is a CPU-only command plan.

--execute-authorized is an operator acknowledgement, NOT PI approval or a claim.
An independent valid PI-approved CODEX claim and green CI are prerequisites.
The 600-pair GPU Logger equivalence is explicitly PI_WAIVED for this P2 only; never forge PASS.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from .manifest import MANIFEST_SHA, check_actual_loader, regenerate
from .storage import read_json, require, sha256, write_json

REPO = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor-reset', choices=['off', 'on'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--physical-gpu', type=int, choices=range(4), default=0)
    parser.add_argument('--num-workers', type=int, choices=[1], default=1)
    parser.add_argument('--seed', type=int, choices=[123], default=123)
    parser.add_argument('--shuffle', action='store_true', required=True)
    parser.add_argument('--greedy', choices=['false'], default='false')
    parser.add_argument('--test-augmentation', action='store_true', required=True)
    parser.add_argument('--episodes', type=int, choices=[200], default=200)
    parser.add_argument('--execute-authorized', action='store_true')
    parser.add_argument('--approval-reference')
    parser.add_argument('--claim-reference')
    parser.add_argument('--control-root', type=Path)
    parser.add_argument('--pi-waive-gpu-equivalence', action='store_true')
    return parser.parse_args(argv)


def original_arguments(args):
    return ['--task_type','ObjectNavType','--eval_subset','minival','--eval_set_size','200',
            '--shuffle','--num_workers','1','--seed','123','--test_augmentation','--max_eps_len','-1',
            '--gpu_devices','0','--house_set','objaverse','--input_sensors','raw_navigation_camera',
            'raw_manipulation_camera','last_actions','an_object_is_in_hand','--ckpt_path',str(args.checkpoint),
            '--output_basedir',str(args.output.parent/(args.output.name+'_official'))]


def code_identity():
    reference = read_json(Path(__file__).with_name('reference_identity.json'))
    changed = {'architecture/models/allenact_transformer_models/inference_agent.py',
               'online_evaluation/online_evaluator_worker.py'}
    for name, digest in reference['source'].items():
        if name not in changed:
            require(sha256(REPO/name) == digest, 'Accepted B0 source drift: '+name)
    tracked = subprocess.check_output(['git','-C',str(REPO),'ls-files','-z']).decode().split('\0')
    status = subprocess.check_output(['git','-C',str(REPO),'status','--porcelain']).decode()
    require(not status, 'Future evaluation requires committed clean implementation')
    return reference, {'git_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD']).decode().strip(),
                       'source_sha256':{p:sha256(REPO/p) for p in tracked if p.endswith(('.py','.json','.sh'))}}


def codex_control_evidence(args):
    """Read exact claimed control HEAD, fail closed; status flags alone are not proof."""
    from .recovery003 import transport_environment, PROXY_URL
    require(args.approval_reference and args.claim_reference and args.control_root,
            'Missing PI approval / Codex claim / control worktree')
    require(args.pi_waive_gpu_equivalence, 'Explicit PI waiver acknowledgement required')
    control = args.control_root.resolve()
    require((control / 'research/LOOP_STATE.json').is_file(), 'Missing control tree')
    def git(*argv):
        return subprocess.check_output(
            ['git', '-c', 'http.proxy='+PROXY_URL, '-c', 'http.sslVerify=true', '-C', str(control), *argv],
            text=True, timeout=60, env=transport_environment()
        ).strip()
    require(git('branch', '--show-current') == 'research-loop', 'Wrong control branch')
    require(not git('status', '--porcelain'), 'Dirty control worktree')
    require(git('rev-parse', 'HEAD') == args.claim_reference, 'Codex claim HEAD mismatch')
    listing = git('ls-remote', '--heads', 'origin', 'refs/heads/research-loop').split()
    require(listing and listing[0] == args.claim_reference, 'Remote Codex claim mismatch')
    approved_code = (control / 'research/P2_CODE_SHA.txt').read_text().strip()
    source_code = subprocess.check_output(
        ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True, timeout=60
    ).strip()
    require(len(approved_code) == 40 and approved_code == source_code,
            'Implementation not pinned to exact PI-approved SHA')
    state = read_json(control / 'research/LOOP_STATE.json')
    require(state.get('status') == 'CODEX_RUNNING' and state.get('next_actor') == 'CODEX'
            and state.get('updated_by') == 'CODEX', 'Not a unique CODEX claim')
    require(state.get('experiment_id') == 'EXP-ACTOR-RESET-ON200-004'
            and state.get('cycle_id') == 'actor-reset-on200-off002-reuse-004-20261015',
            'Wrong approved experiment')
    require(state.get('instruction_commit') == args.approval_reference
            and state.get('claim_id'), 'Claim not bound to approval')
    authorization = state['authorization']
    require(authorization.get('status') == 'APPROVED'
            and authorization.get('approved_by') == 'PI'
            and authorization.get('max_episodes') == 200
            and authorization.get('max_gpu') == 1,
            'P2 authorization mismatch')
    expires = datetime.fromisoformat(authorization['expires_at_utc'].replace('Z', '+00:00'))
    require(datetime.now(timezone.utc) < expires, 'Approval expired')
    design = read_json(control / 'research/NEXT_EXPERIMENT.json')
    require(design.get('status') == 'APPROVED'
            and design.get('experiment_id') == state['experiment_id']
            and design.get('cycle_id') == state['cycle_id']
            and design.get('resources') == {'max_gpu': 1, 'max_episodes': 200},
            'Control design mismatch')
    return {'claim_commit': args.claim_reference, 'claim_id': state['claim_id'],
            'approval_commit': args.approval_reference,
            'approval_expires_at_utc': authorization['expires_at_utc'],
            'gpu_logger_equivalence': 'NOT_RUN_PI_WAIVED'}


def execute(args, tasks, manifest_bytes):
    from .recovery003 import before_model
    control_evidence = before_model(args, tasks, manifest_bytes)
    reference, identity = code_identity()
    require(not args.output.exists(), 'Output exists: no overwrite/resume')
    require(not os.environ.get('ACTION_DICT'), 'Action map override forbidden')
    require(sha256(args.checkpoint) == reference['resources']['checkpoint']['sha256'], 'Checkpoint SHA mismatch')
    dino = Path(os.environ['DINOV2_CKPT'])
    require(sha256(dino) == reference['resources']['DINO']['sha256'], 'DINO weight SHA mismatch')
    dino_repo = Path(os.environ['DINOV2_REPO'])
    for name, digest in reference['dino_source'].items():
        require(sha256(dino_repo/name) == digest, 'DINO source mismatch: '+name)
    for name, version in reference['dependencies'].items():
        require(importlib.metadata.version(name) == version, 'Dependency drift: '+name)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', WANDB_MODE='offline', TOKENIZERS_PARALLELISM='False')
    # All heavyweight imports and CUDA calls are behind the explicit future gate.
    import random
    import numpy as np
    import torch
    from training.online import online_eval as entry
    from .logger import ActorLogitsLogger
    from .reset import reset_root_actor
    require(torch.cuda.device_count() == 1, 'Exactly one visible GPU required')
    random.seed(123); np.random.seed(123); torch.manual_seed(123); torch.cuda.manual_seed_all(123)
    identity.update(actor_reset=args.actor_reset, group='OFF200' if args.actor_reset=='off' else 'ON200',
                    approval_reference=args.approval_reference, control_evidence=control_evidence,
                    checkpoint_sha256=sha256(args.checkpoint), dino_sha256=sha256(dino),
                    manifest_sha256=MANIFEST_SHA, physical_gpu=args.physical_gpu,
                    seed=123, num_workers=1, shuffle=True, greedy=False, test_augmentation=True,
                    official_arguments=original_arguments(args), python=sys.executable,
                    environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','DINOV2_REPO','DINOV2_CKPT','OBJAVERSE_DATA_DIR','OBJAVERSE_HOUSES_DIR','NLTK_DATA','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','WANDB_MODE']})
    manager, agent_type = entry.OnlineEvaluatorManager, entry.InferenceAgentVIDA
    loader = manager.load_minival_eval_samples_per_task
    build_descriptor = agent_type.__dict__['build_agent']
    state = {'checked':False, 'logger':None}
    def checked_loader(self, task_type, use_local_path=None):
        samples = loader(self, task_type, use_local_path)
        check_actual_loader(samples, tasks)
        state['checked'] = True
        return samples
    def build(cls, *positional, **keywords):
        require(state['checked'] and state['logger'] is None, 'Manifest not checked or extra Agent')
        require(keywords['greedy_sampling'] is False and keywords['test_augmentation'] is True, 'Sampling/augmentation mismatch')
        loads = []
        def profile(frame, event, result):
            if event=='return' and frame.f_code.co_name=='load_state_dict' and frame.f_code.co_filename.endswith('/torch/nn/modules/module.py') and hasattr(result,'missing_keys'):
                loads.append({'class':type(frame.f_locals['self']).__name__, 'strict':frame.f_locals['strict'],
                              'nkeys':len(frame.f_locals['state_dict']), 'missing':list(result.missing_keys), 'unexpected':list(result.unexpected_keys)})
        require(sys.getprofile() is None, 'Existing profiler cannot be overwritten')
        identity['pre_model_rng'] = {'python':hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                                    'numpy':hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
                                    'torch_cpu':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                                    'torch_cuda':hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
        try:
            sys.setprofile(profile)
            agent = build_descriptor.__func__(cls, *positional, **keywords)
        finally:
            sys.setprofile(None)
        require(any(x['class']=='SafeDinoLLAMATxNavActorCriticSeparate' and x['nkeys']==417 and not x['missing'] and not x['unexpected'] for x in loads), 'Policy checkpoint load incomplete')
        require(any(x['class']=='DinoVisionTransformer' and x['strict'] and x['nkeys']==175 and not x['missing'] and not x['unexpected'] for x in loads), 'Strict accepted DINO load not observed')
        require(all(not x['missing'] and not x['unexpected'] for x in loads), 'Unexpected load incompatibility')
        names = agent.get_action_list()
        require(len(names)==20 and names.index('end')==4, 'Action map drift')
        identity['load_events'] = loads
        identity['runtime_modules'] = {k:v.__file__ for k,v in sys.modules.items() if getattr(v,'__file__',None) and k.split('.')[0] in ['architecture','training','online_evaluation','tasks','environment','diagnostics','dinov2']}
        for name, path in identity['runtime_modules'].items():
            expected_root = dino_repo if name.startswith('dinov2') else REPO
            require(Path(path).resolve().is_relative_to(expected_root.resolve()), 'Unexpected import: '+path)
        logger = ActorLogitsLogger(agent.actor_critic, args.output, tasks, names, identity)
        state['logger'] = logger
        (args.output/'TASK_MANIFEST.csv').write_bytes(manifest_bytes)
        agent.actor_logits_logger = logger
        agent.actor_boundary_reset = reset_root_actor if args.actor_reset=='on' else None
        return agent
    manager.load_minival_eval_samples_per_task = checked_loader
    agent_type.build_agent = classmethod(build)
    previous_argv = sys.argv
    try:
        sys.argv = ['training/online/online_eval.py'] + original_arguments(args)
        entry.main(entry.parse_args())
        require(state['logger'] is not None, 'No Agent/logger created')
        state['logger'].finish_run()
    except BaseException as exc:
        if state['logger'] is not None:
            state['logger'].abort('evaluation', exc)
            state['logger'].write_index()
        elif not args.output.exists():
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output/'RUN_MANIFEST.json', {'status':'INCOMPLETE','error':str(exc),'episodes_completed':0,'outcomes':None,'provenance':identity})
        raise
    finally:
        sys.argv = previous_argv
        manager.load_minival_eval_samples_per_task = loader
        agent_type.build_agent = build_descriptor
        if state['logger'] is not None:
            state['logger'].detach()


def main(argv=None):
    args = parse_args(argv)
    tasks, content = regenerate(REPO)
    if not args.execute_authorized:
        print(json.dumps({'status':'PLAN_ONLY_NOT_EXECUTED', 'actor_reset':args.actor_reset,
                          'tasks':len(tasks), 'manifest_sha256':MANIFEST_SHA,
                          'official_arguments':original_arguments(args),
                          'GPU_EQUIVALENCE':'NOT_RUN_PI_WAIVED'}, indent=2))
        return
    execute(args, tasks, content)


if __name__ == '__main__':
    main()
