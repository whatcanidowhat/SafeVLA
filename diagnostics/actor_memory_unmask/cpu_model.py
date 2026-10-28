"""CPU fixture: execute source AST forward with genuine Llama decoder.

Only perception is replaced by supplied fixed embeddings. No simulator, model
download, CUDA API, or live actor. AST extraction avoids unrelated T5 imports.
"""
import ast
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from torch import nn
from training.online.third_party_models.llama.model import ModelArgs, TransformerDecoder

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = 'architecture/models/allenact_transformer_models/allenact_dino_transformer.py'


class Output(NS):
    @classmethod
    def __class_getitem__(cls, item):
        return cls


def extract(source, class_name, methods=None, namespace=None):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    if methods is not None:
        cls.bases = [ast.Attribute(value=ast.Name(id='nn', ctx=ast.Load()), attr='Module', ctx=ast.Load())]
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    ns = dict(torch=torch, nn=nn, math=math, ActorCriticOutput=Output,
              MultiAuxTaskNegEntropyLoss=NS(UUID='unused'))
    if namespace:
        ns.update(namespace)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT/MODEL_PATH), 'exec'), ns)
    return ns[class_name]


class ActorHead(nn.Module):
    def __init__(self, dim, actions):
        super().__init__()
        self.linear = nn.Linear(dim, actions)
        self.raw = None

    def forward(self, x):
        self.raw = self.linear(x)
        return torch.distributions.Categorical(logits=self.raw)


class CriticHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        return self.fc(x)


def make_model(dim=16, layers=3, heads=4, max_steps=500, legacy=False):
    source = (ROOT/MODEL_PATH).read_text()
    if legacy:
        source = subprocess.check_output(['git', 'show', '06cde9adeb91d47c062a11e72e90c7edf56fb398:'+MODEL_PATH], cwd=ROOT, text=True)
    cls = extract(source, 'DinoLLAMATxNavActorCritic', {'forward', 'compute_total_grad_norm'})
    model = cls()
    position_source = (ROOT/'architecture/models/transformer_models/text_cond_visual_encoder.py').read_text()
    position = extract(position_source, 'PositionalEncoder')
    model.action_space = NS(n=20)
    model.last_actions_embed = nn.Embedding(22, dim, padding_idx=21)
    model.decoder = TransformerDecoder(ModelArgs(dim=dim, n_layers=layers, n_heads=heads, vocab_size=dim, max_batch_size=1, max_seq_len=max_steps, multiple_of=256))
    model.actor = ActorHead(dim, 20)
    model.critic = CriticHead(dim)
    model.time_encoder = position(dim, max_len=max_steps)
    model.time_step_counter = 0
    model.max_steps = max_steps
    model.time_step_uuid = 'time_step'
    model.traj_idx_uuid = 'traj_index'
    model.an_object_is_in_hand_uuid = None
    model.multiple_beliefs = False
    model.use_linear_belief_adaptor = False
    model.auxiliary_uuids = None
    model.critic_type = 'linear'
    model.forward_encoder = lambda obs: (obs['fixed_embedding'], None)
    model.actor_cross_episode_attention_mask = True
    return model.eval()


def fixed_input(embedding, timestep, episode=0, previous=3):
    return dict(observations={'fixed_embedding':embedding, 'time_step':torch.tensor([[timestep]]),
                              'traj_index':torch.tensor([[episode]])}, memory=None,
                prev_actions=torch.tensor([[previous]]), masks=torch.tensor([[[float(timestep != 0)]]]))


def cache(model):
    return [(l.attention.cache_k.clone(), l.attention.cache_v.clone()) for l in model.decoder.layers]
