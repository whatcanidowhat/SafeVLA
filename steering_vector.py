from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from llm_layers import get_layers, find_module
import pdb

import torchvision.transforms.functional as TF


@dataclass
class ResidualStream:
    hidden: torch.Tensor


class ForwardTrace:
    def __init__(self):
        self.residual_stream: Optional[ResidualStream] = ResidualStream(
            hidden=[],
        )
        self.attentions: Optional[torch.Tensor] = None


class ForwardTracer:
    def __init__(self, model: PreTrainedModel, forward_trace: ForwardTrace):
        self._model = model
        self._forward_trace = forward_trace
        self._layers = get_layers(model)
        self._hooks = []

    def __enter__(self):
        self._register_forward_hooks()


    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()

        if exc_type is None:
            residual_stream = self._forward_trace.residual_stream

            if residual_stream.hidden[0] == []:
                residual_stream.hidden.pop(0)

            for key in residual_stream.__dataclass_fields__.keys():
                acts = getattr(residual_stream, key)
                # TODO: this is a hack, fix it
                if key != "hidden" and not self._with_submodules:
                    continue

                nonempty_layer_acts = [layer_acts for layer_acts in acts if layer_acts != []][0]
                final_shape = torch.cat(nonempty_layer_acts, dim=0).shape
                for i, layer_acts in enumerate(acts):
                    if layer_acts == []:
                        acts[i] = torch.zeros(final_shape)
                    else:
                        acts[i] = torch.cat(layer_acts, dim=0)
                acts = torch.stack(acts).transpose(0, 1)
                setattr(residual_stream, key, acts)

    def _register_forward_hooks(self):
        model = self._model
        hooks = self._hooks

        residual_stream = self._forward_trace.residual_stream

        def store_activations(residual_stream: ResidualStream, acts_type: str, layer_num: int):
            def hook(model, inp, out):
                if isinstance(out, list):
                    out = out[0]

                out = out.float().to("cpu", non_blocking=True)

                acts = getattr(residual_stream, acts_type)
                while len(acts) < layer_num + 1:
                    acts.append([])
                try:
                    acts[layer_num].append(out)
                except IndexError:
                    print(len(acts), layer_num)  
            return hook
        
        for i, layer in enumerate(self._layers):
            hidden_states_hook = layer.register_forward_hook(store_activations(residual_stream, "hidden", i + 1))
            hooks.append(hidden_states_hook)


def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v

class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_


def get_hiddenstates(model, kwargs_list):
    h_all = []
    for example_id in range(len(kwargs_list)):
        embeddings_for_all_styles= []
        for style_id in range(len(kwargs_list[example_id])):
            forward_trace = ForwardTrace()
            context_manager = ForwardTracer(model, forward_trace)
            with context_manager:
                _ = model(
                use_cache=True,
                output_hidden_states=True,
                **kwargs_list[example_id][style_id],
                )
                h = forward_trace.residual_stream.hidden
            embedding_token = []
            for layer in range(len(h)):
                embedding_token.append(h[layer][:,-1])
            embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
            embeddings_for_all_styles.append(embedding_token)
        h_all.append(tuple(embeddings_for_all_styles))
    return h_all


def get_component_outputs(model, kwargs, component_keywords=["self_attn"]):
    """
    通过临时钩子，精确捕获指定层范围内特定子模块的输出。
    """
    layers = get_layers(model)
    component_outputs = []
    handles = [] # 用于存放钩子句柄，以便后续移除

    def hook_fn(module, input, output):
        # 注意力模块的输出可能是元组 (attn_output, attn_weights, ...)，我们只取第一个
        if isinstance(output, tuple):
            component_outputs.append(output[0].cpu())
        else:
            component_outputs.append(output.cpu())

    # 为所有层的目标子模块注册钩子
    for layer in layers:
        component_module = find_module(layer, component_keywords)
        handle = component_module.register_forward_hook(hook_fn)
        handles.append(handle)

    # 执行一次前向传播来触发钩子
    with torch.no_grad():
        model(**kwargs)

    # 立即移除所有钩子，避免对后续操作产生影响
    for handle in handles:
        handle.remove()
        
    # 将捕获到的所有层输出拼接起来
    return torch.cat([out[:, -1, :] for out in component_outputs], dim=0)
    # return torch.stack(component_outputs, dim=0).squeeze(1)

def get_v_proj_inputs(model, kwargs):
    """
    通过临时钩子，精确捕获每一层 v_proj 模块的输入 (即 hidden_states)。
    """
    layers = get_layers(model)
    captured_inputs = []
    handles = []

    # 定义钩子函数，用于捕获输入
    def hook_fn(module, input, output):
        # 对于线性层, input 是一个元组，第一个元素是输入的张量
        if isinstance(input, tuple) and len(input) > 0:
            captured_inputs.append(input[0].cpu())

    attn_keywords = ["self_attn", "attn", "attention"]
    v_proj_keywords = ["v_proj"]

    # 为所有层的 v_proj 子模块注册钩子
    for layer in layers:
        try:
            attn_module = find_module(layer, attn_keywords)
            v_proj_module = find_module(attn_module, v_proj_keywords)
            handle = v_proj_module.register_forward_hook(hook_fn)
            handles.append(handle)
        except (AttributeError, ValueError):
            continue

    # 执行一次前向传播来触发钩子
    with torch.no_grad():
        # 我们只需要进行一次模型调用，不需要完整的generate
        model(**kwargs, output_hidden_states=True)

    # 立即移除所有钩子
    for handle in handles:
        handle.remove()
    
    # 检查是否成功捕获
    if not captured_inputs:
        raise RuntimeError("Failed to capture any v_proj inputs. Check module names and hook registration.")

    # 将所有层的输入堆叠起来，形成 [num_layers, batch, seq_len, dim] 的形状
    # 然后提取最后一个token，并拼接成VSV计算所需的 [num_layers, dim] 形状
    # (假设batch_size=1)
    stacked_inputs = torch.stack(captured_inputs)
    return torch.cat([out[:, -1, :] for out in stacked_inputs], dim=0)


# 提取引导向量
def obtain_vsv(model, kwargs_list, rank=1):
    hidden_states = get_hiddenstates(model, kwargs_list) # each element, layer x len_tokens x dim [样本数, (负样本隐藏状态, 正样本隐藏状态)]
    num_demonstration = len(hidden_states) # 样本数
    neg_all = [] # 负样本隐藏状态
    pos_all = [] # 正样本隐藏状态
    hidden_states_all = [] # 正负隐藏状态差异
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
        neg_all.append(hidden_states[demonstration_id][0].view(-1))
        pos_all.append(hidden_states[demonstration_id][1].view(-1))
    fit_data = torch.stack(hidden_states_all) # 所有样本的视觉差异堆叠为张量 [样本数, 展平后的隐藏状态维度][1, 32x4096]
    neg_emb = torch.stack(neg_all).mean(0)
    pos_emb = torch.stack(pos_all).mean(0)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    # 计算最终VSV 融合PCA主成分（核心差异方向）与均值
    direction = (pca.components_.sum(dim=0,keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1)) # [模型层数, 隐藏层维度]
    return direction, (neg_emb).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))

