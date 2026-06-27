import torch
from torch.nn import functional as F
from torch import nn
from transformers import PreTrainedModel
from torch import Tensor
from typing import Optional, Tuple
import math
import pdb

class VSVLayer(nn.Module):
    def __init__(self, vsv, lam, simple_mode=False):
        super(VSVLayer, self).__init__()
        self.vsv = vsv  # (1, 4096)
        self.lam = lam
        self.simple_mode = simple_mode
    def forward(self, x):
        if self.vsv is not None:
            x = x.float()
            original_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
            y = 0
            if self.simple_mode:
                vsv = self.vsv[0]
                lam_schedule = self.lam[0]
                y = lam_schedule * F.normalize(vsv, dim=-1).repeat(1,x.shape[1],1) 
                x = F.normalize(F.normalize(x, p=2, dim=-1) + y, p=2, dim=-1) * original_norm
            else:
                for i in range(len(self.vsv)):
                    lambda_sim = 1.0 + torch.max(torch.tensor([0.]).to(x.device), F.cosine_similarity(x, -self.vsv[i][None,None,:], dim=-1)).unsqueeze(-1)
                    y += self.lam[i] * lambda_sim * F.normalize(self.vsv[i], dim=-1).repeat(1,x.shape[1],1)
                y = y/len(self.vsv)
                x = F.normalize(F.normalize(x.float(), p=2, dim=-1) + y, p=2, dim=-1) * original_norm
            return x.half()
        else:
            return x

def get_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

def set_nested_attr(obj, attr_path, value):
    attrs = attr_path.split(".")
    parent = get_nested_attr(obj, ".".join(attrs[:-1]))
    setattr(parent, attrs[-1], value)

def find_longest_modulelist(model, path=""):
    """
    Recursively find the longest nn.ModuleList in a PyTorch model.
    Args:
        model: PyTorch model.
        path: Current path in the model (used for recursion).
    Returns:
        Tuple with path and length of the longest nn.ModuleList found.
    """
    longest_path = path
    longest_len = 0

    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name

        # Recursively check the child's children
        child_path, child_len = find_longest_modulelist(child, f"{path}.{name}" if path else name)
        if child_len > longest_len:
            longest_len = child_len
            longest_path = child_path

    return longest_path, longest_len

def find_module(block, keywords):
    """
    Try to find a module in a transformer block.
    Args:
        block: Transformer block (nn.Module).
        keywords: List of possible module names (str).
    Returns:
        The found module if found, else None.
    """
    for name, module in block.named_modules():
        if any(keyword in name for keyword in keywords):
            # print(f"Found module {name}")
            return module
    submodule_names = [name for name, _ in block.named_modules()]
    raise ValueError(f"Could not find keywords {keywords} in: {submodule_names}")

def get_embedding_layer(model: PreTrainedModel):
    keywords = ["emb", "wte"]
    return find_module(model, keywords)

def get_lm_head(model: PreTrainedModel):
    keywords = ["lm_head", "embed_out"]
    return find_module(model, keywords)

def get_lm_pipeline(model: PreTrainedModel):
    model_class = model.__class__.__name__

    if model_class == "LlamaForCausalLM":
        return nn.Sequential(model.model.norm, model.lm_head)
    elif model_class == "RWForCausalLM":
        return nn.Sequential(model.transformer.ln_f, model.lm_head)
    elif model_class == "GPTNeoForCausalLM":
        return nn.Sequential(model.transformer.ln_f, model.lm_head)
    elif model_class == "GPTNeoXForCausalLM":
        return nn.Sequential(model.gpt_neox.final_layer_norm, model.embed_out)
    return get_lm_head(model)

def get_layers_path(model: PreTrainedModel):
    longest_path, longest_len = find_longest_modulelist(model)
    return longest_path

def get_layers(model: PreTrainedModel):
    longest_path = get_layers_path(model)
    return get_nested_attr(model, longest_path)

def get_mlp_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    mlp_layers = [find_module(layer, mlp_keywords) for layer in layers]
    return mlp_layers

def add_vsv_layers(model: PreTrainedModel, vsv: Tensor, alpha: list, tar_layers=None):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    assert len(vsv) == len(layers)
    if tar_layers is not None:
        tar_layers = tar_layers.split(",")
        if len(tar_layers) == 2:
            s_idx, e_idx = int(tar_layers[0]), int(tar_layers[1])
            layers = layers[s_idx:e_idx]
            vsv = vsv[s_idx:e_idx]
        elif len(tar_layers) == 4:
            s_s_idx, s_e_idx, t_s_idx, t_e_idx = tuple(map(int, tar_layers))
            layers = layers[s_s_idx:s_e_idx]
            vsv = vsv[t_s_idx:t_e_idx]
        else:
            raise ValueError("Invalid target layers")
    for i, layer in enumerate(layers):
        original_mlp = find_module(layer, mlp_keywords)
        layer.mlp = nn.Sequential(original_mlp, VSVLayer(vsv[i], alpha))

def remove_vsv_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"] 
    for i, layer in enumerate(layers):
        tar_mlp = find_module(layer, mlp_keywords)
        if isinstance(tar_mlp, nn.Sequential):
            layer.mlp = tar_mlp[0]
        else:
            pass

def add_vsv_layers_to_attn(model: PreTrainedModel, model_name: str, vsv: Tensor, alpha: list, tar_layers=None):
    layers = get_layers(model)
    attn_keywords = ["self_attn"]
    target_attribute_name = 'self_attn'

    if tar_layers is not None:
        if isinstance(tar_layers, str):
            tar_layers_str = tar_layers.split(",")
            s_idx, e_idx = int(tar_layers_str[0]), int(tar_layers_str[1])
            layers = layers[s_idx:e_idx]
            vsv = vsv[s_idx:e_idx]
        elif isinstance(tar_layers, list):
            layers = [layers[i] for i in tar_layers]
            vsv = vsv[tar_layers]

    for i, layer in enumerate(layers):
        original_attn = find_module(layer, attn_keywords)
        if len(alpha) == len(layers) and len(alpha) > 1:
            current_alpha = [alpha[i]] 
        else:
            current_alpha = alpha
        vsv_layer = VSVLayer(vsv[i], current_alpha)
        wrapped_module = AttentionWrapperWithVSV(model_name, original_attn, vsv_layer)
        setattr(layer, target_attribute_name, wrapped_module)

def remove_vsv_layers_from_attn(model: PreTrainedModel):
    layers = get_layers(model)
    attn_keywords = ["self_attn"]
    target_attribute_name = 'self_attn'
    for i, layer in enumerate(layers):
        tar_attn = find_module(layer, attn_keywords)
        if isinstance(tar_attn, AttentionWrapperWithVSV):
            original_module = tar_attn.original_attn
            setattr(layer, target_attribute_name, original_module)
        else:
            pass


class AttentionWrapperWithVSV(nn.Module):
    def __init__(self, model_name, original_attn_module, vsv_layer):
        super().__init__()
        self.model_name = model_name
        self.original_attn = original_attn_module
        self.vsv_layer = vsv_layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if self.model_name == "qwen":
            attn_output, attn_weights = self.original_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
            )
            steered_attn_output = self.vsv_layer(attn_output)             
            return steered_attn_output, attn_weights
        elif self.model_name == "llava":
            attn_output, attn_weights, present_key_value = self.original_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )       
            steered_attn_output = self.vsv_layer(attn_output)             
            return steered_attn_output, attn_weights, present_key_value