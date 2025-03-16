import torch
import torch.nn as nn
import torch.nn.functional as F

from src.module.base import _QBaseLinear
from src.models.lm.retnet import get_activation_fn
from transformers.activations import ACT2FN

class QLlamaMLP(nn.Module):
    def __init__(self, config, rescale_out:bool=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = _QBaseLinear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias, rescale_out=rescale_out)
        self.up_proj = _QBaseLinear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias, rescale_out=rescale_out)
        self.down_proj = _QBaseLinear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias, rescale_out=rescale_out)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
    
class QGLU(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
        rescale_out:bool=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = _QBaseLinear(self.embed_dim, ffn_dim, bias=False, rescale_out=rescale_out)
        self.fc2 = _QBaseLinear(ffn_dim, self.embed_dim, bias=False, rescale_out=rescale_out)
        self.gate = _QBaseLinear(self.embed_dim, ffn_dim, bias=False, rescale_out=rescale_out)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.gate.reset_parameters()

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        g = self.gate(x)
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x) * g
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = x.view(x_shape)
        x = self.dropout_module(x)
        return x
