"""
Loss functions
"""

import torch
from src.module.base import _QBaseConv2d, _QBaseLinear
from src.module.attention import QAttention
from timm.layers.mlp import Mlp

from typing import Union

def lp_loss(pred, tgt, p=2.0):
    """
    loss function
    """
    return (pred - tgt).abs().pow(p).sum(1).mean()

class LinearTempDecay:
    def __init__(self, t_max=20000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))

class AdaRoundLoss:
    def __init__(self, 
            module:Union[_QBaseConv2d, _QBaseLinear, QAttention, Mlp],
            weight:float = 1e-5,
            iters: int = 100,
            b_range: tuple = (4, 2),
            warmup:float = 0.2,
            p: float = 2.
        ):
        
        self.module = module
        self.weight = weight
        self.p = p

        # temperature decay
        self.temp_decay = LinearTempDecay(iters, warm_up=warmup, start_b=b_range[0], end_b=b_range[1])
        self.start = int(warmup * iters)

        self.steps = 0
        self.b = b_range[0]

    def step(self):
        self.b = self.temp_decay(self.steps)

    def attention(self, module:QAttention):
        rqkv = self.weight * (1 - ((module.qkv.wq.h() - .5).abs() * 2).pow(self.b)).sum()
        rproj = self.weight * (1 - ((module.proj.wq.h() - .5).abs() * 2).pow(self.b)).sum()
        return rqkv + rproj
    
    def mlp(self, module:Mlp):
        rfc1 = self.weight * (1 - ((module.fc1.wq.h() - .5).abs() * 2).pow(self.b)).sum()
        rfc2 = self.weight * (1 - ((module.fc2.wq.h() - .5).abs() * 2).pow(self.b)).sum()
        return rfc1 + rfc2
    
    def conv_linear(self, module: Union[_QBaseConv2d, _QBaseLinear]):
        rw = self.weight * (1 - ((module.wq.h() - .5).abs() * 2).pow(self.b)).sum()
        return rw
    
    def __call__(self, pred, target):
        # update b
        self.step()

        rec_loss = lp_loss(pred, target, p=self.p)

        if self.steps > self.start:
            if isinstance(self.module, QAttention):
                round_loss = self.attention(self.module)
            elif isinstance(self.module, Mlp):
                round_loss = self.mlp(self.module)
            elif isinstance(self.module, (_QBaseConv2d, _QBaseLinear)):
                round_loss = self.conv_linear(self.module)

            return rec_loss + round_loss
        else:
            return rec_loss
            
        
