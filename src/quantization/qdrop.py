"""
T2C version of QDrop

https://openreview.net/forum?id=ySQH0oDyp7
"""

import torch
from src.quantization.lsq import LSQ, LSQTokenWise

class QDrop(LSQ):
    def __init__(self, nbit: int = 8, train_flag: bool = True, unsigned: bool = True, drop_prob:float=0.5):
        super().__init__(nbit, train_flag, unsigned)
        self.drop_prob = drop_prob

    def forward(self, input:torch.Tensor):
        xorg = input
        y = super().forward(input)

        if self.drop_prob < 1.0 and self.training:
            x_prob = torch.where(torch.rand_like(input) < self.drop_prob, y, xorg)
            return x_prob
        return y

class QDropTokenWise(LSQTokenWise):
    def __init__(self, nbit: int = 8, train_flag: bool = True, unsigned: bool = True, drop_prob:float=0.5):
        super().__init__(nbit, train_flag, unsigned)
        self.drop_prob=0.5

    def forward(self, input:torch.Tensor):
        xorg = input
        y = super().forward(input)

        if self.drop_prob < 1.0 and self.training:
            x_prob = torch.where(torch.rand_like(input) < self.drop_prob, y, xorg)
            return x_prob
        return y