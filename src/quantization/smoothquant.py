"""
T2C Version of SmoothQuant
"""

import torch
import torch.nn as nn
from src.module.base import _QBase
from src.quantization.observer import BaseTokenWiseObserver, BaseObserver, BaseChannelWiseObserver

class SmoothQChannelObserver(BaseChannelWiseObserver):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = True):
        super().__init__(nbit, train_flag, unsigned)

    def reshape(self, x:torch.Tensor):
        """
        Reshape designed for transformer
        """
        hidden_dim = x.size(-1)
        xr = x.view(-1, hidden_dim).detach()
        return xr
    
    @torch.no_grad
    def get_bound(self, x: torch.Tensor):
        xr = self.reshape(x)

        # get the maximum value
        max_val = xr.abs().max(xr, dim=0)[0].float()
        
        if self.initialize:
            self.lb.data = -max_val
            self.ub.data = max_val

            self.initialize = False
        else:
            ub = torch.max(self.ub, max_val)

            self.lb.copy_(-ub)
            self.ub.copy_(ub)

class SmoothQTokenObserver(BaseTokenWiseObserver):
    def __init__(self, nbit: int, unsigned: bool = True, num_tokens: int = 197):
        super().__init__(nbit, unsigned, num_tokens)
        
    @torch.no_grad
    def get_bound(self, x: torch.Tensor):
        x = x.view(self.num_tokens, -1)

        max_val = x.abs().max(dim=1, keep_dim=True)[0]

        if self.initialize:
            self.ub.data = max_val.unsqueeze(0)
            self.lb.data = max_val.mul(-1.0).unsqueeze(0)
        
        else:
            ub = torch.max(self.ub, max_val)

            # update boundary
            self.lb.copy_(ub.mul(-1.0))
            self.ub.copy_(ub)

    def calculate_qparam(self, x):
        return super().calculate_qparam(x)


class SmoothQTensorObserver(BaseObserver):
    def __init__(self, nbit: int, unsigned: bool = False):
        super().__init__(nbit, unsigned)

    @torch.no_grad
    def get_bound(self, x:torch.Tensor):
        max_val = x.abs().max()

        if self.initialize:
            self.ub.data = max_val
            self.lb.data = max_val.mul(-1.0)

        else:
            ub = torch.max(self.ub, max_val)

            # update boundary
            self.lb.copy_(ub.mul(-1.0))
            self.ub.copy_(ub)

class SmoothQuantizer(_QBase):
    pass