"""
Observer of the high precision floating point distribution
"""

import torch
import torch.nn as nn

class BaseObserver(nn.Module):
    def __init__(self, nbit:int, unsigned:bool=True):
        super().__init__()

        self.nbit = nbit
        self.unsigned = unsigned
        self.initialize = True

        # quantization
        if self.unsigned:
            self.qlb = 0
            self.qub = 2 ** self.nbit - 1
        else:
            self.qlb = -2**(self.nbit-1)
            self.qub = 2**(self.nbit-1) - 1

        # initialize the floating point boundaries
        self.register_range()

    def register_range(self):
        # register buffer for the floating point range
        self.register_buffer("lb", torch.tensor(float("-inf")))
        self.register_buffer("ub", torch.tensor(float("inf")))

    def get_bound(self, x:torch.Tensor):
        min_val = x.min()
        max_val = x.max()

        if self.initialize:
            
            self.lb.data = min_val
            self.ub.data = max_val
            
            self.initialize = False
        else:
            lb = torch.min(self.lb, min_val)
            ub = torch.max(self.ub, max_val)

            # update bound
            self.lb.copy_(lb)
            self.ub.copy_(ub)

    def calculate_qparam(self, x:torch.Tensor):

        if self.unsigned:
            scale = (self.ub - self.lb) / (self.qub - self.qlb)
            zero_point = self.qlb - torch.round(self.lb / scale)
        else:
            max_val_pos = torch.max(-self.lb, self.ub)
            scale = max_val_pos / (float(self.qub - self.qlb) / 2)
            zero_point = torch.zeros(max_val_pos.size(), dtype=torch.float, device=max_val_pos.device)
        
        return scale, zero_point
    
    def forward(self, x:torch.Tensor):
        self.get_bound(x)
        scale, zero_point = self.calculate_qparam(x)
        return scale, zero_point


class BaseChannelWiseObserver(BaseObserver):
    def __init__(self, nbit: int, unsigned: bool = True, num_channels:int=1):
        self.num_channels = num_channels
        super().__init__(nbit, unsigned)

        # register the upper and lower bound
        self.register_range()
        
    def register_range(self):
        # register buffer for the floating point range
        self.register_buffer("lb", torch.ones(self.num_channels, 1).mul(float("-inf")))
        self.register_buffer("ub", torch.ones(self.num_channels, 1).mul(float("inf")))

    def reshape(self, x):
        xr = x.reshape(-1, x.shape[-1])
        return xr

    def get_bound(self, x:torch.Tensor):
        xr = self.reshape(x)
        
        min_val = xr.min(dim=1, keepdim=True)[0]
        max_val = xr.max(dim=1, keepdim=True)[0]

        if self.initialize:
            self.lb.data = min_val
            self.ub.data = max_val
            
            self.initialize = False
        else:
            lb = torch.min(self.lb, min_val)
            ub = torch.max(self.ub, max_val)

            # update bound
            self.lb.copy_(lb)
            self.ub.copy_(ub)


class BaseTokenWiseObserver(BaseObserver):
    def __init__(self, nbit: int, unsigned: bool = True, num_tokens:int=197):
        # number of channels
        self.num_tokens = num_tokens
        super().__init__(nbit, unsigned)

        self.register_range()

    def register_range(self):
        # register buffer for the floating point range
        self.register_buffer("lb", torch.ones(1, self.num_tokens, 1).mul(float("-inf")))
        self.register_buffer("ub", torch.ones(1, self.num_tokens, 1).mul(float("inf")))

    def get_bound(self, x:torch.Tensor):
        # x = x.view(self.num_tokens, -1)
        x = x.reshape(self.num_tokens, -1)


        min_val = x.min(dim=1, keepdim=True)[0]
        max_val = x.max(dim=1, keepdim=True)[0]

        if self.initialize:
            self.lb.data = min_val.unsqueeze(0)
            self.ub.data = max_val.unsqueeze(0)
            
            self.initialize = False
        else:
            lb = torch.min(self.lb, min_val)
            ub = torch.max(self.ub, max_val)

            # update bound
            self.lb.copy_(lb)
            self.ub.copy_(ub)

def lp_loss(pred, target, p=2.0, reduction='none'):
    """
    loss function measured in lp norm
    """
    if reduction == 'none':
        return (pred-target).abs().pow(p).sum(1).mean()
    else:
        return (pred-target).abs().pow(p).mean()