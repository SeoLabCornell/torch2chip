"""
Adaptive Round
"""

import torch
from src.module.base import _QBase
from src.quantization.observer import BaseObserver, lp_loss

class AdaRoundObserver(BaseObserver):
    def __init__(self, nbit: int, unsigned: bool = True):
        super().__init__(nbit, unsigned)
        
    def quantize(self, x:torch.Tensor, xmin, xmax):
        delta = (xmax - xmin) / (self.qub - self.qlb)

        if self.unsigned:
            zero_point = self.qlb - torch.round(xmin / delta)
        else:
            zero_point = torch.tensor(0.0)

        xint = torch.round(x / delta)
        xq = torch.clamp(xint - zero_point, self.qlb, self.qub)
        xdq = (xq + zero_point) * delta
        return xdq, delta, zero_point

    def calculate_qparam(self, x: torch.Tensor):
        # update the quantization boundary
        self.get_bound(x)

        # quantization parameters
        scale, zero_point = torch.tensor(1.0), torch.tensor(0.0)
        
        best_loss = 1e+10
        for i in range(100):
            new_min = self.lb * (1.0 - (i * 0.01))
            new_max = self.ub * (1.0 - (i * 0.01))

            # quantize and dequantize for mse 
            xdq, new_scale, new_zp = self.quantize(x, new_min, new_max)
            loss = lp_loss(xdq, x, p=2.4, reduction='all')

            if loss < best_loss:
                best_loss = loss
                scale, zero_point = new_scale, new_zp
        
        self.lb.data = new_min
        self.ub.data = new_max

        return scale, zero_point


class AdaRound(_QBase):
    """
    Weight quantizer AdaRound: "Up or Down? Adaptive Rounding for Post-Training Quantization"
    https://arxiv.org/abs/2004.10568
    """
    def __init__(self, nbit: int, train_flag: bool = True, weights: torch.Tensor=None, unsigned=False):
        super().__init__(nbit, train_flag, unsigned)
        self.iter = 0

        # initialize the alpha
        self.init_flag = True

        # parameters
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3

        # define the observer
        self.observer = AdaRoundObserver(nbit=self.nbit, unsigned=self.unsigned)
    
        # register the learnable parameters
        self.register_alpha(weights)

    def register_alpha(self, x:torch.Tensor):
        self.register_buffer("delta", torch.tensor(1.0))

        delta, zp = self.observer(x)

        self.delta.copy_(delta)
        self.scale.copy_(1 / delta)
        self.zero_point.copy_(zp)

        # find the optimal scaling factor first
        xfloor = x.div(self.delta).floor()

        # compute alpha
        diff = x.div(self.delta).sub(xfloor)
        alpha = -torch.log((self.zeta - self.gamma) / (diff - self.gamma) - 1)
        self.register_parameter("alpha", torch.nn.Parameter(alpha))

    def h(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    
    def q(self, x:torch.Tensor):
        # quantization
        xfloor = x.mul(self.scale).floor()
        soft_shift = self.h()

        # quantize
        if self.train_flag or self.training:
            xada = xfloor + soft_shift
        else:
            xada = xfloor + self.alpha.ge(0.0).float()

        xq = xada + self.zero_point
        # integer representation
        output = torch.clamp(xq, self.observer.qlb, self.observer.qub).sub(self.zero_point)

        # dequantize
        if self.dequantize:
            output = output.div(self.scale)
        return output
    
    def trainFunc(self, input: torch.Tensor):
        xq = self.q(input)
        return xq

    def evalFunc(self, input: torch.Tensor):
        xq = self.q(input)
        return xq
