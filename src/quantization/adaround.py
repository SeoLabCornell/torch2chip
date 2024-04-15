"""
Adaptive Round
"""

import torch
from src.module.base import _QBase

class AdaRound(_QBase):
    """
    Weight quantizer AdaRound: "Up or Down? Adaptive Rounding for Post-Training Quantization"
    https://arxiv.org/abs/2004.10568
    """
    def __init__(self, nbit: int, train_flag: bool = True, weights: torch.Tensor=None, unsigned=False):
        super().__init__(nbit, train_flag, unsigned)
        self.iter = 0

        self.register_buffer("lb", weights.min())
        self.register_buffer("ub", weights.max())

        # integer boundary
        self.qlb = -2**(self.nbit-1)
        self.qub = 2**(self.nbit-1)-1

        # initialize the alpha
        self.init_flag = True

        # parameters
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3

        # register the learnable parameters
        self.register_alpha(weights)

    def register_alpha(self, x:torch.Tensor):
        xfloor = x.div(self.scale).floor()

        # compute alpha
        diff = x.div(self.scale).sub(xfloor)
        alpha = -torch.log((self.zeta - self.gamma) / (diff - self.gamma) - 1)
        self.register_parameter("alpha", torch.nn.Parameter(alpha))

    def get_qparam(self, x:torch.Tensor):
        lb = torch.min(self.lb, x.min())
        ub = torch.max(self.lb, x.max())

        # update boundary 
        self.lb = lb.clone()
        self.ub = ub.clone()

    def h(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    
    def q(self, x:torch.Tensor):
        # scale = self.ub.sub(self.lb).div(self.qub - self.qlb)
        scale = (self.qub - self.qlb) / self.ub.sub(self.lb)
        zero_point = torch.tensor(0.0)

        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)

        if self.init_flag:
            self.register_alpha(x)
            self.init_flag = False
        
        # quantization
        xfloor = x.mul(self.scale).floor()
        import pdb;pdb.set_trace()
        soft_shift = self.h()

        # quantize
        if self.train_flag or self.training:
            xada = xfloor + soft_shift
        else:
            xada = xfloor + self.alpha.ge(0.0).float()

        xq = xada + self.zero_point
        
        # integer representation
        output = torch.clamp(xq, self.qlb, self.qub).sub(self.zero_point)

        # dequantize
        if self.dequantize:
            output = output.div(self.scale)
        return output
    
    def trainFunc(self, input: torch.Tensor):
        self.get_qparam(input)
        xq = self.q(input)
        return xq

    def evalFunc(self, input: torch.Tensor):
        xq = self.q(input)
        return xq
