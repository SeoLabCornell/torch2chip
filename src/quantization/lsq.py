"""
Learned Step Size Quantization
"""

import torch
from src.module.base import _QBase, round_ste
from src.quantization.observer import BaseObserver, BaseTokenWiseObserver

def lp_loss(pred, target, p=2.0, reduction='none'):
    """
    loss function measured in lp norm
    """
    if reduction == 'none':
        return (pred-target).abs().pow(p).sum(1).mean()
    else:
        return (pred-target).abs().pow(p).mean()

class LSQObserver(BaseObserver):
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

class LSQTokenWiseObserver(BaseTokenWiseObserver):
    def __init__(self, nbit: int, unsigned: bool = True, num_tokens: int = 197):
        super().__init__(nbit, unsigned, num_tokens)

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

    def calculate_qparam(self, x):
        scale, zero_point = torch.ones(x.size(1), device=x.device), torch.zeros(x.size(1), device=x.device)
        
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

class LSQ(_QBase):
    def __init__(self, nbit: int = 8, train_flag: bool = True, unsigned: bool = True):
        super().__init__(nbit, train_flag, unsigned)
        self.train_flag = train_flag
        self.unsigned = unsigned
        
        # initialization flag
        self.initialize = False

        # observer
        self.observer = LSQObserver(nbit=self.nbit, unsigned=self.unsigned)

        # register q parameters
        self.register_qparams()

    def register_qparams(self):
        super().register_qparams()
        # register learnable parameter 
        self.register_parameter("delta", torch.nn.Parameter(torch.tensor(1.0)))

    def q(self, x:torch.Tensor):
        if not self.initialize:
            if self.train_flag:
                with torch.no_grad():
                    print("initialize q params")
                    delta, zero_point = self.observer(x)
                    self.delta.data = delta
                    self.zero_point.data = zero_point
                    self.initialize = True

        # quantize
        xr = round_ste(x / self.delta) + self.zero_point
        xq = torch.clamp(xr, min=self.qlb, max=self.qub)

        # dequantize
        xdq = xq.sub(self.zero_point).clamp(self.qlb, self.qub)
        return xdq.mul(self.delta)
    
    def trainFunc(self, input: torch.Tensor):
        xdq = self.q(input)

        # update the buffer
        self.scale.data = 1 / self.delta.data
        return xdq
    
    def evalFunc(self, input: torch.Tensor):
        xr = round_ste(input * self.scale) + self.zero_point
        xq = torch.clamp(xr, min=self.qlb, max=self.qub)
        xdq = (xq - self.zero_point).clamp(self.qlb, self.qub)

        if self.dequantize:
            xdq = xdq.div(self.scale)

        return xdq
    
    def extra_repr(self) -> str:
        return super().extra_repr() + f", delta={self.delta.data.item():.2e}"


class LSQTokenWise(LSQ):
    def __init__(self, nbit: int = 8, train_flag: bool = True, unsigned: bool = True, num_tokens: int = 197):
        self.num_tokens = num_tokens
        super().__init__(nbit, train_flag, unsigned)

        self.observer = LSQTokenWiseObserver(nbit=self.nbit, unsigned=self.unsigned)

        # register q parameters
        self.register_qparams()
    
    def register_qparams(self):
        super().register_qparams()
        # register learnable parameter 
        self.register_parameter("delta", torch.nn.Parameter(torch.tensor(1.0)))
