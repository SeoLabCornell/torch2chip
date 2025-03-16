"""
T2C version SmoothQuant

Original implementation: https://github.com/mit-han-lab/smoothquant/tree/main
Paper: https://arxiv.org/pdf/2211.10438.pdf
"""

import torch

from src.module.base import _QBase, round_ste
from src.module.fuse import MulShift
from src.quantization.minmax import MinMaxObserver, MinMaxTokenWiseObserver, MinMaxChannelWiseWeightObserver
from src.quantization.observer import BaseChannelWiseObserver
from src.quantization.mxint import MXChannelWiseWeightQuantizer


class SmoothQuantizer(_QBase):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = True):
        super().__init__(nbit, train_flag, unsigned)

        # observer
        self.observer = MinMaxObserver(nbit=self.nbit, unsigned=self.unsigned)

    def q(self, x:torch.Tensor):
        # go through the observer
        delta, zero_point = self.observer(x)

        self.scale.data = delta
        self.zero_point.data = zero_point

        xr = round_ste(x / self.scale) + self.zero_point
        xq = torch.clamp(xr, min=self.qlb, max=self.qub)
        xdq = xq.sub(self.zero_point)

        # dequantize
        if self.dequantize:
            xdq = xdq.mul(self.scale)

        return xdq.to(x.dtype)
    
    def trainFunc(self, input: torch.Tensor):
        xdq = self.q(input)
        return xdq
    
    def evalFunc(self, input: torch.Tensor):
        xdq = self.q(input)
        return xdq


class SmoothQuantChannelWiseWeightQuantizer(SmoothQuantizer):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = False, num_channels:int = 1):
        self.num_channels = num_channels
        super().__init__(nbit, train_flag, unsigned)

        # observer
        self.observer = MinMaxChannelWiseWeightObserver(nbit=self.nbit, unsigned=unsigned)
        self.register_qparams()

    def register_qparams(self):
        self.register_buffer("scale", torch.ones(self.num_channels, 1))
        self.register_buffer("zero_point", torch.zeros(self.num_channels, 1))


class SmoothQuantTokenWiseQuantizer(SmoothQuantizer):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = True, num_tokens: int = 197):
        self.num_tokens = num_tokens
        super().__init__(nbit, train_flag, unsigned)

        # observer
        self.observer = MinMaxTokenWiseObserver(nbit=self.nbit, unsigned=unsigned)
        self.register_qparams()

    def sync_tokens(self):
        if self.num_tokens != self.observer.num_tokens:
            self.observer.num_tokens = self.num_tokens
    
    def update_qparam(self, input:torch.Tensor):
        if len(input.shape) == 4:
            if input.shape[2] != self.num_tokens:
                self.num_tokens = input.shape[2]
                self.register_qparams()
                self.observer.register_range()

        elif len(input.shape) == 3:
            if input.shape[1] != self.num_tokens:
                self.num_tokens = input.shape[1]
                self.register_qparams()
                self.observer.register_range()
        
        self.sync_tokens()

    def register_qparams(self):
        self.register_buffer("scale", torch.ones(1, self.num_tokens, 1))
        self.register_buffer("zero_point", torch.zeros(1, self.num_tokens, 1))

    def trainFunc(self, input: torch.Tensor):
        self.update_qparam(input)

        return super().trainFunc(input)

class ChannelWiseObserver(BaseChannelWiseObserver):
    def __init__(self, nbit: int, unsigned: bool = True, num_channels: int = 1):
        super().__init__(nbit, unsigned, num_channels)

        self.sbit = 4
        self.slb = 0
        self.sub = 2**(self.sbit) - 1

        self.ibit = int(self.nbit - self.sbit)
        self.ilb = -2**(self.ibit-1)
        self.iub = 2**(self.ibit-1) - 1

    @torch.no_grad
    def get_channel_scale(self, x:torch.Tensor):
        super().get_bound(x)

        max_val_pos = torch.max(-self.lb, self.ub)
        sscale = 2 * max_val_pos / self.sub
        zero_point = self.slb - torch.round(self.lb / sscale)

        shared_scale = x.div(sscale).floor().add(zero_point)
        return shared_scale.abs().clamp(self.slb+1, self.sub)
    
    def calculate_qparam(self, x:torch.Tensor):
        max_val_pos = torch.max(-self.lb, self.ub)
        scale = max_val_pos / (float(self.iub - self.ilb) / 2)
        zero_point = torch.zeros(max_val_pos.size(), dtype=x.dtype, device=max_val_pos.device)

        full_scale = max_val_pos / (float(self.qub - self.qlb) / 2)

        return scale, zero_point, full_scale
    
    def forward(self, x:torch.Tensor):
        shared_scale = self.get_channel_scale(x)
        scale, zero_point, full_scale = self.calculate_qparam(x)
        return scale, full_scale, zero_point, shared_scale


class SmoothQuantChannelWeight4Bit(SmoothQuantizer):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = True, num_channels:int = 1):
        self.num_channels = num_channels
        super().__init__(nbit, train_flag, unsigned)
    
        # smoother
        self.smoother = MulShift()

        # observer
        self.observer = ChannelWiseObserver(nbit=self.nbit, unsigned=unsigned)
        self.register_qparams()

        # update the compressed bit precision
        self.nbit = self.nbit - self.observer.sbit

    def register_qparams(self):
        self.register_buffer("scale", torch.ones(self.num_channels, 1))
        self.register_buffer("full_scale", torch.ones(self.num_channels, 1))
        self.register_buffer("zero_point", torch.zeros(self.num_channels, 1))
        self.register_buffer("shared_scale", torch.ones(self.num_channels, 1))

    def q(self, x:torch.Tensor):
        if self.train_flag:
            # go through the observer
            delta, full_scale, zero_point, shared_scale = self.observer(x)

            self.scale.data = delta
            self.full_scale.data = full_scale
            self.zero_point.data = zero_point 
            self.shared_scale = shared_scale

        xr = round_ste(x / self.scale) + self.zero_point
        xr = xr.mul(self.shared_scale)
        xq = torch.clamp(xr, min=self.qlb, max=self.qub)

        xdq = xq.sub(self.zero_point)

        # dequantize
        if self.dequantize:
            xdq = xdq.mul(self.full_scale)

        return xdq
    
    def trainFunc(self, input: torch.Tensor):
        xdq = self.q(input)
        return xdq
    
    def evalFunc(self, input: torch.Tensor):
        xdq = self.q(input)
        return xdq
    
class SmoothQuantMXINTChannelWise(MXChannelWiseWeightQuantizer):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = False, block_size: int = 16):
        super().__init__(nbit, train_flag, unsigned, block_size)