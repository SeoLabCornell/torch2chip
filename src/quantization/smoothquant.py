"""
T2C version SmoothQuant

Original implementation: https://github.com/mit-han-lab/smoothquant/tree/main
Paper: https://arxiv.org/pdf/2211.10438.pdf
"""

import torch

from src.module.base import _QBase, round_ste
from src.module.fuse import MulShift
from src.quantization.minmax import MinMaxObserver, MinMaxTokenWiseObserver, MinMaxChannelWiseWeightObserver


class SmoothQuantizer(_QBase):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = True):
        super().__init__(nbit, train_flag, unsigned)

        # smoother
        self.smoother = MulShift()

        # observer
        self.observer = MinMaxObserver(nbit=self.nbit, unsigned=self.unsigned)

    def q(self, x:torch.Tensor):
        # smooth out the distribution
        x = self.smoother(x)
        
        if self.train_flag:
            # go through the observer
            delta, zero_point = self.observer(x)

            self.scale.data = 1 / delta
            self.zero_point.data = zero_point
        
        xr = round_ste(x * self.scale) + self.zero_point
        xq = torch.clamp(xr, min=self.qlb, max=self.qub)
        xdq = xq.sub(self.zero_point)

        # dequantize
        if self.dequantize:
            xdq = xdq.div(self.scale)
        
        return xdq
    
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

        # smoother
        self.smoother = MulShift()

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

        # smoother
        self.smoother = MulShift()

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
