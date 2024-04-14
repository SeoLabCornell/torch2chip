"""
MinMax observer and quantizer
"""

import torch
from src.quantization.observer import BaseObserver, BaseTokenWiseObserver, BaseChannelWiseObserver
from src.module.base import _QBase, round_ste

class MinMaxObserver(BaseObserver):
    def __init__(self, nbit: int, unsigned: bool = True):
        super().__init__(nbit, unsigned)

    @torch.no_grad
    def get_bound(self, x: torch.Tensor):
        return super().get_bound(x)

class MinMaxTokenWiseObserver(BaseTokenWiseObserver):
    def __init__(self, nbit: int, unsigned: bool = True, num_tokens: int = 197):
        super().__init__(nbit, unsigned, num_tokens)

    @torch.no_grad
    def get_bound(self, x: torch.Tensor):
        return super().get_bound(x)
    
class MinMaxChannelWiseWeightObserver(BaseChannelWiseObserver):
    def __init__(self, nbit: int, unsigned: bool = True, num_channels:int=1):
        super().__init__(nbit, unsigned, num_channels)

    def reshape(self, x):
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], -1)
        elif len(x.shape) == 2:
            x = x.reshape(-1, x.shape[-1])
        else:
            raise NotImplementedError(f"Non-supported Layer Type with the shape of {list(x.size())}!")
        return x
    
    def calculate_qparam(self, x):
        scale, zero_point = super().calculate_qparam(x)
        
        if len(x.shape) == 4:
            scale = scale.unsqueeze(2).unsqueeze(3)
            zero_point = zero_point.unsqueeze(2).unsqueeze(3)
        
        return scale, zero_point
    
    @torch.no_grad
    def get_bound(self, x: torch.Tensor):
        return super().get_bound(x)
    
class MinMaxChannelWiseActObserver(BaseChannelWiseObserver):
    def __init__(self, nbit: int, unsigned: bool = True, num_channels: int = 1):
        super().__init__(nbit, unsigned, num_channels)

    def reshape(self, x):
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1)
        
        x = x.reshape(x.shape[-1], -1)
        return x
    
    def calculate_qparam(self, x):
        scale, zero_point = super().calculate_qparam(x)
        
        if len(x.shape) == 4:
            scale = scale.unsqueeze(0).unsqueeze(3)
            zero_point = zero_point.unsqueeze(0).unsqueeze(3)
        elif len(x.shape) == 2:
            scale = scale.transpose(0,1)
            zero_point = zero_point.transpose(0,1)
        
        return scale, zero_point

class MinMaxQuantizer(_QBase):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = True):
        super().__init__(nbit, train_flag, unsigned)

        self.observer = MinMaxObserver(nbit=self.nbit, unsigned=self.unsigned)

    def q(self, x:torch.Tensor):
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

class MinMaxTokenWiseQuantizer(MinMaxQuantizer):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = True, num_tokens: int = 197):
        self.num_tokens = num_tokens
        super().__init__(nbit, train_flag, unsigned)

        # observer
        self.observer = MinMaxTokenWiseObserver(nbit=self.nbit, unsigned=self.unsigned, num_tokens=num_tokens)        
        self.register_qparams()
    
    def register_qparams(self):
        self.register_buffer("scale", torch.ones(1, self.num_tokens, 1))
        self.register_buffer("zero_point", torch.zeros(1, self.num_tokens, 1))

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

    def trainFunc(self, input: torch.Tensor):
        self.update_qparam(input)
        
        return super().trainFunc(input)

class MinMaxChannelWiseWeightQuantizer(MinMaxQuantizer):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = False, num_channels:int = 1):
        self.num_channels = num_channels
        super().__init__(nbit, train_flag, unsigned)

        # observer
        self.observer = MinMaxChannelWiseWeightObserver(nbit=self.nbit, unsigned=unsigned)

        # qparams
        self.register_qparams()

    def register_qparams(self):
        self.register_buffer("scale", torch.ones(self.num_channels, 1))
        self.register_buffer("zero_point", torch.zeros(self.num_channels, 1))

class MinMaxChannelWiseActQuantizer(MinMaxQuantizer):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = True, num_channels:int=1):
        self.num_channels = num_channels
        super().__init__(nbit, train_flag, unsigned)

        # observer
        self.observer = MinMaxChannelWiseActObserver(nbit=self.nbit, unsigned=unsigned)

        # qparams
        self.register_qparams()

    def register_qparams(self):
        self.register_buffer("scale", torch.ones(1, self.num_channels))
        self.register_buffer("zero_point", torch.zeros(1, self.num_channels))