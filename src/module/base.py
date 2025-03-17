"""
Basic Modules for Low precision and Sparsity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.quantization.observer import BaseObserver, BaseChannelWiseObserver, BaseTokenWiseObserver
from src.module.ops import IntActWeight, FloatActWeight

class ConvOPS(nn.Module):
    def __init__(self, stride:int=1, padding:int=0, dilation:int=1, groups:int=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x:torch.Tensor, y:torch.Tensor, b:torch.Tensor):
        z = F.conv2d(x, y, b, self.stride, self.padding, self.dilation, self.groups)
        return z

class _QBase(nn.Module):
    r"""Base quantization method for weight and activation.

    Args:
    nbit (int): Data precision.
    train_flag (bool): Training mode. 

    Methods:
    trainFunc (input:Tensor): Training function of quantization-aware training (QAT)
    evalFunc (input:Tensor): Forward pass function of inference. 
    inference(): Switch to inference mode. 
    register_qparm(): Register the quantization parameters (scale and zero point) as buffer parameters
    """
    def __init__(self, nbit:int, train_flag:bool=True, unsigned:bool=True):
        super(_QBase, self).__init__()
        self.nbit = nbit
        self.train_flag = train_flag
        self.unsigned = unsigned

        # training flag
        self.train_flag = True
        self.dequantize = True

        # upper and lower bound
        if not self.unsigned:
            self.qlb = -2**(self.nbit-1)
            self.qub = 2**(self.nbit-1) - 1
        else:
            self.qlb = 0
            self.qub = 2**(self.nbit) - 1

        self.register_qparams()
        self.observer = BaseObserver(nbit=self.nbit)

    def register_qparams(self):
        # register quantization scaler and zero point
        self.register_parameter("scale", torch.nn.Parameter(torch.tensor(1.0), requires_grad=False))
        self.register_buffer("zero_point", torch.tensor(0.0))

    def q(self, x:torch.Tensor):
        """
        Quantization operation
        """
        return x
    
    def trainFunc(self, x:torch.Tensor):
        """
        Training path (QAT and PTQ) with quantize + dequantize
        """
        out = self.q(x)
        return out
    
    def evalFunc(self, x:torch.Tensor):
        """
        Evaluation path with integer-only operation only
        """
        return self.trainFunc(x)

    def inference(self):
        """
        Training / Evaluation Enable flag
        """
        self.train_flag = False
        self.dequantize = False

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.train_flag:
            y = self.trainFunc(x)
        else:
            y = self.evalFunc(x)
        return y

    def extra_repr(self) -> str:
        return super().extra_repr() + "nbit={}".format(self.nbit)


class _QBaseConv2d(nn.Conv2d):
    r"""
    Basic low precision convolutional layer

    Inherited from the base nn.Conv2d layer.
    
    Args:
    wbit (int): Weight quantization precision. 
    abit (int): Input quantization precision.
    wq (_QBase): Weight quantizer. 
    aq (_QBase): Activation quantizer.   
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, 
                padding:int=0, dilation:int=1, groups:int=1, bias:bool=True, wbit:int=32, abit:int=32, train_flag=True):
        super(_QBaseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.train_flag = train_flag

        self.wbit = wbit
        self.abit = abit

        # quantizer
        self.wq = _QBase(nbit=wbit)
        self.aq = _QBase(nbit=abit)
        self.yq = _QBase(nbit=abit)

        # masks
        self.register_buffer("mask", torch.ones_like(self.weight))

    def inference(self):
        """
        Training / Evaluation Enable flag
        """
        self.train_flag = False
        self.initialize_qweight = True

        self.wq.inference()
        self.aq.inference()

        # update ops
        self.ops = ConvOPS(self.stride, self.padding, self.dilation, groups=self.groups)

    def trainFunc(self, x:torch.Tensor):
        xq = self.aq(x)
        wq = self.wq(self.weight)
        
        y = F.conv2d(xq, wq.mul(self.mask), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y

    def evalFunc(self, x:torch.Tensor):
        xq = self.aq(x)
        
        if self.initialize_qweight:
            wq = self.wq(self.weight)
            self.weight.data = wq
            self.initialize_qweight = False

        y = self.ops(xq, self.weight, self.bias)
        return y

    def forward(self, x:torch.Tensor):
        if self.train_flag:
            y = self.trainFunc(x)
        else:
            y = self.evalFunc(x)

        y = self.yq(y)
        return y

class _QBaseLinear(nn.Linear):
    r"""Basic low precision linear layer

    Inherited from the base nn.Linear layer.
    
    Args:
    wbit (int): Weight quantization precision. 
    abit (int): Input quantization precision.
    wq (_QBase): Weight quantizer. 
    aq (_QBase): Activation quantizer.
    """
    def __init__(self, 
                in_features: int, 
                out_features: int, 
                bias: bool = True, 
                wbit:int=32, 
                abit:int=32, 
                train_flag=True, 
                rescale_out:bool=False
        ):
        super(_QBaseLinear, self).__init__(in_features, out_features, bias)
        self.train_flag = train_flag

        self.wbit = wbit
        self.abit = abit

        # quantizer
        self.wq = _QBase(nbit=wbit)
        self.aq = _QBase(nbit=abit)
        self.yq = _QBase(nbit=abit)

        # masks
        self.register_buffer("mask", torch.ones_like(self.weight))
        self.initialize_qweight = True

        # rescale the output
        self.rescale_out = rescale_out
    
    def inference(self):
        r"""
        Inference mode
        """
        self.train_flag = False
        self.weight.requires_grad_(False)
        self.wq.inference()
        self.aq.inference()

        if self.aq.nbit > 8 or self.wq.nbit > 8:
            self.ops = FloatActWeight()
        else:
            self.ops = IntActWeight(nbit=8)

    def fetch_yscale(self):
        scale_x = self.aq.scale
        scale_w = self.wq.scale

        if isinstance(self.wq.observer, BaseChannelWiseObserver):
            scale_w = scale_w.permute(1,0)

        if isinstance(self.aq.observer, BaseTokenWiseObserver):
            scale_w = scale_w.unsqueeze(0)

        return scale_x.to(torch.float32), scale_w.to(torch.float32)

    def trainFunc(self, x:torch.Tensor):
        wq = self.wq(self.weight)
        x = self.aq(x)

        y = F.linear(x, wq, self.bias)
        return y

    @torch.no_grad
    def evalFunc(self, x:torch.Tensor):
        x = self.aq(x)

        if self.initialize_qweight:
            wq = self.wq(self.weight)
            self.weight.data = wq
            self.initialize_qweight = False

        y = self.ops(x, self.weight)
        
        if self.rescale_out:
            scale_x, scale_w = self.fetch_yscale()
            y = y.mul(scale_x).mul(scale_w)

        return y.to(x.dtype)

    def forward(self, x:torch.Tensor):
        if self.train_flag:
            y = self.trainFunc(x)
        else:
            y = self.evalFunc(x)

        y = self.yq(y)
        return y

def round_ste(x:torch.Tensor):
    """
    Quantization with STE
    """
    return (x.round() - x).detach() + x
