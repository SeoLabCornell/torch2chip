"""
Basic Modules for Low precision and Sparsity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.quantization.observer import BaseObserver

class IntMatMul(nn.Module):
    def __init__(self, nbit:int):
        super().__init__()
        self.nbit = nbit
        self.register_buffer("x_shape", torch.tensor([1,1]))
        self.register_buffer("y_shape", torch.tensor([1,1]))
        self.register_buffer("z_shape", torch.tensor([1,1]))

    def forward(self, x, y):
        z = torch.matmul(x, y)
        
        self.x_shape.data = torch.tensor(list(x.size()))
        self.y_shape.data = torch.tensor(list(y.size()))
        self.z_shape.data = torch.tensor(list(z.size()))
        return z
    
class ConvOPS(nn.Module):
    def __init__(self, stride:int=1, padding:int=0, dilation:int=1, groups:int=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.register_buffer("x_shape", torch.tensor([1,1]))
        self.register_buffer("y_shape", torch.tensor([1,1]))
        self.register_buffer("z_shape", torch.tensor([1,1]))

    def forward(self, x:torch.Tensor, y:torch.Tensor, b:torch.Tensor):
        z = F.conv2d(x, y, b, self.stride, self.padding, self.dilation, self.groups)
        
        self.x_shape.data = torch.tensor(list(x.size()))
        self.y_shape.data = torch.tensor(list(y.size()))
        self.z_shape.data = torch.tensor(list(z.size()))
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
        self.register_buffer("scale", torch.tensor(1.0))
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

        # integer only weights copy
        self.register_buffer("qweight", torch.ones_like(self.weight))

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
            self.qweight.copy_(wq.data.mul(self.mask))
            self.initialize_qweight = False

        y = self.ops(xq, self.qweight, self.bias)
        return y

    def forward(self, x:torch.Tensor):
        if self.train_flag:
            y = self.trainFunc(x)
        else:
            y = self.evalFunc(x)

        out = self.yq(y)
        return out
    
class _QBaseLinear(nn.Linear):
    r"""Basic low precision linear layer

    Inherited from the base nn.Linear layer.
    
    Args:
    wbit (int): Weight quantization precision. 
    abit (int): Input quantization precision.
    wq (_QBase): Weight quantizer. 
    aq (_QBase): Activation quantizer.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, wbit:int=32, abit:int=32, train_flag=True):
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
    
    def inference(self):
        r"""
        Inference mode
        """
        self.train_flag = False
        self.wq.inference()
        self.aq.inference()

        self.register_buffer("qweight", torch.ones_like(self.weight))
        self.ops = IntMatMul(nbit=self.abit)

    def trainFunc(self, x:torch.Tensor):
        wq = self.wq(self.weight)
        xq = self.aq(x)
        
        y = F.linear(xq, wq.mul(self.mask), self.bias)
        return y

    def evalFunc(self, x:torch.Tensor):
        wq = self.wq(self.weight)

        if self.initialize_qweight:
            wq = self.wq(self.weight)
            self.qweight.copy_(wq.data.mul(self.mask))
            self.initialize_qweight = False

        xq = self.aq(x)
        y = self.ops(xq, self.qweight.transpose(0,1))

        return y

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
