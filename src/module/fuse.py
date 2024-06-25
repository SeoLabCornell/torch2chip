"""
Post-computation scaling
"""

import torch
import torch.nn as nn
from src.module.base import _QBaseConv2d, _QBaseLinear

class MulShift(nn.Module):
    r"""Multiply the scaling factor and add the bias
    
    Attributes:
    scale: Scaling factor with the shape of output channels.
    bias: Bias value. 
    fl: Fractional bits of the high-precision integer.
    """
    def __init__(self):
        super(MulShift, self).__init__()
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("bias", torch.tensor(0.0))

        # fractional bit width
        self.fl = 0.

    def inference(self):
        pass

    def forward(self, x:torch.Tensor):
        out = x.mul(self.scale).add(self.bias)
        out = out.mul(2**(-self.fl))
        return out

class MulQuant(nn.Module):
    r"""Multiply the scaling factor and add the bias, then quantize the output.

    Attributes:
    scale: Scaling factor with the shape of output channels.
    bias: Bias value. 
    fl: Fractional bits of the high-precision integer.
    """
    def __init__(self, nbit:int=8, unsigned=False):
        super(MulQuant, self).__init__()
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("bias", torch.tensor(0.0))
        self.register_buffer("zero_point", torch.tensor(0.0))

        self.nbit = nbit
        self.unsigned = unsigned

        # upper and lower bound
        if not self.unsigned:
            self.qlb = -2**(self.nbit-1)
            self.qub = 2**(self.nbit-1) - 1
        else:
            self.qlb = 0
            self.qub = 2**(self.nbit) - 1

        # fractional bit width
        self.fl = 0.

    def inference(self):
        pass

    def forward(self, x:torch.Tensor):
        # scale
        out = x.mul(self.scale)
        out = out.add(self.bias).mul(2**(-self.fl)).round()
        
        # quant
        out = out.add(self.zero_point)
        out = out.clamp(min=self.qlb, max=self.qub).sub(self.zero_point)
        
        return out.clamp(min=self.qlb, max=self.qub)

class QConvBNReLU(nn.Module):
    r"""
    Template of module fusion
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, 
                padding:int=0, dilation:int=1, groups:int=1, bias:bool=True, wbit:int=32, abit:int=32, train_flag=True, int_out=True):
        super(QConvBNReLU, self).__init__()
        
        # modules
        self.conv = _QBaseConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, wbit, abit, train_flag)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.Identity()

        # flag
        self.int_out = int_out
        self.scale = torch.tensor(1.0)

        # precision
        self.abit = abit
        
        # scaler
        self.scaler = MulShift()

    def quant(self, x:torch.Tensor):
        x = x.mul(self.scale).round().clamp(0, 2**self.abit-1)
        return x.div(self.scale)

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        x = self.conv(inputs)
        x = self.scaler(x)
        x = self.relu(x)
        return x

class QConvReLU(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, 
                padding:int=0, dilation:int=1, groups:int=1, bias:bool=True, wbit:int=32, abit:int=32, train_flag=True, int_out=True):
        super(QConvReLU, self).__init__()
        
        # modules
        self.conv = _QBaseConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, wbit, abit, train_flag)
        
        # relu
        self.relu = nn.ReLU()
        
        # scaler and shifter
        if int_out:
            self.scaler = MulQuant(nbit=abit)
        else:
            self.scaler = MulShift()

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        x = self.conv(inputs)
        x = self.scaler(x)
        x = self.relu(x)
        return x

class ConvReLU(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, 
                padding:int=0, dilation:int=1, groups:int=1, bias:bool=True):
        super(ConvReLU, self).__init__()
        
        # modules
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        # relu
        self.relu = nn.ReLU()
        self.scaler = nn.Identity()

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        x = self.conv(inputs)
        x = self.scaler(x)
        x = self.relu(x)
        return x


class FusedLinear(nn.Module):
    """
    Integer-only linear layer
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                wbit:int=32, abit:int=32, train_flag=True, int_out=False):
        super(FusedLinear, self).__init__()

        # modules
        self.linear = _QBaseLinear(in_features, out_features, bias, wbit=wbit, abit=abit, train_flag=train_flag)
        
        if int_out:
            self.scaler = MulQuant(nbit=abit, unsigned=False)
        else:
            self.scaler = MulShift()

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        x = self.linear(inputs)
        x = self.scaler(x)
        return x
    
class LinearMulShift(nn.Module):
    def __init__(self, in_features: int, out_features: int, wbit:int=32, abit:int=32, train_flag=True, int_out=False, obit:int=32):
        super(LinearMulShift, self).__init__()

        self.linear = _QBaseLinear(in_features, out_features, True, wbit, abit, train_flag)
        
        # scaler and shifter
        if int_out:
            self.scaler = MulQuant(nbit=obit, unsigned=False)
        else:
            self.scaler = MulShift()

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        x = self.linear(inputs)
        x = self.scaler(x)
        return x


class LinearMulShiftReLU(nn.Module):
    def __init__(self, in_features: int, out_features: int, wbit:int=32, abit:int=32, train_flag=True, int_out=False, obit:int=32):
        super(LinearMulShiftReLU, self).__init__()

        self.linear = _QBaseLinear(in_features, out_features, True, wbit, abit, train_flag)
        
        # scaler and shifter
        if int_out:
            self.scaler = MulQuant(nbit=obit, unsigned=True)
        else:
            self.scaler = MulShift()

        # relu
        self.relu = nn.ReLU()

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        x = self.linear(inputs)
        x = self.relu(x)
        x = self.scaler(x)
        return x

