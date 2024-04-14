"""
BatchNorm fusion with full observability
"""

import torch
import torch.nn as nn
from src.module.base import _QBase
from typing import List, Union
from src.module.fuse import QConvReLU, QConvBNReLU, _QBaseConv2d, _QBaseLinear, FusedLinear, MulQuant
from src.quantization.observer import BaseObserver, BaseTokenWiseObserver, BaseChannelWiseObserver


class LayerFuser(object):
    def __init__(self, model:nn.Module):
        self.model = model
        # flag
        self.flag = False
        
        # layers
        self.groups = []
        
        # parameters
        self.xscales = []
        self.xzps = []

        # full precision conv layer
        self.fpl = 1

        # full precision classifier
        self.fpc = False
    
    def inference(self):
        """
        Switch to inference mode
        """
        for n, m in self.model.named_modules():
            if hasattr(m, "inference"):
                m.inference()

    def fuse_linear(self, layer:_QBaseLinear):
        # switch
        layer.inference()

        if layer.bias is not None:
            bias = layer.bias.data

        tmp = FusedLinear(layer.in_features, layer.out_features, True, wbit=layer.wbit, abit=layer.abit, train_flag=False)

        # insert the linear layer
        setattr(tmp, "linear", layer)

        sq = 1 / (tmp.linear.wq.scale.data * tmp.linear.aq.scale.data)

        # assign the scaling factor to the quantizer
        if isinstance(layer.wq.observer, BaseChannelWiseObserver):
            tmp.scaler.scale.data = sq.squeeze(1).unsqueeze(0)
        else:
            tmp.scaler.scale.data = sq

        tmp.scaler.bias.data = bias.unsqueeze(0)
        return tmp

    def quantizer_bn_fuse(self, xq:_QBase, wq:_QBase, bn:Union[nn.BatchNorm1d, nn.BatchNorm2d]):
        sq = 1 / (wq.scale.data * xq.scale.data)

        # bn scaling
        std = torch.sqrt(bn.running_var.data + bn.eps)
        
        if isinstance(wq.observer, BaseChannelWiseObserver):
            sw = wq.scale.data.reshape(bn.weight.shape)
        
        # scaling
        sq = 1 / (sw * xq.scale)
        sbn = bn.weight.data.mul(sq) / std

        # bn bias
        bbn = bn.bias - bn.weight.mul(bn.running_mean.data).div(std)
        
        return sbn.unsqueeze(0).unsqueeze(2).unsqueeze(3), bbn.unsqueeze(0).unsqueeze(2).unsqueeze(3)


    def conv_bn_relu(self, cbr:List, l=-1.0, snxt:float=1.0, zpnxt:float=0.0, int_out:bool=False):
        assert len(cbr) == 3, "The input must include conv, bn, and relu modules"
        conv, bn, _ = cbr

        # fused layer
        tmp = QConvBNReLU(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, groups=conv.groups,
                        wbit=conv.wbit, abit=conv.abit, train_flag=conv.train_flag, int_out=False)
        
        # assign modules
        setattr(tmp, "conv", cbr[0])
        setattr(tmp, "relu", cbr[2])
        tmp.conv.inference()

        sbn, bbn = self.quantizer_bn_fuse(tmp.conv.aq, tmp.conv.wq, bn)
 
        # scale and bias
        tmp.scaler.scale.data = sbn
        tmp.scaler.bias.data = bbn
        
        return tmp
    
    def conv_relu(self, cr:List, l=-1.0, snxt:float=1.0, zpnxt:float=0.0, int_out:bool=False):
        assert len(cr) == 2, "The input must include conv and relu modules"

        conv, relu = cr
        
        # fused layer
        tmp = QConvReLU(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, 
                        wbit=conv.wbit, abit=conv.abit, train_flag=False, int_out=int_out)

        # quantization scalers
        sq = 1 / (conv.wq.scale.data * conv.aq.scale.data)
        
        # scaled bias
        sb = conv.bias.data.div(sq)
        conv.bias.data = sb

        # assign modules
        setattr(tmp, "conv", conv)
        setattr(tmp, "relu", relu)

        # next layer scaler
        tmp.scaler.scale.data = sq.mul(snxt)

        if isinstance(tmp.scaler, MulQuant):
            tmp.scaler.zp.data = zpnxt
        
        # replace the activation quantizer by the Identity module
        if l > self.fpl-1:
            tmp.conv.aq = nn.Identity()
        
        return tmp

    def fuse(self):
        pass
