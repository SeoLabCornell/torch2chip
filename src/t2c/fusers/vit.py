import torch.nn as nn

from src.t2c.convert import get_parent_name
from src.module.attention import QAttention, QWindowAttention
from src.module.base import _QBaseLinear, _QBase
from src.module.fuse import MulQuant, MulShift
from src.quantization.observer import BaseObserver, BaseTokenWiseObserver, BaseChannelWiseObserver

from timm.layers import Mlp

class ViTFuser(object):
    def __init__(self, model: nn.Module, rescale_out:bool=False):
        self.model = model.eval()
        self.rescale_out = rescale_out

    def inference(self):
        """
        Switch to inference mode
        """
        pass

    def layers(self):
        pass

    def quantizer_fuse(self, xq:_QBase, wq:_QBase):
        scale_x = xq.scale
        scale_w = wq.scale
        if isinstance(xq.observer, BaseTokenWiseObserver):
            if isinstance(wq.observer, BaseChannelWiseObserver):
                scale_w = scale_w.unsqueeze(0)
                sw = scale_x @ scale_w.transpose(1,2)
            else:
                sw = scale_x * scale_w
        else:
            if isinstance(wq.observer, BaseChannelWiseObserver):
                scale_w = scale_w.unsqueeze(0).transpose(1,2)
            sw = scale_x * scale_w

        return sw

    def fuse_linear(self, layer:_QBaseLinear):
        # switch to inference mode
        layer.inference()
        scaler = MulShift()

        # fetch the scaling factors
        sw = self.quantizer_fuse(layer.aq, layer.wq)
        bias = getattr(layer, "bias")

        # construct the scalers
        scaler.scale = sw
        scaler.bias.data = bias

        setattr(layer, "yq", scaler)
        return layer

    def qkv_fuser(self, module:QAttention):
        module.inference()

        # fuse the scaling factors
        sw = self.quantizer_fuse(module.qkv.aq, module.qkv.wq)
        sy = module.qkv.yq.scale

        # integer-only attention
        if isinstance(module.qkv.yq.observer, BaseTokenWiseObserver):
            qkscale = (sy @ sy.transpose(-2,-1))
        elif isinstance(module.qkv.yq.observer, BaseObserver):
            qkscale = sy.pow(2)

        module.attn_scale.scale = qkscale.mul(module.attn_scale.scale)

        # replace the simple shifter to quantizer
        scaler = MulQuant(nbit=module.qkv.aq.nbit, unsigned=module.qkv.aq.unsigned)
        scaler.scale.data = sw.div(sy)
        scaler.bias.data = module.qkv.bias.div(sy)

        setattr(module.qkv, "yq", scaler)
        proj = self.fuse_linear(module.proj)
        proj.aq.scale.data.div_(sy)
        setattr(module, "proj", proj)

        return module

    def mlp_fuser(self, module:Mlp):
        fc1 = getattr(module, "fc1")
        fc2 = getattr(module, "fc2")

        ffc1 = self.fuse_linear(fc1)
        ffc2 = self.fuse_linear(fc2)

        setattr(module, "fc1", ffc1)
        setattr(module, "fc2", ffc2)
        return module
    
    def fuse(self):
        modules = dict(self.model.named_modules(remove_duplicate=True))

        for n, m in self.model.named_modules():
            if isinstance(m, QAttention):
                parent_name, name = get_parent_name(n)
                
                module = self.qkv_fuser(m)
                setattr(modules[parent_name], name, module)

            elif isinstance(m, Mlp):
                parent_name, name = get_parent_name(n)

                module = self.mlp_fuser(m)
                setattr(modules[parent_name], name, module)

        return self.model

class SwinFuser(ViTFuser):
    def __init__(self, model: nn.Module, rescale_out:bool=False):
        super().__init__(model)
        self.rescale_out = rescale_out

    def qkv_fuser(self, module: QWindowAttention):
        module = super().qkv_fuser(module)

        return module

    def fuse(self):
        modules = dict(self.model.named_modules(remove_duplicate=True))

        for n, m in self.model.named_modules():
            if isinstance(m, QWindowAttention):
                parent_name, name = get_parent_name(n)
                
                module = self.qkv_fuser(m)
                setattr(modules[parent_name], name, module)
            elif isinstance(m, Mlp):
                parent_name, name = get_parent_name(n)

                module = self.mlp_fuser(m)
                setattr(modules[parent_name], name, module)

        return self.model