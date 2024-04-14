import torch.nn as nn

from src.t2c.convert import get_parent_name
from src.module.attention import QAttention, QWindowAttention
from src.module.base import _QBaseLinear, _QBase
from src.module.fuse import MulQuant, LinearMulShift
from src.quantization.observer import BaseObserver, BaseTokenWiseObserver, BaseChannelWiseObserver

from timm.layers.mlp import Mlp

class ViTFuser(object):
    def __init__(self, model: nn.Module):
        self.model = model.eval()

    def inference(self):
        """
        Switch to inference mode
        """
        for n, m in self.model.named_modules():
            if hasattr(m, "inference"):
                m.inference()

    def layers(self):
        pass

    def quantizer_fuse(self, xq:_QBase, wq:_QBase):
        scale_x = xq.scale
        scale_w = wq.scale
        if isinstance(xq.observer, BaseTokenWiseObserver):
            if isinstance(wq.observer, BaseChannelWiseObserver):
                scale_w = scale_w.unsqueeze(0)
                sall = scale_x @ scale_w.transpose(1,2)
                sw = 1 / sall
            else:
                sw = 1 / (scale_x * scale_w)
        else:
            if isinstance(wq.observer, BaseChannelWiseObserver):
                scale_w = scale_w = scale_w.unsqueeze(0).transpose(1,2)
            sw = 1 / (scale_x * scale_w)
        
        return sw

    def fuse_linear(self, layer:_QBaseLinear):
        new_layer = LinearMulShift(
            layer.in_features,
            layer.out_features,
            wbit=layer.wq.nbit,
            abit=layer.aq.nbit
        )

        # switch to inference mode
        layer.inference()
        
        # fetch the scaling factors
        sw = self.quantizer_fuse(layer.aq, layer.wq)
        bias = getattr(layer, "bias")

        # construct the scalers
        new_layer.scaler.scale = sw
        new_layer.scaler.bias.data = bias

        setattr(new_layer, "linear", layer)

        return new_layer

    def qkv_fuser(self, module:QAttention):
        module.inference()

        # fuse the scaling factors
        sw = self.quantizer_fuse(module.xq, module.qkv.wq)

        sqkv = module.qqkv.scale.mul(sw)
        qbias = module.qkv.bias.mul(module.qqkv.scale)
        q = MulQuant(nbit=module.qqkv.nbit)

        # update scaling and bias
        q.scale.data = sqkv
        q.bias.data = qbias
        q.zp.data = module.qqkv.zero_point

        attn_scale = module.attn_scale.scale
        scale = module.qqkv.scale.unsqueeze(0)

        if isinstance(module.qqkv.observer, BaseTokenWiseObserver):
            qkscale = (scale @ scale.transpose(2,3))
        elif isinstance(module.qqkv.observer, BaseObserver):
            qkscale = scale.pow(2)

        module.attn_scale.scale.data = 1 / (qkscale) * attn_scale

        # update qproj
        sv = 1 / module.qqkv.scale
        ssfmx = 1 / 255

        # output quantizer
        sproj = module.qproj.scale.mul(sv).mul(ssfmx)
        qproj = MulQuant(nbit=module.qproj.nbit)
        qproj.scale.data = sproj
        qproj.zp.data = module.qproj.zero_point

        # fuse the scaling factors
        sdeq = self.quantizer_fuse(module.qproj, module.proj.wq)

        # attention output: dequantize back
        module.qkv_deq.scale.data = sdeq
        module.qkv_deq.bias.data = module.proj.bias

        # remove qqkv
        setattr(module, "qqkv", q)
        setattr(module, "qproj", qproj)

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
                print(f"Fusing {n}")
                parent_name, name = get_parent_name(n)
                
                module = self.qkv_fuser(m)
                setattr(modules[parent_name], name, module)
            elif isinstance(m, Mlp):
                print(f"Fusing {n}")
                parent_name, name = get_parent_name(n)

                module = self.mlp_fuser(m)
                setattr(modules[parent_name], name, module)

        return self.model

class SwinFuser(ViTFuser):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def qkv_fuser(self, module: QWindowAttention):
        module = super().qkv_fuser(module)
        # pos_bias = module._get_rel_pos_bias()

        # update the attn_scale
        # module.attn_scale.bias.data = pos_bias

        return module

    def fuse(self):
        modules = dict(self.model.named_modules(remove_duplicate=True))

        for n, m in self.model.named_modules():
            if isinstance(m, QWindowAttention):
                print(f"Fusing {n}")
                parent_name, name = get_parent_name(n)
                
                module = self.qkv_fuser(m)
                setattr(modules[parent_name], name, module)
            elif isinstance(m, Mlp):
                print(f"Fusing {n}")
                parent_name, name = get_parent_name(n)

                module = self.mlp_fuser(m)
                setattr(modules[parent_name], name, module)

        return self.model