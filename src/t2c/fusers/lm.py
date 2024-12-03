"""
Fuse language model
"""
from typing import Optional
from src.t2c.convert import get_parent_name
from src.module.base import _QBaseLinear, _QBase
from src.quantization.observer import BaseChannelWiseObserver, BaseTokenWiseObserver
from src.module.fuse import MulShift

class LMFuser(object):
    """
    Fuser of language model with the isolated matrix multiplication. 
    """
    def __init__(self, fake_quant_model):
        self.model = fake_quant_model

    def inference(self):
        """
        Switch to inference mode
        """
        pass

    def quantizer_fuse(self, wq: _QBase, aq: _QBase, yq: Optional[_QBase]=None):
        scale_x = getattr(aq, "scale")
        scale_w = getattr(wq, "scale")

        if isinstance(aq.observer, BaseTokenWiseObserver):
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
        hasbias = layer.bias is not None
        scaler = MulShift()
        
        # switch to inference mode
        layer.inference()
        
        sw = self.quantizer_fuse(layer.wq, layer.aq)

        # fused scaling factors
        if hasbias:
            bias = getattr(layer, "bias")
            scaler.bias.data = bias

        layer.ops.scale.data = sw
        return layer

    def fuse(self):
        modules = dict(self.model.named_modules(remove_duplicate=True))

        for n, m in modules.items():
            if isinstance(m, _QBaseLinear):
                parent_name, name = get_parent_name(n)
                new_layer = self.fuse_linear(m)
                setattr(modules[parent_name], name, new_layer)

        return self.model

class LlamaFuser(LMFuser):
    def __init__(self, fake_quant_model):
        super().__init__(fake_quant_model)
