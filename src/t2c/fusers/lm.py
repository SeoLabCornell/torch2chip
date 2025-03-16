"""
Fuse language model
"""
from src.t2c.convert import get_parent_name
from src.module.base import _QBaseLinear, _QBase
from src.quantization.observer import BaseChannelWiseObserver, BaseTokenWiseObserver
from src.module.fuse import Add

class LMFuser(object):
    """
    Fuser of language model with the isolated matrix multiplication. 
    """
    def __init__(self, fake_quant_model, rescale_out:bool=False):
        self.model = fake_quant_model
        self.device = self.model.device
        self.rescale_out = rescale_out

    def inference(self):
        """
        Switch to inference mode
        """
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

        # NOTE: Make sure the performance is not affected by the infinitesimal scales 
        scaler = Add()
        bias = getattr(layer, "bias")

        if bias is not None:
            scaler.bias.data = bias

        setattr(layer, "yq", scaler)
        setattr(layer, "rescale_out", self.rescale_out)
        return layer

    def fuse(self):
        modules = dict(self.model.named_modules(remove_duplicate=True))

        for n, m in modules.items():
            if isinstance(m, _QBaseLinear):
                parent_name, name = get_parent_name(n)
                new_layer = self.fuse_linear(m)
                new_layer = new_layer.to(self.device)
                setattr(modules[parent_name], name, new_layer)

        return self.model

class LlamaFuser(LMFuser):
    def __init__(self, fake_quant_model, rescale_out:bool=False):
        super().__init__(fake_quant_model, rescale_out)
