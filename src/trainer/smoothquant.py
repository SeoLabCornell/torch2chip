"""
Post Training Quantizer of SmoothQuant
"""

import torch
import torch.nn as nn
from typing import List, Union

from src.module.attention import QAttention, QWindowAttention
from src.module.base import _QBaseLinear
from src.trainer.ptq import PTQViT
from timm.layers.mlp import Mlp

class SmoothQuantPTQViT(PTQViT):
    def __init__(self, model: nn.Module, loss_type: str, trainloader, validloader, args, logger):
        super().__init__(model, loss_type, trainloader, validloader, args, logger)

        # smooth coefficient
        self.alpha = self.args.alpha

        # smooth scaling
        self.sscale = None

        # calibration only
        self.layer_train = False
        
    def layer_stat(self, cached_data:List):
        xmax = torch.tensor(0.)
        
        for batch in cached_data:
            x, _ = batch
            xmax = torch.max(xmax, x.max())
        
        return xmax

    def collect_scale(self):
        sscale = {}
        self.logger.info(f"Start Fetching the smooth factor!")
        for n, m in self.model.named_modules():
            if isinstance(m, _QBaseLinear):
                cached_data = self.fetch_layer_data_all(m)
                
                xmax = self.layer_stat(cached_data)
                wmax = m.weight.abs().max()

                scale = (xmax.pow(self.alpha) / wmax.pow(1 - self.alpha)).clamp(1e-5)
                
                sscale[n] = scale

        del cached_data
        self.logger.info(f"Done!\n")
        return sscale
    
    def update_attn(self, layer: Union[QAttention, QWindowAttention], name=None):
        layer = super().update_attn(layer, name)

        # load the smooth factor
        layer.xq.smoother.scale.data.copy_(1 / self.sscale[name+".qkv"])
        layer.qkv.wq.smoother.scale.data.copy_(self.sscale[name+".qkv"])

        layer.qproj.smoother.scale.data.copy_(1 / self.sscale[name+".proj"])
        layer.proj.wq.smoother.scale.data.copy_(self.sscale[name+".proj"])

        return layer

    def update_mlp(self, layer: Mlp, name=None):
        layer = super().update_mlp(layer, name)

        # load the smooth factor
        layer.fc1.wq.smoother.scale.data.copy_(self.sscale[name+".fc1"])
        layer.fc1.aq.smoother.scale.data.copy_(1 / self.sscale[name+".fc1"])

        layer.fc2.wq.smoother.scale.data.copy_(self.sscale[name+".fc2"])
        layer.fc2.aq.smoother.scale.data.copy_(1 / self.sscale[name+".fc2"])

        return layer

    def fit(self):
        if self.sscale is None:
            self.sscale = self.collect_scale()

        super().fit()
        
    
