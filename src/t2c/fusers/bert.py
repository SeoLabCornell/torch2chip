"""
Fuser of BERT
"""

import torch.nn as nn
from src.module.attention import QBertSelfAttention
from src.t2c.convert import get_parent_name
from src.t2c.fusers.vit import ViTFuser
from src.module.fuse import MulQuant
from src.quantization.observer import BaseObserver, BaseTokenWiseObserver

from transformers.models.bert.modeling_bert import BertSelfOutput

class BERTFuser(ViTFuser):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def qkv_fuser(self, module: QBertSelfAttention):
        module.inference()
        
        # fetch modules
        xq = getattr(module, "xq")
        sq = module.qquery.scale
        sk = module.qkey.scale
        sv = module.qvalue.scale

        query = getattr(module, "query")
        key = getattr(module, "key")
        value = getattr(module, "value")

        sxq = self.quantizer_fuse(xq, query.wq)
        sxk = self.quantizer_fuse(xq, key.wq)
        sxv = self.quantizer_fuse(xq, value.wq)

        qquery = MulQuant(nbit=module.qquery.nbit)
        sxq = sq.mul(sxq)
        qbias = query.bias.mul(sq)
        
        setattr(qquery, "scale", sxq)
        setattr(qquery, "bias", qbias)
        setattr(qquery, "zero_point", module.qquery.zero_point)

        qkey = MulQuant(nbit=module.qkey.nbit)        
        sxk = sk.mul(sxk)
        kbias = key.bias.mul(sk)
        
        setattr(qkey, "scale", sxk)
        setattr(qkey, "bias", kbias)
        setattr(qkey, "zero_point", module.qkey.zero_point)

        qvalue = MulQuant(nbit=module.qvalue.nbit)
        sxv = sv.mul(sxv)
        vbias = value.bias.mul(sv)

        setattr(qvalue, "scale", sxv)
        setattr(qvalue, "bias", vbias)
        setattr(qvalue, "zero_point", module.qvalue.zero_point)

        if isinstance(module.qquery.observer, BaseTokenWiseObserver):
            if isinstance(module.qkey.observer, BaseTokenWiseObserver):
                # [B, Head, Token, Token]
                qkscale = (sq @ sk.transpose(-1,-2)).unsqueeze(0)
        elif isinstance(module.qquery.observer, BaseObserver):
            qkscale = sq * sk

        # scale back after q @ k
        module.attn_scale.scale.data = 1 / (qkscale) * module.attn_scale.scale

        # scale back after attention @ v
        ssfmx = 1 / 256
        module.qkv_deq.scale = 1 / sv * ssfmx
        
        # # update the module
        setattr(module, "qquery", qquery)
        setattr(module, "qqkey", qkey)
        setattr(module, "qvalue", qvalue)
        
        return module

    def output_fuser(self, module:BertSelfOutput):
        dense = getattr(module, "dense")

        fdense = self.fuse_linear(dense)
        setattr(module, "dense", fdense)
        return module

    def fuse(self):
        modules = dict(self.model.named_modules(remove_duplicate=True))

        for n, m in self.model.named_modules():
            if isinstance(m, QBertSelfAttention):
                print(f"Fusing {n}")
                parent_name, name = get_parent_name(n)
                
                module = self.qkv_fuser(m)
                setattr(modules[parent_name], name, module)
            
            elif isinstance(m, BertSelfOutput):
                print(f"Fusing {n}")
                parent_name, name = get_parent_name(n)
                
                module = self.output_fuser(m)
                setattr(modules[parent_name], name, module)

        return self.model


