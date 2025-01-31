"""
Vanilla to low precision modules
"""
import torch
import copy
import torch.nn as nn

from typing import Tuple
from src.module.base import _QBaseLinear, _QBaseConv2d, _QBase
from src.module.fuse import MulQuant, MulShift
from src.module.attention import QAttention, QWindowAttention, QBertSelfAttention, QLlamaAttention, QMultiScaleRetention
from src.module.mlp import QLlamaMLP, QGLU
from src.quantization.adaround import AdaRound
from src.quantization.lsq import LSQ, LSQTokenWise
from src.quantization.qdrop import QDrop, QDropTokenWise
from src.quantization.minmax import MinMaxQuantizer, MinMaxTokenWiseQuantizer, MinMaxChannelWiseWeightQuantizer, MinMaxChannelWiseActQuantizer
from src.quantization.observer import BaseObserver, BaseChannelWiseObserver, BaseTokenWiseObserver
from src.quantization.smoothquant import SmoothQuantizer, SmoothQuantChannelWiseWeightQuantizer, SmoothQuantTokenWiseQuantizer
from src.models.lm.retnet import MultiScaleRetention, GLU

from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention
from timm.layers import Mlp
from transformers.models.bert.modeling_bert import BertSelfAttention, BertSelfOutput

from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaMLP


from typing import Union, Dict

weight_quantizer = {
    "adaround": AdaRound,
    "minmax": MinMaxQuantizer,
    "minmax_channel": MinMaxChannelWiseWeightQuantizer,
    "smooth": SmoothQuantizer,
    "smooth_channel": SmoothQuantChannelWiseWeightQuantizer,
    "identity": _QBase
}

input_quantizer = {
    "minmax": MinMaxQuantizer,
    "minmax_token": MinMaxTokenWiseQuantizer,
    "minmax_channel": MinMaxChannelWiseActQuantizer,
    "smooth": SmoothQuantizer,
    "smooth_token": SmoothQuantTokenWiseQuantizer,
    "lsq": LSQ,
    "lsq_token": LSQTokenWise,
    "qdrop": QDrop,
    "qdrop_token": QDropTokenWise,
    "identity": _QBase
}
def get_parent_name(target:str) -> Tuple[str, str]:
    r = target.rsplit(".", 1)
    if len(r) == 1:
        return "", r[0]
    else:
        return r[0], r[1]

class Vanilla4Compress(object):
    def __init__(self, model:nn.Module, wbit:int=8, abit:int=8, state_dict:Dict=None) -> None:
        self.model = model
        self.wbit = wbit
        self.abit = abit
        self.state_dict = state_dict

    def conv(self, layer:nn.Conv2d):
        has_bias = layer.bias is not None

        new_layer = _QBaseConv2d(
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
            bias = has_bias,
            wbit=self.wbit,
            abit=self.abit
        )
        
        # copy the weights and bias to the new layer
        new_layer.weight.data[:] = layer.weight
        
        if has_bias:
            new_layer.bias.data[:] = layer.bias

        return new_layer

    def linear(self, layer:nn.Linear):
        has_bias = layer.bias is not None

        new_layer = _QBaseLinear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=has_bias,
            wbit=self.wbit,
            abit=self.abit
        )

        new_layer.weight.data[:] = layer.weight

        if has_bias:
            new_layer.bias.data[:] = layer.bias
        return new_layer

    def mlp(self, layer:Mlp):
        qfc1 = self.linear(layer.fc1)
        qfc2 = self.linear(layer.fc2)

        setattr(layer, "fc1", qfc1)
        setattr(layer, "fc2", qfc2)

        return layer

    def attn(self, layer:Attention):
        qkv_bias = layer.qkv.bias is not None

        # initialize the attention block
        qattn = QAttention(
            dim=layer.qkv.in_features,
            num_heads=layer.num_heads,
            qkv_bias=qkv_bias,
            qk_norm=False,
            attn_drop=layer.attn_drop.p,
            proj_drop=layer.proj_drop.p
        )

        # conver the linear layer
        qqkv = self.linear(layer.qkv)
        qproj = self.linear(layer.proj)

        # assign the layer back
        setattr(qattn, "qkv", qqkv)
        setattr(qattn, "proj", qproj)
        return qattn
    
    def wattn(self, layer:WindowAttention):
        qkv_bias = layer.qkv.bias is not None
        
        # initialize the attention block
        qattn = QWindowAttention(
            dim=layer.qkv.in_features,
            num_heads=layer.num_heads,
            qkv_bias=qkv_bias,
            attn_drop=layer.attn_drop.p,
            proj_drop=layer.proj_drop.p
        )

        # conver the linear layer
        qqkv = self.linear(layer.qkv)
        qproj = self.linear(layer.proj)
        qattn.relative_position_bias_table = layer.relative_position_bias_table
        qattn.relative_position_index = layer.relative_position_index

        # assign the layer back
        setattr(qattn, "qkv", qqkv)
        setattr(qattn, "proj", qproj)
        return qattn


    def assign_quantizer(self, model, wqtype, xqtype):
        model = copy.deepcopy(model)
        modules = dict(model.named_modules(remove_duplicate=True))

        for n, m in modules.items():
            if isinstance(m, (_QBaseConv2d, _QBaseLinear)):
                if wqtype == "adaround":
                    m.wq = weight_quantizer[wqtype](nbit=self.wbit, train_flag=False, weights=m.weight)
                else:
                    m.wq = weight_quantizer[wqtype](nbit=self.wbit, train_flag=False)
                
                if isinstance(m, _QBaseConv2d):
                    if m.in_channels != 3:
                        m.aq = input_quantizer[xqtype](nbit=self.abit, train_flag=False, unsigned=True)
                else:
                    m.aq = input_quantizer[xqtype](nbit=self.abit, train_flag=False, unsigned=True)

                m = self.reshape_quantizer(m, n)

                parent_name, name = get_parent_name(n)
                setattr(modules[parent_name], name, m)

            elif isinstance(m, (MulQuant, MulShift)):
                parent_name, name = get_parent_name(n)
                m = self.reshape_quantizer(m, n)
                setattr(modules[parent_name], name, m)

        return model
    
    def convert(self):
        model = copy.deepcopy(self.model)
        modules = dict(model.named_modules(remove_duplicate=True))

        for n, m in modules.items():
            parent_name, name = get_parent_name(n)

            if isinstance(m, nn.Conv2d):
                new_layer = self.conv(m)
                setattr(modules[parent_name], name, new_layer)
            
            elif isinstance(m, nn.Linear):
                new_layer = self.linear(m)
                setattr(modules[parent_name], name, new_layer)

        return model
    
    def reshape_quantizer(self, layer:Union[_QBaseLinear, _QBaseConv2d], layer_name:str):
        
        if isinstance(layer, _QBaseLinear):
            layer.wq.num_channels = layer.out_features
            layer.aq.num_channels = layer.in_features

            layer.wq.register_qparams()
            layer.aq.register_qparams()

            layer.wq.observer.num_channels = layer.out_features
            layer.aq.observer.num_channels = layer.in_features

            layer.wq.observer.register_range()
            layer.aq.observer.register_range()
        
        elif isinstance(layer, _QBaseConv2d):
            layer.wq.num_channels = layer.out_channels
            layer.aq.num_channels = layer.in_channels
            
            layer.wq.register_qparams()
            layer.aq.register_qparams()

            layer.wq.observer.num_channels = layer.out_channels
            layer.aq.observer.num_channels = layer.in_channels

            layer.wq.observer.register_range()
            layer.aq.observer.register_range()

            if isinstance(layer.wq.observer, BaseChannelWiseObserver):
                layer.wq.scale.unsqueeze_(2).unsqueeze_(3)
                layer.wq.zero_point.unsqueeze_(2).unsqueeze_(3)

            if isinstance(layer.aq.observer, BaseChannelWiseObserver):
                layer.aq.scale.unsqueeze_(2).unsqueeze_(3)
                layer.aq.zero_point.unsqueeze_(2).unsqueeze_(3)

        elif isinstance(layer, (MulQuant, MulShift)):
            layer.scale.data = torch.ones_like(self.state_dict[layer_name+".scale"])
            layer.bias.data = torch.ones_like(self.state_dict[layer_name+".bias"])

            if isinstance(layer, MulQuant):
                layer.zero_point.data = torch.ones_like(self.state_dict[layer_name+".zero_point"])
        
        return layer

    def reload_fake_quant(self, wqtype, xqtype):
        qmodel = self.convert()
        qmodel = self.assign_quantizer(qmodel, wqtype=wqtype, xqtype=xqtype)
        return qmodel


class ViTV4C(Vanilla4Compress):
    def __init__(self, model: nn.Module, wbit: int = 8, abit: int = 8, state_dict:Dict = None) -> None:
        super().__init__(model, wbit, abit, state_dict)

    def reshape_quantizer(self, layer:Union[_QBaseLinear, _QBaseConv2d, _QBase], layer_name: str):
        if isinstance(layer, (_QBaseConv2d, _QBaseLinear)):
            layer = super().reshape_quantizer(layer, layer_name)
        
        elif isinstance(layer, _QBase):
            layer.scale.data = torch.ones_like(self.state_dict[layer_name+".scale"])
            layer.zero_point.data = torch.zeros_like(self.state_dict[layer_name+".zero_point"])
            
            observer_lb_key = layer_name+".observer.lb"
            observer_ub_key = layer_name+".observer.ub"
            additional_learnable_param = layer_name+".delta"

            if observer_lb_key in self.state_dict.keys():
                layer.observer.lb.data = torch.zeros_like(self.state_dict[observer_lb_key])
            
            if observer_ub_key in self.state_dict.keys():
                layer.observer.ub.data = torch.zeros_like(self.state_dict[observer_ub_key])

            if hasattr(layer, "delta") and additional_learnable_param in self.state_dict.keys():
                layer.delta.data = torch.ones_like(self.state_dict[layer_name+".delta"])

        
        elif isinstance(layer, (MulQuant, MulShift)):
            layer.scale.data = torch.ones_like(self.state_dict[layer_name+".scale"])
            layer.bias.data = torch.ones_like(self.state_dict[layer_name+".bias"])

            if isinstance(layer, MulQuant):
                layer.zero_point.data = torch.ones_like(self.state_dict[layer_name+".zero_point"])
            

        return layer

    def assign_quantizer(self, model, wqtype, xqtype, inference=False):
        model = copy.deepcopy(model)
        modules = dict(model.named_modules(remove_duplicate=True))

        for n, m in modules.items():
            if isinstance(m, (QAttention, QWindowAttention)):
                parent_name, name = get_parent_name(n)
                
                qkvw = m.qkv.weight
                projw = m.proj.weight

                # low precision weights
                if wqtype == "adaround":
                    m.qkv.wq = weight_quantizer[wqtype](nbit=self.wbit, weights=qkvw, train_flag=False).cuda()
                    m.proj.wq = weight_quantizer[wqtype](nbit=self.wbit, weights=projw, train_flag=False).cuda()
                else:
                    m.qkv.wq = weight_quantizer[wqtype](nbit=self.wbit, train_flag=False).cuda()
                    m.proj.wq = weight_quantizer[wqtype](nbit=self.wbit, train_flag=False).cuda()

                # update quantizers
                xq = input_quantizer[xqtype](nbit=self.abit, train_flag=False, unsigned=False)
                qqkv = input_quantizer[xqtype](nbit=self.abit, train_flag=False, unsigned=False)
                qproj = input_quantizer[xqtype](nbit=self.abit, train_flag=False, unsigned=False)
                
                m.xq = self.reshape_quantizer(xq, n+".xq")
                m.qqkv = self.reshape_quantizer(qqkv, n+".qqkv")
                m.qproj = self.reshape_quantizer(qproj, n+".qproj")

                # reshape the q params
                qkv = self.reshape_quantizer(m.qkv, n+".qkv")
                proj = self.reshape_quantizer(m.proj, n+".proj")

                setattr(m, "qkv", qkv)
                setattr(m, "proj", proj)
                setattr(modules[parent_name], name, m)

                if inference:
                    m.inference()

            elif isinstance(m, Mlp):
                mlp = dict(m.named_modules(remove_duplicate=True))
                for subname, sub in m.named_modules():
                    if isinstance(sub, _QBaseLinear):
                        sub_parent, sub_name = get_parent_name(subname)

                        # add quantizers
                        w = sub.weight
                        if wqtype == "adaround":
                            sub.wq = weight_quantizer[wqtype](nbit=self.wbit, weights=w, train_flag=False).cuda()
                        else:
                            sub.wq = weight_quantizer[wqtype](nbit=self.wbit, train_flag=False).cuda()
                        
                        aq = input_quantizer[xqtype](nbit=self.abit, train_flag=False, unsigned=False)

                        sub = self.reshape_quantizer(sub, n+"."+subname)
                        aq = self.reshape_quantizer(aq, n+"."+subname+".aq")
                        
                        setattr(sub, "aq", aq)
                        setattr(mlp[sub_parent], sub_name, sub)

                        if inference:
                            m.inference()

            elif isinstance(m, (MulQuant, MulShift)):
                parent_name, name = get_parent_name(n)
                m = self.reshape_quantizer(m, n)
                setattr(modules[parent_name], name, m)

        return model
    
    def convert(self):
        model = copy.deepcopy(self.model)
        modules = dict(model.named_modules(remove_duplicate=True))

        for n, m in modules.items():
            parent_name, name = get_parent_name(n)
            if isinstance(m, nn.Conv2d):
                new_layer = self.conv(m)
                setattr(modules[parent_name], name, new_layer)
            elif isinstance(m, Attention):
                new_layer = self.attn(m)
                setattr(modules[parent_name], name, new_layer)
            elif isinstance(m, WindowAttention):
                new_layer = self.wattn(m)
                setattr(modules[parent_name], name, new_layer)
            elif isinstance(m, Mlp):
                new_layer = self.mlp(m)
                setattr(modules[parent_name], name, new_layer)

        return model


class BERT4Compress(Vanilla4Compress):
    def __init__(self, model: nn.Module, wbit: int = 8, abit: int = 8, state_dict: Dict = None) -> None:
        super().__init__(model, wbit, abit, state_dict)

        # config
        assert hasattr(model, "config"), "The configuration of the BERT model is missing, are you using hugging face?"
        self.config = self.model.config

    def reshape_quantizer(self, layer:Union[_QBaseLinear, _QBaseConv2d, _QBase], layer_name: str):
        if isinstance(layer, (_QBaseConv2d, _QBaseLinear)):
            layer = super().reshape_quantizer(layer, layer_name)
        
        elif isinstance(layer, _QBase):
            layer.scale.data = torch.ones_like(self.state_dict[layer_name+".scale"])
            layer.zero_point.data = torch.zeros_like(self.state_dict[layer_name+".zero_point"])

            if hasattr(layer, "delta"):
                layer.delta.data = torch.ones_like(self.state_dict[layer_name+".delta"])
            
            layer.observer.lb.data = torch.zeros_like(self.state_dict[layer_name+".observer.lb"])
            layer.observer.ub.data = torch.zeros_like(self.state_dict[layer_name+".observer.ub"])
        
        return layer

    def bert_attn(self, layer:BertSelfAttention):
        qattn = QBertSelfAttention(config=self.config)
    
        # convert the linear layer
        qquery = self.linear(layer.query)
        qkey = self.linear(layer.key)
        qvalue = self.linear(layer.value)

        # assign the layer back
        setattr(qattn, "query", qquery)
        setattr(qattn, "key", qkey)
        setattr(qattn, "value", qvalue)

        return qattn

    def bert_output(self, layer:BertSelfOutput):
        dense = getattr(layer, "dense")
        
        # convert 
        qdense = self.linear(dense)
        setattr(layer, "dense", qdense)
        return layer

    def convert(self):
        model = copy.deepcopy(self.model)
        modules = dict(model.named_modules(remove_duplicate=True))

        for n, m in modules.items():
            parent_name, name = get_parent_name(n)
            
            if isinstance(m, BertSelfAttention):
                new_layer = self.bert_attn(m)
                setattr(modules[parent_name], name, new_layer)
            
            elif isinstance(m, BertSelfOutput):
                new_layer = self.bert_output(m)
                setattr(modules[parent_name], name, new_layer)

        return model
    
    def assign_quantizer(self, model, wqtype, xqtype):
        model = copy.deepcopy(model)
        modules = dict(model.named_modules(remove_duplicate=True))

        for n, m in modules.items():
            if isinstance(m, QBertSelfAttention):
                parent_name, name = get_parent_name(n)

                q = getattr(m, "query")
                k = getattr(m, "key")
                v = getattr(m, "value")

                # low precision weights
                if wqtype == "adaround":
                    qwq = weight_quantizer[wqtype](nbit=self.wbit, weights=q.weight, train_flag=True).cuda()
                    kwq = weight_quantizer[wqtype](nbit=self.wbit, weights=k.weight, train_flag=True).cuda()
                    vwq = weight_quantizer[wqtype](nbit=self.wbit, weights=v.weight, train_flag=True).cuda()
                else:
                    qwq = weight_quantizer[wqtype](nbit=self.wbit, train_flag=True).cuda()
                    kwq = weight_quantizer[wqtype](nbit=self.wbit, train_flag=True).cuda()
                    vwq = weight_quantizer[wqtype](nbit=self.wbit, train_flag=True).cuda()
                
                # tensor quantizer
                xq = input_quantizer[xqtype](nbit=self.abit, train_flag=True, unsigned=False).cuda()
                qquery = input_quantizer[xqtype](nbit=self.abit, train_flag=True, unsigned=False).cuda()
                qkey = input_quantizer[xqtype](nbit=self.abit, train_flag=True, unsigned=False).cuda()
                qvalue = input_quantizer[xqtype](nbit=self.abit, train_flag=True, unsigned=False).cuda()
                
                # reshape the quantizer
                qwq = self.reshape_quantizer(qwq, n+".query.wq")
                kwq = self.reshape_quantizer(kwq, n+".key.wq")
                vwq = self.reshape_quantizer(vwq, n+".value.wq")
                
                xq = self.reshape_quantizer(xq, n+".xq")
                qquery = self.reshape_quantizer(qquery, n+".qquery")
                qkey = self.reshape_quantizer(qkey, n+".qkey")
                qvalue = self.reshape_quantizer(qvalue, n+".qvalue")
                
                # insert the module
                setattr(q, "wq", qwq)
                setattr(k, "wq", kwq)
                setattr(v, "wq", vwq)
                
                setattr(m, "xq", xq)
                setattr(m, "qquery", qquery)
                setattr(m, "qkey", qkey)
                setattr(m, "qvalue", qvalue)

                setattr(modules[parent_name], name, m)

            elif isinstance(m, BertSelfOutput):
                parent_name, name = get_parent_name(n)
                
                dense = getattr(m, "dense")
                weight = getattr(dense, "weight")

                # add quantizers
                if wqtype == "adaround":
                    wq = weight_quantizer[wqtype](nbit=self.wbit, weights=weight, train_flag=False).cuda()
                else:
                    wq = weight_quantizer[wqtype](nbit=self.wbit, train_flag=False).cuda()

                xq = input_quantizer[xqtype](nbit=self.abit, train_flag=True, unsigned=False).cuda()

                xq = self.reshape_quantizer(xq, n+".dense.aq")
                wq = self.reshape_quantizer(wq, n+".dense.wq")
                
                setattr(dense, "wq", wq)
                setattr(dense, "aq", xq)

                setattr(m, "dense", dense)
                setattr(modules[parent_name], name, m)

        return model


class Llama4Compress(Vanilla4Compress):
    def __init__(self, model: nn.Module, wbit: int = 8, abit: int = 8, state_dict: Dict = None) -> None:
        super().__init__(model, wbit, abit, state_dict)

    def to_half(self, module:nn.Module):
        for param in module.parameters():
            param.data = param.data.to(torch.float16)

        return module
    
    def attn(self, attn:LlamaSdpaAttention):
        new_attn = QLlamaAttention(attn.config, attn.layer_idx).cuda()
        new_attn.load_state_dict(attn.state_dict(), strict=False)

        new_attn = self.to_half(new_attn)
        return new_attn

    def mlp(self, mlp:LlamaMLP):
        new_module = QLlamaMLP(config=mlp.config).to(self.model.device)
        new_module = new_module.to(torch.float16)
        new_module.load_state_dict(mlp.state_dict(), strict=False)

        new_module = self.to_half(new_module)
        return new_module

    def convert(self):
        modules = dict(self.model.named_modules(remove_duplicate=True))

        for n, m in modules.items():
            if isinstance(m, LlamaSdpaAttention):
                parent_name, name = get_parent_name(n)
                new_module = self.attn(m)
                setattr(modules[parent_name], name, new_module)

            elif isinstance(m, LlamaMLP):
                parent_name, name = get_parent_name(n)
                new_module = self.mlp(m)
                setattr(modules[parent_name], name, new_module)

        return self.model

    def reload(self, wqtype, xqtype):
        qmodel = self.convert()
        qmodel = self.assign_quantizer(qmodel, wqtype=wqtype, xqtype=xqtype)
        return qmodel
    

class RetNet4Compress(Vanilla4Compress):
    def __init__(self, model: nn.Module, wbit: int = 8, abit: int = 8, state_dict: Dict = None) -> None:
        super().__init__(model, wbit, abit, state_dict)
        self.config = model.config
        
        self.embed_dim = self.config.decoder_embed_dim
        self.ffn_dim = self.config.decoder_ffn_embed_dim

    def attn(self, attn:MultiScaleRetention):
        new_attn = QMultiScaleRetention(attn.config).to(self.model.device)
        new_attn.load_state_dict(attn.state_dict(), strict=False)

        return new_attn
    
    def ffn(self, mlp:GLU):
        new_mlp = GLU(
            self.embed_dim,
            self.ffn_dim,
            self.config.activation_fn,
            self.config.dropout,
            self.config.activation_dropout,
        ).to(self.model.device)

        new_mlp.load_state_dict(mlp.state_dict(), strict=False)
        return new_mlp
    
    def convert(self):
        modules = dict(self.model.named_modules(remove_duplicate=True))

        for n, m in modules.items():
            if isinstance(m, MultiScaleRetention):
                parent_name, name = get_parent_name(n)
                new_module = self.attn(m)
                setattr(modules[parent_name], name, new_module)

            elif isinstance(m, GLU):
                parent_name, name = get_parent_name(n)
                new_module = self.ffn(m)
                setattr(modules[parent_name], name, new_module)

        return self.model