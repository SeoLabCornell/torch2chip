"""
Calibrator of post-training quantization (PTQ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Union
from torch.nn.modules import Module
from tqdm import tqdm

from src.utils.utils import accuracy, AverageMeter
from src.t2c.convert import get_parent_name
from src.module.base import _QBaseConv2d, _QBaseLinear, _QBase
from src.module.attention import QAttention, QWindowAttention, QBertSelfAttention

from src.quantization.adaround import AdaRound
from src.quantization.lsq import LSQ, LSQTokenWise
from src.quantization.qdrop import QDrop, QDropTokenWise
from src.quantization.smoothquant import SmoothQuantChannelWiseWeightQuantizer, SmoothQuantTokenWiseQuantizer, SmoothQuantizer
from src.quantization.minmax import MinMaxQuantizer, MinMaxTokenWiseQuantizer, MinMaxChannelWiseWeightQuantizer, MinMaxChannelWiseActQuantizer

from timm.layers.mlp import Mlp
from transformers.models.bert.modeling_bert import BertSelfOutput

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

class DataSaverHook:
    def __init__(self, store_input=False, store_output=False) -> None:
        self.store_input = store_input
        self.store_output = store_output

        self.input = None
        self.output = None
    
    def __call__(self, module, input_batch, output_batch) -> Any:
        if self.store_input:
            self.input = input_batch
        
        if self.store_output:
            self.output = output_batch

class PTQ(object):
    """
    PTQ trainer
    """
    def __init__(self, 
            model: nn.Module, 
            loss_type: str, 
            trainloader, 
            validloader, 
            args, 
            logger
        ):
                
        # model architecture
        self.model = model

        # args
        self.args = args

        # qtypes
        self.wqtype = self.args.wqtype
        self.xqtype = self.args.xqtype

        # loader
        self.trainloader = trainloader
        self.validloader = validloader

        # max iterations
        self.epochs = self.args.epochs

        # logger
        self.logger = logger
        self.logger_dict = {}

        # trainer
        self.layer_train = self.args.layer_trainer

        # loss func
        if loss_type == "mse":
            self.criterion = torch.nn.MSELoss().cuda()
        elif loss_type == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        else:
            raise NotImplementedError("Unknown loss type")
        
        # cuda
        self.model = self.model.cuda()
        self.steps = len(self.trainloader)

    def freeze(self, layer:Union[nn.Conv2d, nn.Linear]):
        hasbias = layer.bias is not None

        layer.weight.requires_grad_(False)
        if hasbias:
            layer.bias.requires_grad_(False)

    def fetch_layer_data(self, layer:nn.Module, batch):
        hook = DataSaverHook(store_input=True, store_output=True)
        handle = layer.register_forward_hook(hook)

        with torch.no_grad():
            out = self.model(batch)
        
        handle.remove()
        return hook.input[0].detach(), hook.output.detach()

    def fetch_layer_data_all(self, layer:nn.Module):
        cached_data = []
        
        pbar = tqdm(self.trainloader, desc="Fetch Data")
        for idx, (inputs, target) in enumerate(pbar):
            inputs = inputs.cuda()
            
            x, y = self.fetch_layer_data(layer, inputs)
            cached_data.append((x, y))
        
        return cached_data    

    def layer_trainer(self, layer:Union[_QBaseConv2d, _QBaseLinear], cached_data):
        # assign the layer quantizer
        weight = layer.weight
        layer.weight.requires_grad_(False)

        # bias flag
        hasbias = layer.bias is not None
        if hasbias:
            layer.bias.requires_grad_(False)
        
        # quantizer parameters
        qparams = []

        if self.wqtype == "adaround":
            layer.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, weights=weight, train_flag=True).cuda()
        else:
            layer.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, train_flag=True).cuda()
        
        qparams += [
            {'params':layer.wq.parameters(), 'lr': self.args.lr, 'weight_decay': 0.0}, 
        ]

        if isinstance(layer, _QBaseConv2d):
            if layer.in_channels != 3:
                layer.aq = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True).cuda()
        else:
            layer.aq = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True).cuda()
        
        qparams += [
            {'params':layer.aq.parameters(), 'lr': self.args.lr, 'weight_decay': 0.0}, 
        ]

        if self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(qparams)
        elif self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(qparams)
        
        sz = len(cached_data)
        id = torch.randint(0, sz, (self.args.batch_size,))
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.epochs * len(cached_data)), eta_min=0.)
        pbar = tqdm(range(self.epochs), desc=f"Epoch | {self.args.optimizer}")
        for i in pbar:
            calib_loss = AverageMeter()
            for idx, batch in enumerate(cached_data):      
                # fetch the data
                x, y = cached_data[id[idx]]

                # cuda
                x = x.cuda()
                y = y.cuda()

                out = layer(x)

                err = self.criterion(out, y)
                calib_loss.update(err.item())

                try:
                    optimizer.zero_grad()
                    err.backward(retain_graph=True)
                    optimizer.step()
                    scheduler.step()
                except:
                    continue

            pbar.set_postfix(lr=scheduler.get_last_lr()[0], loss=err.item())

        return layer, calib_loss.avg
    
    def layer_calibrator(self, layer:Union[_QBaseConv2d, _QBaseLinear], cached_data):
        self.freeze(layer)
        layer.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, train_flag=True, unsigned=False).cuda()

        if isinstance(layer, _QBaseConv2d):
            if layer.in_channels != 3:
                layer.aq = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True, unsigned=True).cuda()
        else:
            layer.aq = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True, unsigned=True).cuda()

        calib_loss = AverageMeter()
        loss_fn = nn.MSELoss()
        for idx, batch in enumerate(tqdm(cached_data)):
            # fetch the data
            x, y = batch

            # cuda
            x = x.cuda()
            y = y.cuda()

            out = layer(x)
            err = loss_fn(out, y)
            calib_loss.update(err.item())

        return layer, calib_loss.avg
    
    def base_forward(self, inputs, target):
        """
        Foward pass of NN
        """
        out = self.model(inputs)
        loss = F.cross_entropy(out, target)
        return out, loss
    
    def valid_step(self, inputs, target):
        """
        Validation step at each iteration
        """
        out, loss = self.base_forward(inputs, target)
            
        return out, loss

    def valid_epoch(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        self.model.eval()
        
        with torch.no_grad():
            for idx, (inputs, target) in enumerate(tqdm(self.validloader)):
                inputs = inputs.cuda()
                target = target.cuda(non_blocking=True)
                
                out, loss = self.valid_step(inputs, target)
                prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

                losses.update(loss.mean().item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
            
        self.logger_dict["valid_loss"] = losses.avg
        self.logger_dict["valid_top1"] = top1.avg
        self.logger_dict["valid_top5"] = top5.avg

    def fit(self):
        modules = dict(self.model.named_modules(remove_duplicate=False))

        for n, m in modules.items():
            if isinstance(m, (_QBaseConv2d, _QBaseLinear)):
                # fetch data
                cached_data = self.fetch_layer_data_all(m)

                self.logger.info(f"Start Calibration of layer: {n}")

                if self.layer_train:
                    new_layer, calib_err = self.layer_trainer(m, cached_data)
                else:
                    new_layer, calib_err = self.layer_calibrator(m, cached_data)

                self.logger.info(f"Layer {n}: Loss = {calib_err}")

                parent_name, name = get_parent_name(n)
                setattr(modules[parent_name], name, new_layer)

class PTQViT(PTQ):
    def __init__(self, model: Module, loss_type: str, trainloader, validloader, args, logger):
        super().__init__(model, loss_type, trainloader, validloader, args, logger)
    
    def update_attn(self, layer:Union[QAttention, QWindowAttention], name=None):
        # low precision qkv
        qkvw = layer.qkv.weight
        projw = layer.proj.weight
        
        self.freeze(layer.qkv)
        self.freeze(layer.proj)

        # low precision weights
        if self.wqtype == "adaround":
            layer.qkv.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, weights=qkvw, train_flag=True).cuda()
            layer.proj.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, weights=projw, train_flag=True).cuda()
        else:
            layer.qkv.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, train_flag=True).cuda()
            layer.proj.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, train_flag=True).cuda()

        # low precision quantizer
        layer.xq = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True, unsigned=False).cuda()
        layer.qqkv = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True, unsigned=False).cuda()
        layer.qproj = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True, unsigned=False).cuda()
        
        return layer

    def update_mlp(self, layer:Mlp, name=None):
        w1 = layer.fc1.weight
        w2 = layer.fc2.weight

        # freeze weights
        self.freeze(layer.fc1)
        self.freeze(layer.fc2)

        # add quantizers
        if self.wqtype == "adaround":
            layer.fc1.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, weights=w1, train_flag=True).cuda()
            layer.fc2.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, weights=w2, train_flag=True).cuda()
        else:
            layer.fc1.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, train_flag=True).cuda()
            layer.fc2.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, train_flag=True).cuda()

        layer.fc1.aq = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True, unsigned=False).cuda()
        layer.fc2.aq = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True, unsigned=False).cuda()

        return layer

    def layer_trainer(self, layer:Union[QAttention, QWindowAttention, Mlp], name:str, cached_data):
        if isinstance(layer, (QAttention, QWindowAttention)):
            qlayer = self.update_attn(layer, name)
        elif isinstance(layer, Mlp):
            qlayer = self.update_mlp(layer, name)
        
        if self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(qlayer.parameters(), weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(qlayer.parameters(), weight_decay=self.args.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.epochs * len(cached_data)), eta_min=0.)

        loss_fn = nn.MSELoss()
        pbar = tqdm(range(self.epochs), desc="Epoch")
        for i in pbar:
            calib_loss = AverageMeter()
            for idx, batch in enumerate(cached_data):
                # fetch the data
                x, y = batch

                # cuda
                x = x.cuda()
                y = y.cuda()

                out = layer(x)
                
                err = loss_fn(out, y)
                calib_loss.update(err.item())

                optimizer.zero_grad()
                err.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()

            pbar.set_postfix(lr=scheduler.get_last_lr()[0], loss=calib_loss.avg)

        return qlayer, calib_loss.avg
    
    def layer_calibrator(self, layer:Union[QAttention, QWindowAttention, Mlp], name:str, cached_data):
        if isinstance(layer, (QAttention, QWindowAttention)):
            qlayer = self.update_attn(layer, name)
        elif isinstance(layer, Mlp):
            qlayer = self.update_mlp(layer, name)
        
        calib_loss = AverageMeter()
        loss_fn = nn.MSELoss()
        for idx, batch in enumerate(tqdm(cached_data)):
            # fetch the data
            x, y = batch

            # cuda
            x = x.cuda()
            y = y.cuda()

            out = qlayer(x)
            err = loss_fn(out, y)
            calib_loss.update(err.item())

        return qlayer, calib_loss.avg

    def fit(self):
        modules = dict(self.model.named_modules(remove_duplicate=False))

        for n, m in modules.items():
            if isinstance(m, (QAttention, QWindowAttention, Mlp)):
                # fetch data
                cached_data = self.fetch_layer_data_all(m)

                self.logger.info(f"Start Calibration of layer: {n}")
                
                if self.layer_train:
                    new_layer, calib_err = self.layer_trainer(m, n, cached_data)
                else:
                    new_layer, calib_err = self.layer_calibrator(m, n, cached_data)

                self.logger.info(f"Layer {n}: Loss = {calib_err}")

                parent_name, name = get_parent_name(n)
                setattr(modules[parent_name], name, new_layer)

class PTQBERT(PTQ):
    def __init__(self, model: Module, loss_type: str, trainloader, validloader, args, logger):
        super().__init__(model, loss_type, trainloader, validloader, args, logger)

    def fetch_layer_data(self, layer:nn.Module, batch):
        hook = DataSaverHook(store_input=True, store_output=True)
        handle = layer.register_forward_hook(hook)

        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        input_ids = batch['input_ids'].to(self.model.device)
        attention_mask = batch['attention_mask'].to(self.model.device)

        with torch.no_grad():
            out = self.model(input_ids, attention_mask=attention_mask)

        handle.remove()
        return hook.input, hook.output
    
    def fetch_layer_data_all(self, layer:nn.Module):
        cached_data = []
        
        pbar = tqdm(self.trainloader, desc="Fetch Data")
        for i, batch in enumerate(pbar):
            
            x, y = self.fetch_layer_data(layer, batch)
            cached_data.append((x, y))
        
        return cached_data  

    def update_attn(self, layer:QBertSelfAttention, name=None):
        qw = layer.query.weight
        kw = layer.key.weight
        vw = layer.value.weight

        # freeze
        self.freeze(layer.query)
        self.freeze(layer.key)
        self.freeze(layer.value)

        # low precision weights
        if self.wqtype == "adaround":
            layer.query.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, weights=qw, train_flag=True).cuda()
            layer.key.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, weights=kw, train_flag=True).cuda()
            layer.value.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, weights=vw, train_flag=True).cuda()
        else:
            layer.query.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, train_flag=True).cuda()
            layer.key.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, train_flag=True).cuda()
            layer.value.wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, train_flag=True).cuda()
        
        # tensor quantizer
        xq = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True, unsigned=False).cuda()
        qquery = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True, unsigned=False).cuda()
        qkey = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True, unsigned=False).cuda()
        qvalue = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True, unsigned=False).cuda()

        setattr(layer, "xq", xq)
        setattr(layer, "qquery", qquery)
        setattr(layer, "qkey", qkey)
        setattr(layer, "qvalue", qvalue)
        return layer

    def update_output(self, layer:BertSelfOutput, name=None):
        dense = getattr(layer, "dense")
        weight = dense.weight
        
        # freeze weights
        self.freeze(dense)

        # add quantizers
        if self.wqtype == "adaround":
            wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, weights=weight, train_flag=True).cuda()
        else:
            wq = weight_quantizer[self.wqtype](nbit=self.args.wbit, train_flag=True).cuda()
        aq = input_quantizer[self.xqtype](nbit=self.args.abit, train_flag=True, unsigned=False).cuda()
        
        setattr(dense, "wq", wq)
        setattr(dense, "aq", aq)
        setattr(layer, "dense", dense)
        return layer

    def layer_calibrator(self, layer:Union[QBertSelfAttention, BertSelfOutput], cached_data):
        if isinstance(layer, QBertSelfAttention):
            qlayer = self.update_attn(layer)

        elif isinstance(layer, BertSelfOutput):
            qlayer = self.update_output(layer)

        calib_loss = AverageMeter()
        for idx, batch in enumerate(tqdm(cached_data)):
            # fetch the data
            x, y = batch

            out = qlayer(*x)
            
            if isinstance(layer, QBertSelfAttention):
                err = self.criterion(out[0], y[0])
            else:
                err = self.criterion(out, y)

            calib_loss.update(err.item())

        return qlayer, calib_loss.avg

    def fit(self):
        modules = dict(self.model.named_modules(remove_duplicate=False))

        for n, m in modules.items():
            if isinstance(m, (QBertSelfAttention, BertSelfOutput)):
                cached_data = self.fetch_layer_data_all(m)

                self.logger.info(f"Start Calibration of layer: {n}")
                
                new_layer, calib_err = self.layer_calibrator(m, cached_data)
                
                self.logger.info(f"Layer {n}: Loss = {calib_err}")

                parent_name, name = get_parent_name(n)
                setattr(modules[parent_name], name, new_layer)

    def valid_epoch(self):
        pass