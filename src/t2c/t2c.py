import os
import json
import numpy as np 
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List

from src.module.fuse import MulShift
from src.module.base import _QBaseLinear, IntMatMul, ConvOPS
from src.t2c.fusers.resnet import ResNet18Fuser, ResNet34Fuser, ResNet50Fuser
from src.t2c.fusers.vgg import VGGFuser
from src.t2c.fusers.vit import ViTFuser, SwinFuser
from src.t2c.fusers.mobilenet import MobileNetV1Fuser
from src.t2c.fusers.bert import BERTFuser

from fxpmath import Fxp

FUSERS = {
    "resnet18": ResNet18Fuser,
    "resnet34": ResNet34Fuser,
    "resnet50": ResNet50Fuser,
    "vit_large": ViTFuser,
    "vit_base": ViTFuser,
    "vit_small": ViTFuser,
    "vit_tiny": ViTFuser,
    "swin_tiny_patch4_window7_224": SwinFuser,
    "swin_base_patch4_window7_224": SwinFuser,
    "mobilenetv1": MobileNetV1Fuser,
    "vgg16_bn": VGGFuser,
    "bert": BERTFuser,
}

def feature_hook(name, state_dict):
    def hook(model, input, output):
        state_dict[name] = [input[0].detach(), input[1].detach(), output.detach()]
    return hook

class T2C(object):
    def __init__(self, model:nn.Module, swl:int, sfl:int, args):
        self.swl = swl
        self.sfl = sfl
        self.args = args

        self.swl = swl
        self.sfl = sfl
        self.args = args

        # model fusion
        fuser = FUSERS[str(args.model)](model)

        # switch to inference mode
        fuser.inference()
        
        # fuse layers
        fused_model = fuser.fuse()
        self.model = fused_model

    @property
    def sparsity(self):
        return self.compute_sparsity()

    def compute_sparsity(self):
        total_param = 0
        zeros = 0
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                w = m.weight
                total_param += w.numel()
                zeros += w[w.eq(0.)].numel()

        return zeros / total_param
    
    def f2fxp(self, val):
        vfix = Fxp(val, signed=True, n_word=self.swl, n_frac=self.sfl)
        vfix = vfix.base_repr(10)
        vnp = np.array(vfix).astype(float)
        return torch.from_numpy(vnp).cuda()

    def scale_bias2int(self):
        """
        Convert the pre-computed scaling factor and bias to high precision integer
        """

        for n, m in self.model.named_modules():
            if isinstance(m, MulShift):
                m.fl = self.sfl
                scale = m.scale.cpu().numpy()
                bias = m.bias.cpu().numpy()

                # to numpy
                sint = self.f2fxp(scale)
                bint = self.f2fxp(bias)

                # insert back
                m.scale = sint.float()
                m.bias = bint.float()
    
    def hook(self):
        for n, m in self.model.named_modules():
            if isinstance(m, (IntMatMul, ConvOPS)):
                m.register_forward_hook(feature_hook(n, self.node_dict))
    
    def register_node(self):
        self.node_dict = {}
        
        # register hook
        self.hook()
    
    def fused_model(self):
        # self.scale_bias2int()
        return self.model
    
    def get_row_col(self, shape:List):
        if len(shape) == 4:
            scale, row, col = shape[1], shape[2], shape[3]
        elif len(shape) == 3:
            scale, row, col = 1, shape[1], shape[2]
        elif len(shape) == 2:
            scale, row, col = 1, shape[0], shape[1]
        return scale, row, col

    def flops(self, x1:torch.Tensor, x2:torch.Tensor):
        x_shape = list(x1.size())
        y_shape = list(x2.size())
        
        sx, rx, cx = self.get_row_col(x_shape)
        sy, ry, cy = self.get_row_col(y_shape)

        # density
        density = x2.ne(0.).float().sum() / x2.numel()

        assert cx == ry, "ERROR: incorrect MatMul Shape"

        flops = (cx + (cx - 1)) * rx * cy * sx
        return torch.tensor(flops)
            
    def save(self, path:str):
        model_name = os.path.join(path, "t2c_model.pth.tar")
        torch.save(self.model.state_dict(), model_name)

        # create the data path
        tensor_dir = os.path.join(path, "tensors")
        os.makedirs(tensor_dir, exist_ok=True)

        for k, v in tqdm(self.node_dict.items()):
            x1, x2, y = v

            torch.save(x1.int().cpu(), os.path.join(tensor_dir, f"{k}_x1.pt"))
            torch.save(x2.int().cpu(), os.path.join(tensor_dir, f"{k}_x2.pt"))
            torch.save(y.int().cpu(), os.path.join(tensor_dir, f"{k}_y.pt"))

    def dump_size(self, path):
        size_dict = {}
        tensor_dir = os.path.join(path, "tensors")
        json_path = os.path.join(tensor_dir, "matmul.json")

        nz = 0
        total = 0
        
        for n, m in self.model.named_modules():
            if isinstance(m, (IntMatMul, ConvOPS)):
                module_dict = {}

                module_dict["x_shape"] = m.x_shape.tolist()
                module_dict["y_shape"] = m.y_shape.tolist()
                module_dict["z_shape"] = m.z_shape.tolist()

                size_dict[n] = module_dict
            
            elif isinstance(m, _QBaseLinear):
                nz += m.mask.sum().item()
                total += m.mask.numel() 

        with open(json_path, "w") as outfile: 
            json.dump(size_dict, outfile)

        sparsity = 1 - nz / total
        print(f"Sparsity = {sparsity * 100:.2f}")
    
    def export(self, dataloader, path, export_samples):
        print("[T2C EXPORT]: Saving model and tensors!")
        assert export_samples < self.args.batch_size, f"Picking {export_samples} samples from {self.args.batch_size}!"
        
        # select a sample batch for quick inference
        inputs, target = next(iter(dataloader))

        # register hook
        self.register_node()

        # forward pass
        samples = inputs[:export_samples]
        out = self.model(samples.cuda())

        # save model and tensors
        self.save(path)
        self.dump_size(path)

    def bert_export(self, dataloader, path, export_samples):
        print("[T2C EXPORT for BERT]: Saving model and tensors!")
        assert export_samples < self.args.batch_size, f"Picking {export_samples} samples from {self.args.batch_size}!"
        
        batch = next(iter(dataloader))

        # register hook
        self.register_node()

        # forward pass
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        input_ids = batch['input_ids'].to(self.model.device)
        attention_mask = batch['attention_mask'].to(self.model.device)

        out = self.model(input_ids, attention_mask=attention_mask)

        # save model and tensors
        self.save(path)
        self.dump_size(path)