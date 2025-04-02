"""
Vision model stage for compression
"""

import os
import torch
import sys
sys.path.append("../torch2chip/")
import argparse

from collections import OrderedDict
from src.stage.base import Execute
from src.t2c.convert import ViTV4C
from src.trainer.vision.ptq import PTQViT
from src.data.vision.imagenet import ImageNet1K
from src.t2c.t2c import T2C, get_stats_dict
from src.utils.utils import gpufloat2cpuint
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Compress Vision Transformers')
parser.add_argument('--config_dir', type=str, default=None, help="Path to the configuration file (.yaml)")
args = parser.parse_args()

class CompressViT(Execute):
    def __init__(self, config_dir):
        super().__init__(config_dir)

        model = self.create_model()

        qconfig = self.config["quantization"]
        wbit = qconfig["wbit"]
        abit = qconfig["abit"]

        smooth = qconfig.get("smooth", None)
        
        converter = ViTV4C(model=model, wbit=wbit, abit=abit)
        self.model = converter.convert()

        # prepare dataloaders
        trainloader, testloader = self.prepare_dataloader()

        if not smooth:
            self.trainer = PTQViT(
                model=self.model,
                trainloader=trainloader,
                testloader=testloader,
                logger=self.logger,
                config=self.config    
            )
        else:
            pass
    
    def register_run_dir(self):
        super().register_run_dir()
        
        self.t2c_dir = os.path.join(self.run_dir, "t2c")
        self.t2c_model_dir = os.path.join(self.t2c_dir , "t2c_model.pth.tar")

        self.tensors_dir = os.path.join(self.t2c_dir , "tensors")
        if not os.path.isdir(self.tensors_dir):
            os.makedirs(self.tensors_dir, exist_ok=True)

    def prepare_dataloader(self):
        data_gen = ImageNet1K(self.config_dir)
        
        trainloader, testloader = data_gen.run()
        return trainloader, testloader

    def ptq(self):
        wqtype = self.config["quantization"]["wqtype"]
        xqtype = self.config["quantization"]["xqtype"]
        method = f"w{wqtype}_a{xqtype}"
        self.logger.info(f"PTQ start! {method}")
        self.trainer.fit()

        fake_quant_model = getattr(self.trainer, "model")
        self.print_arch(fake_quant_model, "fake_quantized_model")
        
        self.trainer.valid_epoch()
        self.logger.info("[After PTQ] Test accuracy = {:.2f}".format(self.trainer.logger_dict["valid_top1"]))

        return fake_quant_model

    def save(self, t2c:T2C):
        if self.config["t2c"]["save_model"]:
            t2c_model = getattr(t2c, "model")
            torch.save(t2c_model.state_dict(), self.t2c_model_dir)

        # precision
        xdytpe = torch.int8 if self.config["t2c"]["xdtype"] <= 8 else torch.float16
        ydytpe = torch.int32 if self.config["t2c"]["ydtype"] < 32 else torch.float16

        x1_stats_dict = OrderedDict()
        x2_stats_dict = OrderedDict()
        out_stats_dict = OrderedDict()
        for k, v in tqdm(t2c.node_dict.items()):
            x1, x2, y = v

            x1 = gpufloat2cpuint(x1, xdytpe)
            x2 = gpufloat2cpuint(x2, xdytpe)
            y = gpufloat2cpuint(y, ydytpe)

            if self.config["t2c"]["save_tensors"]:
                torch.save(x1, os.path.join(self.tensors_dir, f"{k}_x1.pt"))
                torch.save(x2, os.path.join(self.tensors_dir, f"{k}_x2.pt"))
                torch.save(y, os.path.join(self.tensors_dir, f"{k}_y.pt"))
            
            x1stats = get_stats_dict(x1) 
            x2stats = get_stats_dict(x2)
            ystats = get_stats_dict(y)

            x1_stats_dict[k] = x1stats
            x2_stats_dict[k] = x2stats
            out_stats_dict[k] = ystats

        torch.save(out_stats_dict, os.path.join(self.tensors_dir, f"matmul_output_stats.pt"))
        torch.save(x1_stats_dict, os.path.join(self.tensors_dir, f"matmul_x1_stats.pt"))
        torch.save(x2_stats_dict, os.path.join(self.tensors_dir, f"matmul_x2_stats.pt"))
    
    def t2c(self, fake_quant_model):
        t2c = T2C(model=fake_quant_model, config=self.config)
        fused_model = t2c.fused_model()

        export_samples = self.config["export"]["export_samples"]
        if export_samples > 0:
            t2c.register_node()

        assert hasattr(self, "trainer"), "Trainer must be defined before running T2C fusion!"
        setattr(self.trainer, "model", fused_model.to(self.device))

        self.trainer.valid_epoch()
        self.logger.info("[After fusing]: Test accuracy = {:.2f}".format(self.trainer.logger_dict["valid_top1"]))

        if export_samples > 0:
            self.save(t2c)

        fused_model = getattr(self.trainer, "model")
        self.print_arch(fused_model, "fused_model")

        return fused_model

    def run(self):
        fake_quant_model = self.ptq()
        fused_model = self.t2c(fake_quant_model)


if __name__ == "__main__":
    executor = CompressViT(args.config_dir)
    executor.run()
