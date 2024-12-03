"""
Vision model stage for compression
"""

import sys
sys.path.append("../torch2chip/")

import argparse
from src.stage.base import Execute
from src.t2c.convert import ViTV4C
from src.trainer.vision.ptq import PTQViT
from src.data.vision.imagenet import ImageNet1K
from src.t2c.t2c import T2C

parser = argparse.ArgumentParser(description='Llama')
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

    def t2c(self, fake_quant_model):
        t2c = T2C(model=fake_quant_model, config=self.config)
        fused_model = t2c.fused_model()

        assert hasattr(self, "trainer"), "Trainer must be defined before running T2C fusion!"
        setattr(self.trainer, "model", fused_model.to(self.device))

        self.trainer.valid_epoch()
        self.logger.info("[After fusing]: Test accuracy = {:.3f}".format(self.trainer.logger_dict["valid_top1"]))

        fused_model = getattr(self.trainer, "model")
        self.print_arch(fused_model, "fused_model")

        return fused_model

    def run(self):
        fake_quant_model = self.ptq()
        fused_model = self.t2c(fake_quant_model)


if __name__ == "__main__":
    executor = CompressViT(args.config_dir)
    executor.run()
