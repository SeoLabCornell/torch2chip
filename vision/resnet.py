"""
ResNet-50
"""

import sys
sys.path.append("../torch2chip/")

import argparse
from src.trainer.vision.ptq import PTQ
from src.stage.base import Execute
from src.t2c.convert import Vanilla4Compress
from src.data.vision.imagenet import ImageNet1K
from src.t2c.t2c import T2C

parser = argparse.ArgumentParser(description='Llama')
parser.add_argument('--config_dir', type=str, default=None, help="Path to the configuration file (.yaml)")
args = parser.parse_args()

class CompressResNet(Execute):
    def __init__(self, config_dir):
        super().__init__(config_dir)

        model = self.create_model()
        converter = Vanilla4Compress(model=model, wbit=32, abit=32)
        self.model = converter.convert()

        # prepare dataloaders
        trainloader, self.testloader = self.prepare_dataloader()

        # quantizer
        self.trainer = PTQ(
            model=self.model, 
            trainloader=trainloader,
            testloader=self.testloader,
            config=self.config,
            logger=self.logger
        )

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
        self.logger.info("[After fusing]: Test accuracy = {:.2f}".format(self.trainer.logger_dict["valid_top1"]))

        fused_model = getattr(self.trainer, "model")
        self.print_arch(fused_model, "fused_model")

        # export the files
        t2c.export(self.testloader, path=self.run_dir, export_samples=1)

        return fused_model

    def run(self):
        fake_quant_model = self.ptq()
        fused_model = self.t2c(fake_quant_model)


if __name__ == "__main__":
    executor = CompressResNet(args.config_dir)
    executor.run()