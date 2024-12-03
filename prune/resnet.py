"""
Weight pruning with CNN
"""

import os
import sys
sys.path.append("../torch2chip/")

import argparse
from src.stage.base import Execute
from src.data.vision.imagenet import ImageNet1K
from src.t2c.convert import Vanilla4Compress
from src.trainer.pruning import STrainer

parser = argparse.ArgumentParser(description='Llama')
parser.add_argument('--config_dir', type=str, default=None, help="Path to the configuration file (.yaml)")
args = parser.parse_args()

class PruneResNet(Execute):
    def __init__(self, config_dir):
        super().__init__(config_dir)

        model = self.create_model()
        converter = Vanilla4Compress(model=model, wbit=32, abit=32)
        self.model = converter.convert()

        # prepare dataloaders
        trainloader, testloader = self.prepare_dataloader()

        # trainer
        self.trainer = STrainer(
            model=self.model,
            trainloader=trainloader,
            validloader=testloader,
            config=self.config,
            logger=self.logger
        )

    def prepare_dataloader(self):
        data_gen = ImageNet1K(self.config_dir)

        trainloader, testloader = data_gen.run()
        return trainloader, testloader
    
    def run(self):
        self.trainer.fit()
        self.trainer.valid_epoch()
        self.logger.info("Baseline Model: Test accuracy = {:.3f}".format(self.trainer.logger_dict["valid_top1"]))


if __name__ == "__main__":
    executor = PruneResNet(args.config_dir)
    executor.run()