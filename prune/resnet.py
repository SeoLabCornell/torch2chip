"""
Weight pruning with CNN
"""

import os
import sys
sys.path.append("../torch2chip/")

import argparse
import torchvision
from src.stage.base import Execute
from src.data.vision.imagenet import ImageNet1K
from src.trainer.vision.training import ImageNetTraining

parser = argparse.ArgumentParser(description='Llama')
parser.add_argument('--config_dir', type=str, default=None, help="Path to the configuration file (.yaml)")
args = parser.parse_args()

class PruneResNet(Execute):
    def __init__(self, config_dir):
        super().__init__(config_dir)
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        model = model.to(self.device)
        
        # trainer
        self.trainer = ImageNetTraining(config_dir)
    
    def run(self):
        self.trainer.run()

if __name__ == "__main__":
    executor = PruneResNet(args.config_dir)
    executor.run()