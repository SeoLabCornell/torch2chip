"""
ImageNet data preparation
"""

import os
import torch
import torchvision.transforms as transforms

from torchvision import datasets
from src.data.base import DataStage

class VisionData(DataStage):
    """
    Basic stage for vision dataset
    """
    def __init__(self, config_dir):
        super().__init__(config_dir)
        self.train_dir = self.config["dataset"]["train_dir"]
        self.test_dir = self.config["dataset"]["test_dir"]
        self.num_samples = self.config["dataset"]["samples"]

    def __len__(self):
        return self.num_samples

class ImageNet1K(VisionData):
    def __init__(self, config_dir):
        super().__init__(config_dir)
        self.num_classes = 1000
        self.num_workers = self.config["dataset"]["num_workers"]

    def __name__(self):
        return "ImageNet-1K"
    
    def prepare_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # mean = [0.5, 0.5, 0.5]
        # std = [0.5, 0.5, 0.5]

        train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        return train, test
    
    def prepare_loader(self):
        trtf, tetf = self.prepare_transform()

        trainset = datasets.ImageFolder(self.train_dir, transform=trtf)
        testset = datasets.ImageFolder(self.test_dir, transform=tetf)

        if self.num_samples != -1:
            rand = torch.utils.data.RandomSampler(trainset, num_samples=self.num_samples)
            sampler = torch.utils.data.BatchSampler(rand, batch_size=self.batch_size, drop_last=False)
        else:
            sampler = None
        
        if self.is_ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, sampler=sampler)
            testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

        else:
            if self.num_samples != -1:
                trainloader = torch.utils.data.DataLoader(
                    trainset,
                    batch_sampler=sampler,
                    num_workers=self.num_workers,
                    pin_memory=True
                )
            else:
                trainloader = torch.utils.data.DataLoader(
                    trainset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True
                )

            testloader = torch.utils.data.DataLoader(
                testset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        
        return trainloader, testloader
    
    def run(self):
        self.logger.info("Preparing ImageNet-1K...")
        trainloader, testloader = self.prepare_loader()
        self.logger.info(f"Done | Train size: {len(trainloader)} | Test size: {len(testloader)}")

        return trainloader, testloader