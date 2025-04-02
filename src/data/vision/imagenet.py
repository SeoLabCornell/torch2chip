"""
ImageNet data preparation
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from torchvision import datasets
from src.data.base import DataStage
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class ClassificationPresetTrain:
    """
    Adopted from 
    """
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        backend="pil",
        use_v2=False,
    ):

        trans = []
        backend = backend.lower()
        if backend == "tensor":
            trans.append(transforms.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        trans.append(transforms.v2.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))
        if hflip_prob > 0:
            trans.append(transforms.v2.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(transforms.v2.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                trans.append(transforms.v2.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                trans.append(transforms.v2.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = transforms.v2.AutoAugmentPolicy(auto_augment_policy)
                trans.append(transforms.v2.AutoAugment(policy=aa_policy, interpolation=interpolation))

        if backend == "pil":
            trans.append(transforms.v2.PILToTensor())

        trans.extend(
            [
                transforms.v2.ToDtype(torch.float, scale=True) if use_v2 else transforms.ConvertImageDtype(torch.float),
                transforms.v2.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.v2.RandomErasing(p=random_erase_prob))

        if use_v2:
            trans.append(transforms.ToPureTensor())

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class VisionData(DataStage):
    """
    Basic stage for vision dataset
    """
    def __init__(self, config_dir):
        super().__init__(config_dir)
        self.train_dir = self.config["dataset"]["train_dir"]
        self.test_dir = self.config["dataset"]["test_dir"]
        self.num_samples = self.config["dataset"]["samples"]
        self.mean = self.config["dataset"].get("mean", IMAGENET_DEFAULT_MEAN)
        self.std = self.config["dataset"].get("std", IMAGENET_DEFAULT_STD)

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
        train = ClassificationPresetTrain(
            crop_size=224
        )

        test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
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
                    shuffle=True,
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