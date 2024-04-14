"""
Prepare the dataset and dataloader
"""
import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets

from src.utils.aug import CIFAR10Policy

def get_loader(args):
    # Preparing data
    if args.dataset == 'cifar10':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        if "vit" in args.model:
            print("Augmentation for ViT!")
            train_xforms = [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip()]
            
            test_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
            ])
        else:
            train_xforms = [transforms.RandomHorizontalFlip(), 
                            transforms.RandomCrop(32, padding=4)]
            
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        train_xforms += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        train_transform = transforms.Compose(train_xforms)

        trainset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        
        # test loader
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        num_classes, img_size = 10, 32
    
    elif 'imagenet' in args.dataset:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        # read the dataset
        trainset = datasets.ImageFolder(args.train_dir, transform=transform_train)
        testset = datasets.ImageFolder(args.val_dir, transform=transform_val)
        
        drop_last = True if args.mixup_active else False
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, drop_last=drop_last, sampler=train_sampler)        
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        
        num_classes = 100 if args.dataset == "imagenet-100" else 1000
        img_size = 224
    else:
        raise ValueError("Unrecegonized dataset!")
    
    return trainloader, testloader, num_classes, img_size

def get_ptq_dataloader(args):
    if args.dataset == 'cifar10':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        if "vit" in args.model:
            print("Augmentation for ViT!")
            train_xforms = [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip()]
            
            test_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
            ])
        else:
            train_xforms = [transforms.RandomHorizontalFlip(), 
                            transforms.RandomCrop(32, padding=4)]
            
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        train_xforms += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

        train_transform = transforms.Compose(train_xforms)

        trainset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)
        num_classes = 10

    elif args.dataset == 'imagenet':
        if not "vit" in args.model:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset

        trainset = datasets.ImageFolder(args.train_dir, transform=train_transform)
        testset = datasets.ImageFolder(args.val_dir, transform=test_transform)
        num_classes = 1000

    # random sampler on training set
    rand = torch.utils.data.RandomSampler(trainset, num_samples=args.num_samples)
    sampler = torch.utils.data.BatchSampler(rand, batch_size=args.batch_size, drop_last=False)

    trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_sampler=sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)

    testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=args.batch_size, 
                                            shuffle=False, 
                                            num_workers=args.workers, 
                                            pin_memory=True)
    return trainloader, testloader, num_classes