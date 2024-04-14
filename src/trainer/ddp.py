"""
Distributed Data Parallel Training
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.optim.optim_factory as optim_factory

from tqdm import tqdm
from timm.loss import SoftTargetCrossEntropy
from src.utils.utils import accuracy, AverageMeter, print_table, convert_secs2time, save_checkpoint

class DDPTrainer(object):
    def __init__(self,
        model: nn.Module,
        loss_type: str, 
        trainloader, 
        validloader,
        args,
        logger,
        mixup_fn=None
    ):  
        # model architecture
        self.model = model
        
        # logger
        self.logger = logger
        self.logger_dict = {}

        # args
        self.args = args

        # loader
        self.trainloader = trainloader
        self.validloader = validloader
        
        # loss func
        if loss_type == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        elif loss_type == "soft_ce":
            self.criterion = SoftTargetCrossEntropy()

        logger.info("Prepare the deployment to DDP!")
        self.local_rank = int(os.environ["LOCAL_RANK"])

        self.device = torch.device(args.device)
        self.model = self.model.to(self.device)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.gpu])
        self.model_without_ddp = self.model.module

        # optimizer
        param_groups = optim_factory.add_weight_decay(self.model_without_ddp, args.weight_decay)
        if self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(param_groups, lr=self.args.lr, momentum=0.9)
        elif args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(param_groups, lr=self.args.lr, betas=(self.args.beta1, self.args.beta2), weight_decay=self.args.weight_decay)
        elif self.args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(param_groups, lr=self.args.lr, betas=(0.9, 0.95))
        
        # learning rate scheduler
        if args.lr_sch == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.schedule, last_epoch=-1)
        elif args.lr_sch == "cos":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=1e-5)

        # amp
        self.scaler = torch.cuda.amp.GradScaler()

        # mix up
        self.mixup = mixup_fn

    def base_forward(self, x:torch.Tensor, y:torch.Tensor):
        # distributed sampler
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = self.model(x)
            loss = self.criterion(out, y)
        
        return out, loss

    def base_eval(self, x:torch.Tensor, y:torch.Tensor):
        # distributed sampler
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = self.model(x)
            loss = F.cross_entropy(out, y)
        
        return out, loss
    
    def base_backward(self, loss):
        self.optimizer.zero_grad()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def train_step(self, inputs, target):
        out, loss = self.base_forward(inputs, target)
        self.base_backward(loss)
        return out, loss
    
    def valid_step(self, inputs, target):
        out, loss = self.base_eval(inputs, target)
        return out, loss
    
    def train_epoch(self, epoch:int):
        # meter
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # update sampling
        self.trainloader.sampler.set_epoch(epoch)

        self.model.train()
        pbar = tqdm(self.trainloader, desc="Train")
        for idx, (inputs, target) in enumerate(pbar):
            inputs = inputs.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            
            if self.mixup is not None:
                inputs, target = self.mixup(inputs, target)

            out, loss = self.train_step(inputs, target)

            if self.local_rank == 0:
                losses.update(loss.mean().item(), inputs.size(0))
        
            pbar.set_postfix(loss = loss.mean().item())

        # logger dict
        self.logger_dict["ep"] = epoch+1
        self.logger_dict["lr"] = self.optimizer.param_groups[0]['lr']
        self.logger_dict["train_loss"] = losses.avg

    def valid_epoch(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.eval()
        pbar = tqdm(self.validloader, desc="Valid")
        with torch.no_grad():
            for idx, (inputs, target) in enumerate(pbar):
                inputs = inputs.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                out, loss = self.valid_step(inputs, target)

                if self.local_rank == 0:
                    prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

                    losses.update(loss.mean().item(), inputs.size(0))
                    top1.update(prec1.item(), inputs.size(0))
                    top5.update(prec5.item(), inputs.size(0))
                
                pbar.set_postfix(loss = loss.mean().item())
    
        self.logger_dict["valid_loss"] = losses.avg
        self.logger_dict["valid_top1"] = top1.avg
        self.logger_dict["valid_top5"] = top5.avg

    def fit(self):
        self.logger.info(f"Training Start! | Epochs: {self.args.epochs} | Batch_size = {self.args.batch_size}")

        start_time = time.time()
        epoch_time = AverageMeter()

        for epoch in tqdm(range(self.args.epochs)):
            # training and validation
            self.train_epoch(epoch)
            self.valid_epoch()
            self.lr_scheduler.step()

            if self.local_rank == 0:
                state = {
                    'state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'optimizer': self.optimizer.state_dict(),
                }

                filename=f"checkpoint.pth.tar"
                save_checkpoint(state, False, self.args.save_path, filename=filename)

                # terminal log
                columns = list(self.logger_dict.keys())
                values = list(self.logger_dict.values())
                print_table(values, columns, epoch, self.logger)

                # record time
                e_time = time.time() - start_time
                epoch_time.update(e_time)
                start_time = time.time()

                need_hour, need_mins, need_secs = convert_secs2time(
                epoch_time.avg * (self.args.epochs - epoch))
                print('[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs))