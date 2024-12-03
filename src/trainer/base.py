import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from src.utils.utils import accuracy, AverageMeter, print_table, convert_secs2time, save_checkpoint
from src.trainer.scheduler import LabelSmoothingCrossEntropyLoss, LinearWarmupCosineAnnealingLR
from tqdm import tqdm

class Trainer(object):
    r"""Base deep neural network trainer. 
    """
    def __init__(self,
        model: nn.Module,
        trainloader, 
        validloader,
        config,
        logger,
    ):  
        # model architecture
        self.model = model

        # args
        self.config = config
        self.run_dir = config["save"]["run_dir"]

        # loader
        self.trainloader = trainloader
        self.validloader = validloader

        # optimizer time
        train_config = self.config["train"]
        loss_type = train_config["loss_type"]
        optim_type = train_config["optim_type"]
        lr_sch = train_config["lr_sch"]
        mix_prec = train_config["mix_prec"]
        self.epochs = train_config["epochs"]

        # loss func
        if loss_type == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        elif loss_type == "mse":
            self.criterion = torch.nn.MSELoss().cuda()
        elif loss_type == "smooth_ce":
            self.criterion = LabelSmoothingCrossEntropyLoss(self.config["dataset"]["num_classes"], smoothing=train_config["smoothing"])
        else:
            raise NotImplementedError("Unknown loss type")

        if optim_type == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=train_config["lr"], momentum=train_config["momentum"], weight_decay=train_config["weight_decay"])
        elif optim_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_config["lr"], betas=(0.9, 0.999), weight_decay=train_config["weight_decay"])
        elif optim_type == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_config["lr"], betas=(0.9, 0.95))

        # learning rate scheduler
        if lr_sch == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=train_config["schedule"], last_epoch=-1)
        elif lr_sch == "cos":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-5)
        elif lr_sch == "cos_warmup":
            warmup = self.config["train"].get("warmup", 1)
            self.lr_scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, warmup, max_epochs=self.epochs, warmup_start_lr=train_config["lr"], eta_min=1e-5)

        # cuda
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.model = self.model.cuda()

        if mix_prec:
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None

        # logger
        self.logger = logger
        self.logger_dict = {}
        self.logger.info("\nStart training: lr={}, loss={}, optim={}, run_dir={}".format(train_config["lr"], loss_type, optim_type, self.run_dir))
        self.logger.info("Total Epochs = {}".format(train_config["epochs"]))

    def base_forward(self, inputs, target):
        """Foward pass of NN
        """
        out = self.model(inputs)
        loss = self.criterion(out, target)
        return out, loss
    
    def amp_forward(self, inputs, target):
        """Mixed precision forward
        """
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = self.model(inputs)
            loss = self.criterion(out, target)
        return out, loss

    def base_backward(self, loss):
        """Basic backward pass
        """
        # zero grad
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def amp_backward(self, loss):
        """Mixed precision backward
        """
        self.optimizer.zero_grad()
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
    
    def train_step(self, inputs, target):
        """Training step at each iteration
        """
        if isinstance(self.criterion, nn.MSELoss):
            target = F.one_hot(target, 10).float()

        if self.scaler is not None:
            out, loss = self.amp_forward(inputs, target)
            self.amp_backward(loss)
        else:
            out, loss = self.base_forward(inputs, target)
            self.base_backward(loss)
        
        return out, loss

    def valid_step(self, inputs, target):
        """validation step at each iteration
        """
        if isinstance(self.criterion, nn.MSELoss):
            target = F.one_hot(target, 10).float()

        out, loss = self.base_forward(inputs, target)
            
        return out, loss

    def train_epoch(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        beta = AverageMeter()

        self.model.train()

        for idx, (inputs, target) in enumerate(tqdm(self.trainloader)):
            if self.use_cuda:
                inputs = inputs.cuda()
                target = target.cuda(non_blocking=True)
            
            out, loss = self.train_step(inputs, target)
            prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

            losses.update(loss.mean().item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        
        for name, param in self.model.named_parameters():
            if 'beta' in name:
                beta.update(param.item())

        self.logger_dict["train_loss"] = losses.avg
        self.logger_dict["train_top1"] = top1.avg
        self.logger_dict["train_top5"] = top5.avg

    def valid_epoch(self):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        self.model.eval()
        with torch.no_grad():
            for idx, (inputs, target) in enumerate(tqdm(self.validloader)):
                if self.use_cuda:
                    inputs = inputs.cuda()
                    target = target.cuda(non_blocking=True)

                out, loss = self.valid_step(inputs, target)
                prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

                losses.update(loss.mean().item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

        self.logger_dict["valid_loss"] = losses.avg
        self.logger_dict["valid_top1"] = top1.avg
        self.logger_dict["valid_top5"] = top5.avg

    def fit(self):
        start_time = time.time()
        epoch_time = AverageMeter()
        best_acc = 0.
        for epoch in range(self.epochs):
            self.logger_dict["ep"] = epoch+1
            self.logger_dict["lr"] = self.optimizer.param_groups[0]['lr']
            
            # training and validation
            self.train_epoch()
            self.valid_epoch()
            self.lr_scheduler.step()

            is_best = self.logger_dict["valid_top1"] > best_acc
            if is_best:
                best_acc = self.logger_dict["valid_top1"]

            state = {
                'state_dict': self.model.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
                'optimizer': self.optimizer.state_dict(),
            }

            filename=f"checkpoint.pth.tar"
            save_checkpoint(state, is_best, self.run_dir, filename=filename)

            # terminal log
            columns = list(self.logger_dict.keys())
            values = list(self.logger_dict.values())
            print_table(values, columns, epoch, self.logger)

            # record time
            e_time = time.time() - start_time
            epoch_time.update(e_time)
            start_time = time.time()

            need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (self.epochs - epoch))
            print('[Need: {:02d}:{:02d}:{:02d}]'.format(
                need_hour, need_mins, need_secs))