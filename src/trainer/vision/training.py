import torch
import torch.optim as optim
import torchvision
from tqdm import tqdm
from src.utils.utils import accuracy, AverageMeter
from src.data.vision.imagenet import ImageNet1K
from src.stage.base import Execute

class ImageNetTraining(Execute):
    def __init__(self, config_dir, model):
        super().__init__(config_dir)
        self.model = model

        datastage = ImageNet1K(config_dir)
        self.trainloader, self.testloader = datastage.run()

    def prepare_training(self):
        # optimizer time
        train_config = self.config["train"]

        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=train_config["lr"], momentum=train_config["momentum"], weight_decay=train_config["weight_decay"])
        return criterion, optimizer

    def train_epoch(self, model, criterion, optimizer):
        scaler = torch.amp.GradScaler()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        model.train()

        for idx, (images, target) in enumerate(tqdm(self.trainloader)):
            images = images.to(self.device)
            target = target.to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(images)
                loss = criterion(out, target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

            losses.update(loss.mean().item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
        
        return losses.avg, top1.avg
    
    def test_epoch(self, model, criterion):
        losses = AverageMeter()
        top1 = AverageMeter()

        model.eval()
        for idx, (images, target) in enumerate(tqdm(self.testloader)):
            images = images.to(self.device)
            target = target.to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(images)
                loss = criterion(out, target)

            prec1, prec5 = accuracy(out.data, target, topk=(1, 5))
            losses.update(loss.mean().item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
        
        return losses.avg, top1.avg

    def run(self):
        criterion, optimizer = self.prepare_training()
        for i in range(50):
            train_loss, train_acc = self.train_epoch(model, criterion, optimizer)
            test_loss, test_acc = self.test_epoch(model, criterion)

            print(f"Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Test loss: {test_loss:.3f} | Test Acc = {test_acc:.3f}")            