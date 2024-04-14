import os
import sys

import logging
import argparse
import torchvision

sys.path.append("../torch2chip/")

from src.models.imagenet.mobilenetv1 import mobilenetv1

from timm.data.mixup import Mixup
from src.utils.ddp import init_distributed_mode, get_world_size
from src.utils.utils import str2bool
from src.utils.get_data import get_loader
from src.trainer.ddp import DDPTrainer

parser = argparse.ArgumentParser('DDP Training', add_help=False)
parser.add_argument('--model', type=str, help='model architecture')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_sch', type=str, default='step', help='learning rate scheduler')
parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120], help='Decrease learning rate at these epochs.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--weight-decay', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--log_file', type=str, default=None, help='path to log file')

# loss and gradient
parser.add_argument('--loss_type', type=str, default='cross_entropy', help='loss func')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers (default: 2)')

# dataset
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: CIFAR10 / ImageNet_1k')
parser.add_argument('--train_dir', type=str, default='./data/', help='training data directory')
parser.add_argument('--val_dir', type=str, default='./data/', help='test/validation data directory')

# model saving
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
parser.add_argument('--save_param', action='store_true', help='save the model parameters')

# Fine-tuning
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                    help='fine tuning from the pre-trained model, force the start epoch be zero')
parser.add_argument('--resume', default='', type=str, help='path of the pretrained model')

# ddp
parser.add_argument("--ddp", type=str2bool, nargs='?', const=True, default=False, help="enable DDP")
parser.add_argument('--seed', type=int, default=0, help='use random seed to make sure all the processes has the same model')
parser.add_argument('--device', default='cuda', help='device to use for training / testing')

# distributed training parameters
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--local-rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

# fine-tuning
parser.add_argument('--finetune', type=str, default='./', help='pre-trained model directory')

# mixup
parser.add_argument('--mixup', type=float, default=0,
                    help='mixup alpha, mixup enabled if > 0.')
parser.add_argument('--cutmix', type=float, default=0,
                    help='cutmix alpha, cutmix enabled if > 0.')
parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup_prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup_mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

# Augmentation parameters
parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                    help='Color jitter factor (enabled only when not using Auto/RandAug)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')

# * Random Erase params
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')

args = parser.parse_args()

def main():
    init_distributed_mode(args)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    logger = logging.getLogger('training')
    if args.log_file is not None:
        fileHandler = logging.FileHandler(args.save_path+args.log_file)
        fileHandler.setLevel(0)
        logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)
    logger.info(args)
    
    # get the effective batch size
    eff_batch_size = args.batch_size * get_world_size()
    logger.info(f"Effective Batch Size with DDP: {eff_batch_size}")

    # timm Mixup
    mixup_fn = None
    args.mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None

    # get the loader
    trainloader, testloader, num_classes, img_size = get_loader(args)

    if args.mixup_active:
        logger.info("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=num_classes)
    
    # model
    if args.model == "mobilenetv1":
        model = mobilenetv1(num_classes=num_classes)
    elif args.model == "resnet18":
        model = torchvision.models.resnet18()
    elif args.model == "resnet50":
        model = torchvision.models.resnet50()

    trainer = DDPTrainer(
        model=model,
        loss_type=args.loss_type,
        trainloader=trainloader,
        validloader=testloader,
        args=args,
        logger=logger,
        mixup_fn=mixup_fn
    )

    trainer.fit()


if __name__ == '__main__':
    main()

