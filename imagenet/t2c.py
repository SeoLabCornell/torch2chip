"""
T2C Example on the ImagetNet-1K dataset
"""

import os
import sys
import logging
import argparse

sys.path.append("../torch2chip/")

from src.utils.utils import str2bool, load_checkpoint
from src.utils.get_data import get_ptq_dataloader
from src.trainer.base import Trainer
from src.trainer.pruning import STrainer
from src.t2c.t2c import T2C
from src.trainer.vision.ptq import PTQ, PTQViT
from src.t2c.convert import Vanilla4Compress, ViTV4C
from src.models.imagenet.mobilenetv1 import mobilenetv1

from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from timm.models.vision_transformer import vit_tiny_patch16_224, vit_base_patch16_224, vit_small_patch16_224
from timm.models.swin_transformer import swin_tiny_patch4_window7_224, swin_base_patch4_window7_224
from torchvision.models import vgg16_bn, VGG16_BN_Weights

TRAINERS = {
    "base": Trainer,
    "sparse": STrainer,
    "ptq": PTQ,
    "qattn": PTQViT
}

parser = argparse.ArgumentParser(description='T2C Training')
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
parser.add_argument('--workers', type=int, default=16,help='number of data loading workers (default: 2)')

# dataset
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: CIFAR10 / ImageNet_1k')
parser.add_argument('--data_path', type=str, default='./data/', help='data directory')
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

# amp training
parser.add_argument("--mixed_prec", type=str2bool, nargs='?', const=True, default=False, help="enable amp")

# trainer
parser.add_argument('--trainer', type=str, default='base', help='trainer type')

# prune
parser.add_argument('--pruner', type=str, default='element', help='trainer type')
parser.add_argument('--prune_ratio', default=0.9, type=float, help='target prune ratio')
parser.add_argument('--drate', default=0.5, type=float, help='additional pruning rate before regrow')
parser.add_argument('--warmup', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--prune_freq', type=int, default=1000, help='Iteration gap between sparsity update')
parser.add_argument('--final_epoch', type=int, default=160, help='Final pruning epoch')

# ptq
parser.add_argument('--wbit', type=int, default=8, help="Weight Precision")
parser.add_argument('--abit', type=int, default=8, help="Input Precision")
parser.add_argument('--swl', type=int, default=32, help="Precision of scaling factor")
parser.add_argument('--sfl', type=int, default=26, help="Fractional bits")
parser.add_argument('--wqtype', type=str, default="adaround", help='Weight quantizer')
parser.add_argument('--xqtype', type=str, default="lsq", help='Input quantizer')
parser.add_argument('--num_samples', type=int, default=1024, help="Number of samples for calibration")
parser.add_argument('--export_samples', type=int, default=10, help="Number of samples for export")
parser.add_argument("--layer_trainer", type=str2bool, nargs='?', const=True, default=False, help="enable layer-wise training / calibration")

args = parser.parse_args()

def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    # initialize terminal logger
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

    args.mixup_active = False
    trainloader, testloader, num_classes = get_ptq_dataloader(args)

    # model
    if args.model == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        wrapper = Vanilla4Compress

    elif args.model == "resnet34":
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        wrapper = Vanilla4Compress
    
    elif args.model == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        wrapper = Vanilla4Compress

    elif args.model == "mobilenetv1":
        model = mobilenetv1()
        wrapper = Vanilla4Compress

    elif args.model == "vgg16_bn":
        model = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        wrapper = Vanilla4Compress

    elif args.model == "vit_tiny":
        model = vit_tiny_patch16_224(pretrained=True)
        wrapper = ViTV4C

    elif args.model == "vit_small":
        model = vit_small_patch16_224(pretrained=True)
        wrapper = ViTV4C

    elif args.model == "vit_base":
        model = vit_base_patch16_224(pretrained=True)
        wrapper = ViTV4C

    elif args.model == "swin_tiny_patch4_window7_224":
        model = swin_tiny_patch4_window7_224(pretrained=True)
        wrapper = ViTV4C
    
    elif args.model == "swin_base_patch4_window7_224":
        model = swin_base_patch4_window7_224(pretrained=True)
        wrapper = ViTV4C

    else:
        raise NotImplementedError(f"Unknown model architecture: {args.model}")
    
    # load the state_dict
    logger.info("=> loading checkpoint...")
    state_tmp = load_checkpoint(ckpt=args.resume, state=model.state_dict())

    # convert the model to the compression-ready model
    if args.wbit < 32 or args.wbit < 32:
        converter = wrapper(model, wbit=args.wbit, abit=args.abit, state_dict=state_tmp)
        model = converter.reload_fake_quant(wqtype=args.wqtype, xqtype=args.xqtype)

    # resume from the checkpoint
    model.load_state_dict(state_tmp)
    logger.info(f"Loaded checkpoint from: {args.resume}")

    # define the trainer
    trainer = TRAINERS[args.trainer](
        model=model,
        loss_type=args.loss_type,
        trainloader=trainloader,
        validloader=testloader,
        args=args,
        logger=logger,
    )

    if args.evaluate:
        # pre-trained baseline
        trainer.valid_epoch()
        logger.info("[Pre-trained Model]: Test accuracy = {:.3f}".format(trainer.logger_dict["valid_top1"]))

        # t2c and model fuse
        t2c = T2C(model=model, swl=args.swl, sfl=args.sfl, args=args)
        qmodel = t2c.fused_model()

        # update model
        setattr(trainer, "model", qmodel.cuda())
        trainer.valid_epoch()
        logger.info("[After fusing]: Test accuracy = {:.3f}".format(trainer.logger_dict["valid_top1"]))

        # export the files
        t2c.export(testloader, path=args.save_path, export_samples=args.export_samples)

    else:
        exit()

if __name__ == '__main__':
    main()