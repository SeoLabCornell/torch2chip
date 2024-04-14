"""
PTQ Example of BERT on MRPC
"""

import sys
sys.path.append("../torch2chip/")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import argparse

import torch
import torch.nn.functional as F
import evaluate

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.utils import str2bool, save_checkpoint, load_checkpoint, AverageMeter
from src.t2c.convert import BERT4Compress
from src.trainer.ptq import PTQBERT

parser = argparse.ArgumentParser(description='T2C for BERT')
# parser.add_argument('--model', type=str, help='model architecture')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_sch', type=str, default='step', help='learning rate scheduler')
parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120], help='Decrease learning rate at these epochs.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--weight-decay', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--log_file', type=str, default=None, help='path to log file')

# loss and gradient
parser.add_argument('--loss_type', type=str, default='mse', help='loss func')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
parser.add_argument('--workers', type=int, default=16,help='number of data loading workers (default: 2)')

parser.add_argument("--mixed_prec", type=str2bool, nargs='?', const=True, default=False, help="enable amp")

# model saving
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
parser.add_argument('--save_param', action='store_true', help='save the model parameters')

# ptq
parser.add_argument('--wbit', type=int, default=8, help="Weight Precision")
parser.add_argument('--abit', type=int, default=8, help="Input Precision")
parser.add_argument('--wqtype', type=str, default="adaround", help='Weight quantizer')
parser.add_argument('--xqtype', type=str, default="lsq", help='Input quantizer')
parser.add_argument('--num_samples', type=int, default=1024, help="Number of samples for calibration")
parser.add_argument("--layer_trainer", type=str2bool, nargs='?', const=True, default=False, help="enable layer-wise training / calibration")

args = parser.parse_args()

def process_mrpc(dataset, tokenizer):
    def encode(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')

    dataset = dataset.map(encode, batched=True)
    
    # reformat the dataset
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    
    return dataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model = AutoModelForSequenceClassification.from_pretrained('Intel/bert-base-uncased-mrpc')
    tokenizer = AutoTokenizer.from_pretrained('Intel/bert-base-uncased-mrpc')

    wrapper = BERT4Compress(model=model)
    model = wrapper.convert()

    # load and encode the data
    trainset = load_dataset('glue', 'mrpc', split='train')
    validset = load_dataset('glue', 'mrpc', split='validation')
    
    trainset = process_mrpc(trainset, tokenizer)
    validset = process_mrpc(validset, tokenizer)
    
    # dataloader
    # random samples on training set
    rand = torch.utils.data.RandomSampler(trainset, num_samples=args.num_samples)
    sampler = torch.utils.data.BatchSampler(rand, batch_size=args.batch_size, drop_last=False)

    # loader
    trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=sampler, num_workers=args.workers, pin_memory=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=32)
    model.to(device)

    trainer = PTQBERT(
        model = model,
        loss_type=args.loss_type,
        trainloader=trainloader,
        validloader=validloader,
        args=args,
        logger=logger
    )

    # PTQ start
    trainer.fit()
    model = getattr(trainer, "model")
    
    # metric
    metric = evaluate.load('glue', 'mrpc')

    # meter
    f1 = AverageMeter()
    accuracy = AverageMeter()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(validloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]

            outputs = model(**batch)
            prediction = F.softmax(outputs.logits, dim=-1)
            _, pred = torch.max(prediction, dim=1)

            final_score = metric.compute(predictions=pred, references=labels)
            f1.update(final_score["f1"])
            accuracy.update(final_score["accuracy"])

    print(f"Average F1 Score = {f1.avg:.3f}")
    print(f"Average Accuracy = {accuracy.avg:.3f}")

    state = {
        'state_dict': model.state_dict(),
        'acc': f1.avg
    }

    # save the checkpoint
    filename=f"checkpoint.pth.tar"
    save_checkpoint(state, True, args.save_path, filename=filename)
    logger.info("Model Saved")

if __name__ == "__main__":
    main()