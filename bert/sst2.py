"""
PTQ Example of BERT on SST2
"""

import sys
sys.path.append("../torch2chip/")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import argparse

import torch
import torch.nn.functional as F

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.t2c.convert import BERT4Compress
from src.trainer.ptq import PTQBERT
from src.utils.utils import str2bool, save_checkpoint

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

def process_sst(dataset, tokenizer):
    def encode(examples):
        return tokenizer(examples['sentence'], truncation=True, padding='max_length')
    
    dataset = dataset.map(encode, batched=True)
    
    # reformat the dataset
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
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


    # define the model
    # model = AutoModelForSequenceClassification.from_pretrained('gokuls/bert-base-sst2')
    # tokenizer = AutoTokenizer.from_pretrained('gokuls/bert-base-sst2')

    model = AutoModelForSequenceClassification.from_pretrained('gchhablani/bert-base-cased-finetuned-sst2')
    tokenizer = AutoTokenizer.from_pretrained('gchhablani/bert-base-cased-finetuned-sst2')

    wrapper = BERT4Compress(model=model)
    model = wrapper.convert()
    
    # load and encode the data
    trainset = load_dataset('glue', 'sst2', split='train') 
    validset = load_dataset('glue', 'sst2', split='validation')

    trainset = process_sst(trainset, tokenizer)
    validset = process_sst(validset, tokenizer)

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
    
    total = 0
    correct = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(validloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            
            prediction = F.softmax(outputs.logits, dim=-1)
            _, pred = torch.max(prediction, dim=1) 

            total += labels.size(0)
            correct += (pred == labels).sum().item()

    accuracy = correct / total
    logger.info(f'Test Accuracy After PTQ: {accuracy:.4f}')
    
    state = {
        'state_dict': trainer.model.state_dict(),
        'acc': accuracy
    }

    # save the checkpoint
    filename=f"checkpoint.pth.tar"
    save_checkpoint(state, True, args.save_path, filename=filename)
    logger.info("Model Saved")

if __name__ == "__main__":
    main()