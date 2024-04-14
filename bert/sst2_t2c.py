"""
Torch2Chip: Model Conversion
"""

import sys
sys.path.append("../torch2chip/")
import os
import logging
import argparse

import torch
import torch.nn.functional as F

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.t2c.convert import BERT4Compress
from src.utils.utils import str2bool, save_checkpoint, load_checkpoint
from src.t2c.t2c import T2C

parser = argparse.ArgumentParser(description='T2C for BERT')
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
parser.add_argument('--loss_type', type=str, default='mse', help='loss func')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
parser.add_argument('--workers', type=int, default=16,help='number of data loading workers (default: 2)')

parser.add_argument("--mixed_prec", type=str2bool, nargs='?', const=True, default=False, help="enable amp")

# model saving
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
parser.add_argument('--save_param', action='store_true', help='save the model parameters')
parser.add_argument('--export_samples', type=int, default=1, help="Number of samples for export")

# Fine-tuning
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                    help='fine tuning from the pre-trained model, force the start epoch be zero')
parser.add_argument('--resume', default='', type=str, help='path of the pretrained model')

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

def evaluate(model, validloader, logger, device):
    total = 0
    correct = 0
    
    model.eval()
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
    model = AutoModelForSequenceClassification.from_pretrained('gchhablani/bert-base-cased-finetuned-sst2')
    tokenizer = AutoTokenizer.from_pretrained('gchhablani/bert-base-cased-finetuned-sst2')

    # model = AutoModelForSequenceClassification.from_pretrained('gokuls/bert-base-sst2')
    # tokenizer = AutoTokenizer.from_pretrained('gokuls/bert-base-sst2')

    state_tmp = load_checkpoint(ckpt=args.resume, state=model.state_dict())
    wrapper = BERT4Compress(model=model, state_dict=state_tmp)
    model = wrapper.reload(wqtype=args.wqtype, xqtype=args.xqtype)

    # resume from the checkpoint
    model.load_state_dict(state_tmp)
    logger.info(f"Loaded checkpoint from: {args.resume}")
    
    # # load and encode the data
    validset = load_dataset('glue', 'sst2', split='validation')

    # trainset = process_sst(trainset, tokenizer)
    validset = process_sst(validset, tokenizer)
    validloader = torch.utils.data.DataLoader(validset, batch_size=32)
    
    model.to(device)

    # evaluate the pre-trained model
    evaluate(model, validloader, logger, device)

    # convert the model
    t2c = T2C(model=model, swl=32, sfl=26, args=args)
    model = t2c.fused_model()

    # evaluate the pre-trained model
    evaluate(model, validloader, logger, device)

    t2c.bert_export(validloader, path=args.save_path, export_samples=args.export_samples)
    

if __name__ == "__main__":
    main()