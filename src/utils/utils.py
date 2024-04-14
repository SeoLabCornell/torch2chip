"""
Utilities
"""
import shutil
import torch
import tabulate
import argparse
from collections import OrderedDict

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def print_table(values, columns, epoch, logger):
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.6f')
    if epoch == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    logger.info(table)

def lr_schedule(epoch):
    if epoch >= 100:
        factor = 0.1
        if epoch >= 150:
            factor = 0.01
    else:
        factor = 1.0
    return factor

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs

def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, save_path+filename)
    if is_best:
        shutil.copyfile(save_path+filename, save_path+'model_best.pth.tar')

def load_checkpoint(ckpt, state):
    checkpoint = torch.load(ckpt)
    sdict = checkpoint['state_dict']

    new_state_dict = OrderedDict()
    
    for k, v in sdict.items():
        name = k
        new_state_dict[name] = v

    state.update(new_state_dict)
    return state

def load_ddp_checkpoint(ckpt, state):
    checkpoint = torch.load(ckpt)
    sdict = checkpoint['state_dict']

    new_state_dict = OrderedDict()
    
    for k, v in sdict.items():
        name = k[7:]
        new_state_dict[name] = v

    state.update(new_state_dict)
    return state