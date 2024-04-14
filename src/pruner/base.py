"""
Base pruner
"""

import torch
import torch.nn as nn
from torch import optim

class CosineDecay(object):
    def __init__(self, prune_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']

class Pruner(object):
    """
    Base pruner
    """
    def __init__(self, 
                model:nn.Module, 
                prune_ratio:float, 
                warmup:int, 
                final_epoch: int,
                dataloader, 
                prune_freq:float,
                prune_decay=None,
                regrow:bool=True
            ):
        
        self.model = model
        self.prune_ratio = prune_ratio
        self.final_density = 1 - self.prune_ratio
        
        self.curr_pr = 0.0
        self.prune_rate_decay = prune_decay

        # mask buffer
        self.masks = {}

        # iterations
        self.steps = 0

        # loader and warmup
        self.iter_per_ep = len(dataloader)
        self.warmup = warmup
        self.final_epoch = final_epoch

        # pruning frequency
        self.prune_freq = prune_freq

        # regrow
        self.regrow = regrow

    @property
    def pr(self):
        return self.curr_pr
    
    @property
    def sparsity(self):
        self.compute_sparsity()
        return self.current_sparsity
    
    def init_schedule(self):
        self.final_step = int((self.final_epoch * self.iter_per_ep) / self.prune_freq)
        self.start_step = int((self.warmup * self.iter_per_ep) / self.prune_freq)
        self.total_step = self.final_step - self.start_step
    
    def _param_stats(self):
        total_params = 0
        spars_params = 0
        for n, m in self.model.named_modules():
            if hasattr(m, "mask"):
                mask = m.mask.data
                total_params += mask.numel()
                spars_params += mask[mask.eq(0)].numel()
        return total_params, spars_params

    def compute_sparsity(self):
        total_params = 0
        ones = 0
        for n, m in self.model.named_modules():
            if hasattr(m, "mask"):
                mask = m.mask.data
                total_params += mask.numel()
                ones += mask.sum()
        self.current_sparsity = 1 - ones / total_params
    
    def register_masks(self):
        for n, m in self.model.named_modules():
            if hasattr(m, "mask"):
                self.masks[n] = m.mask

    def apply_masks(self):
        for n, m in self.model.named_modules():
            if hasattr(m, "mask"):
                m.mask.data.copy_(self.masks[n])

    def get_weight_grad(self, weight:torch.Tensor):
        grad = weight.grad.clone()
        return grad
    
    def step(self):
        # update the current prune_rate / probability
        self.prune_rate_decay.step()
        self.dr = self.prune_rate_decay.get_dr()
        
        # increment
        self.steps += 1

        if self.steps >= int(self.warmup * self.iter_per_ep) and self.steps % self.prune_freq == 0:
            if self.steps != 0:
                self.pruning()
                if self.regrow:
                    self.prune_and_regrow()
    
    def pruning(self):
        """
        Pruning method
        """
        pass

    def prune_and_regrow(self):
        """
        Plasticity of sparsity
        """
        pass