"""
Pruning with Structured Fine-grained Sparsity
"""

import math
import torch
import torch.nn as nn
from torch.nn.modules import Module
from src.pruner.base import Pruner

class NMPruner(Pruner):
    def __init__(self, 
                model: Module, 
                prune_ratio: float, 
                warmup: int, 
                final_epoch: int, 
                dataloader, 
                prune_freq: float, 
                prune_decay=None, 
                regrow: bool = True,
                M: int = 4,
                N: int = 2,
                init_sparsity: float = 0.0
            ):
        super().__init__(model, prune_ratio, warmup, final_epoch, dataloader, prune_freq, prune_decay, regrow)
        # group size
        self.M = M
        self.N = N

        # pruning probability
        self.init_sparsity = init_sparsity
        self.init_density = 1 - self.init_sparsity
        self.init_schedule()
    
    def get_groups(self, tensor:torch.Tensor):
        length = tensor.numel()
        group = int(length / self.M)
        return group
    
    def layer_grp_stats(self):
        self.name2nzgrp = {}
        self.name2zerogrp = {}

        for name, mask in self.masks.items():
            ngroups = self.get_groups(mask)

            if len(mask.size()) == 4:
                m = mask.permute(0,2,3,1).reshape(ngroups, int(self.M))
            elif len(mask.size()) == 2:
                m = mask.reshape(ngroups, int(self.M))

            gsum = m.sum(dim=1)
            nzgrp = gsum[gsum.eq(self.M)].numel()

            # sparse and non sparse groups
            self.name2nzgrp[name] = nzgrp
            self.name2zerogrp[name] = ngroups - nzgrp
    
    def update_mask(self, prob):
        for n, m in self.model.named_modules():
            if hasattr(m, "mask"):
                weight = m.weight.clone()

                # number of groups
                ngroups = self.get_groups(weight)

                if isinstance(m, nn.Conv2d):
                    w = weight.detach().abs().permute(0,2,3,1).reshape(ngroups, int(self.M))
                elif isinstance(m, nn.Linear):
                    w = weight.detach().abs().reshape(ngroups, int(self.M))

                # generate the binary masks
                index = torch.argsort(w, dim=1)[:, :int(self.M-self.N)]
                w_b = torch.ones(w.shape, device=w.device)
                w_b = w_b.scatter_(dim=1, index=index, value=0)

                # probability sampling
                if prob < 1.0:
                    rnumel = math.ceil((1-prob)*ngroups)
                    ridx = torch.randperm(rnumel)
                    w_b[ridx] = 1.

                # reshape the mask back
                if isinstance(m, nn.Conv2d):
                    w_b = w_b.reshape(weight.permute(0,2,3,1).shape)
                    w_b = w_b.permute(0,3,1,2)
                elif isinstance(m, nn.Linear):
                    w_b = w_b.reshape(weight.shape)

                # update the mask
                self.masks[n] = w_b

    def group_death(self, weight:torch.Tensor, name, module):
        ngroups = self.get_groups(weight)

        # number of groups with further death
        nremove = int(self.dr * self.name2nzgrp[name])

        if nremove == 0:
            w_b = weight.data.ne(0.).float()
        else:
            nzeros = self.name2zerogrp[name]
            
            # total number of sparse groups
            k = int(nzeros + nremove)

            if isinstance(module, nn.Conv2d):
                w = weight.detach().abs().permute(0,2,3,1).reshape(ngroups, int(self.M))
            elif isinstance(module, nn.Linear):
                w = weight.detach().abs().reshape(ngroups, int(self.M))

            index = torch.argsort(w, dim=1)[:, :int(self.M-self.N)]

            # initialize the mask
            w_b = torch.ones(w.shape, device=weight.device)
            w_b = w_b.scatter_(dim=1, index=index, value=0) # fill all the groups with N:M sparsity

            # group-wise sum
            wgsum = torch.sum(w_b.abs(), dim=1)
            y, idx = torch.sort(torch.abs(wgsum).flatten())
            w_b[idx[:(wgsum.size(0)-k)]] = 1.

            # reshape the mask back
            if isinstance(module, nn.Conv2d):
                w_b = w_b.reshape(weight.permute(0,2,3,1).shape)
                w_b = w_b.permute(0,3,1,2)
            elif isinstance(module, nn.Linear):
                w_b = w_b.reshape(weight.shape)

        return w_b, nremove

    def grp_grad_regrow(self, weight, mask, ngrow):
        grad = self.get_weight_grad(weight)
        ngroups = self.get_groups(grad)

        if len(mask.size()) == 4:
            ggrad = grad.abs().permute(0,2,3,1).reshape(ngroups, int(self.M))
            m = mask.permute(0,2,3,1).reshape(ngroups, int(self.M))
        elif len(mask.size()) == 2:
            ggrad = grad.abs().reshape(ngroups, int(self.M))
            m = mask.reshape(ngroups, int(self.M))

        # only grow the weights within the current sparsity
        msum = torch.sum(m, dim=1)
        sidx = msum.eq(self.N).float()

        ggrad = ggrad*sidx[:, None]
        gsum = torch.sum(ggrad, dim=1)
        y, idx = torch.sort(gsum.flatten(), descending=True)

        # regrow
        m[idx[:ngrow]] = 1.0
        msum = torch.sum(m, dim=1)

        # reshape
        if len(mask.size())==4:
            rgmask = m.reshape(mask.permute(0,2,3,1).shape)
            rgmask = rgmask.permute(0,3,1,2)
        elif len(mask.size())==2:
            rgmask = m.reshape(mask.shape)

        return rgmask, msum[msum.eq(self.N)].numel()
    
    def update_schedule(self):
        ramping_decay = (1 - ((self.current_step - self.start_step) / self.total_step)) ** 3
        curr_prune_rate = (1 - self.init_density) + (self.init_density - self.final_density) * (1 - ramping_decay)
        return curr_prune_rate

    def pruning(self):
        self.current_step = int(self.steps / self.prune_freq)
        if self.current_step >= self.start_step and self.current_step < self.final_step:
            # update the pruning probability
            self.curr_pr = self.update_schedule()

            # update the N:M mask
            self.update_mask(self.curr_pr)
            self.apply_masks()
    
    def prune_and_regrow(self):
        # update the stats
        self.layer_grp_stats()

        # prune
        for n, m in self.model.named_modules():
            if hasattr(m, "mask"):
                weight = m.weight

                # prune additional groups
                pr_mask, nremove = self.group_death(weight, n, m)

                # regrow
                gr_mask, remained = self.grp_grad_regrow(weight, pr_mask, nremove)
                
                # apply mask
                if nremove > 0:
                    m.mask = gr_mask.clone()
                    # record mask
                    self.masks[n] = gr_mask

        self.apply_masks()
        # update sparsity
        self.compute_sparsity()
        print(f"Pruning Progress: {self.current_step - self.start_step} / {self.total_step} ; Sparsity = {self.sparsity}")
