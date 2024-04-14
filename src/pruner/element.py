"""
Element-wise pruning

Example method: Sparse Training via Boosting Pruning Plasticity with Neuroregeneration
https://arxiv.org/abs/2106.10404
"""

import math
import torch
import numpy as np
from torch.nn.modules import Module
from src.pruner.base import Pruner, CosineDecay

class ElementPrune(Pruner):
    def __init__(self, 
                model: Module, 
                prune_ratio: float, 
                warmup: int, 
                final_epoch: int, 
                dataloader, 
                prune_freq: float, 
                prune_decay = None, 
                regrow: bool = True,
                init_sparsity: float = 0.0
            ):
        super().__init__(model, prune_ratio, warmup, final_epoch, dataloader, prune_freq, prune_decay, regrow)
        self.init_sparsity = init_sparsity
        self.init_density = 1 - self.init_sparsity
        
        self.init_schedule()
        
        if self.init_density < 1.0:
            self.erk(self.init_density)

    def layer_stats(self):
        self.name2nonzeros = {}
        self.name2zeros = {}

        for name, mask in self.masks.items():
            self.name2nonzeros[name] = mask.sum().item()
            self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]


    def erk(self, density:float, erk_power_scale:float=1.0):
        print('initialize by ERK')
        self.total_params, _ = self._param_stats()

        is_epsilon_valid = False
        dense_layers = set()
        while not is_epsilon_valid:
            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, mask in self.masks.items():
                n_param = mask.numel()
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                    rhs -= n_zeros

                else:
                    # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                    # equation above.
                    rhs += n_ones
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    raw_probabilities[name] = (np.sum(list(mask.size())) / mask.numel()) ** erk_power_scale
                    # Note that raw_probabilities[mask] * n_param gives the individual
                    # elements of the divisor.
                    divisor += raw_probabilities[name] * n_param
            # By multipliying individual probabilites with epsilon, we should get the
            # number of parameters per layer correctly.
            epsilon = rhs / divisor
            # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
            # mask to 0., so they become part of dense_layers sets.
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        density_dict = {}
        total_nonzero = 0.0

        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, mask in self.masks.items():
            n_param = mask.numel()
            if name in dense_layers:
                density_dict[name] = 1.0
            else:
                probability_one = epsilon * raw_probabilities[name]
                density_dict[name] = probability_one
            print(
                f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
            )
            self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

            total_nonzero += density_dict[name] * mask.numel()
        print(f"Overall sparsity {1 - total_nonzero / self.total_params}")
    
    def register_masks(self):
        super().register_masks()
        
        # initialize the mask by ERK
        self.init_mask(self.init_density)

        # apply the mask
        self.apply_masks()

    def collect_score(self):
        mp_score = []
        for n, m in self.model.named_modules():
            if hasattr(m, "mask"):
                mp_score.append(m.weight.data.abs())
        mp_scores = torch.cat([torch.flatten(x) for x in mp_score])
        return mp_scores
    
    def update_mask(self, threshold):
        for n, m in self.model.named_modules():
            if hasattr(m, "mask"):
                self.masks[n] = m.weight.abs().gt(threshold).float()

    def update_schedule(self):
        ramping_decay = (1 - ((self.current_step - self.start_step) / self.total_step)) ** 3
        curr_prune_rate = (1 - self.init_density) + (self.init_density - self.final_density) * (1 - ramping_decay)
        return curr_prune_rate
    
    def get_threshold(self):
        # get scores
        mp_scores = self.collect_score()
        num_params_to_keep = int(len(mp_scores) * (1 - self.curr_pr))
        topkscore, _ = torch.topk(mp_scores, num_params_to_keep, sorted=True)
        return topkscore[-1]
    
    def pruning(self):
        self.current_step = int(self.steps / self.prune_freq)
        if self.current_step >= self.start_step and self.current_step < self.final_step:
            # update the pruning rate with the defined schedule
            self.curr_pr = self.update_schedule()

            # compute the score threshold based on the current pruning rate
            threshold = self.get_threshold()

            # update and apply the masks
            self.update_mask(threshold)
            self.apply_masks()

        # update sparsity
        self.compute_sparsity()

    def gradient_regrow(self, weight:torch.Tensor, mask:torch.Tensor, ngrow):
        grad = self.get_weight_grad(weight)
        grad = grad.mul(mask.eq(0.).float())

        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        mask.data.view(-1)[idx[:ngrow]] = 1.0
        return mask

    def magnitude_death(self, weight:torch.Tensor, name:str):
        num_remove = math.ceil(self.dr * self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)
        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        threshold = x[k-1].item()
        return (torch.abs(weight.data) > threshold)

    def prune_and_regrow(self):
        # layer statistics
        self.layer_stats()

        for n, m in self.model.named_modules():
            if hasattr(m, "mask"):
                weight = m.weight

                # prune additional parameters
                pr_mask = self.magnitude_death(weight, n)

                # regrow
                gr_count = int(self.name2nonzeros[n] - pr_mask.sum().item())
                gr_mask = self.gradient_regrow(weight, pr_mask, gr_count)
                
                # update mask
                self.masks[n] = gr_mask

        # update sparsity
        self.compute_sparsity()
        print(f"Pruning Progress: {self.current_step - self.start_step} / {self.total_step} ; Sparsity = {self.sparsity}")