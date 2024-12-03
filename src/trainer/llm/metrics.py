"""
Metric for different llm tasks
"""

import torch

class Metric(object):
    def __init__(self):
        pass

class Perplexity(Metric):
    def __init__(self, chunk_size:int=2048, n_samples:int=40):
        super().__init__()

        self.loss = []
        self.chunk_size = chunk_size
        self.n_samples = n_samples

    def func(self, pred:torch.Tensor, target:torch.Tensor):
        target = target.long()

        loss_fn = torch.nn.CrossEntropyLoss()
        loss_val = loss_fn(pred.view(-1, pred.size(-1)), target.view(-1))
        return loss_val
    
    def update(self, pred:torch.Tensor, target:torch.Tensor):
        loss_val = self.func(pred, target)
        neg_log_likelihood = loss_val.float() * self.chunk_size
        self.loss.append(neg_log_likelihood)

    def reduce(self):
        ppl = torch.exp(torch.stack(self.loss).sum() / (self.n_samples * self.chunk_size))
        return ppl
