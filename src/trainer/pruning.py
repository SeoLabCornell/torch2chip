"""
Sparse Trainer
"""

from torch.nn.modules import Module
from src.trainer.base import Trainer
from src.pruner.base import CosineDecay
from src.pruner.element import ElementPrune
from src.pruner.nm import NMPruner

PRUNER = {
    "element": ElementPrune,
    "nm": NMPruner
}

class STrainer(Trainer):
    def __init__(self, model: Module, loss_type: str, trainloader, validloader, args, logger):
        super().__init__(model, loss_type, trainloader, validloader, args, logger)

        # decay
        pr_decay = CosineDecay(args.drate, T_max=int(len(trainloader)*args.epochs))

        # pruner
        self.pruner = PRUNER[str(self.args.pruner)](
            model=model,
            prune_ratio=self.args.prune_ratio,
            warmup=self.args.swarmup,
            final_epoch=self.args.final_epoch,
            dataloader=trainloader,
            prune_freq=self.args.prune_freq,
            prune_decay=pr_decay,
        )

    def train_step(self, inputs, target):
        out, loss = super().train_step(inputs, target)
        self.pruner.step()
        return out, loss
    
    def train_epoch(self):
        super().train_epoch()
        self.logger_dict["sparsity"] = self.pruner.sparsity
        self.logger_dict["pr"] = self.pruner.pr
        self.logger_dict["dr"] = self.pruner.dr
