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
    def __init__(self, model: Module, trainloader, validloader, config, logger):
        super().__init__(model, trainloader, validloader, config, logger)

        prune_config = self.config["prune"]
        pruner = prune_config["type"]

        # decay
        pr_decay = CosineDecay(prune_config["drate"], T_max=int(len(trainloader)*self.epochs))

        # pruner
        self.pruner = PRUNER[str(pruner)](
            model=model,
            prune_ratio=prune_config["prune_ratio"],
            warmup=prune_config["warmup"],
            final_epoch=prune_config["final_epoch"],
            dataloader=trainloader,
            prune_freq=prune_config["prune_freq"],
            prune_decay=pr_decay,
        )

        if str(pruner) == "nm":
            self.pruner.M = prune_config.get("M", 4)
            self.pruner.N = prune_config.get("N", 2)

    def train_step(self, inputs, target):
        out, loss = super().train_step(inputs, target)
        self.pruner.step()
        return out, loss

    def train_epoch(self):
        super().train_epoch()
        self.logger_dict["sparsity"] = self.pruner.sparsity
        self.logger_dict["pr"] = self.pruner.pr
        self.logger_dict["dr"] = self.pruner.dr
