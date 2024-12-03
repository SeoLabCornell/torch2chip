"""
Base dataset configuration
"""

import torch
import transformers
import datasets

from src.stage.base import Execute

# language model dataset mapping: path, package, function, separate path flag
LANGUAGE_DATASET_MAP = {
    'wikitext': ('wikitext-2-raw-v1', 'datasets', 'load_dataset', False),
    'boolq': ('json', 'datasets', 'load_dataset', True),
    'openbookqa': ('json', 'datasets', 'load_dataset', True),
    'piqa': ('json', 'datasets', 'load_dataset', True),
    'winogrande': ('json', 'datasets', 'load_dataset', True),
    'commonsense_reasoning': ('json', 'datasets', 'load_dataset', True),
}

class DataStage(Execute):
    """
    Base dataset stage for vision and language datasets
    """
    def __init__(self, config_dir):
        super().__init__(config_dir)

        self.dataset_name = self.config["dataset"]["name"]
        self.data_split = self.config["dataset"]["split"]
        self.batch_size = self.config["train"]["batch_size"]

        # ddp flag
        self.is_ddp = torch.distributed.is_initialized()

    def __len__(self):
        return len(self.dataset)

    def __name__(self):
        return "BaseDataStage"

    def load_dataset(self):
        return []

    def prepare_transform(self):
        pass

    def prepare_loader(self):
        pass

    def run(self):
        self.logger.info(f"Preparing dataset {self.dataset_name}")