"""
HuggingFace Stage
"""

from src.stage.base import Execute
from transformers import AutoTokenizer

class HFExecute(Execute):
    def __init__(self, config_dir):
        super().__init__(config_dir)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["model_type"])

    def commonsense_qa(self):
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left"

    def gsm8k(self):
        if self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.tokenizer.pad_token_id = 0
