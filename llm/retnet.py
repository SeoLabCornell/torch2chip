"""
Language model
"""

import os
import sys
sys.path.append("../torch2chip/")

import torch
import argparse

from transformers import AutoTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer
from src.stage.base import Execute
from src.trainer.llm.evaluator import WikiText
from src.t2c.convert import RetNet4Compress
from src.trainer.llm.ptq import SmoothQuantRetNet

parser = argparse.ArgumentParser(description='Llama')
parser.add_argument('--config_dir', type=str, default=None, help="Path to the configuration file (.yaml)")
args = parser.parse_args()

class CompressRetNet(Execute):
    def __init__(self, config_dir):
        super().__init__(config_dir)

        self.tokenizer = self.prepare_tokenizer()
        model = self.create_model()

        wbit = self.config["quantization"]["wbit"]
        abit = self.config["quantization"]["abit"]

        converter = RetNet4Compress(model, wbit, abit)
        self.model = converter.convert()

        # define the target task
        self.task = SmoothQuantRetNet(config_dir, self.model, self.tokenizer, self.logger)

    
    def prepare_tokenizer(self):
        tok_type = self.config["model"]["tokenizer"]
        tokenizer = AutoTokenizer.from_pretrained(tok_type, trust_remote_code=True)

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0

        return tokenizer
        
    def ptq(self):
        fake_quantized_model = self.task.run()
        evaluator = WikiText(self.config_dir, fake_quantized_model, self.tokenizer)
        evaluator.run()

        return fake_quantized_model

    def run(self):
        fake_quantized_model = self.ptq()

def starter():
    executor = CompressRetNet(args.config_dir)
    executor.run()

if __name__ == "__main__":
    starter()