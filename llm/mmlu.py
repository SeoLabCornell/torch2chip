"""
LLM Evaluation with MMLU
"""

import os
import sys
sys.path.append("../torch2chip/")

import argparse

from transformers import AutoTokenizer
from src.stage.base import Execute
from src.trainer.llm.evaluator import MMLU

parser = argparse.ArgumentParser(description='Llama')
parser.add_argument('--config_dir', type=str, default=None, help="Path to the configuration file (.yaml)")
args = parser.parse_args()

class MMLUEval(Execute):
    def __init__(self, config_dir):
        super().__init__(config_dir)
        self.model = self.create_model()
        self.tokenizer = self.prepare_tokenizer()

        self.task = MMLU(config_dir, self.model, self.tokenizer)
    
    def register_run_dir(self):
        super().register_run_dir()
        
        self.t2c_dir = os.path.join(self.run_dir, "t2c")
        self.t2c_model_dir = os.path.join(self.t2c_dir , "t2c_model.pth.tar")

        self.tensors_dir = os.path.join(self.t2c_dir , "tensors")
        if not os.path.isdir(self.tensors_dir):
            os.makedirs(self.tensors_dir, exist_ok=True)

    def prepare_tokenizer(self):
        model_type = self.config["model"]["model_type"]
        tokenizer = AutoTokenizer.from_pretrained(model_type, padding_side="left")

        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1

        return tokenizer
    
    def run(self):
        self.task.run()

def starter():
    executor = MMLUEval(args.config_dir)
    executor.run()

if __name__ == "__main__":
    starter()    
