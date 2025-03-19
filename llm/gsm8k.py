"""
Math Task (GSM8K, MATH)
"""

import os
import sys
sys.path.append("../torch2chip/")
import argparse

from transformers import AutoTokenizer
from src.stage.base import Execute
from src.trainer.llm.ptq import SmoothQuant
from src.t2c.convert import Llama4Compress
from src.trainer.llm.evaluator import GSM8K
from src.t2c.t2c import T2C

parser = argparse.ArgumentParser(description='LLM model evaluation against the GSM8K benchmark')
parser.add_argument('--config_dir', type=str, default=None, help="Path to the configuration file (.yaml)")
args = parser.parse_args()

class GSM8KEval(Execute):
    def __init__(self, config_dir):
        super().__init__(config_dir)
        model = self.create_model()
        self.tokenizer = self.prepare_tokenizer()

        wbit = self.config["quantization"]["wbit"]
        abit = self.config["quantization"]["abit"]
        converter = Llama4Compress(model, wbit=wbit, abit=abit)

        # convert model
        self.model = converter.convert()
        self.model = self.model.to(self.device)

        # initialize logging
        self.logger = self.initialize_logger()
        self.task = SmoothQuant(config_dir, self.model, self.tokenizer, self.logger)

    def register_run_dir(self):
        super().register_run_dir()
        
        self.t2c_dir = os.path.join(self.run_dir, "t2c")
        self.t2c_model_dir = os.path.join(self.t2c_dir , "t2c_model.pth.tar")

        self.tensors_dir = os.path.join(self.t2c_dir , "tensors")
        if not os.path.isdir(self.tensors_dir):
            os.makedirs(self.tensors_dir, exist_ok=True)

    def prepare_tokenizer(self):
        model_type = self.config["model"]["model_type"]
        tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0

        return tokenizer

    def ptq(self):
        fake_quantized_model = self.task.run()
        return fake_quantized_model

    def t2c(self, fake_quant_model):
        t2c = T2C(model=fake_quant_model, config=self.config)
        fused_model = t2c.fused_model()
        self.print_arch(fused_model, "fused_model")

        evaluator = GSM8K(self.config_dir, fused_model, self.tokenizer)
        evaluator.run()

    def run(self):
        fake_quant_model = self.ptq()
        fused_model = self.t2c(fake_quant_model)

def starter():
    executor = GSM8KEval(args.config_dir)
    executor.run()

if __name__ == "__main__":
    starter()