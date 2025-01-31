"""
Compression of llama model series
"""

import os
import sys
sys.path.append("../torch2chip/")

import torch
import argparse

from tqdm import tqdm
from transformers import AutoTokenizer
from src.stage.base import Execute
from src.trainer.llm.ptq import SmoothQuant
from src.t2c.convert import Llama4Compress
from src.trainer.llm.evaluator import WikiText
from src.t2c.t2c import T2C
from src.utils.utils import gpufloat2cpuint

parser = argparse.ArgumentParser(description='Llama')
parser.add_argument('--config_dir', type=str, default=None, help="Path to the configuration file (.yaml)")
args = parser.parse_args()

class CompressLlama(Execute):
    def __init__(self, config_dir):
        super().__init__(config_dir)
        model = self.create_model()
        self.tokenizer = self.prepare_tokenizer()

        wbit = self.config["quantization"]["wbit"]
        abit = self.config["quantization"]["abit"]
        converter = Llama4Compress(model, wbit=wbit, abit=abit)

        # convert model
        self.model = converter.convert()

        # define the target task
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
        evaluator = WikiText(self.config_dir, fake_quantized_model, self.tokenizer)
        evaluator.run()

        return fake_quantized_model

    def save(self, t2c:T2C):
        t2c_model = getattr(t2c, "model")
        torch.save(t2c_model.state_dict(), self.t2c_model_dir)

        for k, v in tqdm(t2c.node_dict.items()):
            x1, x2, y = v

            x1 = gpufloat2cpuint(x1, torch.int8)
            x2 = gpufloat2cpuint(x2, torch.int8)
            y = gpufloat2cpuint(y, torch.int32)

            torch.save(x1, os.path.join(self.tensors_dir, f"{k}_x1.pt"))
            torch.save(x2, os.path.join(self.tensors_dir, f"{k}_x2.pt"))
            torch.save(y, os.path.join(self.tensors_dir, f"{k}_y.pt"))


    def t2c(self, fake_quant_model):
        t2c = T2C(model=fake_quant_model, config=self.config)
        fused_model = t2c.fused_model()
        self.print_arch(fused_model, "fused_model")

        evaluator = WikiText(self.config_dir, fused_model, self.tokenizer)
        self.logger.info(f"\n Evaluating the fused model...")
        evaluator.run()

        export_samples = self.config["export"]["export_samples"]
        t2c.register_node()

        if export_samples > 0:
            evaluator.export_run(export_samples)
            self.save(t2c)

        return fused_model

    def run(self):
        fake_quant_model = self.ptq()
        fused_model = self.t2c(fake_quant_model)

def starter():
    executor = CompressLlama(args.config_dir)
    executor.run()

if __name__ == "__main__":
    starter()