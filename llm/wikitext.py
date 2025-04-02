"""
Compression of llama model series
"""

import os
import sys
sys.path.append("../torch2chip/")

import torch
import argparse

from tqdm import tqdm
from collections import OrderedDict
from transformers import AutoTokenizer
from src.stage.base import Execute
from src.trainer.llm.ptq import SmoothQuant
from src.trainer.llm.evaluator import WikiText
from src.t2c.t2c import T2C, get_stats_dict
from src.utils.utils import gpufloat2cpuint
from src.t2c.convert import CONVERTNN
from transformers import set_seed

parser = argparse.ArgumentParser(description='LLM model evaluation against the WikiText benchmark')
parser.add_argument('--config_dir', type=str, default=None, help="Path to the configuration file (.yaml)")
args = parser.parse_args()

class CompressLlama(Execute):
    def __init__(self, config_dir):
        super().__init__(config_dir)
        set_seed(self.config["seed"])
        
        model_type = self.config["model"]["model_type"]
        model = self.create_model()
        self.tokenizer = self.prepare_tokenizer()

        wbit = self.config["quantization"]["wbit"]
        abit = self.config["quantization"]["abit"]
        converter = CONVERTNN[model_type](model, wbit=wbit, abit=abit)

        # convert model
        self.model = converter.convert()
        self.model = self.model.to(self.device)

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

        return fake_quantized_model

    def save(self, t2c:T2C):
        if self.config["t2c"]["save_model"]:
            t2c_model = getattr(t2c, "model")
            torch.save(t2c_model.state_dict(), self.t2c_model_dir)

        x1_stats_dict = OrderedDict()
        x2_stats_dict = OrderedDict()
        out_stats_dict = OrderedDict()
        for k, v in tqdm(t2c.node_dict.items()):
            x1, x2, y = v

            x1 = gpufloat2cpuint(x1, torch.float16)
            x2 = gpufloat2cpuint(x2, torch.float16)
            y = gpufloat2cpuint(y, torch.float32)

            if self.config["t2c"]["save_tensors"]:
                torch.save(x1, os.path.join(self.tensors_dir, f"{k}_x1.pt"))
                torch.save(x2, os.path.join(self.tensors_dir, f"{k}_x2.pt"))
                torch.save(y, os.path.join(self.tensors_dir, f"{k}_y.pt"))
            
            x1stats = get_stats_dict(x1) 
            x2stats = get_stats_dict(x2)
            ystats = get_stats_dict(y)

            x1_stats_dict[k] = x1stats
            x2_stats_dict[k] = x2stats
            out_stats_dict[k] = ystats

        torch.save(out_stats_dict, os.path.join(self.tensors_dir, f"matmul_output_stats.pt"))
        torch.save(x1_stats_dict, os.path.join(self.tensors_dir, f"matmul_x1_stats.pt"))
        torch.save(x2_stats_dict, os.path.join(self.tensors_dir, f"matmul_x2_stats.pt"))

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