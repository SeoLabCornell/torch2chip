"""
LLM Evaluation with MMLU
"""

import os
import torch
import sys
sys.path.append("../torch2chip/")
import argparse

from collections import OrderedDict
from transformers import AutoTokenizer, set_seed
from src.stage.base import Execute
from src.trainer.llm.evaluator import MMLU
from src.t2c.convert import Llama4Compress
from src.trainer.llm.ptq import SmoothQuant
from src.t2c.t2c import T2C, get_stats_dict
from src.utils.utils import gpufloat2cpuint
from tqdm import tqdm

parser = argparse.ArgumentParser(description='LLM model evaluation against the MMLU benchmark')
parser.add_argument('--config_dir', type=str, default=None, help="Path to the configuration file (.yaml)")
args = parser.parse_args()

class MMLUEval(Execute):
    def __init__(self, config_dir):
        super().__init__(config_dir)
        set_seed(self.config["seed"])
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
        tokenizer = AutoTokenizer.from_pretrained(model_type, padding_side="left")

        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1

        return tokenizer
    
    def t2c(self, fake_quant_model):
        t2c = T2C(model=fake_quant_model, config=self.config)
        fused_model = t2c.fused_model()
        self.print_arch(fused_model, "fused_model")
        
        export_samples = self.config["export"]["export_samples"]
        if export_samples > 0:
            t2c.register_node()

        evaluator = MMLU(self.config_dir, fused_model, self.tokenizer)
        evaluator.run()

        if export_samples > 0:
            self.save(t2c)
    
    def ptq(self):
        fake_quantized_model = self.task.run()
        return fake_quantized_model

    def save(self, t2c:T2C):
        if self.config["t2c"]["save_model"]:
            t2c_model = getattr(t2c, "model")
            torch.save(t2c_model.state_dict(), self.t2c_model_dir)

        # precision
        xdytpe = torch.int8 if self.config["t2c"]["xdtype"] <= 8 else torch.float16
        ydytpe = torch.int32 if self.config["t2c"]["ydtype"] < 32 else torch.float16

        x1_stats_dict = OrderedDict()
        x2_stats_dict = OrderedDict()
        out_stats_dict = OrderedDict()
        for k, v in tqdm(t2c.node_dict.items()):
            x1, x2, y = v

            x1 = gpufloat2cpuint(x1, xdytpe)
            x2 = gpufloat2cpuint(x2, xdytpe)
            y = gpufloat2cpuint(y, ydytpe)

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
    
    def run(self):
        fake_quant_model = self.ptq()
        fused_model = self.t2c(fake_quant_model)

def starter():
    executor = MMLUEval(args.config_dir)
    executor.run()

if __name__ == "__main__":
    starter()    
