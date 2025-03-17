"""
Post-training quantization of language models
"""
import os
import torch

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from src.module.base import _QBaseLinear, _QBase
from src.module.attention import QLlamaAttention
from src.module.mlp import QLlamaMLP
from src.trainer.llm.metrics import Perplexity
from src.data.llm.hf import PileSubset
from src.stage.base import Execute
from src.t2c.convert import get_parent_name
from src.quantization.smoothquant import SmoothQuantizer, SmoothQuantChannelWiseWeightQuantizer, SmoothQuantTokenWiseQuantizer, SmoothQuantChannelWeight4Bit, SmoothQuantMXINTChannelWise
from src.quantization.mxint import MXChannelWiseWeightQuantizer
from tqdm import tqdm
from typing import Dict
from collections import OrderedDict

WEIGHT_QUANTIZER_MAP = {
    "smooth_quant": SmoothQuantChannelWiseWeightQuantizer,
    "smooth_quant_4bit": SmoothQuantChannelWeight4Bit,
    "smooth_quant_mxint": SmoothQuantMXINTChannelWise,
    "mxint_quant": MXChannelWiseWeightQuantizer,
    "identity": _QBase
}

INPUT_QUANTIZER_MAP = {
    "smooth_quant": SmoothQuantizer,
    "smooth_quant_token": SmoothQuantTokenWiseQuantizer,
    "identity": _QBase
}


class MinMaxHook:
    def __init__(self) -> None:
        self.max = None
        self.min = None

    def __call__(self, module, input_batch, output_batch):
        hidden_dim = input_batch[0].size(-1)
        tensor = input_batch[0].view(-1, hidden_dim).abs().detach()

        curr_max = torch.max(tensor, dim=0)[0].float()
        curr_min = torch.min(tensor, dim=0)[0].float()
        
        if self.max is None:
            self.max = curr_max
        else:
            self.max = torch.max(curr_max, self.max)

        if self.min is None:
            self.min = curr_min
        else:
            self.min = torch.min(curr_min, self.min)


class LMPTQ(Execute):
    def __init__(self, config_dir, model, tokenizer, logger):
        super().__init__(config_dir)
        self.model = model
        self.tokenizer = tokenizer
        
        # logging
        self.logger = logger

        # method
        self.wqtype = self.config["quantization"]["wqtype"]
        self.xqtype = self.config["quantization"]["xqtype"]

        # precision
        self.wbit = self.config["quantization"]["wbit"]
        self.abit = self.config["quantization"]["abit"]

        # subset of the Pile dataset
        self.datastage = PileSubset(config_dir)
        trainset, _ = self.datastage.run()

        self.calib_set = self.tokenize(trainset)
        print(f"Length of Calibration Set = {len(self.calib_set)}")

        self.metric = Perplexity(self.datastage.batch_size, 512)

    def tokenize(self, trainset):
        """
        Partially adopted from AWQ of MIT-HAN-Lab: 
        https://github.com/mit-han-lab/llm-awq
        """

        calib_set = []
        batch_size = self.datastage.batch_size
        calib_samples = self.config["quantization"]["num_samples"]

        for idx, data in enumerate(trainset):
            line = data["text"]
            line = line.strip()
            
            line_enc = self.tokenizer.encode(line)
            
            if len(line_enc) > 512:
                continue

            sample = torch.tensor([line_enc])
            calib_set.append(sample)
            
            if idx + 1 > calib_samples:
                break

        # split into batches
        cat_samples = torch.cat(calib_set, dim=1)
        n_split = calib_samples // batch_size
        dataset = [cat_samples[:, i * batch_size : (i+1) * batch_size] for i in range(n_split)]
        return dataset

    def forward_step(self, batch):
        logits = self.model(batch).logits
        return logits
    
    def inject_quantizers(self):
        modules = dict(self.model.named_modules(remove_duplicate=True))

        for n, m in self.model.named_modules():
            if isinstance(m, _QBaseLinear):
                parent_name, name = get_parent_name(n)

                weight_quantizer = WEIGHT_QUANTIZER_MAP[self.wqtype](nbit=self.wbit, train_flag=True, unsigned=False, num_channels=1).to(torch.float16)
                input_quantizer = INPUT_QUANTIZER_MAP[self.xqtype](nbit=self.abit, train_flag=True, unsigned=False).to(torch.float16)

                setattr(m, "wq", weight_quantizer.to(self.device))
                setattr(m, "aq", input_quantizer.to(self.device))

                parent_name, name = get_parent_name(n)
                setattr(modules[parent_name], name, m)

    @torch.no_grad
    def forward_model(self):
        for batch in tqdm(self.calib_set):
            batch = batch.to(self.device)

            shift_labels = batch[:, 1:]
            logits = self.forward_step(batch)

            shift_logits = logits[:, :-1, :].contiguous().float()
            self.metric.update(shift_logits, shift_labels)

        ppl = self.metric.reduce()
        self.logger.info(f"[Sanity Check] Perpleixty on the Calib Set = {ppl.item():.3f}")

    def run(self):
        self.logger.info(f"PTQ Start!")
        self.model.eval()

class SmoothQuant(LMPTQ):
    def __init__(self, config_dir, model, tokenizer, logger):
        super().__init__(config_dir, model, tokenizer, logger)
        self.alpha = self.config["smooth"]["alpha"]
        self.smooth = self.config["smooth"].get("flag", True)
        print(f"Smooth Quant: alpha = {self.alpha}")

    def smooth_factor(self, xmax, wmax):
        scale = (xmax.pow(self.alpha) / wmax.pow(1 - self.alpha)).clamp(1e-5)
        return scale.to(torch.float16)

    def profile_smooth(self):
        hooks = {}
        handles = {}
        act_scales = OrderedDict()

        for n, m in self.model.named_modules():
            if isinstance(m, _QBaseLinear):
                hook = MinMaxHook()
                handle = m.register_forward_hook(hook)

                hooks[n] = hook
                handles[n] = handle

        # forward pass
        self.forward_model()

        # compute smooth factor
        for n, m in hooks.items():
            if not "down_proj" in n:
                xmax = hooks[n].max
                if self.smooth:
                    act_scales[n] = xmax
                else:    
                    act_scales[n] = torch.ones_like(xmax)
                
                handles[n].remove()

        del handles
        del hooks

        # save the smooth factor
        smooth_path = os.path.join(self.config["save"]["run_dir"], "act_scales.pt")
        torch.save(act_scales, smooth_path)

        return act_scales

    @torch.no_grad()
    def smooth_fcs_llama_like(self, fcs, act_scales, alpha=0.5):
        if not isinstance(fcs, list):
            fcs = [fcs]

        for fc in fcs:
            assert isinstance(fc, _QBaseLinear)
            fc.wq = WEIGHT_QUANTIZER_MAP[self.wqtype](nbit=self.wbit, train_flag=True, unsigned=False).to(torch.float16)
            fc.aq = INPUT_QUANTIZER_MAP[self.xqtype](nbit=self.abit, train_flag=True, unsigned=False).to(torch.float16)

        device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
        act_scales = act_scales.to(device=device, dtype=dtype)

        weight_scales = torch.cat(
            [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
        )
        weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
        scales = (
            (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
            .clamp(min=1e-5)
            .to(device)
            .to(dtype)
        )

        for fc in fcs:
            fc.weight.mul_(scales.view(1, -1))
    
        return scales

    def llama_attn(self, module:QLlamaAttention, name:str, scales:Dict):
        qkv = [module.q_proj, module.k_proj, module.v_proj]
        qkv_input_scales = scales[name + ".self_attn.q_proj"]
        
        qkv_smooth_scale = self.smooth_fcs_llama_like(qkv, qkv_input_scales, self.alpha)

        # special case for the down_proj
        module.o_proj.wq = WEIGHT_QUANTIZER_MAP[self.wqtype](nbit=self.wbit, train_flag=True, unsigned=False).to(self.device)
        module.o_proj.aq = INPUT_QUANTIZER_MAP[self.xqtype](nbit=self.abit, train_flag=True, unsigned=False).to(self.device)
        return qkv_smooth_scale

    def llama_mlp(self, module:QLlamaMLP, name:str, scales:Dict):
        fcs = [module.gate_proj, module.up_proj]
        fcs_input_scales = scales[name + ".mlp.gate_proj"]
        mlp_smooth_scale = self.smooth_fcs_llama_like(fcs, fcs_input_scales, self.alpha)

        # special case for the down_proj
        module.down_proj.wq = WEIGHT_QUANTIZER_MAP[self.wqtype](nbit=self.wbit, train_flag=True, unsigned=False).to(self.device)
        module.down_proj.aq = INPUT_QUANTIZER_MAP[self.xqtype](nbit=self.abit, train_flag=True, unsigned=False).to(self.device)
        return mlp_smooth_scale

    @torch.no_grad
    def inject_quantizers(self, act_scales:Dict):
        for n, m in self.model.named_modules():
            if isinstance(m, LlamaDecoderLayer):
                qkv_smooth_scales = self.llama_attn(m.self_attn, n, act_scales)
                mlp_smooth_scale = self.llama_mlp(m.mlp, n, act_scales)

                m.input_layernorm.weight.div_(qkv_smooth_scales)
                m.post_attention_layernorm.weight.div_(mlp_smooth_scale)

    def run(self):
        super().run()
        self.model.eval()

        # run through the forward step
        act_scales = self.profile_smooth()

        # add quantizers to the model
        self.inject_quantizers(act_scales)
        return self.model

class SmoothQuantRetNet(SmoothQuant):
    def __init__(self, config_dir, model, tokenizer, logger):
        super().__init__(config_dir, model, tokenizer, logger)

    @torch.no_grad
    def forward_model(self):
        for batch in tqdm(self.calib_set):
            batch = batch.to(self.device)

            shift_labels = batch[:, 1:]
            logits = self.forward_step(batch)

            shift_logits = logits[:, :-1, :].contiguous()
            self.metric.update(shift_logits, shift_labels)

        ppl = self.metric.reduce()
        self.logger.info(f"[Sanity Check] Perpleixty on the Calib Set = {ppl.item():.3f}")

    def forward_step(self, batch:torch.Tensor):
        logits = self.model(batch.long()).logits
        return logits
    
    def run(self):
        super().run()
        self.model.eval()

        # run through the forward step
        act_scales = self.profile_smooth()